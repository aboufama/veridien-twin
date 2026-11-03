// Where the twin page's tips, camera images and white balance come from (the seam of docs/WEB_TWIN.md §21):
//   'server'   the multi-camera preview (veridien multicam preview, http://127.0.0.1:8833): today's proven path, unchanged
//              (rig_link.js RigFeed, real_cams.js /api/frame, calib_embed.js's camera-page iframe and /api/wb, /api/record).
//   'browser'  no server at all: this page opens the cameras, runs TipNet + the RCM solve + the filters itself
//              (twinpage/browser_source.js -> tracker_worker.js -> web/twin/tracker/), and feeds the same consumers
//              the identical /api/tips payload through a LocalFeed, camera views through a FramesRing.
// ?tracker=server|browser|auto picks it; ?replay=<session>|synthetic means 'browser' (a replay never reaches a server).
// The defaults (ADR-147, amended):
//   the local twin server (veridien webtwin serve; twin.html has no veridien-tracker meta): 'server', exactly as before
//     (the camera-page iframe mounts at once and the feed reconnects whenever the server comes back);
//   the static site (scripts/build_twin_site.py writes <meta name="veridien-tracker" content="auto">): 'auto' probes the
//     CV server once at boot; it answers -> 'server' (today's hosted behaviour, d0b750c), else 'browser'. While the
//     browser tracker has no cameras open, watchServer keeps asking and the page reloads onto the server once it answers
//     (never two capture owners: a page whose own cameras run keeps them).
// Pure parts are node-tested (web/twin/tests/browser_source.test.mjs).

import { RigFeed, RIG_DEFAULT_URL, rigUrlFromQuery } from './rig_link.js';

/** The FeedHub key of the in-page tracker's feed (never fetched). */
export const LOCAL_BASE = 'local://tracker';

const MODES = ['auto', 'server', 'browser'];

/**
 * The mode the page asks for: ?tracker= wins; else ?replay= means 'browser'; else ?rig=<url> (an explicit server) means
 * 'server'; else the site's default (twin.html's veridien-tracker meta), else 'server' (the local twin server's page).
 */
export function trackerModeFromQuery(q, siteDefault = null) {
  const v = (q.get('tracker') || '').toLowerCase();
  if (MODES.includes(v)) return v;
  if (v === 'local' || v === 'page') return 'browser';
  if (q.get('replay')) return 'browser';
  const rig = q.get('rig');
  if (rig && !['0', '1', 'true', 'false', ''].includes(rig)) return 'server';
  return MODES.includes(siteDefault) ? siteDefault : 'server';
}

/** auto -> 'server' when the server answered at boot, else 'browser'; the others as asked. */
export function resolveTrackerMode({ asked, serverUp }) {
  if (asked === 'server' || asked === 'browser') return asked;
  return serverUp ? 'server' : 'browser';
}

/** Does the CV server answer? One GET of its tips (the cheapest endpoint every version has), with a timeout. */
export async function serverAnswers(base, { fetch: f = globalThis.fetch?.bind(globalThis), timeoutMs = 1500 } = {}) {
  if (!f) return false;
  const ac = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const to = setTimeout(() => ac?.abort(), timeoutMs);
  try {
    const r = await f(`${String(base).replace(/\/+$/, '')}/api/tips`, { cache: 'no-store', mode: 'cors', signal: ac?.signal });
    if (!r.ok) return false;
    const j = await r.json();
    return !!j && Array.isArray(j.instruments);
  } catch { return false; } finally { clearTimeout(to); }
}

/**
 * Keep asking whether the CV server answers (an 'auto' page that fell back to the browser tracker). opts: {when: () =>
 * bool (ask only while true: the page's own cameras are off), onUp: () => void (called once), everyMs (first gap; it
 * grows x1.5 to maxMs), fetch, timeoutMs, setTimeout}. -> stop().
 */
export function watchServer(base, { when = () => true, onUp, everyMs = 3000, maxMs = 15000, fetch: f = globalThis.fetch?.bind(globalThis),
  timeoutMs = 1500, setTimeout: st = globalThis.setTimeout, clearTimeout: ct = globalThis.clearTimeout } = {}) {
  let stopped = false, timer = null, gap = everyMs;
  const tick = async () => {
    timer = null;
    if (stopped) return;
    if (when() && await serverAnswers(base, { fetch: f, timeoutMs }) && !stopped && when()) { stopped = true; onUp?.(); return; }
    gap = Math.min(maxMs, gap * 1.5);
    if (!stopped) timer = st(tick, gap);
  };
  timer = st(tick, gap);
  return () => { stopped = true; if (timer != null) ct(timer); };
}

/**
 * LocalFeed: a RigFeed fed by push(payload) instead of a network transport. Everything after the transport (dedup by
 * frame_id|t, out-of-order drop, rate, status, listeners) is RigFeed's own code. FeedHub's stop() (page hidden) only
 * gates delivery: the tracker keeps running in its worker.
 */
export class LocalFeed extends RigFeed {
  constructor(base = LOCAL_BASE, opts = {}) { super(base, { ...opts, fetch: null, EventSource: null, stream: false }); this.label = 'Browser tracker'; }
  host() { return 'this page'; }
  transport() { return this.running ? 'local' : 'off'; }
  start() { if (this.running) return; this.running = true; this.startedAt = this.now(); }
  stop() { this.running = false; }
  /** One tracker result (the /api/tips shape). Returns whether it was new. */
  push(j) {
    if (!this.running || !j || !Array.isArray(j.instruments)) { if (j && !Array.isArray(j.instruments)) this.stats.bad++; return false; }
    this.stats.stream_msgs++;
    return this._accept(j);
  }
}

/**
 * FramesRing: the camera views of the last n frame sets, by frame_id (the server keeps 16: FRAME_HIST). Each entry
 * {views: {CAMk: bitmap}, size: {CAMk: [w, h]}}; a bitmap is closed when its set leaves the ring.
 * get(cam, id) -> {bitmap, id, size, exact} (that set, else the newest with that camera), or null.
 */
export class FramesRing {
  constructor(n = 8) { this.n = n; this.map = new Map(); this.newest = null; this.listeners = new Set(); }
  put(id, views, size = {}) {
    const k = String(id);
    const old = this.map.get(k);
    if (old) this._close(old);
    this.map.set(k, { views, size }); this.newest = k;
    while (this.map.size > this.n) { const first = this.map.keys().next().value; this._close(this.map.get(first)); this.map.delete(first); }
    for (const fn of this.listeners) fn(k);
  }
  _close(e) { for (const b of Object.values(e?.views || {})) try { b?.close?.(); } catch { /* closed */ } }
  get(cam, id = null) {
    const want = id != null ? this.map.get(String(id)) : null;
    if (want?.views?.[cam]) return { bitmap: want.views[cam], id: String(id), size: want.size?.[cam] ?? null, exact: true };
    for (const k of [...this.map.keys()].reverse()) {
      const e = this.map.get(k);
      if (e.views?.[cam]) return { bitmap: e.views[cam], id: k, size: e.size?.[cam] ?? null, exact: false };
    }
    return null;
  }
  cams() { const e = this.newest != null ? this.map.get(this.newest) : null; return e ? Object.keys(e.views) : []; }
  onPut(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  clear() {
    for (const e of this.map.values()) this._close(e);
    this.map.clear(); this.newest = null;
    for (const fn of this.listeners) fn(null);
  }
}

/** rig.json (scripts/export_rig_json.py) in the shape of GET /api/calibration (preview.py twin_calibration). */
export function calibrationFromRig(rig) {
  if (!rig) return null;
  return {
    type: 'calibration',
    source: 'rig.json',
    cameras: (rig.cameras || []).map((c) => ({ name: c.name, size: c.size, lens: c.lens, T_D_C: c.T_D_C })),
    ports: (rig.ports || []).map((p) => ({ id: p.name, pivot_cad_D: p.pivot_cad_D, pivot_D: p.pivot_D, pivot_fit: p.pivot_fit })),
    placement: { residuals: rig.align_to_design ?? [] },
  };
}

/**
 * The server source: today's code, unchanged. base is the rig feed's URL (?rig=<url>, else 127.0.0.1:8833); camBase is
 * calib_embed's camera-page origin (?cams=<url>, else localhost:8833).
 */
export class ServerSource {
  constructor({ base = RIG_DEFAULT_URL, camBase = 'http://localhost:8833', fetch: f = globalThis.fetch?.bind(globalThis) } = {}) {
    this.kind = 'server'; this.base = base; this.camBase = camBase; this.fetch = f; this.frames = null;
  }
  async calibration() {
    const ac = new AbortController(); const to = setTimeout(() => ac.abort(), 1500);
    try { const r = await this.fetch(`${this.base}/api/calibration`, { cache: 'no-store', signal: ac.signal }); return r.ok ? await r.json() : null; } catch { return null; } finally { clearTimeout(to); }
  }
  info() { return { mode: 'server', base: this.base }; }
}

/** The camera-page origin calib_embed iframes in server mode: ?cams=<base>, else localhost:8833 (exactly as before). */
export function camBaseFromQuery(q) {
  return q.get('cams') || 'http://localhost:8833';
}

/**
 * Pick the page's source. opts: {siteDefault, fetch, makeBrowser: async () => BrowserSource}. The browser source module
 * (workers, cameras) is imported only when it is used.
 */
export async function makeSource(q, { siteDefault = null, fetch: f = globalThis.fetch?.bind(globalThis), makeBrowser, probeMs = 1500 } = {}) {
  const asked = trackerModeFromQuery(q, siteDefault);
  const base = rigUrlFromQuery(q) ?? RIG_DEFAULT_URL;
  const serverUp = asked === 'auto' ? await serverAnswers(base, { fetch: f, timeoutMs: probeMs }) : asked === 'server';
  const mode = resolveTrackerMode({ asked, serverUp });
  const src = mode === 'server' ? new ServerSource({ base, camBase: camBaseFromQuery(q), fetch: f }) : await makeBrowser();
  src.asked = asked;
  return src;
}
