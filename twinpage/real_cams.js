// The REAL camera images in the side column (docs/WEB_TWIN.md §21, ADR-132): CAM1-CAM3 as the CV server sees them, each
// with the newest tips feed's keypoints_px drawn over it. The simulated tiles (§18-§19) are opt-in (?cams=<dir>).
// The server is the multi-camera preview (veridien multicam preview, default http://127.0.0.1:8833; ?rig=<url> moves it):
//   GET /api/frame/<cam>.jpg     newest JPEG (CORS, X-Frame-Id, X-T, X-Size)      polled here at <= REAL_CAMS.hz
//   GET /api/frame/<cam>.jpg?id=<frame_id>   that frame set's JPEG (the last 16 per camera, ~0.6 s; 404 once gone):
//                                the overlay asks for the frame of the tips result it draws, so the two match exactly
//   GET /api/stream/<cam>.mjpg   multipart/x-mixed-replace (the server's rate)     the fallback, in an <img>
//   GET /api/tips                {source: "live"|"replay", keypoints_px: {CAM1: {L: {tip: [u, v], hinge?: [u, v]}}}, image_size}
//   GET /api/tips/stream         the same, pushed (SSE): rig_link.js's shared feed (feedHub), one EventSource per server
//   GET /api/calibration         {cameras: [{name, size: [w, h], lens}]}
// Without a server (twinpage/tracker_source.js 'browser' mode) the same column shows the in-page tracker's views: RealCams
// takes a FramesRing (`frames`) instead of probing and fetching, and the rig's calibration (`calib`) instead of /api/calibration.
// Pure parts first (node tests: web/twin/tests/real_cams.test.mjs); the DOM class RealCams last.

import { feedHub } from './rig_link.js';

/** Every path the page uses on the CV server: change the endpoints here (one line each). */
export const REAL_CAMS = Object.freeze({
  cams: Object.freeze(['CAM1', 'CAM2', 'CAM3']),
  health: Object.freeze(['/api/health', '/api/tips']),     // the first that answers 200 JSON means "reachable"
  calibration: '/api/calibration',
  tips: '/api/tips',
  stream: '/api/stream/{cam}.mjpg',
  frame: '/api/frame/{cam}.jpg',
  // frame first: polled at `hz` (the server's MJPEG runs at 15 fps, and three long-lived HTTP/1.0 streams plus the tips
  // poll come close to the browser's 6 connections per host); stream only when the server has no frame endpoint
  prefer: Object.freeze(['frame', 'stream']),
  hz: 10,
  probeEveryMs: 10000,
  timeoutMs: 1500,
  kpMaxAgeS: 1.0,                                         // keypoints older than this are not drawn
});

/** A camera's URL from a template with {cam}. */
export function camUrl(base, tmpl, cam) {
  const b = String(base).replace(/\/+$/, '');
  const p = String(tmpl).replace('{cam}', encodeURIComponent(cam));
  return /^https?:\/\//.test(p) ? p : `${b}${p.startsWith('/') ? '' : '/'}${p}`;
}

/** The JPEG of one frame set: <frame url>?id=<frame_id> (or &id= when the template has a query). */
export function frameIdUrl(base, tmpl, cam, frameId) {
  const u = camUrl(base, tmpl, cam);
  return `${u}${u.includes('?') ? '&' : '?'}id=${encodeURIComponent(frameId)}`;
}

/** The last n tips payloads by frame_id, each with its receive time: the keypoints that belong to a shown image. */
export class TipsRing {
  constructor(n = 32) { this.n = n; this.map = new Map(); }
  add(payload, at) {
    if (!payload || payload.frame_id == null) return;
    const k = String(payload.frame_id);
    this.map.delete(k); this.map.set(k, { payload, at });
    while (this.map.size > this.n) this.map.delete(this.map.keys().next().value);
  }
  get(frameId) { return frameId == null ? null : this.map.get(String(frameId)) ?? null; }
}

/** The frame_id to ask the image for: the shown tips result's, while it is fresh and not stale; else null (newest). */
export function frameToRequest(tips, now, maxAgeS = REAL_CAMS.kpMaxAgeS) {
  const p = tips?.payload;
  if (!p || p.frame_id == null || p.stale || !Number.isFinite(tips.at) || now - tips.at > maxAgeS) return null;
  return p.frame_id;
}

/** Letterbox (object-fit: contain) of a sw x sh image in a bw x bh box: {s, ox, oy, w, h}. */
export function fitContain(sw, sh, bw, bh) {
  if (!(sw > 0 && sh > 0 && bw > 0 && bh > 0)) return { s: 0, ox: 0, oy: 0, w: 0, h: 0 };
  const s = Math.min(bw / sw, bh / sh);
  const w = sw * s, h = sh * s;
  return { s, ox: (bw - w) / 2, oy: (bh - h) / 2, w, h };
}

const isPt = (p) => Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]);

/** One camera's keypoints {L: {tip, hinge?}, R: ...} from full-resolution px (native [w, h]) to a bw x bh box. */
export function scaleKeypoints(kp, native, bw, bh) {
  const out = {};
  if (!kp || !native) return out;
  const f = fitContain(native[0], native[1], bw, bh);
  if (!f.s) return out;
  const map = (p) => [f.ox + p[0] * f.s, f.oy + p[1] * f.s];
  for (const [id, k] of Object.entries(kp)) {
    if (!k || !isPt(k.tip)) continue;
    out[id] = { tip: map(k.tip) };
    if (isPt(k.hinge)) out[id].hinge = map(k.hinge);
  }
  return out;
}

const okSize = (s) => Array.isArray(s) && s.length >= 2 && s[0] > 0 && s[1] > 0 ? [Number(s[0]), Number(s[1])] : null;

/** "1280x720" -> [1280, 720]. */
export function parseSize(h) {
  const m = /^(\d+)\s*[x×]\s*(\d+)$/.exec(String(h ?? '').trim());
  return m ? [Number(m[1]), Number(m[2])] : null;
}

/** A camera's full-resolution size: the tips payload's image_size, the calibration, the X-Size header, the image's own. */
export function nativeSize(cam, { payload = null, calib = null, header = null, natural = null } = {}) {
  return okSize(payload?.image_size?.[cam])
    ?? okSize((calib?.cameras || []).find((c) => c?.name === cam)?.size)
    ?? okSize((calib?.cameras || []).find((c) => c?.name === cam)?.lens?.size)
    ?? parseSize(header)
    ?? okSize(natural);
}

/** "Camera 1 · real", or "Camera 1 · replay" when the feed says it replays a recording. */
export function camLabel(cam, source) {
  const m = /^CAM(\d+)$/.exec(cam);
  return `${m ? `Camera ${m[1]}` : cam} · ${source === 'replay' ? 'replay' : 'real'}`;
}

/** Image endpoints a /api/health or /api/calibration reply advertises: {stream, frame} templates with {cam}, or {}. */
export function advertisedEndpoints(j) {
  const out = {};
  if (!j || typeof j !== 'object') return out;
  const ep = j.endpoints || j.images || j.image_endpoints || {};
  const tmpl = (v) => (typeof v === 'string' && v.includes('{cam}') ? v : null);
  out.stream = tmpl(ep.stream) ?? tmpl(ep.mjpeg) ?? tmpl(ep.mjpg) ?? undefined;
  out.frame = tmpl(ep.frame) ?? tmpl(ep.jpeg) ?? tmpl(ep.jpg) ?? undefined;
  // per camera: cameras: [{name, stream: "/api/stream/CAM1.mjpg", frame: "/api/frame/CAM1.jpg"}]
  for (const c of Array.isArray(j.cameras) ? j.cameras : []) {
    if (!c?.name) continue;
    for (const [k, keys] of [['stream', ['stream', 'mjpeg', 'mjpg']], ['frame', ['frame', 'jpeg', 'jpg']]]) {
      const v = keys.map((x) => c[x]).find((x) => typeof x === 'string' && x.includes(c.name));
      if (v && !out[k]) out[k] = v.replace(c.name, '{cam}');
    }
  }
  for (const k of Object.keys(out)) if (!out[k]) delete out[k];
  return out;
}

async function timed(fetchFn, url, ms, init = {}) {
  const ac = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const to = setTimeout(() => ac?.abort(), ms);
  try { return { res: await fetchFn(url, { cache: 'no-store', mode: 'cors', ...init, signal: ac?.signal }), ac }; } finally { clearTimeout(to); }
}

const ctypeOf = (res) => String(res?.headers?.get?.('content-type') ?? '');

/**
 * Probe the CV server: reachable (a health path answers JSON), its cameras and sizes (calibration), and which image
 * endpoints answer. A stream is accepted on its headers (multipart) and the request aborted at once.
 * Returns {reachable, health, calib, cams, sizes, available: {stream: tmpl|null, frame: tmpl|null}, mode, source}.
 */
export async function probeServer(base, { fetch: f = globalThis.fetch?.bind(globalThis), cfg = REAL_CAMS, timeoutMs = cfg.timeoutMs } = {}) {
  const out = { reachable: false, health: null, healthPath: null, calib: null, cams: [...cfg.cams], sizes: {}, available: { stream: null, frame: null }, mode: null, source: null };
  for (const p of cfg.health) {
    try {
      const { res } = await timed(f, camUrl(base, p, ''), timeoutMs);
      if (!res.ok) continue;                               // 404: try the next path
      out.health = await res.json(); out.healthPath = p; out.reachable = true;
      break;
    } catch { break; }                                     // refused / timed out: the server is down, stop here
  }
  if (!out.reachable) return out;
  if (out.health?.type === 'predictions') out.source = out.health.source ?? null;
  if (cfg.calibration) {
    try {
      const { res } = await timed(f, camUrl(base, cfg.calibration, ''), timeoutMs);
      if (res.ok) out.calib = await res.json();
    } catch { /* optional */ }
  }
  const names = (out.calib?.cameras || []).map((c) => c?.name).filter((n) => typeof n === 'string' && /^CAM\d+$/.test(n));
  if (names.length) out.cams = names;
  for (const c of out.cams) { const s = nativeSize(c, { payload: out.health, calib: out.calib }); if (s) out.sizes[c] = s; }
  const adv = { ...advertisedEndpoints(out.calib), ...advertisedEndpoints(out.health) };
  const cand = { stream: [adv.stream, cfg.stream].filter(Boolean), frame: [adv.frame, cfg.frame].filter(Boolean) };
  const cam = out.cams[0];
  for (const tmpl of cand.stream) {                       // stream first (as asked), then frame
    try {
      const { res, ac } = await timed(f, camUrl(base, tmpl, cam), timeoutMs);
      const ok = res.ok && /multipart\/x-mixed-replace/i.test(ctypeOf(res));
      try { ac?.abort(); res.body?.cancel?.().catch?.(() => {}); } catch { /* closed */ }
      if (ok) { out.available.stream = tmpl; break; }
    } catch { /* next */ }
  }
  for (const tmpl of cand.frame) {
    try {
      const { res } = await timed(f, camUrl(base, tmpl, cam), timeoutMs);
      const ok = res.ok && /^image\//i.test(ctypeOf(res));
      try { res.body?.cancel?.().catch?.(() => {}); } catch { /* closed */ }
      if (ok) { out.available.frame = tmpl; break; }
    } catch { /* next */ }
  }
  out.mode = cfg.prefer.find((k) => out.available[k]) ?? null;
  return out;
}

/**
 * probeServer's answer for the in-page tracker (no server): reachable once its ring holds views; the cameras are the
 * calibration's (rig.json), in rig order, else the ring's. mode 'frame' (images come from frames.get, never a URL).
 */
export function localProbe(frames, calib, cfg = REAL_CAMS) {
  const have = frames?.cams?.() ?? [];
  const named = (calib?.cameras || []).map((c) => c?.name).filter((n) => typeof n === 'string' && /^CAM\d+$/.test(n));
  const cams = named.length ? named : (have.length ? [...have].sort() : [...cfg.cams]);
  const sizes = {};
  for (const c of cams) { const s = nativeSize(c, { calib }); if (s) sizes[c] = s; }
  const reachable = have.length > 0;
  return { reachable, health: null, healthPath: null, calib, cams, sizes, available: { stream: null, frame: reachable ? 'local' : null },
    mode: reachable ? 'frame' : null, source: null, local: true };
}

/** Which cameras to fetch: all while the column shows, else only the one shown large. */
export function camsToFetch({ cams, visible, large = null }) {
  if (visible) return [...cams];
  return large && cams.includes(large) ? [large] : [];
}

/** The keypoints to draw: the payload's for `cam` while it is younger than maxAgeS, else none. */
export function freshKeypoints(tips, cam, now, maxAgeS = REAL_CAMS.kpMaxAgeS) {
  if (!tips?.payload || !Number.isFinite(tips.at) || now - tips.at > maxAgeS || tips.payload.stale) return null;
  return tips.payload.keypoints_px?.[cam] ?? null;
}

// ------------------------------------------------------------------ drawing (2-D canvas)

export const KP_STYLE = Object.freeze({
  L: { color: 'rgba(242, 200, 121, 0.95)', dash: [] },          // soft amber, solid ring
  R: { color: 'rgba(143, 211, 255, 0.95)', dash: [3, 2.5] },     // soft cyan, dashed ring
  other: { color: 'rgba(214, 214, 206, 0.9)', dash: [1.5, 2] },
  halo: 'rgba(0, 0, 0, 0.55)',
});

/** Draw scaled keypoints (CSS px) on a 2-D context whose transform is CSS px -> backing px. */
export function drawKeypoints(g, pts, { boxW = 300 } = {}) {
  const r = Math.max(5, Math.min(12, boxW / 40));
  g.lineCap = 'round';
  const both = (fn, color, w) => {
    g.strokeStyle = KP_STYLE.halo; g.lineWidth = w + 2; fn(); g.stroke();
    g.strokeStyle = color; g.lineWidth = w; fn(); g.stroke();
  };
  for (const [id, k] of Object.entries(pts)) {
    const st = KP_STYLE[id] ?? KP_STYLE.other;
    const [x, y] = k.tip;
    if (k.hinge) {
      const [hx, hy] = k.hinge, a = r * 0.55;
      g.setLineDash([]);
      both(() => { g.beginPath(); g.moveTo(hx - a, hy - a); g.lineTo(hx + a, hy + a); g.moveTo(hx - a, hy + a); g.lineTo(hx + a, hy - a); }, st.color, 1.4);
    }
    g.setLineDash(st.dash);
    both(() => { g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); }, st.color, 1.4);
    g.setLineDash([]);
    g.fillStyle = KP_STYLE.halo; g.beginPath(); g.arc(x, y, 2.6, 0, 2 * Math.PI); g.fill();
    g.fillStyle = st.color; g.beginPath(); g.arc(x, y, 1.8, 0, 2 * Math.PI); g.fill();
    g.font = `600 ${Math.round(Math.max(10, r * 1.05))}px ui-sans-serif, system-ui, sans-serif`;
    g.textBaseline = 'bottom';
    g.lineWidth = 3; g.strokeStyle = KP_STYLE.halo; g.strokeText(id, x + r * 0.8, y - r * 0.6);
    g.fillStyle = st.color; g.fillText(id, x + r * 0.8, y - r * 0.6);
  }
}

// ------------------------------------------------------------------ the DOM part

const dprOf = () => Math.min(2, (typeof window !== 'undefined' && window.devicePixelRatio) || 1);

/** Size a canvas's backing store to its CSS box x dpr; returns the CSS box. */
function fitCanvas(c) {
  const w = c.clientWidth, h = c.clientHeight, d = dprOf();
  const bw = Math.max(1, Math.round(w * d)), bh = Math.max(1, Math.round(h * d));
  if (c.width !== bw || c.height !== bh) { c.width = bw; c.height = bh; }
  return { w, h, d };
}

/**
 * RealCams: probes the server at start and every probeEveryMs; while it is reachable, one tile per camera in `side`
 * (prepended) and a large view in `stage`. Polls each camera's newest JPEG at <= hz with no overlapping requests and
 * decodes it off the main thread (createImageBitmap), or shows the MJPEG stream in an <img>. Never blocks.
 *   tips(): {payload, at (s, Date.now() / 1000)} from the rig feed, or null. Either way, while anything is shown it holds
 *   the page's shared feed of `base` (hub: rig_link.js feedHub; pushed, polled as the fallback): rig mode's feed when
 *   that is on, so the page never opens a second EventSource to the same server.
 *   Frame-exact overlay: each camera asks for /api/frame/<cam>.jpg?id=<the newest tips frame_id> and draws that result's
 *   keypoints over it; on a 404 (gone) the newest frame, with the keypoints of its frame_id when the ring has them.
 *   onAvail(bool), onLarge(name|null).
 */
export class RealCams {
  constructor({ base, side, stage, tips = () => null, onAvail = () => {}, onLarge = () => {}, cfg = REAL_CAMS, fetch: f = null, log = () => {}, hub = feedHub,
    frames = null, calib = null }) {
    this.base = String(base).replace(/\/+$/, ''); this.side = side; this.stage = stage; this.cfg = cfg;
    // frames: a FramesRing (tracker_source.js) of the in-page tracker; then no server is probed or fetched: the views
    // come from the ring (get(cam, frame_id) -> {bitmap, id, size}) and calib is the rig's /api/calibration-shaped copy
    this.framesSrc = frames; this.localCalib = calib;
    this.fetch = f ?? globalThis.fetch.bind(globalThis); this.extTips = tips; this.onAvail = onAvail; this.onLarge = onLarge; this.log = log;
    this.probe = null; this.reachable = false; this.visible = false; this.large = null; this.tiles = {}; this.loops = {};
    this.hub = hub; this.tipsSub = null; this.ring = new TipsRing(32); this.byId = true; this.idMisses = 0; this.idStats = { exact: 0, gone: 0, newest: 0 };
    this.frames = 0; this.errors = 0; this.drawn = {}; this.lastProbeAt = 0; this.misses = 0;
    this.section = null; this.big = null; this.stopped = false;
  }

  start() {
    // the in-page source: re-check as soon as its first views arrive (or the cameras stop), not on the 10 s probe
    if (this.framesSrc) this.offFrames = this.framesSrc.onPut((id) => { if ((id == null) === this.available()) this.check(); });
    this.check();
    this.probeTimer = setInterval(() => this.check(), this.cfg.probeEveryMs);
    this.overlayTimer = setInterval(() => this.paintOverlays(), 66);
    this.tipsTimer = setInterval(() => this.syncTips(), 250);
  }

  available() { return this.reachable && !!this.probe?.mode; }

  async check() {
    if (this.checking) return;
    this.checking = true;
    try {
      let p;
      if (this.framesSrc) {
        p = localProbe(this.framesSrc, this.localCalib, this.cfg);
        const was = this.available();
        this.probe = p; this.reachable = p.reachable; this.misses = 0; this.lastProbeAt = Date.now();
        if (this.available() && !was) { this.build(); this.log(`real cameras: ${p.cams.join(', ')} from this page's tracker`); }
        if (!this.available() && was) this.teardown();
        if (this.available() !== was) this.onAvail(this.available());
        return;
      }
      if (this.available()) {
        // already up: a health GET only; two misses in a row (20 s) take the column down
        const health = [this.probe.healthPath, ...this.cfg.health.filter((x) => x !== this.probe.healthPath)];
        const up = await probeServer(this.base, { fetch: this.fetch, cfg: { ...this.cfg, health, stream: null, frame: null, calibration: null } }).catch(() => null);
        if (up?.reachable) { this.misses = 0; this.probe.source = up.source ?? this.probe.source; return; }
        if (++this.misses < 2) return;
        p = { reachable: false, available: {}, mode: null };
      } else {
        p = await probeServer(this.base, { fetch: this.fetch, cfg: this.cfg });
      }
      this.lastProbeAt = Date.now();
      const was = this.available();
      this.probe = p; this.reachable = !!p.reachable; this.misses = 0;
      if (this.available() && !was) { this.build(); this.log(`real cameras: ${p.cams.join(', ')} via ${p.mode} (${p.available[p.mode]})`); }
      if (!this.available() && was) this.teardown();
      if (this.available() !== was) this.onAvail(this.available());
    } finally { this.checking = false; }
  }

  // ---------------------------------------------------------------- DOM

  build() {
    this.teardown();
    this.byId = true; this.idMisses = 0;                    // a new server session: try frame-exact images again
    const sec = this.section = document.createElement('div');
    sec.className = 'real-cams';
    for (const cam of this.probe.cams) {
      const t = document.createElement('button');
      t.type = 'button'; t.className = 'tile real'; t.dataset.view = cam; t.dataset.real = '1';
      const stream = this.probe.mode === 'stream';
      t.innerHTML = `<span class="label"></span><span class="frame">${stream ? '<img alt="" draggable="false">' : '<canvas class="img"></canvas>'}<canvas class="ov"></canvas><span class="note">Waiting for images</span></span>`;
      t.addEventListener('click', (e) => { e.currentTarget.blur(); this.setLarge(this.large === cam ? null : cam); });
      sec.appendChild(t);
      this.tiles[cam] = { el: t, label: t.querySelector('.label'), img: t.querySelector('img'), canvas: t.querySelector('canvas.img'), ov: t.querySelector('canvas.ov'),
        note: t.querySelector('.note'), size: this.probe.sizes[cam] ?? null, frameId: null, shown: 0 };
    }
    // after the column's header (the Real / Sim switch) when the page has one, else first
    const head = this.side.querySelector(':scope > .col-head');
    if (head) head.after(sec); else this.side.prepend(sec);
    const big = this.big = document.createElement('div');
    big.id = 'real-large'; big.hidden = true; big.tabIndex = 0; big.setAttribute('aria-label', 'Camera image; click or Esc returns');
    big.innerHTML = `${this.probe.mode === 'stream' ? '<img alt="" draggable="false">' : '<canvas class="img"></canvas>'}<canvas class="ov"></canvas><div class="real-large-label"></div>`;
    big.addEventListener('click', () => this.setLarge(null));
    this.stage.appendChild(big);
    this.bigEls = { img: big.querySelector('img'), canvas: big.querySelector('canvas.img'), ov: big.querySelector('canvas.ov'), label: big.querySelector('.real-large-label') };
    this.paintLabels();
    this.sync();
  }

  teardown() {
    for (const cam of Object.keys(this.loops)) this.loops[cam].on = false;
    this.loops = {};
    for (const t of Object.values(this.tiles)) if (t.img) t.img.removeAttribute('src');
    if (this.bigEls?.img) this.bigEls.img.removeAttribute('src');
    if (this.large) { this.large = null; this.onLarge(null); }
    this.section?.remove(); this.big?.remove();
    this.section = null; this.big = null; this.bigEls = null; this.tiles = {}; this.drawn = {};
  }

  setVisible(on) { this.visible = !!on; this.sync(); }

  setLarge(cam) {
    if (cam && !this.tiles[cam]) return;
    if (cam === this.large) return;
    this.large = cam;
    if (this.big) this.big.hidden = !cam;
    for (const [n, t] of Object.entries(this.tiles)) t.el.classList.toggle('current', n === cam);
    if (this.bigEls?.img) { if (cam) this.bigEls.img.src = camUrl(this.base, this.probe.available.stream, cam); else this.bigEls.img.removeAttribute('src'); }
    if (this.bigEls?.canvas && cam) { const g = this.bigEls.canvas.getContext('2d'); g.clearRect(0, 0, this.bigEls.canvas.width, this.bigEls.canvas.height); }
    this.paintLabels();
    this.sync();
    this.onLarge(cam);
    if (cam) this.big?.focus({ preventScroll: true });
  }

  source() { return (this.tipsNow()?.payload?.source ?? this.probe?.source) || 'live'; }

  paintLabels() {
    const src = this.source();
    for (const [cam, t] of Object.entries(this.tiles)) { const s = camLabel(cam, src); if (t.label.textContent !== s) t.label.textContent = s; }
    if (this.bigEls) this.bigEls.label.textContent = this.large ? `${camLabel(this.large, src)} · Esc returns` : '';
  }

  /** Start or stop each camera's fetch loop (or stream) to match visibility and the large view. */
  sync() {
    this.syncTips();
    if (!this.available()) return;
    const want = new Set(camsToFetch({ cams: Object.keys(this.tiles), visible: this.visible, large: this.large }));
    if (this.probe.mode === 'stream') {
      for (const [cam, t] of Object.entries(this.tiles)) {
        const url = camUrl(this.base, this.probe.available.stream, cam);
        if (want.has(cam) && this.visible) { if (t.img.getAttribute('src') !== url) { t.img.src = url; t.img.onload = () => { t.note.hidden = true; t.shown++; t.size ??= [t.img.naturalWidth, t.img.naturalHeight]; }; } }
        else t.img.removeAttribute('src');
      }
      return;
    }
    for (const cam of Object.keys(this.tiles)) {
      if (want.has(cam) && !this.loops[cam]?.on) { const L = this.loops[cam] = { on: true }; this.loop(cam, L); }
      if (!want.has(cam) && this.loops[cam]) { this.loops[cam].on = false; delete this.loops[cam]; }
    }
  }

  async loop(cam, L) {
    const period = 1000 / this.cfg.hz;
    while (L.on && !this.stopped) {
      const t0 = performance.now();
      try { await this.fetchOne(cam); } catch (e) { this.errors++; if (this.errors < 4) console.warn('[twin] real camera', cam, e?.message || e); }
      const wait = Math.max(0, period - (performance.now() - t0)) + (this.lastErrAt && performance.now() - this.lastErrAt < 1000 ? 400 : 0);
      await new Promise((r) => setTimeout(r, wait));
    }
  }

  async fetchOne(cam) {
    const t = this.tiles[cam];
    if (!t) return;
    const target0 = cam === this.large && this.bigEls ? this.bigEls.canvas : t.canvas;
    const boxKey = `${t.canvas.clientWidth}x${t.canvas.clientHeight}|${target0.clientWidth}x${target0.clientHeight}|${this.large === cam}`;
    // frame-exact: the image of the tips result the overlay will draw (one request; on 404 a second for the newest)
    const tips = this.tipsNow();
    const want = this.byId ? frameToRequest(tips, Date.now() / 1000, this.cfg.kpMaxAgeS) : null;
    if (want != null && `${want}|${boxKey}` === t.drawnKey) return;   // that frame set is already shown
    if (this.framesSrc) return this.drawLocal(cam, t, tips, want, boxKey);
    let res = null, exact = false;
    try {
      if (want != null) {
        ({ res } = await timed(this.fetch, frameIdUrl(this.base, this.probe.available.frame, cam, want), this.cfg.timeoutMs));
        if (res.ok && String(res.headers.get('x-frame-id')) === String(want)) { exact = true; this.idMisses = 0; this.idStats.exact++; }
        else if (res.ok) this.byId = false;                // a server that ignores ?id: the newest it is, from now on
        else { res.body?.cancel?.(); res = null; this.idStats.gone++; if (++this.idMisses >= 5) this.byId = false; }
      }
      if (!res) { ({ res } = await timed(this.fetch, camUrl(this.base, this.probe.available.frame, cam), this.cfg.timeoutMs)); this.idStats.newest++; }
    } catch (e) { this.lastErrAt = performance.now(); throw e; }
    if (!res.ok) { this.lastErrAt = performance.now(); res.body?.cancel?.(); return; }
    const id = res.headers.get('x-frame-id');
    const key = `${id}|${boxKey}`;
    if (id != null && key === t.drawnKey) { res.body?.cancel?.(); return; }  // same frame, same boxes: nothing to draw
    t.size = nativeSize(cam, { payload: this.tipsNow()?.payload, calib: this.probe.calib, header: res.headers.get('x-size') }) ?? t.size;
    const blob = await res.blob();
    // decode off the main thread at the size it is shown (the large view when this camera is large, else the tile)
    const target = cam === this.large && this.bigEls ? this.bigEls.canvas : t.canvas;
    const box = { w: target.clientWidth, h: target.clientHeight, d: dprOf() };
    const opts = {};
    if (t.size && box.w > 0) {
      const f = fitContain(t.size[0], t.size[1], box.w * box.d, box.h * box.d);
      if (f.w >= 1 && f.w < t.size[0]) { opts.resizeWidth = Math.round(f.w); opts.resizeHeight = Math.round(f.h); opts.resizeQuality = 'medium'; }
    }
    const bmp = await createImageBitmap(blob, opts);
    if (!t.size) t.size = [bmp.width, bmp.height];
    t.frameId = id; t.drawnKey = key;
    // the keypoints that belong to this image (kept with it while it is shown); null: the newest tips, as before
    const own = exact ? tips : this.ring.get(id);
    t.tips = own ? { payload: own.payload, at: Date.now() / 1000 } : null;
    this.drawImage(t.canvas, bmp, t.size);
    if (cam === this.large && this.bigEls) this.drawImage(this.bigEls.canvas, bmp, t.size);
    bmp.close?.();
    t.shown++; this.frames++;
    if (!t.note.hidden) t.note.hidden = true;
    this.paintOverlay(cam);
  }

  /** The in-page source's view of `cam` (that frame set's when the ring still has it, else the newest). The ring owns
   * the bitmap: it is drawn, never closed here. */
  drawLocal(cam, t, tips, want, boxKey) {
    const got = this.framesSrc.get(cam, want);
    if (!got?.bitmap) return;
    const key = `${got.id}|${boxKey}`;
    if (key === t.drawnKey) return;
    t.size = got.size ?? nativeSize(cam, { payload: tips?.payload, calib: this.probe?.calib }) ?? t.size ?? [got.bitmap.width, got.bitmap.height];
    const exact = want != null && got.exact && String(got.id) === String(want);
    if (exact) this.idStats.exact++; else this.idStats.newest++;
    t.frameId = got.id; t.drawnKey = key;
    const own = exact ? tips : this.ring.get(got.id);
    t.tips = own ? { payload: own.payload, at: Date.now() / 1000 } : null;
    this.drawImage(t.canvas, got.bitmap, t.size);
    if (cam === this.large && this.bigEls) this.drawImage(this.bigEls.canvas, got.bitmap, t.size);
    t.shown++; this.frames++;
    if (!t.note.hidden) t.note.hidden = true;
    this.paintOverlay(cam);
  }

  drawImage(c, bmp, native) {
    const { w, h, d } = fitCanvas(c);
    const g = c.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = 'rgb(12, 13, 12)'; g.fillRect(0, 0, c.width, c.height);
    const f = fitContain(native[0], native[1], w * d, h * d);
    g.drawImage(bmp, f.ox, f.oy, f.w, f.h);
  }

  // ---------------------------------------------------------------- keypoints

  tipsNow() {
    const ext = this.extTips?.();
    if (ext) return ext;
    const f = this.tipsSub?.feed;
    return f?.last ? { payload: f.last, at: f.lastAt } : null;
  }

  /** Hold the page's shared tips feed of this server while anything is shown (pushed; polled as the fallback). */
  syncTips() {
    const want = !this.stopped && this.available() && (this.visible || !!this.large);
    if (want && !this.tipsSub) this.tipsSub = this.hub.acquire(this.base, (j) => this.ring.add(j, Date.now() / 1000), { hz: this.cfg.hz, fetch: this.fetch });
    else if (!want && this.tipsSub) { this.tipsSub.release(); this.tipsSub = null; }
  }

  paintOverlays() {
    if (!this.available()) return;
    this.paintLabels();
    for (const cam of Object.keys(this.tiles)) this.paintOverlay(cam);
  }

  paintOverlay(cam) {
    const t = this.tiles[cam];
    if (!t) return;
    const tips = t.tips ?? this.tipsNow();              // the shown image's own result when known (frame-exact)
    const kp = freshKeypoints(tips, cam, Date.now() / 1000);
    const size = nativeSize(cam, { payload: tips?.payload, calib: this.probe?.calib }) ?? t.size ?? (t.img?.naturalWidth ? [t.img.naturalWidth, t.img.naturalHeight] : null);
    const targets = [[t.ov, 'tile']];
    if (cam === this.large && this.bigEls) targets.push([this.bigEls.ov, 'large']);
    for (const [ov, where] of targets) {
      const { w, h, d } = fitCanvas(ov);
      const g = ov.getContext('2d');
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, ov.width, ov.height);
      if (!kp || !size || this.visible === false && where === 'tile') { if (where === 'tile') this.drawn[cam] = null; continue; }
      const pts = scaleKeypoints(kp, size, w, h);
      g.setTransform(d, 0, 0, d, 0, 0);
      drawKeypoints(g, pts, { boxW: w });
      const d0 = { pts, box: [w, h], size, frame_id: tips.payload.frame_id, image_id: t.frameId ?? null, exact: t.tips != null && String(t.frameId) === String(tips.payload.frame_id) };
      if (where === 'tile') this.drawn[cam] = d0; else this.drawn[`${cam}:large`] = d0;
    }
  }

  info() {
    return { base: this.base, reachable: this.reachable, available: this.available(), mode: this.probe?.mode ?? null, endpoints: this.probe?.available ?? null,
      healthPath: this.probe?.healthPath ?? null, cams: this.probe?.cams ?? [], sizes: this.probe?.sizes ?? {}, visible: this.visible, large: this.large,
      fetching: Object.keys(this.loops), frames: this.frames, errors: this.errors, source: this.available() ? this.source() : null,
      shown: Object.fromEntries(Object.entries(this.tiles).map(([c, t]) => [c, t.shown])), drawn: this.drawn,
      tipsFrom: this.extTips?.() ? 'rig feed' : this.tipsSub ? `shared feed (${this.tipsSub.feed.transport()})` : null, byId: this.byId, idStats: { ...this.idStats } };
  }

  stop() { this.stopped = true; this.offFrames?.(); clearInterval(this.probeTimer); clearInterval(this.overlayTimer); clearInterval(this.tipsTimer); this.syncTips(); this.teardown(); }
}
