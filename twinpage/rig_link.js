// Rig link: the twin's instruments driven by the REAL rig's tool-tip predictions (docs/WEB_TWIN.md §7, §14, §19).
// The feed is GET <url>/api/tips (the multi-camera preview session, default http://127.0.0.1:8833):
//   {type: "predictions", frame_id, t, model, instruments: [{id: "L"|"R", tip_D, shaft_dir_D, hinge_D?, conf,
//    jaw_rad?, roll_rad?}], keypoints_px, stale?}
// Pure core (no DOM in the math): stateFromTip / solveTip invert control/kinematics.js exactly, OneEuro smooths (the old
// default; snapshot interpolation in twinpage/rig_smooth.js replaced it, ADR-133), and
// RigFeed takes the pushed stream GET <url>/api/tips/stream (Server-Sent Events, ADR-133) and falls back to polling
// /api/tips with no overlapping requests; FeedHub shares one feed per server across the page. The page hook lives in
// twinpage/main.js ("rig mode").
//
// Inversion (frame D, metres): truthFromState's tip = hinge + R_D_S . o with o = [tx, cos(jaw/2) ty, cos(jaw/2) tz]
// (the midpoint of the two jaw tips, tip_in_jaw_frame = [tx, ty, tz] in each jaw frame). With the hinge on the shaft
// axis through the CAD pivot, tip - pivot - R . [ox, oy, 0] = d . (insertion + oz); the shaft direction d is the fixed
// point of that equation (R depends on d and roll; |o_perp| / |tip - pivot| ~ 0.02, so it converges in a few steps).
// The fed tip is kept exact; the fed shaft direction is used only to report how far its axis passes from the CAD
// pivot (the real pivot sits 1.5-4 mm off the CAD one), never to bend the shaft off the RCM.

import { port, spec, shaftRotation, inverseDirection, mulv, sub, add, scale, dot, norm, normalize, coneClamp, DEG } from '../control/kinematics.js';

export const RIG_DEFAULT_URL = 'http://127.0.0.1:8833';

/** Rig feed base URL from the page query: ?rig=1 -> the default, ?rig=<url> -> that url, else null. */
export function rigUrlFromQuery(q) {
  const v = q.get('rig');
  if (v == null || v === '0' || v === 'false') return null;
  if (v === '' || v === '1' || v === 'true') return RIG_DEFAULT_URL;
  return v.replace(/\/+$/, '');
}

const prevOf = (prevState, id) => (prevState?.instruments ? prevState.instruments.find((q) => q.id === id) : prevState) ?? null;

/** Tip offset from the hinge in the shaft frame S for jaw opening `jaw` (rad). */
export function tipOffsetS(ins, sp, jaw) {
  const tl = ins.tip_in_jaw_frame ?? [0, 0, sp.jaw_length];
  const c = Math.cos(jaw / 2);
  return [tl[0], c * tl[1], c * tl[2]];
}

/** Perpendicular distance (m) of point p from the line through a along unit u. */
export function pointLineDistance(p, a, u) {
  const w = sub(p, a);
  return norm(sub(w, scale(u, dot(w, u))));
}

/**
 * solveTip(scene, id, tipD, dirD, prevState, opts) -> {q, residual_mm, dir_err_deg, limited}
 *   q: {id, yaw, pitch, roll, insertion, jaw} about the instrument's CAD port pivot, with truthFromState's tip on tipD.
 *   residual_mm: distance from the CAD pivot to the fed shaft axis (hinge_D, or tip - R . o, along dirD); null without dirD.
 *   opts: {roll, jaw} (rad; default prev or 0 / 0.35), hingeD, limits (a control/limits.js Limits; else cone/insertion/jaw only).
 */
export function solveTip(scene, id, tipD, dirD, prevState = null, opts = {}) {
  const ins = scene.instruments.find((i) => i.id === id);
  if (!ins) throw new Error(`scene has no instrument ${id}`);
  const p = port(scene, ins.port);
  const sp = spec(ins);
  const prev = prevOf(prevState, id);
  const roll = Number.isFinite(opts.roll) ? opts.roll : (prev?.roll ?? 0);
  const jaw = Math.max(0, Math.min(sp.jaw_max, Number.isFinite(opts.jaw) ? opts.jaw : (prev?.jaw ?? 0.35)));
  const o = tipOffsetS(ins, sp, jaw);
  const oPerp = [o[0], o[1], 0];
  const r = sub(tipD, p.pivot);
  let d = dirD && norm(dirD) > 1e-9 && dot(dirD, r) > 0 ? normalize(dirD) : normalize(r);
  let yaw = 0, pitch = 0, R = null, v = r;
  for (let k = 0; k < 30; k++) {
    [yaw, pitch] = inverseDirection(p.R, d);
    R = shaftRotation(p.R, { yaw, pitch, roll });
    v = sub(r, mulv(R, oPerp));
    const dn = normalize(v);
    const step = norm(sub(dn, d));
    d = dn;
    if (step < 1e-14) break;
  }
  [yaw, pitch] = inverseDirection(p.R, d);
  let q = { id, yaw, pitch, roll, insertion: norm(v) - o[2], jaw };
  let residual_mm = null, dir_err_deg = null;
  if (dirD && norm(dirD) > 1e-9) {
    const u = normalize(dirD);
    let a = opts.hingeD;
    if (!a) {
      const [y2, p2] = inverseDirection(p.R, u);
      a = sub(tipD, mulv(shaftRotation(p.R, { yaw: y2, pitch: p2, roll }), o));
    }
    residual_mm = pointLineDistance(p.pivot, a, u) * 1000;
    dir_err_deg = Math.acos(Math.max(-1, Math.min(1, dot(u, d)))) / DEG;
  }
  let limited = [];
  if (opts.limits) {
    const c = opts.limits.clampInstrument(id, { ...q });
    q = { id, yaw: c.q.yaw, pitch: c.q.pitch, roll: c.q.roll, insertion: c.q.insertion, jaw: c.q.jaw };
    limited = c.limited;
  } else {
    const c = coneClamp(q, p.cone);
    if (c._cone) limited.push('cone');
    q = { id, yaw: c.yaw, pitch: c.pitch, roll, insertion: q.insertion, jaw };
    const lo = sp.min_insertion ?? 0.005, hi = sp.max_insertion ?? 0.33;
    if (q.insertion < lo) { q.insertion = lo; limited.push('insertion_min'); }
    if (q.insertion > hi) { q.insertion = hi; limited.push('insertion_max'); }
  }
  return { q, residual_mm, dir_err_deg, limited };
}

/** The twin's instrument state for a fed tip (see solveTip). posesFromState(stateFromTip(...)) puts the tip on tipD. */
export function stateFromTip(scene, id, tipD, dirD, prevState = null, opts = {}) {
  return solveTip(scene, id, tipD, dirD, prevState, opts).q;
}

// ------------------------------------------------------------------ smoothing (one-euro, Casiez et al. 2012)

const alpha = (cutoff, dt) => { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); };

/** One-euro filter on a vector. minCutoff (Hz), beta (1 / unit of x), dCutoff (Hz). t in seconds. */
export class OneEuro {
  constructor({ minCutoff = 1.5, beta = 60, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.x = null; this.dx = null; this.t = null;
  }
  reset() { this.x = null; this.dx = null; this.t = null; }
  filter(x, t) {
    if (this.x == null || this.t == null || !(t > this.t) || t - this.t > 1.0) {
      this.x = [...x]; this.dx = x.map(() => 0); this.t = t; return [...x];
    }
    const dt = t - this.t;
    const dxRaw = x.map((v, i) => (v - this.x[i]) / dt);
    const ad = alpha(this.dCutoff, dt);
    this.dx = dxRaw.map((v, i) => this.dx[i] + ad * (v - this.dx[i]));
    const speed = Math.hypot(...this.dx);
    const a = alpha(this.minCutoff + this.beta * speed, dt);
    this.x = x.map((v, i) => this.x[i] + a * (v - this.x[i]));
    this.t = t;
    return [...this.x];
  }
}

/** Per-instrument smoothing of tip (m) and shaft direction (unit), keyed by id. smooth=false passes values through. */
export class TipSmoother {
  constructor({ smooth = true, tip = { minCutoff: 1.5, beta: 60 }, dir = { minCutoff: 1.5, beta: 10 } } = {}) {
    this.smooth = smooth; this.cfg = { tip, dir }; this.f = {};
  }
  apply(id, tipD, dirD, t) {
    if (!this.smooth) return { tipD, dirD };
    const f = (this.f[id] ??= { tip: new OneEuro(this.cfg.tip), dir: new OneEuro(this.cfg.dir) });
    const tip = f.tip.filter(tipD, t);
    const dir = dirD ? normalize(f.dir.filter(dirD, t)) : null;
    return { tipD: tip, dirD: dir };
  }
  reset() { this.f = {}; }
}

/**
 * Apply one feed result to a twin state: every instrument the feed names (L, R) gets solveTip's state; the rest of the
 * state (scope, boards, other tools) is kept. A frame the feed marks stale holds the last pose (holdStale, default):
 * its instruments count as present (seen) but move nothing. Returns {state, residual_mm: {L, R}, limited: {L, R},
 * used: [ids moved], present: [ids in the frame], held: bool}.
 */
export function applyPredictions(scene, state, pred, { smoother = null, limits = null, minConf = 0, holdStale = true } = {}) {
  const out = { state, residual_mm: {}, dir_err_deg: {}, limited: {}, used: [], present: [], held: false };
  if (!pred || !Array.isArray(pred.instruments)) return out;
  const t = Number.isFinite(pred.t) ? pred.t : 0;
  let instruments = state.instruments.map((q) => ({ ...q }));
  for (const rec of pred.instruments) {
    if (!rec || !Array.isArray(rec.tip_D) || rec.tip_D.length !== 3 || !rec.tip_D.every(Number.isFinite)) continue;
    if (!scene.instruments.some((i) => i.id === rec.id)) continue;
    if (Number.isFinite(rec.conf) && rec.conf < minConf) continue;
    out.present.push(rec.id);
    if (pred.stale && holdStale) { out.held = true; continue; }
    const dir = Array.isArray(rec.shaft_dir_D) && rec.shaft_dir_D.every(Number.isFinite) ? rec.shaft_dir_D : null;
    const sm = smoother ? smoother.apply(rec.id, rec.tip_D, dir, t) : { tipD: rec.tip_D, dirD: dir };
    const r = solveTip(scene, rec.id, sm.tipD, sm.dirD, state, {
      roll: rec.roll_rad, jaw: rec.jaw_rad, limits,
      hingeD: !smoother?.smooth && Array.isArray(rec.hinge_D) ? rec.hinge_D : null,
    });
    const k = instruments.findIndex((q) => q.id === rec.id);
    if (k >= 0) instruments[k] = { ...instruments[k], ...r.q }; else instruments.push(r.q);
    out.residual_mm[rec.id] = r.residual_mm;
    out.dir_err_deg[rec.id] = r.dir_err_deg;
    out.limited[rec.id] = r.limited;
    out.used.push(rec.id);
  }
  out.state = { ...state, instruments };
  return out;
}

// ------------------------------------------------------------------ the feed: push (SSE) first, poll as the fallback

export const TIPS_PATH = '/api/tips';
export const TIPS_STREAM_PATH = '/api/tips/stream';

/**
 * RigFeed: the tips feed of <base>. It prefers PUSH, an EventSource on <base>/api/tips/stream (Server-Sent Events: one
 * `data:` line per published result, the same JSON as GET /api/tips, `id:` a sequence number, ": keep-alive" comments
 * every 5 s). It falls back to the poller (GET <base>/api/tips at ~hz, no overlapping requests, a per-request timeout)
 * when EventSource is missing, when the stream does not open within openTimeoutMs, when the browser gives the stream up
 * (404 / not text/event-stream from an older server), or after maxStreamErrors errors in a row; while polling it
 * retries the stream every retryStreamMs and stops polling once it opens.
 * Every message, pushed or polled, goes through one path (_accept): parsed JSON, de-duplicated by frame_id (and t; a
 * restarted server counts frame ids again from zero but its t moves on), out-of-order frames from the hand-over dropped,
 * then onResult(pred) and every listener, once per new frame.
 * Status: 'connecting' (nothing yet), 'live' (a result younger than staleS), 'stale' (the last result is older, or the
 * feed says stale), 'down' (no result within staleS and an error: refused, 404, timeout, the stream reconnecting).
 * Keep-alives are SSE comments: EventSource never surfaces them, so they never count as data.
 * Injectable fetch / EventSource / clock / timers for the tests.
 */
export class RigFeed {
  constructor(base, { fetch: f = globalThis.fetch?.bind(globalThis), EventSource: ES = globalThis.EventSource, stream = true,
    now = () => Date.now() / 1000, hz = 15, timeoutMs = 400, staleS = 0.6, openTimeoutMs = 1500, maxStreamErrors = 3, retryStreamMs = 10000,
    setTimer = (fn, ms) => setTimeout(fn, ms), clearTimer = (h) => clearTimeout(h), onResult = null } = {}) {
    this.base = base.replace(/\/+$/, ''); this.url = `${this.base}${TIPS_PATH}`; this.streamUrl = `${this.base}${TIPS_STREAM_PATH}`;
    this.fetch = f; this.ES = typeof ES === 'function' ? ES : null; this.streamWanted = stream !== false;
    this.now = now; this.hz = hz; this.timeoutMs = timeoutMs; this.staleS = staleS;
    this.openTimeoutMs = openTimeoutMs; this.maxStreamErrors = maxStreamErrors; this.retryStreamMs = retryStreamMs;
    this.setTimer = setTimer; this.clearTimer = clearTimer; this.onResult = onResult; this.listeners = new Set();
    this.inFlight = false; this.running = false; this.timer = null; this.polling = false; this.pollGen = 0;
    this.es = null; this.streamOpen = false; this.everOpened = false; this.streamErrors = 0; this.openTimer = null; this.retryTimer = null;
    this.last = null; this.lastAt = null; this.lastKey = null; this.lastT = null; this.lastError = null; this.errorAt = null;
    this.startedAt = null; this.frames = []; this.requests = 0; this.overlaps = 0; this.recent = [];
    this.stats = { stream_msgs: 0, poll_replies: 0, new_frames: 0, dups: 0, late: 0, bad: 0, opens: 0, stream_fails: 0, retries: 0 };
    this.fallbackReason = null; this.lastEventId = null;
  }
  host() { try { return new URL(this.base).host; } catch { return this.base; } }
  /** Can this feed push? (EventSource exists and the stream was not switched off.) */
  streamable() { return !!this.ES && this.streamWanted; }
  /** 'stream' | 'poll' | 'connecting' (a stream opening, no poll) | 'off'. */
  transport() { return !this.running ? 'off' : this.streamOpen ? 'stream' : this.polling ? 'poll' : 'connecting'; }
  start() {
    if (this.running) return;
    this.running = true; this.startedAt = this.now();
    if (this.streamable()) this._openStream(); else this._startPoll(this.ES ? 'stream off' : 'no EventSource');
  }
  /** Close everything: the stream, the poll loop (its request aborted) and the timers. start() opens it again. */
  stop() {
    this.running = false;
    this._closeStream();
    if (this.retryTimer != null) this.clearTimer(this.retryTimer); this.retryTimer = null;
    this._stopPoll();
    this.abort?.abort();
  }
  addListener(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }

  // ---------------------------------------------------------------- push
  _openStream() {
    if (!this.running || this.es) return;
    let es;
    try { es = new this.ES(this.streamUrl); } catch (e) { this._streamFailed(`EventSource: ${e?.message || e}`); return; }
    this.es = es; this.streamOpen = false; this.everOpened = false; this.streamErrors = 0;
    this.openTimer = this.setTimer(() => { if (this.es === es && !this.everOpened) this._streamFailed(`stream did not open in ${this.openTimeoutMs} ms`); }, this.openTimeoutMs);
    es.onopen = () => {
      if (this.es !== es) return;
      if (this.openTimer != null) { this.clearTimer(this.openTimer); this.openTimer = null; }
      this.streamOpen = true; this.everOpened = true; this.streamErrors = 0; this.stats.opens++;
      this.lastError = null; this.errorAt = null; this.fallbackReason = null;
      if (this.retryTimer != null) { this.clearTimer(this.retryTimer); this.retryTimer = null; }
      this._stopPoll();                                     // pushed now: the poller (a fallback) goes
    };
    es.onmessage = (ev) => {
      if (this.es !== es) return;
      this.streamErrors = 0; this.stats.stream_msgs++;
      if (ev?.lastEventId) this.lastEventId = ev.lastEventId;
      let j = null;
      try { j = JSON.parse(ev.data); } catch { this.stats.bad++; return; }
      if (!j || !Array.isArray(j.instruments)) { this.stats.bad++; return; }
      this._accept(j);
    };
    es.onerror = () => {
      if (this.es !== es) return;
      this.streamOpen = false; this.streamErrors++;
      this.lastError = 'stream reconnecting'; this.errorAt ??= this.now();
      // CLOSED: the browser gave up (an older server: 404, or not text/event-stream); else it reconnects by itself
      if (es.readyState === 2) this._streamFailed('stream closed by the browser');
      else if (this.streamErrors >= this.maxStreamErrors) this._streamFailed(`${this.streamErrors} stream errors in a row`);
    };
  }
  _closeStream() {
    if (this.openTimer != null) { this.clearTimer(this.openTimer); this.openTimer = null; }
    const es = this.es;
    this.es = null; this.streamOpen = false;
    if (es) { es.onopen = es.onmessage = es.onerror = null; try { es.close(); } catch { /* closed */ } }
  }
  _streamFailed(why) {
    this._closeStream();
    this.stats.stream_fails++; this.fallbackReason = why;
    if (!this.running) return;
    this._startPoll(why);
    if (this.retryTimer == null && this.streamable()) {
      this.retryTimer = this.setTimer(() => { this.retryTimer = null; if (this.running && !this.es) { this.stats.retries++; this._openStream(); } }, this.retryStreamMs);
    }
  }

  // ---------------------------------------------------------------- poll (the fallback, and older servers)
  _startPoll(why = null) {
    if (why) this.fallbackReason = why;
    if (this.polling || !this.running) return;
    this.polling = true;
    this._loop(++this.pollGen);
  }
  _stopPoll() {
    this.polling = false; this.pollGen++;
    if (this.timer != null) this.clearTimer(this.timer);
    this.timer = null;
  }
  async _loop(gen = this.pollGen) {
    if (!this.running || !this.polling || gen !== this.pollGen) return;
    const t0 = this.now();
    await this.poll();
    if (!this.running || !this.polling || gen !== this.pollGen) return;
    const wait = Math.max(0, 1000 / this.hz - (this.now() - t0) * 1000);
    this.timer = this.setTimer(() => this._loop(gen), wait);
  }
  /** One request (skipped, returning null, when another is still in flight). */
  async poll() {
    if (this.inFlight) { this.overlaps++; return null; }
    this.inFlight = true; this.requests++;
    const ac = typeof AbortController !== 'undefined' ? new AbortController() : null;
    this.abort = ac;
    const to = this.setTimer(() => ac?.abort(), this.timeoutMs);
    try {
      const res = await this.fetch(this.url, { cache: 'no-store', signal: ac?.signal, mode: 'cors' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = await res.json();
      if (!j || !Array.isArray(j.instruments)) throw new Error('no instruments in the reply');
      this.stats.poll_replies++;
      this._accept(j);
      return j;
    } catch (e) {
      // a poll that ends after the stream took over says nothing about the feed
      if (!this.streamOpen) { this.lastError = e?.name === 'AbortError' ? 'timeout' : (e?.message || String(e)); this.errorAt ??= this.now(); }
      return null;
    } finally {
      this.clearTimer(to);
      this.inFlight = false;
    }
  }

  // ---------------------------------------------------------------- one path for every message
  /** A valid payload from either transport: dedup (frame_id|t), drop out-of-order, then the callbacks. */
  _accept(j) {
    const at = this.now();
    this.last = j; this.lastAt = at; this.lastError = null; this.errorAt = null;
    const key = `${j.frame_id ?? ''}|${j.t ?? ''}`;
    if (key === this.lastKey || this.recent.includes(key)) { this.stats.dups++; return false; }
    // a frame older than the newest one (by < 1 s: a hand-over between poll and stream; a bigger jump is a new clock)
    if (Number.isFinite(j.t) && Number.isFinite(this.lastT) && j.t < this.lastT && this.lastT - j.t < 1.0) { this.stats.late++; return false; }
    this.lastKey = key; this.recent.push(key); if (this.recent.length > 32) this.recent.shift();
    if (Number.isFinite(j.t)) this.lastT = j.t;
    this.frames.push(at); this.stats.new_frames++;
    while (this.frames.length && this.frames[0] < at - 2) this.frames.shift();
    this.onResult?.(j);
    for (const fn of this.listeners) fn(j);
    return true;
  }
  /** Frames per second over the last 2 s of new results. */
  rate() {
    const now = this.now();
    const f = this.frames.filter((x) => x >= now - 2);
    if (f.length < 2) return 0;
    return (f.length - 1) / Math.max(1e-3, f[f.length - 1] - f[0]);
  }
  status() {
    const now = this.now();
    if (this.lastAt == null) return this.lastError ? 'down' : 'connecting';
    if (now - this.lastAt > this.staleS) return this.lastError ? 'down' : 'stale';
    if (this.last?.stale) return 'stale';
    return 'live';
  }
  /** Numbers for the page's test hook. */
  info() {
    return { transport: this.transport(), status: this.status(), rate: this.rate(), fallback: this.fallbackReason, last_event_id: this.lastEventId,
      requests: this.requests, ...this.stats };
  }
}

/**
 * FeedHub: one RigFeed per base URL for the whole page, reference-counted. Rig mode and the camera overlay acquire the
 * same feed, so the page opens ONE EventSource per server (Chrome allows 6 connections per host, and the camera images
 * need theirs). The poll rate is the highest any holder asked for. While the page is hidden every feed is closed; it
 * reopens (stream first) when the page shows again.
 *   acquire(base, onResult?, opts?) -> {feed, release()}     configure(opts): defaults for feeds made from now on
 */
export class FeedHub {
  constructor({ make = (base, opts) => new RigFeed(base, opts), doc = undefined, win = undefined, defaults = {} } = {}) {
    this.make = make; this.doc = doc; this.win = win; this.defaults = { ...defaults }; this.feeds = new Map(); this.hidden = false; this.bound = false;
  }
  configure(opts) { Object.assign(this.defaults, opts); return this; }
  _bind() {
    if (this.bound) return;
    this.bound = true;
    const doc = this.doc === undefined ? (typeof document !== 'undefined' ? document : null) : this.doc;
    const win = this.win === undefined ? (typeof window !== 'undefined' ? window : null) : this.win;
    if (doc?.addEventListener) {
      this.hidden = doc.visibilityState === 'hidden';
      doc.addEventListener('visibilitychange', () => this.setHidden(doc.visibilityState === 'hidden'));
    }
    if (win?.addEventListener) {
      win.addEventListener('pagehide', () => this.setHidden(true));
      win.addEventListener('pageshow', () => this.setHidden(doc ? doc.visibilityState === 'hidden' : false));
    }
  }
  setHidden(h) {
    this.hidden = !!h;
    for (const e of this.feeds.values()) { if (this.hidden) e.feed.stop(); else if (e.subs.size) e.feed.start(); }
  }
  acquire(base, onResult = null, opts = {}) {
    this._bind();
    const key = String(base).replace(/\/+$/, '');
    let e = this.feeds.get(key);
    if (!e) { e = { feed: this.make(key, { ...this.defaults, ...opts }), subs: new Map() }; this.feeds.set(key, e); }
    const id = Symbol('sub');
    e.subs.set(id, { hz: Number.isFinite(opts.hz) ? opts.hz : null, fn: onResult });
    if (onResult) e.feed.addListener(onResult);
    this._hz(e);
    if (!this.hidden) e.feed.start();
    let done = false;
    return {
      feed: e.feed,
      release: () => {
        if (done) return; done = true;
        if (onResult) e.feed.listeners.delete(onResult);
        e.subs.delete(id);
        if (!e.subs.size) { e.feed.stop(); if (this.feeds.get(key) === e) this.feeds.delete(key); } else this._hz(e);
      },
    };
  }
  _hz(e) { const hz = [...e.subs.values()].map((s) => s.hz).filter(Number.isFinite); if (hz.length) e.feed.hz = Math.max(...hz); }
  /** The feed currently held for base, or null. */
  get(base) { return this.feeds.get(String(base).replace(/\/+$/, ''))?.feed ?? null; }
}

/** The page's hub (twinpage/main.js rig mode and twinpage/real_cams.js's overlay). */
export const feedHub = new FeedHub();

const mean = (o) => { const v = Object.values(o || {}).filter(Number.isFinite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };

/** The quiet line for rig mode. info: {label ('Real rig'), waiting (the text while no tips came), status, host, rate, smooth_s (snapshot interpolation's delay, s), residual_mm:
 * {L, R}, fit_mm: {L, R} (the feed's own residual_mm), lost: [ids not seen lately], model}. */
export function rigLine(info) {
  const head = info.label || 'Real rig';                   // 'Browser tracker' for the in-page tracker (tracker_source.js)
  if (info.status === 'down' || info.status === 'connecting') return info.waiting ?? `${head}: no tips from ${info.host}`;
  const parts = [head, `${Math.round(info.rate || 0)} Hz`];
  if (Number.isFinite(info.smooth_s)) parts.push(`${info.smooth_s.toFixed(1)} s smooth`);   // rig_smooth.js's delay
  if (info.status === 'stale') parts.push('stale');
  const res = mean(info.residual_mm), fit = mean(info.fit_mm), fu = info.fit_unit || 'mm';
  if (res != null) parts.push(`axis vs pivot ${res.toFixed(1)} mm${fit != null ? `, fit ${fit.toFixed(1)} ${fu}` : ''}`);
  else if (fit != null) parts.push(`fit ${fit.toFixed(1)} ${fu}`);
  for (const id of info.lost || []) parts.push(`${id} not seen`);
  if (info.model) parts.push(`model ${info.model}`);
  return parts.join(' · ');
}

/** How long an omitted instrument holds before it shows as lost: 0.6 s, or three frame intervals of a slow feed. */
export const lostAfter = (rate) => Math.max(0.6, 3 / Math.max(rate || 0, 1));

/** Instruments omitted from the feed for longer than `after` s (they hold their last pose; the line says "R not seen"). */
export function lostIds(lastSeen, now, ids, after = 0.6) {
  return ids.filter((id) => lastSeen[id] == null || now - lastSeen[id] > after);
}

/**
 * Distance (mm) between each working port's CAD pivot and the calibrated pivot_D in /api/calibration (informational).
 * calib.ports may be a list [{id, pivot_D}] or a map {id: {pivot_D}}; ids are port ids or instrument ids.
 */
export function pivotOffsets(scene, calib) {
  const out = {};
  const ports = calib?.ports;
  if (!ports) return out;
  const list = Array.isArray(ports) ? ports : Object.entries(ports).map(([id, v]) => ({ id, ...(v || {}) }));
  for (const c of list) {
    const pv = c?.pivot_D;
    if (!Array.isArray(pv) || pv.length !== 3 || !pv.every(Number.isFinite)) continue;
    const ins = scene.instruments.find((i) => i.id === c.id || i.port === c.id);
    const pid = scene.ports.some((p) => p.id === c.id) ? c.id : ins?.port;
    if (!pid) continue;
    out[c.id] = norm(sub(pv, port(scene, pid).pivot)) * 1000;
  }
  return out;
}

export { add };
