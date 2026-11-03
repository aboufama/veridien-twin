// Snapshot (entity) interpolation of the rig feed (docs/WEB_TWIN.md §20, ADR-133): how networked games draw remote
// entities (Source's cl_interp, Gambetta's "entity interpolation"), applied to the CV model's tool tips.
//
// The feed delivers predictions at irregular times (5 Hz in replay, 15-30 Hz live). Each tool's snapshots are kept with
// the SENDER's t. The page renders at render_t = now - L, where L = (receive - send offset, a low percentile of recent
// frames) + delay (1.5 x the median interval + 2 x the jitter, 60-400 ms: about 85 ms on the live 24 Hz feed, 0.4 s on
// the 5 Hz replay). intervalK = 2 would keep render time behind every provisional end tangent (ADR-133 has the numbers).
// L slews by at most `warp` (10 %) of elapsed time, so render time never jumps and never runs backwards.
//   tip   : a cubic Hermite through every snapshot AT ITS OWN t (exact: h00(0) = h01(1) = 1), tangents from the
//           non-uniform three-point (Catmull-Rom) derivative (C1). limiter: true adds Steffen's limiter (no overshoot).
//   dir   : slerp between the neighbouring snapshots.   jaw, roll : linear (roll the short way round).
// Past the newest snapshot it extrapolates linearly for <= extrapMax (100 ms), then holds; a feed that stays silent
// for settleS (0.4 s) more eases the tool back onto its last snapshot, so a stopped feed leaves it exactly there. When the curve under the
// current render time changes (a new snapshot replaced the provisional end tangent, or data resumed after a stall),
// the difference is blended out over `blend` (150 ms, smoothstep, so velocity stays continuous): nothing snaps back.
// A jump > 25 mm or 15 deg between consecutive snapshots, or a tool back after > lostS, is a cut: interpolation never
// crosses it and the drawn tool snaps when render time reaches it.
// Pure: no DOM, no clocks of its own (times are passed in, seconds).

export const SMOOTH_DEFAULTS = Object.freeze({
  maxSnaps: 8,            // per tool, kept while older ones are still needed for the segment under render time
  minDelay: 0.06, maxDelay: 0.4, fixedDelay: null,   // s; fixedDelay (e.g. ?rigDelay=200 -> 0.2) overrides the estimate
  intervalK: 1.5, jitterK: 2,                         // delay = intervalK * median interval + jitterK * jitter
  defaultInterval: 0.1,   // s, until two frames have been seen
  extrapMax: 0.1,         // s of linear extrapolation past the newest snapshot, then hold
  settleS: 0.4,           // s of that hold before the tool eases (over `blend`) onto the newest snapshot exactly
  blend: 0.15,            // s to blend a correction out
  teleportM: 0.025, teleportRad: 15 * Math.PI / 180,  // a jump bigger than this between consecutive snapshots snaps
  lostS: 0.8,             // s without a snapshot (sender time; or 4 intervals of a slow feed) -> the next one snaps
  warp: 0.1,              // |dL/dt| <= warp: render time runs at 0.9-1.1 x real time while the delay adapts
  window: 64,             // frames in the clock statistics
  offsetPct: 0.1,         // the offset is this percentile of (receive - send) over the window
  resyncS: 1.0,           // an offset change bigger than this (a restarted sender clock) resets everything
  limiter: false,         // true: Steffen's limiter on the tip tangents (default: plain non-uniform Catmull-Rom)
});

const sortNum = (a) => [...a].sort((x, y) => x - y);
const pct = (s, p) => (s.length ? s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))] : NaN);
const median = (a) => pct(sortNum(a), 0.5);
const mad = (a) => { if (a.length < 2) return 0; const m = median(a); return median(a.map((x) => Math.abs(x - m))); };
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm3 = (a) => Math.hypot(a[0], a[1], a[2]);
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit3 = (a) => { const n = norm3(a); return n > 0 ? [a[0] / n, a[1] / n, a[2] / n] : null; };
const finite3 = (a) => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);
const angleBetween = (a, b) => Math.acos(Math.max(-1, Math.min(1, dot3(a, b))));
const wrapPi = (x) => x - 2 * Math.PI * Math.round(x / (2 * Math.PI));
const smoothstep = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

/** Slerp between unit vectors a and b (s in [0, 1]); exact at both ends. */
export function slerpDir(a, b, s) {
  if (s <= 0) return [...a];
  if (s >= 1) return [...b];
  const c = Math.max(-1, Math.min(1, dot3(a, b)));
  const th = Math.acos(c);
  if (th < 1e-9) return unit3([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s]);
  const sa = Math.sin((1 - s) * th) / Math.sin(th), sb = Math.sin(s * th) / Math.sin(th);
  return unit3([a[0] * sa + b[0] * sb, a[1] * sa + b[1] * sb, a[2] * sa + b[2] * sb]);
}

/** Steffen (1990) tangent from the secants either side (sl, sr over hl, hr): monotone, C1, exact for lines. */
function steffen(sl, sr, hl, hr) {
  const p = (sl * hr + sr * hl) / (hl + hr);
  if (sl * sr <= 0) return 0;
  const m = Math.min(Math.abs(sl), Math.abs(sr), 0.5 * Math.abs(p));
  return Math.sign(sl) * 2 * m;
}

/** The feed's clock: offset = local receive - sender t (low percentile), interval median and jitter. */
export class FeedClock {
  constructor(cfg = {}) { this.cfg = { ...SMOOTH_DEFAULTS, ...cfg }; this.reset(); }
  reset() { this.lastT = null; this.offs = []; this.ivs = []; this._cache = null; this.frames = 0; }
  /** One frame (sender t, local receive time). Returns 'resync' when the sender's clock jumped, else true/false. */
  observe(t, recv) {
    if (!Number.isFinite(t) || !Number.isFinite(recv)) return false;
    const o = recv - t;
    if (this.offs.length && Math.abs(o - this.offset()) > this.cfg.resyncS) { this.reset(); this.lastT = t; this.offs.push(o); this.frames = 1; return 'resync'; }
    if (this.lastT != null && !(t > this.lastT)) return false;
    if (this.lastT != null && t - this.lastT < 1.0) { this.ivs.push(t - this.lastT); if (this.ivs.length > this.cfg.window) this.ivs.shift(); }
    this.lastT = t;
    this.offs.push(o); if (this.offs.length > this.cfg.window) this.offs.shift();
    this.frames++; this._cache = null;
    return true;
  }
  _stats() {
    if (this._cache) return this._cache;
    const so = sortNum(this.offs);
    const offset = pct(so, this.cfg.offsetPct);
    const interval = this.ivs.length ? median(this.ivs) : this.cfg.defaultInterval;
    const late = this.offs.map((o) => o - offset);
    const sInt = 1.4826 * mad(this.ivs), sLate = 1.4826 * mad(late);
    const jitter = Math.hypot(sInt, sLate);
    return (this._cache = { offset, interval, jitter, sInt, sLate });
  }
  offset() { return this.offs.length ? this._stats().offset : NaN; }
  interval() { return this._stats().interval; }
  jitter() { return this._stats().jitter; }
  /** The delay the statistics ask for (s), clamped; fixedDelay when set. */
  targetDelay() {
    const c = this.cfg;
    if (Number.isFinite(c.fixedDelay)) return c.fixedDelay;
    const { interval, jitter } = this._stats();
    return Math.max(c.minDelay, Math.min(c.maxDelay, c.intervalK * interval + c.jitterK * jitter));
  }
}

/**
 * SnapshotInterp: per-tool snapshot buffers and the render clock.
 *   pushFrame(pred, recv, ids?)  one feed payload received at local time recv (s); ids limits which tools are taken.
 *   push(id, t, recv, {tip, dir, jaw, roll})  one tool snapshot (sender time t).
 *   advance(now) -> render_t   moves the render clock (call once per rendered frame, before sample).
 *   sample(id, now) -> {tip, dir, jaw, roll, mode, corr_m} | null   mode: 'wait' | 'interp' | 'extrap' | 'hold'.
 *   curveAt(id, t)   the curve itself at sender time t (no correction): equals each snapshot at its t.
 */
export class SnapshotInterp {
  constructor(cfg = {}) {
    this.cfg = { ...SMOOTH_DEFAULTS, ...cfg };
    this.clock = new FeedClock(this.cfg);
    this.reset();
  }
  reset() { this.clock.reset(); this.tools = {}; this.L = null; this.lastNow = null; this.renderT = null; this.delayS = null; this.cuts = 0; this.corrections = 0; }
  ids() { return Object.keys(this.tools); }
  ready() { return this.L != null; }
  lostS() { return Math.max(this.cfg.lostS, 4 * this.clock.interval()); }

  pushFrame(pred, recv, ids = null) {
    if (!pred || !Array.isArray(pred.instruments) || !Number.isFinite(pred.t)) return [];
    const ok = this.clock.observe(pred.t, recv);
    if (ok === 'resync') { const c = this.clock; this.tools = {}; this.L = null; this.lastNow = null; this.renderT = null; this.clock = c; }
    else if (!ok) return [];
    const used = [];
    for (const rec of pred.instruments) {
      if (!rec || !finite3(rec.tip_D) || (ids && !ids.includes(rec.id))) continue;
      const dir = finite3(rec.shaft_dir_D) ? unit3(rec.shaft_dir_D) : null;
      if (this.push(rec.id, pred.t, recv, { tip: rec.tip_D, dir, jaw: Number.isFinite(rec.jaw_rad) ? rec.jaw_rad : null, roll: Number.isFinite(rec.roll_rad) ? rec.roll_rad : null })) used.push(rec.id);
    }
    return used;
  }

  push(id, t, recv, snap) {
    const tool = (this.tools[id] ??= { buf: [], corr: null, last: null });
    const buf = tool.buf, prev = buf[buf.length - 1];
    if (prev && !(t > prev.t)) return false;
    let cut = !prev;
    if (prev) {
      const jump = norm3(sub3(snap.tip, prev.tip)) > this.cfg.teleportM;
      const turn = snap.dir && prev.dir && angleBetween(snap.dir, prev.dir) > this.cfg.teleportRad;
      cut = jump || turn || t - prev.t > this.lostS();
    }
    const s = { t, tip: [...snap.tip], dir: snap.dir ? [...snap.dir] : null, jaw: snap.jaw ?? null, roll: snap.roll ?? null, cut };
    const L = tool.last;   // what was drawn last: keep it continuous under the new curve (unless this is a cut it already passed)
    const before = L && !cut ? this._tipAt(tool, L.r) : null;
    buf.push(s);
    if (cut) this.cuts += prev ? 1 : 0;
    if (L && cut && t <= L.r) tool.corr = null;
    else if (before) {
      const after = this._tipAt(tool, L.r);
      const d = sub3(before, after);
      if (norm3(d) > 1e-12) {
        const cur = this._corrAt(tool, L.now);
        tool.corr = { v: [cur[0] + d[0], cur[1] + d[1], cur[2] + d[2]], at: L.now };
        this.corrections++;
      }
    }
    // keep maxSnaps, but never drop the nodes the segment under render time (and its left tangent) still needs
    const r = L?.r ?? -Infinity;
    while ((buf.length > this.cfg.maxSnaps && buf[2] && buf[2].t <= r) || buf.length > 64) buf.shift();
    return true;
  }

  /** Move the render clock to local time `now` (s); returns render_t (sender time). */
  advance(now) {
    const off = this.clock.offset();
    if (!Number.isFinite(off)) return null;
    const target = off + this.clock.targetDelay();
    if (this.L == null || this.lastNow == null) this.L = target;
    else {
      const dt = Math.max(0, now - this.lastNow), step = this.cfg.warp * dt;
      this.L += Math.max(-step, Math.min(step, target - this.L));
    }
    if (this.lastNow == null || now >= this.lastNow) this.lastNow = now;
    this.delayS = this.L - off;
    this.renderT = now - this.L;
    return this.renderT;
  }

  _corrAt(tool, now) {
    const c = tool.corr;
    if (!c) return [0, 0, 0];
    const w = 1 - smoothstep((now - c.at) / this.cfg.blend);
    return [c.v[0] * w, c.v[1] * w, c.v[2] * w];
  }

  /** Index i of the snapshot at or before t (-1 before the first). */
  _seg(buf, t) {
    let i = buf.length - 1;
    while (i >= 0 && buf[i].t > t) i--;
    return i;
  }

  /** Tip tangent (m/s) at node i, using only neighbours in the same run (no cut between). */
  _tangent(buf, i) {
    const a = buf[i];
    const hasPrev = i > 0 && !a.cut, hasNext = i + 1 < buf.length && !buf[i + 1].cut;
    if (!hasPrev && !hasNext) return [0, 0, 0];
    if (!hasPrev || !hasNext) {
      const [p, q] = hasNext ? [a, buf[i + 1]] : [buf[i - 1], a];
      const h = q.t - p.t;
      return [(q.tip[0] - p.tip[0]) / h, (q.tip[1] - p.tip[1]) / h, (q.tip[2] - p.tip[2]) / h];
    }
    const p = buf[i - 1], q = buf[i + 1];
    const hl = a.t - p.t, hr = q.t - a.t;
    const m = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      const sl = (a.tip[k] - p.tip[k]) / hl, sr = (q.tip[k] - a.tip[k]) / hr;
      m[k] = this.cfg.limiter ? steffen(sl, sr, hl, hr) : (sl * hr + sr * hl) / (hl + hr);
    }
    return m;
  }

  /** The tip curve at sender time t, with its mode. */
  _eval(tool, t) {
    const buf = tool.buf;
    if (!buf.length) return null;
    const i = this._seg(buf, t);
    if (i < 0) return { tip: [...buf[0].tip], mode: 'wait', i: 0, s: 0 };
    const a = buf[i];
    if (t === a.t) return { tip: [...a.tip], mode: 'interp', i, s: 0 };
    const b = buf[i + 1];
    if (!b) {                                  // past the newest: linear extrapolation, then hold
      const over = t - a.t, dt = Math.min(over, this.cfg.extrapMax);
      const v = i > 0 && !a.cut ? this._tangent(buf, i) : [0, 0, 0];
      // a feed that stops leaves the tool on its last prediction, exactly: after settleS the guess eases back
      const w = smoothstep((over - this.cfg.extrapMax - this.cfg.settleS) / this.cfg.blend);
      const x = [a.tip[0] + v[0] * dt, a.tip[1] + v[1] * dt, a.tip[2] + v[2] * dt];
      const tip = w >= 1 ? [...a.tip] : x.map((xk, k) => xk + (a.tip[k] - xk) * w);
      return { tip, mode: over <= this.cfg.extrapMax ? 'extrap' : 'hold', i, s: 1 };
    }
    if (b.cut) return { tip: [...a.tip], mode: 'hold', i, s: 0 };   // a cut ahead: hold, then snap when render time reaches it
    const h = b.t - a.t, s = (t - a.t) / h;
    const m0 = this._tangent(buf, i), m1 = this._tangent(buf, i + 1);
    const s2 = s * s, s3 = s2 * s;
    const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
    const tip = [0, 1, 2].map((k) => h00 * a.tip[k] + h10 * h * m0[k] + h01 * b.tip[k] + h11 * h * m1[k]);
    return { tip, mode: 'interp', i, s };
  }
  _tipAt(tool, t) { return this._eval(tool, t)?.tip ?? null; }

  /** The curve at sender time t (no correction). */
  curveAt(id, t) { const tool = this.tools[id]; return tool ? this._tipAt(tool, t) : null; }

  /** The drawn sample for tool id at local time now (call advance(now) first; it is called here if needed). */
  sample(id, now) {
    const tool = this.tools[id];
    if (!tool || !tool.buf.length) return null;
    if (this.lastNow !== now || this.renderT == null) this.advance(now);
    const r = this.renderT;
    if (r == null) return null;
    const e = this._eval(tool, r);
    const buf = tool.buf;
    // crossing a cut drops any correction: a teleport snaps
    if (tool.last && buf.some((x) => x.cut && x.t > tool.last.r && x.t <= r)) tool.corr = null;
    const c = this._corrAt(tool, now);
    if (tool.corr && now - tool.corr.at >= this.cfg.blend) tool.corr = null;
    const tip = [e.tip[0] + c[0], e.tip[1] + c[1], e.tip[2] + c[2]];
    const a = buf[e.i], b = e.mode === 'interp' && e.s > 0 ? buf[e.i + 1] : null;
    let dir = a.dir, jaw = a.jaw, roll = a.roll;
    if (b) {
      if (a.dir && b.dir) dir = slerpDir(a.dir, b.dir, e.s);
      if (a.jaw != null && b.jaw != null) jaw = a.jaw + (b.jaw - a.jaw) * e.s;
      if (a.roll != null && b.roll != null) roll = a.roll + wrapPi(b.roll - a.roll) * e.s;
    }
    tool.last = { r, now, tip };
    return { tip, dir: dir ? [...dir] : null, jaw, roll, mode: e.mode, corr_m: norm3(c), render_t: r };
  }

  /** sample() for every tool: {id: sample}. */
  sampleAll(now) {
    this.advance(now);
    const out = {};
    for (const id of this.ids()) { const s = this.sample(id, now); if (s) out[id] = s; }
    return out;
  }

  /** Numbers for the page's test hook and the quiet line. */
  info() {
    return { delay_s: this.delayS, target_delay_s: this.clock.frames ? this.clock.targetDelay() : null, interval_s: this.clock.interval(),
      jitter_s: this.clock.jitter(), offset_s: this.clock.offset(), frames: this.clock.frames, cuts: this.cuts, corrections: this.corrections };
  }
}
