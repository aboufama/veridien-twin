// Calm output for the twin, after the tip filter: the JS twin of src/veridien/multicam/steady.py (ghost veto, teleport
// guard, hidden tip kept on its shaft, soft re-entry) and shaftdir.py (a port's shaft direction from the shaft
// silhouettes alone). numpy quirks kept on purpose so the output matches the live server: np.gradient over the whole
// flattened (angle, side, sample) grid, float32 contrast arithmetic with pairwise sums, first-maximum argmax and
// floor-mod angles. Rig frame, metres, seconds; gray images at scale `sc` of the full-res frame (net input: 0.4).

import { project, toCam } from './lens.js';
import { add, sub, scale, dot, norm, cross, linspace, arange, floorMod, radians, eigSym3, npSum32, f32 } from './math.js';
import { sampleBilinear } from './preprocess.js';

export const STEADY_DEFAULTS = Object.freeze({ ghost_m: 0.006, jump_m: 0.030, confirm: 3, confirm_m: 0.010, blend_sets: 4,
  alpha_u: 0.3, max_est_s: 4.0, shaft: true, min_contrast: 0.25 });

export const S_ALONG = linspace(0.03, 0.22, 24);

/** shaftdir._sample: float32 bilinear samples; NaN outside 2 < u < w-3, 2 < v < h-3. */
function sample(g, u, v) {
  if (!(Number.isFinite(u) && Number.isFinite(v) && u > 2 && v > 2 && u < g.width - 3 && v < g.height - 3)) return NaN;
  return sampleBilinear(g, u, v, 'constant');
}

/** shaftdir._contrast: per-sample (side - centre) / side along rig points X[] (NaN where invalid). */
export function contrast(g, cam, X, radius = 0.0025, sc = 1.0) {
  const n = X.length;
  const Xc = X.map((p) => toCam(cam, p));
  const front = Xc.map((p) => p[2] > 0.01);
  const uv = Xc.map((p, i) => {
    if (!front[i]) return [NaN, NaN];
    const q = project(cam.lens, p);
    return [(q[0] + 0.5) * sc - 0.5, (q[1] + 0.5) * sc - 0.5];
  });
  const z = uv.map((p) => p.map((x) => (Number.isNaN(x) ? 0 : x)));        // np.nan_to_num
  const out = new Array(n);
  const K00 = cam.lens.fx;
  for (let i = 0; i < n; i++) {
    let t;                                                                 // np.gradient(axis=0), edge_order 1
    if (n === 1) t = [0, 0];
    else if (i === 0) t = [z[1][0] - z[0][0], z[1][1] - z[0][1]];
    else if (i === n - 1) t = [z[n - 1][0] - z[n - 2][0], z[n - 1][1] - z[n - 2][1]];
    else t = [(z[i + 1][0] - z[i - 1][0]) / 2.0, (z[i + 1][1] - z[i - 1][1]) / 2.0];
    const tl = Math.max(Math.hypot(t[0], t[1]), 1e-9);
    t = [t[0] / tl, t[1] / tl];
    const nr = [-t[1], t[0]];
    const off = Math.min(Math.max(2.5 * radius * K00 * sc / Math.max(Xc[i][2], 0.01), 2.0), 40.0);
    const [u, v] = uv[i];
    const c = sample(g, u, v), l = sample(g, u + off * nr[0], v + off * nr[1]), r = sample(g, u - off * nr[0], v - off * nr[1]);
    const side = 0.5 * (l + r);
    const con = f32((side - c) / Math.max(side, 8.0));
    out[i] = Number.isFinite(con) && front[i] ? con : NaN;
  }
  return out;
}

function planeBasis(cam, P) {
  let a = sub(P, cam.C);
  a = scale(a, 1 / norm(a));
  let e1 = cross(a, Math.abs(a[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0]);
  e1 = scale(e1, 1 / norm(e1));
  return { a, e1, e2: cross(a, e1) };
}

/** shaftdir._plane_scores -> {N[f], W[f], mean[f][side], cnt[f][side]}. */
export function planeScores(g, cam, P, phis, s, sc) {
  const { a, e1, e2 } = planeBasis(cam, P);
  const N = phis.map((f) => add(scale(e1, Math.cos(f)), scale(e2, Math.sin(f))));
  const W = N.map((nv) => cross(nv, a));
  const X = [];
  for (let f = 0; f < phis.length; f++) for (const sg of [1.0, -1.0]) for (const sj of s) X.push(add(P, scale(W[f], sg * sj)));
  const con = contrast(g, cam, X, 0.0025, sc);
  const S = s.length, mean = [], cnt = [];
  for (let f = 0; f < phis.length; f++) {
    mean.push([]); cnt.push([]);
    for (let side = 0; side < 2; side++) {
      const seg = con.slice((f * 2 + side) * S, (f * 2 + side + 1) * S);
      const c = seg.filter((x) => !Number.isNaN(x)).length;
      const sum = npSum32(seg.map((x) => (Number.isNaN(x) ? 0 : x)));
      mean[f].push(c > 0 ? sum / Math.max(c, 1) : -1.0);
      cnt[f].push(c);
    }
  }
  return { N, W, mean, cnt };
}

/** shaftdir.camera_plane -> [n, w, contrast] or null. */
export function cameraPlane(g, cam, P, priorU = null, s = S_ALONG, sc = 1.0, minValid = 6) {
  const { a, e1, e2 } = planeBasis(cam, P);
  let phis;
  if (priorU) {
    let n0 = cross(a, priorU);
    n0 = scale(n0, 1 / Math.max(norm(n0), 1e-9));
    const phi0 = floorMod(Math.atan2(dot(n0, e2), dot(n0, e1)), Math.PI);
    phis = arange(-12, 12.1, 1.5).map((x) => floorMod(phi0 + radians(x), Math.PI));
  } else phis = arange(0, 180, 3.0).map(radians);
  let best = null;
  for (let round = 0; round < 2; round++) {
    const { N, W, mean, cnt } = planeScores(g, cam, P, phis, s, sc);
    let bf = 0, bs = 0, bv = -Infinity;
    for (let f = 0; f < phis.length; f++) for (let side = 0; side < 2; side++) {
      const m = cnt[f][side] >= minValid ? mean[f][side] : -1.0;
      if (m > bv) { bv = m; bf = f; bs = side; }
    }
    if (bv <= -1.0) return null;
    best = [N[bf], scale(W[bf], bs === 0 ? 1.0 : -1.0), bv];
    phis = arange(-2.0, 2.01, 0.25).map((x) => floorMod(phis[bf] + radians(x), Math.PI));
  }
  return best;
}

let _tieCount = 0;
/** How many times shaftDirection met a 0-0 side vote (Python keeps LAPACK's arbitrary sign there; here: toward prior). */
export const shaftTieCount = () => _tieCount;

/** shaftdir.shaft_direction -> {u, cams, visible_m} or null. grays: {camera index: gray}. */
export function shaftDirection(grays, cams, P, priorU = null, minContrast = 0.25, sc = 1.0) {
  const got = [];
  for (const k of Object.keys(grays).map(Number)) {
    const r = cameraPlane(grays[k], cams[k], P, priorU, S_ALONG, sc);
    if (r !== null && r[2] >= minContrast) got.push([k, r]);
  }
  if (got.length < 2) return null;
  const N = got.map(([, r]) => r[0]);
  let u;
  if (N.length === 2) {
    u = cross(N[0], N[1]);
    u = scale(u, 1 / norm(u));
  } else {
    const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const nv of N) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) S[i][j] += nv[i] * nv[j];
    u = eigSym3(S).vectors[0];
  }
  let votes = 0;
  for (const [, r] of got) votes += Math.sign(dot(u, r[1]));
  if (votes < 0) u = scale(u, -1);
  else if (votes === 0) {
    _tieCount += 1;
    if (priorU && dot(u, priorU) < 0) u = scale(u, -1);
  }
  const s = linspace(0.02, 0.40, 77);
  let vis = 0.0;
  for (const [k] of got) {
    const con = contrast(grays[k], cams[k], s.map((si) => add(P, scale(u, si))), 0.0025, sc);
    let end = 0.0, started = false;
    for (let j = 0; j < s.length; j++) {
      const seen = Number.isFinite(con[j]);
      if (seen && con[j] > 0.5 * minContrast) { started = true; end = s[j]; }
      else if (seen && started) break;
    }
    vis = Math.max(vis, end);
  }
  return { u, cams: got.map(([k]) => k).sort((a, b) => a - b), visible_m: vis };
}

/** steady.SteadyTips: per port, filtered tip -> (tip, mode) with mode "measured" | "blend" | "estimated", or null. */
export class SteadyTips {
  constructor(ports, cams, cfg = {}) {
    this.ports = ports;
    this.cams = cams;
    this.cfg = { ...STEADY_DEFAULTS, ...cfg };
    this.reset();
  }

  reset() {
    this.s = Object.fromEntries(this.ports.map((p) => [p.name, { tip: null, meas: null, u: null, d: null, t_meas: -1e9,
      pending: [], blend_from: null, blend_k: 0 }]));
  }

  /** tips: {port: tip or null}; grays: {camera index: gray}. -> {port: [tip, mode] or null} */
  step(tips, t, grays = null, sc = 1.0) {
    const c = this.cfg;
    const z = {};
    for (const [n, v] of Object.entries(tips)) z[n] = v == null ? null : v.slice();
    const names = Object.keys(z).filter((n) => z[n] !== null);
    for (let i = 0; i < names.length; i++) for (let j = i + 1; j < names.length; j++) {
      const a = names[i], b = names[j];
      if (z[a] === null || z[b] === null || norm(sub(z[a], z[b])) >= c.ghost_m) continue;
      const ja = this.s[a].meas === null ? Infinity : norm(sub(z[a], this.s[a].meas));
      const jb = this.s[b].meas === null ? Infinity : norm(sub(z[b], this.s[b].meas));
      z[ja > jb ? a : b] = null;
    }
    const out = {};
    for (const port of this.ports) {
      const n = port.name, st = this.s[n];
      let m = z[n] ?? null;
      if (m !== null && st.meas !== null && t - st.t_meas < 1.0 && norm(sub(m, st.meas)) > c.jump_m) {
        if (st.pending.length && norm(sub(m, st.pending[st.pending.length - 1])) > c.confirm_m) st.pending = [];
        st.pending.push(m);
        if (st.pending.length < c.confirm) m = null;
        else { st.pending = []; st.blend_from = st.tip; st.blend_k = 0; }
      } else st.pending = [];
      if (m !== null) {
        const v = sub(m, port.pivot);
        st.d = norm(v);
        st.u = scale(v, 1 / Math.max(st.d, 1e-9));
        st.meas = m; st.t_meas = t;
        let tip = m, mode = 'measured';
        if (st.blend_from !== null) {
          st.blend_k += 1;
          const w = Math.min(1.0, st.blend_k / c.blend_sets);
          tip = add(st.blend_from, scale(sub(m, st.blend_from), w));
          mode = w < 1.0 ? 'blend' : 'measured';
          if (w >= 1.0) st.blend_from = null;
        }
        st.tip = tip;
        out[n] = [tip, mode];
        continue;
      }
      if (st.u === null || t - st.t_meas > c.max_est_s) {
        st.tip = null; st.blend_from = null;
        out[n] = null;
        continue;
      }
      let uNew = null, lo = 0.0;
      if (c.shaft && grays && Object.keys(grays).length) {
        const sd = shaftDirection(grays, this.cams, port.pivot, st.u, c.min_contrast, sc);
        if (sd !== null && dot(sd.u, st.u) > Math.cos(radians(25.0))) { uNew = sd.u; lo = sd.visible_m; }
      }
      if (uNew !== null) {
        const u = add(st.u, scale(sub(uNew, st.u), c.alpha_u));
        st.u = scale(u, 1 / norm(u));
      }
      const d = Math.min(Math.max(st.d, lo), port.depth_m[1]);
      let tip = add(port.pivot, scale(st.u, d));
      if (st.tip !== null && norm(sub(tip, st.tip)) > 0.004) {
        const dd = norm(sub(tip, st.tip));
        tip = add(st.tip, scale(sub(tip, st.tip), 0.004 / dd));
      }
      st.tip = tip;
      st.blend_from = tip; st.blend_k = 0;
      out[n] = [tip, 'estimated'];
    }
    return out;
  }
}
