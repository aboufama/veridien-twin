// The RCM solve of the in-browser tracker: the JS twin of tipnet.py's make_obs, tip_hypotheses, solve_rcm (Gauss-
// Newton on the remote centre of motion: tip = pivot + d·u), shaft_darkness and solve_frame (assignment of tip
// hypotheses to ports by confidence, reprojection RMS, matched hinges, shaft darkness and continuity). Same control
// flow and tie-breaking as Python ("first wins" everywhere), float64. Rig frame, metres; pixels are full-res fisheye.
// A view set (Python dict {camera: candidate index}, insertion-ordered) is an array of [camera, index] pairs here.

import { projectPx, normalized, jacobian, toCam, project } from './lens.js';
import { add, sub, scale, dot, norm, cross, mulMtv, solveLinear, linspace, npSum, npSum32, f32 } from './math.js';
import { sampleBilinear } from './preprocess.js';

export const JAW_M = 0.018;
export const COLLAR_M = 0.007;

export const SOLVE_DEFAULTS = Object.freeze({
  gate_px: 8, extend_gate_px: 8, hinge_gate_px: 12, min_angle_deg: 3, max_rms_px: 4, hinge_w: 0.5, w_rms: 0.15,
  w_hinge: 0.5, w_dark: 1, w_prev: 0.5, prev_scale_m: 0.015, prev_cap: 2, dark_len_m: 0.08, dark_samples: 12,
  dark_top: 2, min_dark: -Infinity,
});

/** One camera's candidates for one channel, undistorted once: {cam, uv[], conf[], xn[], J[], ray[] (rig frame)}. */
export function makeObs(k, cam, cands) {
  const uv = cands.map((c) => c.uv), conf = cands.map((c) => c.conf);
  const xn = uv.map((p) => normalized(cam.lens, p));
  const ray = xn.map((x) => {
    const n = Math.sqrt(x[0] * x[0] + x[1] * x[1] + 1);
    return mulMtv(cam.R, [x[0] / n, x[1] / n, 1 / n]);
  });
  return { cam: k, uv, conf, xn, J: xn.map((x) => jacobian(cam.lens, x)), ray };
}

/** Pixel residual [2] of rig point X against an undistorted observation (first order; NaN behind the camera). */
export function resid(cam, X, xn, J) {
  const Xc = toCam(cam, X);
  const z = Xc[2] > 1e-6 ? Xc[2] : NaN;
  const d0 = Xc[0] / z - xn[0], d1 = Xc[1] / z - xn[1];
  return [J[0][0] * d0 + J[0][1] * d1, J[1][0] * d0 + J[1][1] * d1];
}

/** Least-squares point closest to rays C_i + s r_i. */
export function triangulateRays(Cs, rs) {
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  Cs.forEach((C, m) => {
    const r = rs[m];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) A[i][j] += (i === j ? 1 : 0) - r[i] * r[j];
    for (let i = 0; i < 3; i++) {
      let s = 0;
      for (let j = 0; j < 3; j++) s += ((i === j ? 1 : 0) - r[i] * r[j]) * C[j];
      b[i] += s;
    }
  });
  return solveLinear(A, b);
}

/** Closest-approach midpoint of two rays -> {X, s1, s2, sinang}. */
export function pairMidpoint(C1, r1, C2, r2) {
  const w = sub(C1, C2);
  const b = dot(r1, r2), d = dot(r1, w), e = dot(r2, w);
  const den = Math.max(1 - b * b, 1e-12);
  const s1 = (b * e - d) / den, s2 = (e - b * d) / den;
  const X = [0, 1, 2].map((i) => 0.5 * (C1[i] + s1 * r1[i] + C2[i] + s2 * r2[i]));
  return { X, s1, s2, sinang: Math.sqrt(den) };
}

export function tangentBasis(u) {
  const a = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  let e1 = cross(u, a);
  e1 = scale(e1, 1 / Math.sqrt(dot(e1, e1)));
  return [e1, cross(u, e1)];
}

/** tipnet.solve_rcm: tip = pivot + d·u minimising reprojection error. tipObs / hingeObs: [{k, xn, J}].
 *  -> {tip, u, d, rms_tip_px, rms_hinge_px, n_tip, n_hinge, ok}. */
export function solveRcm(pivot, cams, tipObs, hingeObs = [], X0 = null, { jaw = JAW_M, hinge_w = 1, iters = 8 } = {}) {
  if (X0 == null) {
    X0 = triangulateRays(tipObs.map((o) => cams[o.k].C), tipObs.map((o) => {
      const n = Math.sqrt(o.xn[0] * o.xn[0] + o.xn[1] * o.xn[1] + 1);
      return mulMtv(cams[o.k].R, [o.xn[0] / n, o.xn[1] / n, 1 / n]);
    }));
  }
  const v = sub(X0, pivot);
  let d = norm(v);
  let u = scale(v, 1 / Math.max(d, 1e-9));
  const obs = [...tipObs.map((o) => ({ ...o, off: 0, ws: 1 })), ...hingeObs.map((o) => ({ ...o, off: -jaw, ws: Math.sqrt(hinge_w) }))];
  const n = obs.length;
  if (n === 0) return { tip: X0.slice(), u, d, rms_tip_px: NaN, rms_hinge_px: NaN, n_tip: 0, n_hinge: 0, ok: false };
  let lam = 1e-6;
  const residuals = (u_, d_) => {
    const Xc = [], r = [];
    for (const o of obs) {
      const s = d_ + o.off;
      const X = [pivot[0] + s * u_[0], pivot[1] + s * u_[1], pivot[2] + s * u_[2]];
      const c = toCam(cams[o.k], X);
      const e0 = c[0] / c[2] - o.xn[0], e1 = c[1] / c[2] - o.xn[1];
      Xc.push(c);
      r.push([(o.J[0][0] * e0 + o.J[0][1] * e1) * o.ws, (o.J[1][0] * e0 + o.J[1][1] * e1) * o.ws]);
    }
    return { Xc, r };
  };
  const flat = (r) => r.flatMap((x) => x);
  let { Xc, r } = residuals(u, d);
  let cost = npSum(flat(r).map((x) => (Number.isNaN(x * x) ? 0 : x * x)));
  for (let it = 0; it < iters; it++) {
    const [e1, e2] = tangentBasis(u);
    const A = [];
    obs.forEach((o, i) => {
      const [x, y, z] = Xc[i];
      const R = cams[o.k].R, J = o.J;
      // G = ws * J . dpi . R  (2x3)
      const dpi = [[1 / z, 0, -x / (z * z)], [0, 1 / z, -y / (z * z)]];
      const JD = [0, 1].map((a) => [0, 1, 2].map((c) => J[a][0] * dpi[0][c] + J[a][1] * dpi[1][c]));
      const G = [0, 1].map((a) => [0, 1, 2].map((c) => (JD[a][0] * R[0][c] + JD[a][1] * R[1][c] + JD[a][2] * R[2][c]) * o.ws));
      const s = d + o.off;
      const se1 = scale(e1, s), se2 = scale(e2, s);
      for (let a = 0; a < 2; a++) A.push([dot(G[a], se1), dot(G[a], se2), dot(G[a], u)]);
    });
    const rr = flat(r);
    if (!rr.every(Number.isFinite)) break;
    const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], g = [0, 0, 0];
    for (let m = 0; m < A.length; m++) for (let i = 0; i < 3; i++) {
      g[i] += A[m][i] * rr[m];
      for (let j = 0; j < 3; j++) H[i][j] += A[m][i] * A[m][j];
    }
    const M = H.map((row, i) => row.map((x, j) => (i === j ? x + lam * (H[i][i] + 1e-12) : x)));
    const step = solveLinear(M, g).map((x) => -x);
    let un = add(add(u, scale(e1, step[0])), scale(e2, step[1]));
    un = scale(un, 1 / norm(un));
    const dn = d + step[2];
    const nr = residuals(un, dn);
    const rn = flat(nr.r);
    const cn = rn.every(Number.isFinite) ? npSum(rn.map((x) => x * x)) : Infinity;
    if (cn <= cost) {
      u = un; d = dn; Xc = nr.Xc; r = nr.r; cost = cn;
      lam = Math.max(lam * 0.3, 1e-9);
      if (Math.max(Math.abs(step[0]), Math.abs(step[1])) * d < 1e-6 && Math.abs(step[2]) < 1e-6) break;
    } else {
      lam *= 10;
      if (lam > 1e4) break;
    }
  }
  const e = r.map((x, i) => Math.sqrt(x[0] * x[0] + x[1] * x[1]) / obs[i].ws);
  const nt = tipObs.length;
  const mean = (a) => npSum(a) / a.length;
  const rmsT = nt ? Math.sqrt(mean(e.slice(0, nt).map((x) => x * x))) : NaN;
  const rmsH = n > nt ? Math.sqrt(mean(e.slice(nt).map((x) => x * x))) : NaN;
  return { tip: add(pivot, scale(u, d)), u, d, rms_tip_px: rmsT, rms_hinge_px: rmsH, n_tip: nt, n_hinge: n - nt, ok: Number.isFinite(cost) };
}

const viewKey = (m) => m.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(([k, i]) => `${k}:${i}`).join(',');

/** tipnet.tip_hypotheses: consistent candidate-to-camera sets -> [[[cam, idx], ...]] (>= 2 cameras). */
export function tipHypotheses(obs, cams, cfg = SOLVE_DEFAULTS) {
  const ks = Object.keys(obs).map(Number).filter((k) => obs[k].conf.length).sort((a, b) => a - b);
  const seen = new Set(), out = [];
  const smin = Math.sin(cfg.min_angle_deg * (Math.PI / 180));
  for (let ai = 0; ai < ks.length; ai++) for (let bi = ai + 1; bi < ks.length; bi++) {
    const a = ks[ai], b = ks[bi], oa = obs[a], ob = obs[b];
    for (let ia = 0; ia < oa.conf.length; ia++) for (let ib = 0; ib < ob.conf.length; ib++) {
      const { X, s1, s2, sinang } = pairMidpoint(cams[a].C, oa.ray[ia], cams[b].C, ob.ray[ib]);
      if (!(s1 > 0 && s2 > 0 && sinang > smin)) continue;
      const ra = resid(cams[a], X, oa.xn[ia], oa.J[ia]), rb = resid(cams[b], X, ob.xn[ib], ob.J[ib]);
      if (!(Math.hypot(ra[0], ra[1]) < cfg.gate_px && Math.hypot(rb[0], rb[1]) < cfg.gate_px)) continue;
      const mem = [[a, ia], [b, ib]];
      for (const c of ks) {
        if (c === a || c === b) continue;
        const uv = projectPx(cams[c], X);
        if (!Number.isFinite(uv[0]) || !Number.isFinite(uv[1])) continue;
        let best = 0, bd = Infinity;
        obs[c].uv.forEach((p, i) => { const dd = Math.hypot(p[0] - uv[0], p[1] - uv[1]); if (dd < bd) { bd = dd; best = i; } });
        if (bd < cfg.extend_gate_px) mem.push([c, best]);
      }
      const key = viewKey(mem);
      if (!seen.has(key)) { seen.add(key); out.push(mem); }
    }
  }
  const sets = out.map((m) => new Set(m.map(([k, i]) => `${k}:${i}`)));
  return out.filter((m, i) => !sets.some((o, j) => j !== i && o.size > sets[i].size && [...sets[i]].every((x) => o.has(x))));
}

/** tipnet.shaft_darkness: contrast of the black shaft along pivot -> collar in one gray image (any resampling of
 *  the full frame; float32 arithmetic as numpy). null when fewer than 3 samples land in the image. */
export function shaftDarkness(gray, cam, pivot, u, d, cfg = SOLVE_DEFAULTS, radius = 0.0025) {
  const sEnd = d - JAW_M - COLLAR_M - 0.004;
  if (sEnd <= 0.005) return null;
  const s = linspace(Math.max(sEnd - cfg.dark_len_m, 0.003), sEnd, cfg.dark_samples);
  const w = gray.width, h = gray.height, sc = w / cam.size[0];
  const uvs = [], zs = [];
  for (const si of s) {
    const X = [pivot[0] + si * u[0], pivot[1] + si * u[1], pivot[2] + si * u[2]];
    const p = projectPx(cam, X);
    const uv = [(p[0] + 0.5) * sc - 0.5, (p[1] + 0.5) * sc - 0.5];
    if (Number.isFinite(uv[0]) && Number.isFinite(uv[1]) && uv[0] > 2 && uv[0] < w - 3 && uv[1] > 2 && uv[1] < h - 3) {
      uvs.push(uv); zs.push(toCam(cam, X)[2]);
    }
  }
  if (uvs.length < 3) return null;
  let t = [uvs[uvs.length - 1][0] - uvs[0][0], uvs[uvs.length - 1][1] - uvs[0][1]];
  const tl = Math.max(Math.hypot(t[0], t[1]), 1e-9);
  t = [t[0] / tl, t[1] / tl];
  const nrm = [-t[1], t[0]];
  const K00 = cam.lens.fx;
  const samp = (x, y) => (x < 0 || y < 0 || x > w - 1 || y > h - 1) ? -1 : sampleBilinear(gray, x, y, 'replicate');
  const vals = [];
  uvs.forEach((uv, i) => {
    const off = Math.min(Math.max(2.0 * radius * K00 * sc / zs[i], 2.0), 60.0);
    const c = samp(uv[0], uv[1]);
    const l = samp(uv[0] + off * nrm[0], uv[1] + off * nrm[1]);
    const r = samp(uv[0] - off * nrm[0], uv[1] - off * nrm[1]);
    if (c >= 0 && l >= 0 && r >= 0) {
      const side = 0.5 * (l + r);                       // exact in float32 (integers and halves)
      vals.push(f32((side - c) / Math.max(side, 8.0)));
    }
  });
  if (vals.length < 3) return null;
  return f32(npSum32(vals) / vals.length);
}

/** tipnet.solve_frame. cands: {camera index: {tip: [{uv, conf}], hinge: [...]}} (rig order); grays: {camera index:
 *  gray image} or null; prev: {port: tip} -> {port: PortTip or null}; PortTip = {port, tip, u, d, rms_px,
 *  rms_hinge_px, views, hinges, score, dark}. */
export function solveFrame(cands, cams, ports, grays = null, prev = null, cfg = SOLVE_DEFAULTS) {
  const order = Object.keys(cands).map(Number);
  const tipObs = {}, hinObs = {};
  for (const k of order) {
    tipObs[k] = makeObs(k, cams[k], cands[k].tip ?? []);
    hinObs[k] = makeObs(k, cams[k], cands[k].hinge ?? []);
  }
  const hyps = tipHypotheses(tipObs, cams, cfg);
  const scored = Object.fromEntries(ports.map((p) => [p.name, []]));
  const bad = (f) => !f.ok || !Number.isFinite(f.rms_tip_px) || f.rms_tip_px > cfg.max_rms_px;
  const tobsOf = (mem) => mem.map(([k, i]) => ({ k, xn: tipObs[k].xn[i], J: tipObs[k].J[i] }));
  hyps.forEach((mem, hi) => {
    let mem0 = mem.map((x) => x.slice());
    let tobs0 = tobsOf(mem0);
    let fit0 = solveRcm(ports[0].pivot, cams, tobs0);
    while (bad(fit0) && mem0.length >= 3) {
      let worst = -1, we = -Infinity;
      mem0.forEach(([k, i], j) => {
        const r = resid(cams[k], fit0.tip, tipObs[k].xn[i], tipObs[k].J[i]);
        let e = Math.hypot(r[0], r[1]);
        if (!Number.isFinite(e)) e = Infinity;
        if (e > we) { we = e; worst = j; }
      });
      mem0.splice(worst, 1);
      tobs0 = tobsOf(mem0);
      fit0 = solveRcm(ports[0].pivot, cams, tobs0);
    }
    if (bad(fit0)) return;
    const X0 = fit0.tip;
    for (const p of ports) {
      const v = sub(X0, p.pivot);
      const d0 = norm(v);
      if (!(p.depth_m[0] <= d0 && d0 <= p.depth_m[1])) continue;
      const u0 = scale(v, 1 / d0);
      const Xh = add(p.pivot, scale(u0, d0 - JAW_M));
      let hmem = [];
      const hobs = [];
      for (const k of order) {
        const ho = hinObs[k];
        if (!ho.conf.length) continue;
        const uvh = projectPx(cams[k], Xh);
        if (!Number.isFinite(uvh[0]) || !Number.isFinite(uvh[1])) continue;
        let bi = 0, bd = Infinity;
        ho.uv.forEach((q, i) => { const dd = Math.hypot(q[0] - uvh[0], q[1] - uvh[1]); if (dd < bd) { bd = dd; bi = i; } });
        if (bd < cfg.hinge_gate_px) { hmem.push([k, bi]); hobs.push({ k, xn: ho.xn[bi], J: ho.J[bi] }); }
      }
      let fit = null;
      if (hobs.length) {
        fit = solveRcm(p.pivot, cams, tobs0, hobs, X0, { hinge_w: cfg.hinge_w });
        if (bad(fit) || !Number.isFinite(fit.rms_hinge_px) || fit.rms_hinge_px > cfg.hinge_gate_px) { fit = null; hmem = []; }
      }
      if (fit === null) fit = { tip: X0, u: u0, d: d0, rms_tip_px: fit0.rms_tip_px, rms_hinge_px: NaN };
      if (!(p.depth_m[0] <= fit.d && fit.d <= p.depth_m[1])) continue;
      let score = 0;
      for (const [k, i] of mem0) score += tipObs[k].conf[i];
      score -= cfg.w_rms * fit.rms_tip_px;
      let hs = 0;
      for (const [k, i] of hmem) hs += hinObs[k].conf[i];
      score += cfg.w_hinge * hs;
      if (prev && prev[p.name] != null) {
        const dist = norm(sub(fit.tip, prev[p.name])) / cfg.prev_scale_m;
        score -= cfg.w_prev * Math.min(dist ** 2, cfg.prev_cap);
      }
      scored[p.name].push([score, hi, { port: p.name, tip: fit.tip, u: fit.u, d: fit.d, rms_px: fit.rms_tip_px,
        rms_hinge_px: fit.rms_hinge_px, views: mem0, hinges: hmem, score, dark: 0 }]);
    }
  });
  if (grays) {
    const gk = Object.keys(grays).map(Number);
    for (const p of ports) {
      const lst = scored[p.name];
      lst.sort((a, b) => b[0] - a[0]);
      for (let j = 0; j < Math.min(cfg.dark_top, lst.length); j++) {
        const [sc0, hi, pt] = lst[j];
        const ds = gk.map((k) => shaftDarkness(grays[k], cams[k], p.pivot, pt.u, pt.d, cfg)).filter((x) => x !== null);
        pt.dark = ds.length ? npSum(ds) / ds.length : 0;
        pt.score = sc0 + cfg.w_dark * pt.dark;
        lst[j] = [pt.score, hi, pt];
      }
      lst.length = Math.min(lst.length, cfg.dark_top);
      scored[p.name] = lst.filter((x) => x[2].dark >= cfg.min_dark);
    }
  }
  const names = ports.map((p) => p.name);
  const choices = names.map((n) => [null, ...scored[n].slice().sort((a, b) => b[0] - a[0]).slice(0, 8)]);
  let best = Object.fromEntries(names.map((n) => [n, null])), bestS = 0;
  const combo = new Array(names.length);
  const walk = (pi) => {
    if (pi === names.length) {
      const used = new Set();
      let tot = 0;
      for (const c of combo) {
        if (c === null) continue;
        const items = [...c[2].views.map(([k, i]) => `t${k}:${i}`), ...c[2].hinges.map(([k, i]) => `h${k}:${i}`)];
        if (items.some((x) => used.has(x))) return;
        items.forEach((x) => used.add(x));
        tot += c[0];
      }
      if (tot > bestS) { bestS = tot; best = Object.fromEntries(names.map((n, i) => [n, combo[i] ? combo[i][2] : null])); }
      return;
    }
    for (const c of choices[pi]) { combo[pi] = c; walk(pi + 1); }
  };
  walk(0);
  return best;
}

export { project };
