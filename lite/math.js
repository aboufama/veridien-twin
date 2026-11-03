// CPU-side math of the lite renderer (lite/renderer.js), pure ES module (node tests: tests/lite_math.test.mjs).
//   * kb4ProjectF32: the per-vertex KB4 projection exactly as the WGSL vertex shader does it (float32 steps), to be
//     judged against gpu/lens.js project (float64, = cv2.fisheye);
//   * scopeCamera: the SCOPE view (docs/WEB_TWIN.md §18.1) from the scope body pose;
//   * emitterSamples: the light outlet as M area-weighted points (k-means on its triangles), its centroid and the
//     radius of the disk with the same second moment (for the soft capsule shadows);
//   * bodyCapsules / posedCapsules: one occluding capsule per posed body, clipped to the cavity (gpu/scene_gpu.js
//     cavityCapsule), with the body's mean hemispherical albedo;
//   * liteViews: typed-array views of web/twin/assets/lite_*.bin (scripts/build_lite_twin.mjs);
//   * lerpPose: mid-exposure pose between poses0 and poses1 (slerp + lerp).

import { lensParams } from '../gpu/lens.js';
import { cavityCapsule, fitCapsule, unwrap } from '../gpu/scene_gpu.js';

export const DEG = Math.PI / 180;
const f32 = Math.fround;

/** KB4 projection of a camera-frame point, in float32 steps as the vertex shader (lite/shaders.js vsProject).
 * Returns [u, v, theta]. */
export function kb4ProjectF32(L, X) {
  const x = f32(X[0]), y = f32(X[1]), z = f32(X[2]);
  const rho = f32(Math.sqrt(f32(f32(x * x) + f32(y * y))));
  const th = f32(Math.atan2(rho, z));
  const t2 = f32(th * th);
  const k = L.k.map(f32);
  const thd = f32(th * f32(1 + f32(t2 * f32(k[0] + f32(t2 * f32(k[1] + f32(t2 * f32(k[2] + f32(t2 * k[3])))))))));
  if (rho <= 1e-12) return [L.cx, L.cy, th];
  const s = f32(thd / rho);
  const xd = f32(x * s), yd = f32(y * s);
  return [f32(f32(L.fx) * f32(xd + f32(f32(L.alpha) * yd)) + f32(L.cx)), f32(f32(L.fy) * yd + f32(L.cy)), th];
}

/** KB4 lens of a scene camera scaled to an output size (the grid renders 640x360 from the 1280x720 calibration). */
export function scaledLens(lens, size) {
  const L = lensParams(lens);
  const [W0, H0] = L.size || size;
  const sx = size[0] / W0, sy = size[1] / H0;
  // pixel centres at integer coordinates: u' = (u + 0.5) s - 0.5
  return { ...L, fx: L.fx * sx, fy: L.fy * sy, cx: (L.cx + 0.5) * sx - 0.5, cy: (L.cy + 0.5) * sy - 0.5, size: [...size] };
}

/** The scope camera's assumed optics (physics.yaml scope.camera, §18.1 defaults when absent). */
export function scopeOptics(physics) {
  const c = physics?.scope?.camera || {};
  return { hfovDeg: Number(unwrap(c.hfov_deg) ?? 75), circleFrac: Number(unwrap(c.image_circle_frac) ?? 0.95) };
}

/**
 * SCOPE view from the scope body pose T_D_K (column-major 16). Centre: scope.objective_centre_K; axis:
 * scope.view_dir_K (the 30 deg DOV); image up = the side the tilt looks toward (light post up on a tower monitor):
 * camera y (down) = -(component of +y_K normal to the view axis), x = y cross z. Returns {R (3x3 rows, R_D_C),
 * centre (D), fx, fy, cx, cy, circleR (px), size, T_D_C (4x4 rows)}.
 */
export function scopeCamera(scene, physics, T_D_K, size = [1280, 720]) {
  const sc = scene.scope || {};
  const dov = Number(sc.direction_of_view_deg ?? 30) * DEG;
  const zK = sc.view_dir_K ? normalize(sc.view_dir_K) : [0, Math.sin(dov), Math.cos(dov)];
  const up = normalize(sub([0, 1, 0], scale(zK, zK[1])));       // +y_K made normal to the axis
  const yK = scale(up, -1);                                       // OpenCV y is image down
  const xK = cross(yK, zK);
  const oK = sc.objective_centre_K || [0, 0, 0];
  const R_D_K = [[T_D_K[0], T_D_K[4], T_D_K[8]], [T_D_K[1], T_D_K[5], T_D_K[9]], [T_D_K[2], T_D_K[6], T_D_K[10]]];
  const t = [T_D_K[12], T_D_K[13], T_D_K[14]];
  const colD = (v) => [0, 1, 2].map((i) => R_D_K[i][0] * v[0] + R_D_K[i][1] * v[1] + R_D_K[i][2] * v[2]);
  const x = colD(xK), y = colD(yK), z = colD(zK);
  const centre = colD(oK).map((v, i) => v + t[i]);
  const R = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
  const { hfovDeg, circleFrac } = scopeOptics(physics);
  const [w, h] = size;
  const fx = (w / 2) / Math.tan((hfovDeg * DEG) / 2);
  return {
    R, centre, fx, fy: fx, cx: (w - 1) / 2, cy: (h - 1) / 2, circleR: (circleFrac * w) / 2, size: [w, h], hfovDeg, circleFrac,
    T_D_C: [[...R[0], centre[0]], [...R[1], centre[1]], [...R[2], centre[2]], [0, 0, 0, 1]],
  };
}

/** Pinhole projection of a D point into the scope camera: [u, v, z] or null behind it. */
export function scopeProject(cam, pD) {
  const d = sub(pD, cam.centre);
  const X = [0, 1, 2].map((j) => cam.R[0][j] * d[0] + cam.R[1][j] * d[1] + cam.R[2][j] * d[2]);
  if (!(X[2] > 1e-9)) return null;
  return [cam.fx * X[0] / X[2] + cam.cx, cam.fy * X[1] / X[2] + cam.cy, X[2]];
}

/**
 * The light outlet (mesh in the scope frame K) as M area-weighted points: k-means (Lloyd, 24 iterations, farthest-
 * point init from the area centroid) on the triangle centroids weighted by area. Returns {points: [{p, a}], area,
 * centroid, normal (area-weighted, oriented by the vertex normals), radius (second-moment disk radius), extent}.
 */
export function emitterSamples(md, M = 8) {
  const { P, N, I } = md;
  const tris = [];
  let area = 0;
  const nAcc = [0, 0, 0];
  for (let k = 0; k < I.length; k += 3) {
    const a = 3 * I[k], b = 3 * I[k + 1], c = 3 * I[k + 2];
    const A = [P[a], P[a + 1], P[a + 2]], B = [P[b], P[b + 1], P[b + 2]], C = [P[c], P[c + 1], P[c + 2]];
    let n = cross(sub(B, A), sub(C, A));
    const ar = 0.5 * Math.hypot(...n);
    if (!(ar > 0)) continue;
    const vn = [N[a] + N[b] + N[c], N[a + 1] + N[b + 1] + N[c + 1], N[a + 2] + N[b + 2] + N[c + 2]];
    if (dot(n, vn) < 0) n = scale(n, -1);
    for (let q = 0; q < 3; q++) nAcc[q] += n[q];                   // |n| = 2 area: area-weighted
    tris.push({ c: [(A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3, (A[2] + B[2] + C[2]) / 3], a: ar });
    area += ar;
  }
  const centroid = [0, 1, 2].map((q) => tris.reduce((s, t) => s + t.c[q] * t.a, 0) / area);
  let m2 = 0, ext = 0;
  for (const t of tris) { const d2 = dist2(t.c, centroid); m2 += d2 * t.a; ext = Math.max(ext, Math.sqrt(d2)); }
  m2 /= area;
  // farthest-point init, then Lloyd
  const cen = [centroid];
  while (cen.length < M) {
    let best = null, bd = -1;
    for (const t of tris) { const d = Math.min(...cen.map((c) => dist2(c, t.c))); if (d > bd) { bd = d; best = t.c; } }
    cen.push([...best]);
  }
  let assign = new Int32Array(tris.length);
  for (let it = 0; it < 24; it++) {
    tris.forEach((t, i) => { let b = 0, bd = Infinity; cen.forEach((c, j) => { const d = dist2(c, t.c); if (d < bd) { bd = d; b = j; } }); assign[i] = b; });
    for (let j = 0; j < M; j++) {
      let s = 0; const acc = [0, 0, 0];
      tris.forEach((t, i) => { if (assign[i] === j) { s += t.a; for (let q = 0; q < 3; q++) acc[q] += t.c[q] * t.a; } });
      if (s > 0) cen[j] = acc.map((x) => x / s);
    }
  }
  const points = cen.map((p, j) => ({ p, a: tris.reduce((s, t, i) => s + (assign[i] === j ? t.a : 0), 0) })).filter((x) => x.a > 0);
  return { points, area, centroid, normal: normalize(nAcc), radius: Math.sqrt(2 * m2), extent: ext };
}

/** The emitter profile p(theta) as a uniform table: {step (rad), values: Float32Array(16)} (linear, 0 past the end). */
export function profileTable(theta, p) {
  const n = Math.min(16, theta.length);
  const step = theta.length > 1 ? theta[1] - theta[0] : 1;
  for (let i = 1; i < theta.length; i++) if (Math.abs(theta[i] - theta[i - 1] - step) > 1e-6) throw new Error('profile: theta must be uniform');
  const values = new Float32Array(16);
  for (let i = 0; i < n; i++) values[i] = p[i];
  return { step, count: n, values };
}

/** p(theta) from profileTable, as the shader evaluates it. */
export function profileAt(tab, th) {
  const x = th / tab.step;
  if (!(x >= 0) || x >= tab.count - 1) return x >= 0 && x <= tab.count - 1 ? tab.values[tab.count - 1] : 0;
  const i = Math.floor(x), f = x - i;
  return tab.values[i] * (1 - f) + tab.values[i + 1] * f;
}

/** Group of a posed body for shadows: an instrument, its port's cannula; the scope with port C's cannula. */
export function bodyGroup(scene, key) {
  const head = key.split('/')[0];
  if (key === 'scope') return 3;
  const scopePort = scene.scope?.port ?? 'C';
  if (head === scopePort) return 3;
  const ids = (scene.instruments || []).map((i) => i.id);
  const portOf = Object.fromEntries((scene.instruments || []).map((i) => [i.port, i.id]));
  const id = ids.includes(head) ? head : portOf[head];
  return id ? 1 + ids.indexOf(id) : 0;
}

/**
 * One occlusion capsule per dynamic body, in its body frame, from the scene's original meshes (overview_only meshes
 * left out: a trocar housing outside the dome hides no light inside it). r: the shaft radius for instrument shafts,
 * the RMS distance from the axis for jaws (the max would bound the serrations); cannulas get the port's outer radius
 * when posed. rhoEff: area-weighted hemispherical albedo (luminance) of the body's materials.
 */
export function bodyCapsules(scene, meshes, materials) {
  const bodyOf = {};
  for (const ins of scene.instruments || []) for (const p of ins.parts || []) bodyOf[p.mesh] = `${ins.id}/${p.body}`;
  for (const m of scene.scope?.mesh_parts || []) bodyOf[m] = 'scope';
  for (const b of scene.boards || []) bodyOf[b.mesh] = `board/${b.name}`;
  for (const m of scene.meshes) if (m.dynamic && !bodyOf[m.name] && m.body) bodyOf[m.name] = m.body;
  const acc = {};
  for (const m of scene.meshes) {
    const key = bodyOf[m.name];
    if (!key || !m.dynamic || m.overview_only || !meshes[m.name]) continue;
    (acc[key] ??= []).push(m);
  }
  const out = [];
  for (const [key, list] of Object.entries(acc)) {
    let nv = 0;
    for (const m of list) nv += meshes[m.name].P.length;
    const P = new Float32Array(nv);
    let o = 0;
    for (const m of list) { P.set(meshes[m.name].P, o); o += meshes[m.name].P.length; }
    const cap = fitCapsule(P);
    const ax = normalize(sub(cap.b, cap.a));
    let ar = 0, rho = 0, r2 = 0;
    for (const m of list) {
      const { P: Q, I } = meshes[m.name];
      const mat = materials?.[m.material];
      const hh = mat?.table?.hh ?? [0, 0, 0];
      const lum = 0.2126 * hh[0] + 0.7152 * hh[1] + 0.0722 * hh[2];
      for (let k = 0; k < I.length; k += 3) {
        const a = 3 * I[k], b = 3 * I[k + 1], c = 3 * I[k + 2];
        const A = [Q[a], Q[a + 1], Q[a + 2]], B = [Q[b], Q[b + 1], Q[b + 2]], C = [Q[c], Q[c + 1], Q[c + 2]];
        const t = 0.5 * Math.hypot(...cross(sub(B, A), sub(C, A)));
        const cen = [(A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3, (A[2] + B[2] + C[2]) / 3];
        const d = sub(cen, cap.a), s = dot(d, ax);
        r2 += t * (dot(d, d) - s * s); ar += t; rho += t * lum;
      }
    }
    const kind = key === 'scope' ? 'scope' : key.startsWith('board/') ? 'board' : /\/jaw_[ab]$/.test(key) ? 'jaw' : /\/shaft$/.test(key) ? 'shaft' : 'fixture';
    let r = cap.r;
    if (kind === 'jaw') r = Math.sqrt(r2 / Math.max(ar, 1e-30));
    if (kind === 'shaft') {
      const ins = (scene.instruments || []).find((i) => key === `${i.id}/shaft`);
      r = Number(ins?.spec?.shaft_radius ?? r);
    }
    if (kind === 'board') r = 0;
    out.push({ key, kind, a: cap.a, b: cap.b, r, rhoEff: ar ? rho / ar : 0, group: bodyGroup(scene, key) });
  }
  return out;
}

/** Capsules posed and clipped to the cavity (scene_gpu.js cavityCapsule): [{a, b, r, rhoEff, group}]. Hidden bodies
 * (no pose) and empty capsules are dropped. */
export function posedCapsules(scene, caps, bodies) {
  const ports = Object.fromEntries((scene.ports || []).map((p) => [p.id, p]));
  const out = [];
  for (const c of caps) {
    const T = bodies[c.key];
    if (!T || !(c.r > 0)) continue;
    const a = applyT(T, c.a), b = applyT(T, c.b);
    const head = c.key.split('/')[0];
    const port = c.key === 'scope' ? ports[scene.scope?.port ?? 'C'] : ports[head] ?? ports[(scene.instruments || []).find((i) => i.id === head)?.port];
    const kind = c.kind === 'fixture' ? 'fixture' : c.kind === 'board' ? 'board' : 'instrument';
    const cc = cavityCapsule(kind, a, b, c.r, port);
    if (!(cc.r > 0)) continue;
    out.push({ key: c.key, a: cc.a, b: cc.b, r: cc.r, rhoEff: c.rhoEff, group: c.group });
  }
  return out;
}

/** Mid-exposure pose of a rigid T (column-major 16): lerp the translation, slerp the rotation. */
export function lerpPose(T0, T1, w = 0.5) {
  if (!T0 || !T1 || T0 === T1) return T0 || T1;
  const q0 = quatFromCols(T0), q1 = quatFromCols(T1);
  const q = slerp(q0, q1, w);
  const R = matFromQuat(q);
  const out = new Float64Array(16);
  for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) out[4 * j + i] = R[i][j];
  for (let i = 0; i < 3; i++) out[12 + i] = T0[12 + i] + (T1[12 + i] - T0[12 + i]) * w;
  out[15] = 1;
  return out;
}

/** Views of the lite asset buffers: {static: {pos, nrm, nodes, w, index}, dynamic: {pos, nrm, uv, index},
 * nodes: {nodes, gather, env}} as typed arrays over the fetched ArrayBuffers. */
export function liteViews(meta, bufs) {
  const out = {};
  for (const [part, info] of Object.entries(meta.files)) {
    const buf = bufs[part];
    const v = {};
    for (const [k, [off, n, type]] of Object.entries(info.layout)) {
      const C = { Uint16Array, Int16Array, Int8Array, Uint8Array, Float32Array, Uint32Array }[type];
      v[k] = new C(buf, off, n);
    }
    out[part] = v;
  }
  return out;
}

/** Decode one static vertex (as the vertex shader): position (m, D), normal, mesh id. */
export function decodeStaticVertex(meta, views, i) {
  const s = meta.files.static, q = views.static.pos, n8 = views.static.nrm;
  const p = [0, 1, 2].map((k) => s.bmin[k] + (q[4 * i + k] / 65535) * (s.bmax[k] - s.bmin[k]));
  const n = normalize([n8[4 * i] / 127, n8[4 * i + 1] / 127, n8[4 * i + 2] / 127]);
  return { p, n, mesh: q[4 * i + 3] };
}

// ------------------------------------------------------------------ small vector helpers
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const normalize = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
export const applyT = (T, p) => [T[0] * p[0] + T[4] * p[1] + T[8] * p[2] + T[12], T[1] * p[0] + T[5] * p[1] + T[9] * p[2] + T[13], T[2] * p[0] + T[6] * p[1] + T[10] * p[2] + T[14]];

function quatFromCols(m) {
  const r00 = m[0], r10 = m[1], r20 = m[2], r01 = m[4], r11 = m[5], r21 = m[6], r02 = m[8], r12 = m[9], r22 = m[10];
  const tr = r00 + r11 + r22;
  let x, y, z, w;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (r21 - r12) / s; y = (r02 - r20) / s; z = (r10 - r01) / s; }
  else if (r00 > r11 && r00 > r22) { const s = Math.sqrt(1 + r00 - r11 - r22) * 2; w = (r21 - r12) / s; x = 0.25 * s; y = (r01 + r10) / s; z = (r02 + r20) / s; }
  else if (r11 > r22) { const s = Math.sqrt(1 + r11 - r00 - r22) * 2; w = (r02 - r20) / s; x = (r01 + r10) / s; y = 0.25 * s; z = (r12 + r21) / s; }
  else { const s = Math.sqrt(1 + r22 - r00 - r11) * 2; w = (r10 - r01) / s; x = (r02 + r20) / s; y = (r12 + r21) / s; z = 0.25 * s; }
  const l = Math.hypot(x, y, z, w);
  return [x / l, y / l, z / l, w / l];
}
function matFromQuat([x, y, z, w]) {
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
}
function slerp(a, b, t) {
  let c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (c < 0) { bb = b.map((x) => -x); c = -c; }
  if (c > 0.9995) { const r = a.map((x, i) => x + (bb[i] - x) * t); const l = Math.hypot(...r); return r.map((x) => x / l); }
  const th = Math.acos(c), s = Math.sin(th);
  return a.map((x, i) => (x * Math.sin((1 - t) * th) + bb[i] * Math.sin(t * th)) / s);
}
