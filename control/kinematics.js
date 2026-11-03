// Instrument, scope and board kinematics of the web twin: the ONE implementation of state -> body poses and
// state -> truth (docs/WEB_TWIN.md §5, §6, §8). The renderer never does kinematics.
//
// Conventions, ported exactly from src/veridien/twin/instrument.py (judged by tests/control_kinematics.test.mjs
// against twin.instrument.forward to <= 1e-9 m):
//   * Frame D (metres), port frame P from scene.ports[].R_D_P (its z points into the dome).
//   * Shaft frame S: z along the shaft toward the tip, jaws open in the S y-z plane, hinge axis = S x.
//     R_D_S = R_D_P . Ry(yaw) . Rx(pitch) . Rz(roll).
//   * insertion = pivot -> jaw hinge (m); jaw = full opening angle (rad), split +-jaw/2 (double action).
//     jaw A on the +y_S side: R_D_S . [0, sin(jaw/2), cos(jaw/2)]; jaw B on the -y_S side.
//   * tip = midpoint of the two jaw tips (hinge + jaw_length * jaw direction).
//   * Body poses (scene_export.py, kinematics_fixtures): "<id>/shaft" = [R_D_S | hinge - R_D_S hinge_in_S],
//     "<id>/jaw_a" = shaft . Rx(-jaw/2), "<id>/jaw_b" = shaft . Rx(+jaw/2) (both at the hinge).
//   * Scope frame K: the same RCM kinematics through port C, R_D_K = R_D_C . Ry . Rx . Rz, origin = lens point =
//     pivot + insertion * z_K (where the shaft axis meets the oblique tip face). View direction = z turned by the
//     direction of view about +x toward +y.
//   * Board B: OpenCV ChArUco object frame (corner k at (x_k, y_k, 0)), pose T_D_B straight from the state.
// Poses are column-major 4x4 Float64Array(16) T_D_body (p_D = T p_body). Units SI.

import { lensParams, project as kb4Project } from '../gpu/lens.js';

export const DEG = Math.PI / 180;

// ------------------------------------------------------------------ small linear algebra (row-major 3x3)
export const rotX = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[1, 0, 0], [0, c, -s], [0, s, c]]; };
export const rotY = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; };
export const rotZ = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };
export function matmul(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
}
export const mulv = (A, v) => [A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2], A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2], A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]];
export const mulTv = (A, v) => [A[0][0] * v[0] + A[1][0] * v[1] + A[2][0] * v[2], A[0][1] * v[0] + A[1][1] * v[1] + A[2][1] * v[2], A[0][2] * v[0] + A[1][2] * v[1] + A[2][2] * v[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a) => scale(a, 1 / norm(a));
export const col = (R, j) => [R[0][j], R[1][j], R[2][j]];

/** Column-major 4x4 from a row-major 3x3 R and translation t. */
export function makeT(R, t) {
  const T = new Float64Array(16);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) T[j * 4 + i] = R[i][j];
  T[12] = t[0]; T[13] = t[1]; T[14] = t[2]; T[15] = 1;
  return T;
}
/** Column-major 4x4 from a row-major nested 4x4 (scene / state JSON). */
export function fromRows(M) {
  const T = new Float64Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) T[j * 4 + i] = M[i][j];
  return T;
}
/** Row-major nested 4x4 from a column-major one. */
export function toRows(T) {
  return [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => T[j * 4 + i]));
}
export const applyT = (T, p) => [T[0] * p[0] + T[4] * p[1] + T[8] * p[2] + T[12], T[1] * p[0] + T[5] * p[1] + T[9] * p[2] + T[13], T[2] * p[0] + T[6] * p[1] + T[10] * p[2] + T[14]];
/** Inverse of a rigid column-major 4x4. */
export function invT(T) {
  const R = [[T[0], T[4], T[8]], [T[1], T[5], T[9]], [T[2], T[6], T[10]]];
  const Rt = [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]];
  return makeT(Rt, scale(mulv(Rt, [T[12], T[13], T[14]]), -1));
}

/** frame_from_z (veridien.geometry.se3): rotation whose z column is z and whose x is closest to xHint. */
export function frameFromZ(z, xHint = [1, 0, 0]) {
  const zz = normalize(z);
  let x = sub(xHint, scale(zz, dot(xHint, zz)));
  if (norm(x) < 1e-9) x = cross(zz, [0, 1, 0]);
  x = normalize(x);
  const y = cross(zz, x);
  return [[x[0], y[0], zz[0]], [x[1], y[1], zz[1]], [x[2], y[2], zz[2]]];
}

// ------------------------------------------------------------------ scene access

/** Port record {id, pivot, R (row-major R_D_P), cone (rad), role} for a port id. */
export function port(scene, id) {
  const p = scene.ports.find((q) => q.id === id);
  if (!p) throw new Error(`scene has no port ${id}`);
  return { id, pivot: p.pivot.map(Number), R: p.R_D_P.map((r) => r.map(Number)), cone: (p.cone_half_deg ?? 30) * DEG, role: p.role, trocar_d: p.trocar_d, cannula_body: p.cannula_body ?? null };
}

const SPEC_DEFAULTS = { shaft_radius: 0.0025, housing_length: 0.010, jaw_length: 0.016, jaw_radius: 0.0012, jaw_max: 60 * DEG, max_insertion: 0.33 };
/** Instrument spec with the instrument-level min_insertion (§13: jaws past the cannula toe) folded in. */
export function spec(ins) {
  const sp = { ...SPEC_DEFAULTS, ...(ins.spec || {}) };
  if (ins.min_insertion != null && sp.min_insertion == null) sp.min_insertion = ins.min_insertion;
  return sp;
}

// ------------------------------------------------------------------ forward kinematics (twin.instrument.forward)

export function shaftRotation(R_D_P, q) {
  return matmul(matmul(matmul(R_D_P, rotY(q.yaw)), rotX(q.pitch)), rotZ(q.roll));
}

/** twin.instrument.forward, exactly. Returns {R, d, hinge, housing_start, jaw_tips: [a, b], tip, jaw_dirs}. */
export function forward(pivot, R_D_P, sp, q) {
  const R = shaftRotation(R_D_P, q);
  const d = col(R, 2);
  const hinge = add(pivot, scale(d, q.insertion));
  const half = q.jaw / 2;
  const a = mulv(R, [0, Math.sin(half), Math.cos(half)]);
  const b = mulv(R, [0, -Math.sin(half), Math.cos(half)]);
  const ta = add(hinge, scale(a, sp.jaw_length)), tb = add(hinge, scale(b, sp.jaw_length));
  return {
    R, d, hinge, housing_start: sub(hinge, scale(d, sp.housing_length)), jaw_tips: [ta, tb],
    tip: [(ta[0] + tb[0]) / 2, (ta[1] + tb[1]) / 2, (ta[2] + tb[2]) / 2], jaw_dirs: [a, b],
  };
}

/** twin.instrument.inverse_direction: shaft direction (D) -> [yaw, pitch] in the port frame. */
export function inverseDirection(R_D_P, dD) {
  const d = mulTv(R_D_P, normalize(dD));
  const pitch = -Math.asin(Math.max(-1, Math.min(1, d[1])));
  const yaw = Math.atan2(d[0], d[2]);
  return [yaw, pitch];
}

/** twin.instrument.aim_at: the state that puts the jaw hinge at hingeTarget. */
export function aimAt(pivot, R_D_P, hingeTarget, roll = 0, jaw = 0) {
  const v = sub(hingeTarget, pivot);
  const [yaw, pitch] = inverseDirection(R_D_P, v);
  return { yaw, pitch, roll, insertion: norm(v), jaw };
}

/** Scope direction-of-view (rad) from the scene (scope.direction_of_view_deg or kinematics block), default 30 deg. */
export function scopeDov(scene, physics) {
  const fromScene = scene.scope?.direction_of_view_deg ?? scene.kinematics?.scope?.direction_of_view_deg;
  const ph = physics?.scope?.direction_of_view_deg;
  const v = fromScene ?? (ph && typeof ph === 'object' ? ph.value : ph) ?? 30;
  return Number(v) * DEG;
}

/** Scope forward kinematics: {R (R_D_K), d (shaft axis), lens (origin of K), view_dir}. */
export function scopeForward(scene, q, dov = scopeDov(scene)) {
  const p = port(scene, scene.scope?.port ?? 'C');
  const R = shaftRotation(p.R, q);
  const d = col(R, 2);
  const lens = add(p.pivot, scale(d, q.insertion));
  const view = mulv(R, [0, Math.sin(dov), Math.cos(dov)]);
  return { R, d, lens, view_dir: view, pivot: p.pivot };
}

/** The scope's default state in its port frame: scene.scope.default {yaw, pitch, roll, insertion} or, when only
 * dir_D is given (the stub scene), the yaw/pitch of dir_D with the default roll. */
export function scopeDefault(scene) {
  const df = scene.scope?.default ?? {};
  const p = port(scene, scene.scope?.port ?? 'C');
  let yaw = df.yaw, pitch = df.pitch;
  if (yaw == null || pitch == null) [yaw, pitch] = df.dir_D ? inverseDirection(p.R, df.dir_D) : [0, 0];
  return { yaw, pitch, roll: df.roll ?? 0, insertion: df.insertion ?? 0.016 };
}

/** The scope's insertion floor: its whole tip face past the cannula toe (scene.scope.default.min_insertion, §13). */
export function scopeMinInsertion(scene) { return scene.scope?.default?.min_insertion ?? scene.scope?.min_insertion ?? 0.004; }

/** Clamp a joint state's shaft direction into a port cone of half-angle `cone` (rad), keeping its azimuth.
 * Returns a copy with `_cone: true` when it had to move. */
export function coneClamp(q, cone) {
  const cp = Math.cos(q.pitch);
  const dP = [cp * Math.sin(q.yaw), -Math.sin(q.pitch), cp * Math.cos(q.yaw)];
  const ang = Math.acos(Math.max(-1, Math.min(1, dP[2])));
  if (ang <= cone + 1e-12) return { ...q, _cone: false };
  const l = Math.hypot(dP[0], dP[1]) || 1;
  const s = Math.sin(cone), c = Math.cos(cone);
  const d = [dP[0] / l * s, dP[1] / l * s, c];
  return { ...q, yaw: Math.atan2(d[0], d[2]), pitch: -Math.asin(Math.max(-1, Math.min(1, d[1]))), _cone: true };
}

/** The start state: scene.default_state (§13) when the export gives one, else both graspers aimed at the workspace
 * centre (inside 90 % of their cones), jaws slightly open, the scope at its default. */
export function defaultState(scene, t = 0) {
  if (scene.default_state) {
    const d = scene.default_state;
    return {
      t, instruments: d.instruments.map((q) => ({ ...q })), scope: d.scope ? { ...d.scope } : scopeDefault(scene),
      boards: (scene.boards || []).map((b) => ({ name: b.name, T_D_B: null, visible: false })),
    };
  }
  const target = workspaceCentre(scene);
  const instruments = scene.instruments.map((ins) => {
    const p = port(scene, ins.port);
    const sp = spec(ins);
    const v = sub(target, p.pivot);
    const dist = norm(v);
    const q = coneClamp(aimAt(p.pivot, p.R, add(p.pivot, scale(normalize(v), Math.min(dist * 0.72, sp.max_insertion))), 0, 0.35), p.cone * 0.9);
    delete q._cone;
    return { id: ins.id, ...q };
  });
  return { t, instruments, scope: scopeDefault(scene), boards: (scene.boards || []).map((b) => ({ name: b.name, T_D_B: null, visible: false })) };
}

/** Where the tools work: scene.workspace.centre when present, else the point closest (least squares) to every
 * tracking camera's optical axis (the pockets are aimed at the uterus, ADR-123), kept above the bench. */
export function workspaceCentre(scene) {
  if (scene.workspace?.centre) return scene.workspace.centre.map(Number);
  // sum (I - z z^T) p = sum (I - z z^T) o over the camera axes (o, z)
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  for (const c of scene.cameras) {
    const T = fromRows(c.T_D_C);
    const o = [T[12], T[13], T[14]], z = normalize([T[8], T[9], T[10]]);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const m = (i === j ? 1 : 0) - z[i] * z[j];
      A[i][j] += m; b[i] += m * o[j];
    }
  }
  const p = solve3(A, b);
  const bz = (scene.bench?.z ?? 0) + 0.03;
  if (p && Number.isFinite(p[0])) return [p[0], p[1], Math.max(p[2], bz)];
  if (scene.scope?.default?.lens_D && scene.scope?.default?.view_dir_D) return add(scene.scope.default.lens_D, scale(scene.scope.default.view_dir_D, 0.08));
  return [0, 0, bz];
}

function solve3(A, b) {
  const det = (M) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const d = det(A);
  if (Math.abs(d) < 1e-12) return null;
  return [0, 1, 2].map((k) => det(A.map((row, i) => row.map((v, j) => (j === k ? b[i] : v)))) / d);
}

// ------------------------------------------------------------------ poses (renderer API §8)

/** posesFromState(scene, state) -> {bodies: {"L/shaft": T, "L/jaw_a": T, "L/jaw_b": T, "R/...", scope: T, "board/<name>": T|null}}. */
export function posesFromState(scene, state) {
  const bodies = {};
  for (const ins of scene.instruments) {
    const q = state.instruments.find((s) => s.id === ins.id);
    if (!q) continue;
    const p = port(scene, ins.port);
    const sp = spec(ins);
    const f = forward(p.pivot, p.R, sp, q);
    const h = ins.hinge_in_S || [0, 0, 0];
    const origin = sub(f.hinge, mulv(f.R, h));
    bodies[`${ins.id}/shaft`] = makeT(f.R, origin);
    bodies[`${ins.id}/jaw_a`] = makeT(matmul(f.R, rotX(-q.jaw / 2)), f.hinge);
    bodies[`${ins.id}/jaw_b`] = makeT(matmul(f.R, rotX(q.jaw / 2)), f.hinge);
    if (p.cannula_body) bodies[p.cannula_body] = cannulaT(p, q);
  }
  if (scene.scope && state.scope) {
    const s = scopeForward(scene, state.scope);
    bodies.scope = makeT(s.R, s.lens);
    const p = port(scene, scene.scope.port ?? 'C');
    if (p.cannula_body) bodies[p.cannula_body] = cannulaT(p, state.scope);
  }
  for (const b of scene.boards || []) {
    const sb = (state.boards || []).find((x) => x.name === b.name);
    bodies[`board/${b.name}`] = sb && sb.visible !== false && sb.T_D_B ? fromRows(sb.T_D_B) : null;
  }
  return { bodies };
}

/** The trocar cannula in a port (§13, frame Q): R_D_Q = R_D_P . Ry(yaw) . Rx(pitch) of the tool in it, no roll,
 * origin at the pivot. */
export function cannulaT(p, q) {
  return makeT(matmul(matmul(p.R, rotY(q.yaw)), rotX(q.pitch)), p.pivot);
}

/** Linear interpolation of two states (instrument joints, scope joints; boards: rigid interpolation by slerp). */
export function lerpState(a, b, w) {
  if (w <= 0) return a;
  if (w >= 1) return b;
  const L = (x, y) => x + (y - x) * w;
  const J = (qa, qb) => ({ ...qa, yaw: L(qa.yaw, qb.yaw), pitch: L(qa.pitch, qb.pitch), roll: L(qa.roll, qb.roll), insertion: L(qa.insertion, qb.insertion), ...(qa.jaw != null ? { jaw: L(qa.jaw, qb.jaw ?? qa.jaw) } : {}) });
  return {
    t: L(a.t ?? 0, b.t ?? 0),
    instruments: a.instruments.map((qa) => { const qb = b.instruments.find((x) => x.id === qa.id) || qa; return J(qa, qb); }),
    scope: a.scope && b.scope ? J(a.scope, b.scope) : a.scope,
    boards: (a.boards || []).map((ba) => {
      const bb = (b.boards || []).find((x) => x.name === ba.name);
      if (!bb || !ba.T_D_B || !bb.T_D_B || ba.visible === false || bb.visible === false) return w < 0.5 ? ba : (bb || ba);
      return { ...ba, T_D_B: lerpPose(ba.T_D_B, bb.T_D_B, w), visible: true };
    }),
  };
}

function lerpPose(Ma, Mb, w) {
  const Ra = [0, 1, 2].map((i) => Ma[i].slice(0, 3)), Rb = [0, 1, 2].map((i) => Mb[i].slice(0, 3));
  const qa = quatFromR(Ra); let qb = quatFromR(Rb);
  if (qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3] < 0) qb = qb.map((x) => -x);
  const q = slerp(qa, qb, w), R = rFromQuat(q);
  const t = [0, 1, 2].map((i) => Ma[i][3] + (Mb[i][3] - Ma[i][3]) * w);
  return [[...R[0], t[0]], [...R[1], t[1]], [...R[2], t[2]], [0, 0, 0, 1]];
}
export function quatFromR(R) {
  const tr = R[0][0] + R[1][1] + R[2][2];
  let w, x, y, z;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (R[2][1] - R[1][2]) / s; y = (R[0][2] - R[2][0]) / s; z = (R[1][0] - R[0][1]) / s; }
  else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) { const s = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2; w = (R[2][1] - R[1][2]) / s; x = 0.25 * s; y = (R[0][1] + R[1][0]) / s; z = (R[0][2] + R[2][0]) / s; }
  else if (R[1][1] > R[2][2]) { const s = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2; w = (R[0][2] - R[2][0]) / s; x = (R[0][1] + R[1][0]) / s; y = 0.25 * s; z = (R[1][2] + R[2][1]) / s; }
  else { const s = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2; w = (R[1][0] - R[0][1]) / s; x = (R[0][2] + R[2][0]) / s; y = (R[1][2] + R[2][1]) / s; z = 0.25 * s; }
  return [w, x, y, z];
}
export function rFromQuat([w, x, y, z]) {
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
}
function slerp(a, b, w) {
  const c = Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  if (c > 0.9995) { const q = a.map((x, i) => x + (b[i] - x) * w); const n = Math.hypot(...q); return q.map((x) => x / n); }
  const th = Math.acos(c), s = Math.sin(th);
  const ka = Math.sin((1 - w) * th) / s, kb = Math.sin(w * th) / s;
  return a.map((x, i) => ka * x + kb * b[i]);
}

// ------------------------------------------------------------------ truth (§6, without `visible`)

const KP = ['tip', 'hinge', 'jaw_a', 'jaw_b'];

/** Camera records with prepared lens + T_C_D, cached per scene object. */
const camCache = new WeakMap();
export function cameras(scene) {
  let c = camCache.get(scene);
  if (!c) {
    c = scene.cameras.map((cam) => {
      const T_D_C = fromRows(cam.T_D_C);
      const lens = lensParams(cam.lens);
      // §13: px/visible only for theta < theta_fold (KB4 fold, from the export); never past 90 deg (behind the lens plane)
      if (Number.isFinite(cam.theta_fold_rad)) lens.thetaMax = Math.min(cam.theta_fold_rad, Math.PI / 2);
      return { name: cam.name, size: cam.size, lens, T_D_C, T_C_D: invT(T_D_C), centre: [T_D_C[12], T_D_C[13], T_D_C[14]], foldFromScene: Number.isFinite(cam.theta_fold_rad) };
    });
    camCache.set(scene, c);
  }
  return c;
}

/** cameras(scene) with each lens's monotone field (lens.thetaMax) computed. */
export function ensureLenses(scene) {
  const cs = cameras(scene);
  for (const c of cs) prepareThetaMax(c);
  return cs;
}

/** KB4 pixel of a D-frame point in one camera, or null when it is behind the lens or past the monotone field. */
export function projectD(cam, pD) {
  const X = applyT(cam.T_C_D, pD);
  const th = Math.atan2(Math.hypot(X[0], X[1]), X[2]);
  if (!(th < cam.lens.thetaMax)) return null;
  return kb4Project(cam.lens, X);
}

/** Largest field angle the cameras' KB4 maps stay monotone over (computed once per lens). */
function prepareThetaMax(cam) {
  if (cam.lens.thetaMax != null) return;
  let lim = Math.PI / 2;
  const k = cam.lens.k;
  for (let i = 1; i <= 2048; i++) {
    const th = (i / 2048) * Math.PI;
    const t2 = th * th;
    const der = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
    if (der <= 0) { lim = Math.min(lim, ((i - 1) / 2048) * Math.PI); break; }
  }
  cam.lens.thetaMax = lim;
}

/**
 * truthFromState(scene, state) -> {t, instruments: [{id, tip_D, hinge_D, jaw_tips_D, shaft_dir_D, jaw, jaw_tips_cad_D?}],
 *   scope: {lens_D, shaft_dir_D, view_dir_D}, px: {CAM1: {L: {tip: [u, v]|null, hinge, jaw_a, jaw_b}}},
 *   in_image: {CAM1: {L: {tip: bool, ...}}}, points_D: {L: {tip, hinge, jaw_a, jaw_b}}}
 * `visible` is not here: it comes from renderer.visible(name, pointsD) (occlusion in the renderer's own BVH).
 */
export function truthFromState(scene, state) {
  const out = { t: state.t ?? 0, instruments: [], px: {}, in_image: {}, points_D: {} };
  for (const ins of scene.instruments) {
    const q = state.instruments.find((s) => s.id === ins.id);
    if (!q) continue;
    const p = port(scene, ins.port);
    const sp = spec(ins);
    const f = forward(p.pivot, p.R, sp, q);
    // §13: each jaw tip is the CAD's tip_in_jaw_frame in its jaw frame (the Maryland's sits 3.17 mm off the jaw
    // plane); for straight jaws that is [0, 0, L] and equals forward(). tip = midpoint of the two jaw tips.
    let tips = f.jaw_tips;
    if (ins.tip_in_jaw_frame) {
      const tl = ins.tip_in_jaw_frame;
      tips = [add(f.hinge, mulv(matmul(f.R, rotX(-q.jaw / 2)), tl)), add(f.hinge, mulv(matmul(f.R, rotX(q.jaw / 2)), tl))];
    }
    const tip = [(tips[0][0] + tips[1][0]) / 2, (tips[0][1] + tips[1][1]) / 2, (tips[0][2] + tips[1][2]) / 2];
    const rec = { id: ins.id, tip_D: tip, hinge_D: f.hinge, jaw_tips_D: tips, shaft_dir_D: f.d, jaw: q.jaw };
    out.instruments.push(rec);
    out.points_D[ins.id] = { tip, hinge: f.hinge, jaw_a: tips[0], jaw_b: tips[1] };
  }
  if (scene.scope && state.scope) {
    const s = scopeForward(scene, state.scope);
    out.scope = { lens_D: s.lens, shaft_dir_D: s.d, view_dir_D: s.view_dir };
  }
  for (const cam of cameras(scene)) {
    prepareThetaMax(cam);
    const [W, H] = cam.size;
    const px = {}, inside = {};
    for (const id of Object.keys(out.points_D)) {
      px[id] = {}; inside[id] = {};
      for (const k of KP) {
        const uv = projectD(cam, out.points_D[id][k]);
        px[id][k] = uv;
        inside[id][k] = !!uv && uv[0] >= -0.5 && uv[1] >= -0.5 && uv[0] <= W - 0.5 && uv[1] <= H - 0.5;
      }
    }
    out.px[cam.name] = px;
    out.in_image[cam.name] = inside;
  }
  return out;
}

/** The four keypoints of every instrument as a flat list [{id, kp, p}] (for renderer.visible). */
export function keypointList(truth) {
  const out = [];
  for (const id of Object.keys(truth.points_D)) for (const k of KP) out.push({ id, kp: k, p: truth.points_D[id][k] });
  return out;
}
export const KEYPOINTS = KP;
