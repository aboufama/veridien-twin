// View-relative input (docs/WEB_TWIN.md §18.2): drags and arrows act in the main view's image directions, so moving a
// tool right on screen moves it right in that view. A view's screen basis is its camera's image axes at the image
// centre, in D: `right` (+u) and `up` (-v), plus `forward` (the optical axis). control/input.js takes it through its
// screenBasis() callback and moves the tip in that plane, perpendicular to the shaft, as it did for the overview.
//
//   CAMk   the tracking camera's own pose: right = R_D_C x, up = -R_D_C y (OpenCV camera: x right, y down, z forward).
//   SCOPE  the laparoscope view (§18.1): it follows the scope body K. forward = the 30 deg direction of view
//          R_D_K [0, sin dov, cos dov]; image up is the side the tilt looks toward, R_D_K [0, cos dov, -sin dov]
//          (perpendicular to forward, on the +y side of the shaft, as a tower monitor shows a 30 deg scope with the
//          light post up); right = forward x up.
//
// A renderer may publish its own basis (r.screenBasis(name) -> {right, up}); the page prefers it, so the input always
// matches the pixels on screen. Pure functions: no DOM, no GPU (web/twin/tests/control_view_basis.test.mjs).
import { add, sub, scale, cross, dot, norm, normalize, posesFromState, scopeDov, scopeForward, port, spec, forward, ensureLenses, projectD } from './kinematics.js';

/** Rows (4x4 nested) or a 16-element column-major array -> 3x3 rotation columns [x, y, z] (each in D). */
export function rotationColumns(T) {
  if (Array.isArray(T) && Array.isArray(T[0])) return [0, 1, 2].map((j) => [T[0][j], T[1][j], T[2][j]]);
  return [0, 1, 2].map((j) => [T[4 * j], T[4 * j + 1], T[4 * j + 2]]);
}

/** A tracking camera's screen basis from scene.cameras[].T_D_C (row-major nested, as scene.json stores it). */
export function cameraBasis(cam) {
  const [x, y, z] = rotationColumns(cam.T_D_C);
  return { right: normalize(x), up: normalize(scale(y, -1)), forward: normalize(z) };
}

/** The scope view's basis from the scope body pose T_D_K (column-major 16) and its direction of view (rad). */
export function scopeBasisFromPose(T_D_K, dov) {
  const [, y, z] = rotationColumns(T_D_K);
  const s = Math.sin(dov), c = Math.cos(dov);
  const forward = normalize(add(scale(y, s), scale(z, c)));
  const up = normalize(add(scale(y, c), scale(z, -s)));
  return { right: normalize(cross(forward, up)), up, forward, origin: T_D_K.length === 16 ? [T_D_K[12], T_D_K[13], T_D_K[14]] : null };
}

/** The scope view's basis for a state. */
export function scopeBasis(scene, state, physics = null) {
  const T = posesFromState(scene, state).bodies.scope;
  if (!T) return null;
  return scopeBasisFromPose(T, scopeDov(scene, physics));
}

/**
 * The screen basis of a named view ('SCOPE', 'CAM1', ...): the renderer's own when it publishes one, otherwise the
 * kinematics'. Returns {right, up, forward} (unit vectors in D) or null when the view is unknown.
 */
export function viewBasis(scene, state, name, { renderer = null, physics = null } = {}) {
  if (renderer && typeof renderer.screenBasis === 'function') {
    try {
      const b = renderer.screenBasis(name);
      if (b && b.right && b.up) return { right: normalize(b.right), up: normalize(b.up), forward: b.forward ? normalize(b.forward) : normalize(cross(b.up, b.right)) };
    } catch { /* fall through to the kinematics */ }
  }
  if (name === 'SCOPE') return state?.scope ? scopeBasis(scene, state, physics) : null;
  const cam = scene.cameras.find((c) => c.name === name);
  return cam ? cameraBasis(cam) : null;
}

/** A view's projection D -> image px (u right, v down): the renderer's own (r.project(name, points)) when it has one
 * for this view, else KB4 for CAMk and a pinhole along the scope basis for SCOPE. */
export function viewProjector(scene, state, name, basis, renderer = null) {
  if (renderer && typeof renderer.project === 'function') {
    const f = (p) => renderer.project(name, [p])?.[0] ?? null;
    try { f([0, 0, 0.1]); return f; } catch { /* no such view in the renderer (the tracer's SCOPE): the kinematics' */ }
  }
  if (name === 'SCOPE') {
    if (!basis || !state?.scope) return null;
    const lens = scopeForward(scene, state.scope).lens;
    return (p) => pinholeOffset(basis, lens, p);
  }
  const cam = ensureLenses(scene).find((c) => c.name === name);
  return cam ? (p) => projectD(cam, p) : null;
}

/**
 * The selected tool's drag basis in a view: two displacements in D, in the plane perpendicular to the tool's shaft
 * (the only directions a swing about the port can move it), chosen so the tool's hinge moves along image right and
 * image up of that view at the speed a free move along the view's own axes would have. Solved from the view's image
 * Jacobian at the hinge (KB4 or pinhole), damped near the pose where the shaft lies along the drag in the image (there
 * the port allows little image motion, and the drag moves the tool less instead of sideways). Without the Jacobian a
 * drag up in CAM3 moved the Maryland 45 deg off the drag (tests/control_view_basis.test.mjs).
 * The scope in its own view keeps the view basis (moving it moves the view). Falls back to viewBasis.
 */
export function toolBasis(scene, state, name, tool, opts = {}) {
  const vb = viewBasis(scene, state, name, opts);
  if (!vb || !state || (tool === 'scope' && name === 'SCOPE')) return vb;
  const proj = viewProjector(scene, state, name, vb, opts.renderer);
  if (!proj) return vb;
  let p, d;
  if (tool === 'scope') {
    if (!state.scope) return vb;
    const s = scopeForward(scene, state.scope);
    p = s.lens; d = s.d;
  } else {
    const ins = scene.instruments.find((i) => i.id === tool);
    const q = state.instruments?.find((x) => x.id === tool);
    if (!ins || !q) return vb;
    const pt = port(scene, ins.port);
    const f = forward(pt.pivot, pt.R, spec(ins), q);
    p = f.hinge; d = f.d;
  }
  const perp = (v) => sub(v, scale(d, dot(v, d)));
  let e1 = perp(vb.right);
  if (norm(e1) < 0.2) e1 = perp(vb.up);
  e1 = normalize(e1);
  const e2 = cross(d, e1);
  const eps = 5e-4;
  const u0 = proj(p), u1 = proj(add(p, scale(e1, eps))), u2 = proj(add(p, scale(e2, eps)));
  const ur = proj(add(p, scale(vb.right, eps))), uu = proj(add(p, scale(vb.up, eps)));
  if (!u0 || !u1 || !u2 || !ur || !uu) return vb;
  // M: image px per m of hinge motion along e1, e2 (columns); s: the image speed of a free move along the view axes
  const a = (u1[0] - u0[0]) / eps, b = (u2[0] - u0[0]) / eps, c = (u1[1] - u0[1]) / eps, e = (u2[1] - u0[1]) / eps;
  const s = 0.5 * (Math.hypot(ur[0] - u0[0], ur[1] - u0[1]) + Math.hypot(uu[0] - u0[0], uu[1] - u0[1])) / eps;
  if (!(s > 0)) return vb;
  // damped least squares x = M^T (M M^T + lam I)^-1 t, for t = (s, 0) (image right) and (0, -s) (image up)
  const lam = (0.15 * s) ** 2;
  const A00 = a * a + b * b + lam, A01 = a * c + b * e, A11 = c * c + e * e + lam;
  const det = A00 * A11 - A01 * A01;
  if (!(Math.abs(det) > 0)) return vb;
  const solve = (t0, t1) => {
    const y0 = (A11 * t0 - A01 * t1) / det, y1 = (-A01 * t0 + A00 * t1) / det;
    const x1 = a * y0 + c * y1, x2 = b * y0 + e * y1;           // M^T y
    let v = add(scale(e1, x1), scale(e2, x2));
    const n = norm(v);
    if (n > 3) v = scale(v, 3 / n);                              // never more than 3x the free speed
    return v;
  };
  return { right: solve(s, 0), up: solve(0, -s), forward: vb.forward, tool };
}

/** A pinhole projection through a basis (for checks): pixel offset (u right, v down) of point p seen from `origin`. */
export function pinholeOffset(basis, origin, p, f = 1000) {
  const d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
  const z = dot(d, basis.forward);
  if (z <= 1e-9) return null;
  return [f * dot(d, basis.right) / z, -f * dot(d, basis.up) / z];
}
