// The page's start state (docs/WEB_TWIN.md §18.2: "the user opens it and immediately sees one big laparoscope view").
// The export's default scope pose looks 29 deg past the graspers (they sit at the top edge of the scope image), so the
// page re-aims the scope once at load: yaw and pitch about port C so that the 30 deg direction of view passes through
// the midpoint of the two grasper tips, roll and insertion unchanged, then the limits clamp it. Pure (node-tested).
// ?aim=0 keeps the export's pose. During autoplay the scope follows the tools the same way (a camera assistant):
// re-aimed at 10 Hz, smoothed with a 0.35 s time constant (view.js); ?autoscope=1 lets the script drive it instead.
import * as K from '../control/kinematics.js';

/** Angle (rad) between the scope's direction of view and the direction from its lens to `target` (D). */
export function aimError(scene, state, target) {
  const sc = K.truthFromState(scene, state).scope;
  if (!sc) return null;
  const d = K.normalize(K.sub(target, sc.lens_D));
  return Math.acos(Math.max(-1, Math.min(1, K.dot(d, sc.view_dir_D))));
}

/** The grasper tips' midpoint (D), or the workspace centre when there are no graspers. */
export function toolsMidpoint(scene, state) {
  const tr = K.truthFromState(scene, state);
  const tips = tr.instruments.map((i) => i.tip_D).filter(Boolean);
  if (!tips.length) return K.workspaceCentre(scene);
  return K.scale(tips.reduce((a, b) => K.add(a, b), [0, 0, 0]), 1 / tips.length);
}

/**
 * Re-aim the scope at `target` (default: the tools' midpoint) by Gauss-Newton on (yaw, pitch) with a numeric
 * Jacobian of the view direction, at most 30 steps, each clamped by `limits`. Returns {state, before, after} (rad).
 */
export function aimScope(scene, limits, state, target = null, iters = 30) {
  if (!state?.scope) return { state, before: null, after: null };
  const tgt = target ?? toolsMidpoint(scene, state);
  const before = aimError(scene, state, tgt);
  const clamp = (s) => (limits ? limits.clampState(s).state : s);
  const withYP = (s, yaw, pitch) => ({ ...s, scope: { ...s.scope, yaw, pitch } });
  const resid = (s) => {
    const sc = K.truthFromState(scene, s).scope;
    const d = K.normalize(K.sub(tgt, sc.lens_D));
    return K.sub(sc.view_dir_D, d);
  };
  let s = state;
  const h = 1e-5;
  for (let it = 0; it < iters; it++) {
    const r0 = resid(s);
    if (Math.hypot(...r0) < 1e-6) break;
    const { yaw, pitch } = s.scope;
    const ry = resid(withYP(s, yaw + h, pitch)), rp = resid(withYP(s, yaw, pitch + h));
    const J = [0, 1, 2].map((i) => [(ry[i] - r0[i]) / h, (rp[i] - r0[i]) / h]);
    // normal equations (2x2)
    let a = 0, b = 0, c = 0, g0 = 0, g1 = 0;
    for (let i = 0; i < 3; i++) { a += J[i][0] ** 2; b += J[i][0] * J[i][1]; c += J[i][1] ** 2; g0 += J[i][0] * r0[i]; g1 += J[i][1] * r0[i]; }
    const det = a * c - b * b;
    if (!(Math.abs(det) > 1e-18)) break;
    let dy = -(c * g0 - b * g1) / det, dp = -(a * g1 - b * g0) / det;
    const step = Math.hypot(dy, dp), maxStep = 0.2;
    if (step > maxStep) { dy *= maxStep / step; dp *= maxStep / step; }
    const next = clamp(withYP(s, yaw + dy, pitch + dp));
    if (Math.hypot(...resid(next)) >= Math.hypot(...r0) - 1e-12) { s = next; break; }
    s = next;
  }
  const after = aimError(scene, s, tgt);
  return after != null && before != null && after < before ? { state: s, before, after } : { state, before, after: before };
}

/** One camera-assistant step: move scope yaw/pitch from `cur` toward `goal` with time constant tau (s). */
export function followStep(cur, goal, dt, tau = 0.35) {
  const k = 1 - Math.exp(-Math.max(0, dt) / tau);
  return { ...cur, yaw: cur.yaw + (goal.yaw - cur.yaw) * k, pitch: cur.pitch + (goal.pitch - cur.pitch) * k };
}
