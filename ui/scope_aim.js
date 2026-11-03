// The live page opens on the scope view, so the scope starts aimed at the work (docs/WEB_TWIN.md §18.2): its 30 deg
// direction of view points at the midpoint of the two instrument tips, by yaw and pitch about port C (roll and
// insertion kept), within the limits. The scene's default scope pose aims at the fundus target, which put both tips
// about 28 deg above the view centre, outside a 75 deg (16:9) image. Live pages only: the judges, scripts and
// datasets keep the scene's default state. Pure (kinematics only; tests/ui_scope_aim.test.mjs).
import { scopeForward, truthFromState, sub, add, scale, normalize, dot } from '../control/kinematics.js';

/** The midpoint of the instruments' tips in a state (D). */
export function toolsTarget(scene, state) {
  const tips = truthFromState(scene, state).instruments.map((i) => i.tip_D);
  if (!tips.length) return null;
  return scale(tips.reduce((a, b) => add(a, b), [0, 0, 0]), 1 / tips.length);
}

/**
 * Aim the scope's view direction at `target` (D). clamp(state) -> state applies the limits (control/limits.js
 * clampState(...).state); a pattern search on yaw and pitch, at most `iters` steps. Returns {state, deg} with the
 * angle left between the view direction and the target.
 */
export function aimScope(scene, state, target, { clamp = (s) => s, iters = 80 } = {}) {
  if (!state?.scope || !target) return { state, deg: null };
  const cost = (st) => {
    const s = scopeForward(scene, st.scope);
    return 1 - dot(normalize(sub(target, s.lens)), s.view_dir);
  };
  let st = clamp(state), c = cost(st), step = 0.08;
  for (let i = 0; i < iters && step > 2e-4; i++) {
    let better = null;
    for (const [dy, dp] of [[step, 0], [-step, 0], [0, step], [0, -step]]) {
      const cand = clamp({ ...st, scope: { ...st.scope, yaw: st.scope.yaw + dy, pitch: st.scope.pitch + dp } });
      const cc = cost(cand);
      if (cc < c - 1e-12 && (!better || cc < better.c)) better = { st: cand, c: cc };
    }
    if (better) { st = better.st; c = better.c; } else step /= 2;
  }
  return { state: st, deg: Math.acos(Math.max(-1, Math.min(1, 1 - c))) * 180 / Math.PI };
}
