// Rigid tool colliders for the jelly (docs/JELLY.md contact): the graspers' jaws as convex pieces (k-DOPs of slab hulls
// of the CAD jaw meshes, baked by tools/bake_colliders.py into colliders.json) and each shaft + collar as a capsule,
// posed per frame from control/kinematics.js posesFromState().bodies. Pure.

/** Steel jaws on PVA (docs/JELLY.md material: wet pin-on-plate, 0.15 static / 0.08 kinetic). */
export const STEEL = Object.freeze({ mu_s: 0.15, mu_k: 0.08 });

/** baked = colliders.json; bodies = posesFromState(scene, state).bodies (column-major 4x4 per "<id>/<body>").
 * Returns the collider list step() takes: [{id, kind, planes|a,b,r, center, radius, T, mu_s, mu_k}]. */
export function toolColliders(baked, bodies, { ids = null, friction = STEEL } = {}) {
  const out = [];
  for (const [id, ins] of Object.entries(baked.instruments)) {
    if (ids && !ids.includes(id)) continue;
    for (const body of ['jaw_a', 'jaw_b']) {
      const T = bodies[`${id}/${body}`];
      if (!T || !ins[body]) continue;
      ins[body].pieces.forEach((p, k) => {
        out.push({ id: `${id}/${body}#${k}`, kind: 'convex', planes: p._f32 ?? (p._f32 = Float64Array.from(p.planes)), center: p.center, radius: p.radius, T, ...friction });
      });
    }
    const T = bodies[`${id}/shaft`];
    if (T && ins.shaft) out.push({ id: `${id}/shaft`, kind: 'capsule', a: ins.shaft.a, b: ins.shaft.b, r: ins.shaft.r, T, ...friction });
  }
  return out;
}

/** Axis-aligned box as a convex collider (world-posed by T): half extents e, centred on the body origin. */
export function boxCollider(id, e, T, friction = {}) {
  const planes = [1, 0, 0, e[0], -1, 0, 0, e[0], 0, 1, 0, e[1], 0, -1, 0, e[1], 0, 0, 1, e[2], 0, 0, -1, e[2]];
  return { id, kind: 'convex', planes, center: [0, 0, 0], radius: Math.hypot(...e) + 1e-6, T, ...friction };
}
