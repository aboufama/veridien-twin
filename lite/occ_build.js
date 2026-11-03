// The occlusion BVHs for LiteRenderer.visible() (bvh.js, the tracer's own builder): one over the static triangles
// (with the owning camera of each lens-front triangle, so that camera's rays skip it), one per posed body in its
// body frame. Used by occ_worker.js (off the main thread) and as the fallback when module workers are unavailable.
import { buildBVH, packTriangles } from '../gpu/bvh.js';

/** data: {static: {P, I, owner: Uint8Array per triangle (0 none, k = camera index + 1)}, bodies: [{key, P, I}]}. */
export function build(data) {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const st = data.static;
  const b = buildBVH(st.P, st.I, { leafSize: 4 });
  const T = packTriangles(st.P, st.I, b.order);
  const owner = new Uint8Array(b.order.length);
  for (let k = 0; k < b.order.length; k++) owner[k] = st.owner[b.order[k]];
  const out = { static: { nodes: b.nodes, T, owner }, bodies: [] };
  const transfer = [b.nodes.buffer, T.buffer, owner.buffer];
  for (const body of data.bodies) {
    const bb = buildBVH(body.P, body.I, { leafSize: 4 });
    const TT = packTriangles(body.P, body.I, bb.order);
    out.bodies.push({ key: body.key, nodes: bb.nodes, T: TT });
    transfer.push(bb.nodes.buffer, TT.buffer);
  }
  out.ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
  return { out, transfer };
}
