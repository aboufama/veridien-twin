// Prepared-scene cache of the godview_pt candidate: prepareScene (gpu/scene_gpu.js: weld, subdivide, SAH BVH) costs ~4 s
// of CPU at load; its geometric output depends only on scene.json + meshes, the god's-eye bench extent and the bake
// spacing, so tools/build_cache.mjs writes it once (prepared.json + prepared.bin) and the page loads it in ~0.3 s.
// Materials and the emitter's radiance are recomputed from physics.yaml at every load, so a physics change never
// needs a rebuild. Any key mismatch falls back to prepareScene.
import { materialsFromPhysics, emitterFromPhysics } from '../gpu/scene_gpu.js';

const ARRAYS = [['nodes', Float32Array], ['tris', Float32Array], ['verts', Float32Array], ['adjStart', Uint32Array], ['adjList', Uint32Array]];

export function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, '0');
}

export function cacheKey(scene, extent, spacing) {
  return { scene: fnv1a(JSON.stringify(scene)), extent, spacing, version: 1 };
}

/** -> { json: object, bin: Uint8Array } */
export function serializePrepared(S, key) {
  const arrs = { nodes: S.nodes, tris: S.tris, verts: S.verts, adjStart: S.adjacency.start, adjList: S.adjacency.list };
  let off = 0;
  const layout = {};
  for (const [k] of ARRAYS) { layout[k] = [off, arrs[k].length]; off += arrs[k].byteLength; off = Math.ceil(off / 16) * 16; }
  const bin = new Uint8Array(off);
  for (const [k] of ARRAYS) bin.set(new Uint8Array(arrs[k].buffer, arrs[k].byteOffset, arrs[k].byteLength), layout[k][0]);
  const em = S.emitter;
  const json = {
    key, layout, nodeCount: S.nodeCount, triCount: S.triCount, vertCount: S.vertCount, staticVerts: S.staticVerts, staticTris: S.staticTris,
    blas: S.blas, meshTable: S.meshTable, materialNames: S.materials.map((m) => m.name),
    emitter: { tris: em.tris, area: em.area, axis: em.axis, normal: em.normal, centroid: em.centroid, radius: em.radius, shadowSkipsScope: em.shadowSkipsScope },
    stats: S.stats,
  };
  return { json, bin };
}

/** Load the cache from `base` when its key matches; returns the prepared object or null. */
export async function loadPrepared(base, scene, physics, extent, spacing, fetchFn = fetch) {
  try {
    const r = await fetchFn(base + 'prepared.json', { cache: 'no-store' });
    if (!r.ok) return null;
    const J = await r.json();
    const key = cacheKey(scene, extent, spacing);
    if (JSON.stringify(J.key) !== JSON.stringify(key)) return null;
    const mats = materialsFromPhysics(physics, J.materialNames);
    if (mats.list.length !== J.materialNames.length || mats.list.some((m, i) => m.name !== J.materialNames[i])) return null;
    const b = await fetchFn(base + 'prepared.bin');
    if (!b.ok) return null;
    const buf = await b.arrayBuffer();
    const A = {};
    for (const [k, T] of ARRAYS) { const [o, n] = J.layout[k]; A[k] = new T(buf, o, n); }
    const ep = emitterFromPhysics(physics);
    return {
      nodes: A.nodes, tris: A.tris, verts: A.verts, adjacency: { start: A.adjStart, list: A.adjList },
      nodeCount: J.nodeCount, triCount: J.triCount, vertCount: J.vertCount, staticVerts: J.staticVerts, staticTris: J.staticTris,
      blas: J.blas, meshTable: J.meshTable, materials: mats.list, materialIndex: mats.index,
      emitter: { ...J.emitter, L0: ep.L0, theta: ep.theta, p: ep.p }, stats: { ...J.stats, cached: true },
    };
  } catch (e) {
    console.warn('[godview_pt] cache not used:', e.message);
    return null;
  }
}
