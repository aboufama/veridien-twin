// lite_pt2 CPU worker: the load-time CPU work of lite_pt, off the main thread and unchanged in result.
//   prepare    : gpu/scene_gpu.js prepareScene (weld, subdivide, SAH BVH; ~4 s)            -> the tracer's prepared scene
//   raster     : cams.js rasterGeometry (G-buffer meshes, edge subdivision; ~0.5 s)
//   neighbours : TwinRenderer._neighbours (node smoothing neighbours; copy of gpu/renderer.js)
//   loadPrepared: cache.js loadPrepared (cache/prepared.*, ~0.1-0.3 s) -> the same prepared scene, or null on a miss
//   ri         : gpu/sensor.js relativeIlluminationMap of the full-res cameras (~80 ms each at 1280x720)
//   interior   : TwinRenderer._buildInteriorBVH's CPU part (the interior BVH; copy of gpu/renderer.js)
import { prepareScene, MESH_FLAG } from '../gpu/scene_gpu.js';
import { buildBVH, relocate } from '../gpu/bvh.js';
import { rasterGeometry } from './cams.js';
import { cacheKey, loadPrepared } from './cache.js';
import { relativeIlluminationMap, sensorParams } from '../gpu/sensor.js';

function transferables(o, out = new Set(), seen = new Set()) {
  if (!o || typeof o !== 'object' || seen.has(o)) return out;
  seen.add(o);
  if (ArrayBuffer.isView(o)) { out.add(o.buffer); return out; }
  for (const v of Object.values(o)) transferables(v, out, seen);
  return out;
}

function neighbours({ V, act, K, R, sg }) {
  const n = act.length;
  const out = new Uint32Array(n * K * 2), outF = new Float32Array(out.buffer);
  if (!(sg > 0)) { for (let k = 0; k < n; k++) { out[k * K * 2] = k; outF[k * K * 2 + 1] = 1; } return out; }
  const cell = (x) => Math.floor(x / R) + 512;
  const keyOf = (x, y, z) => (x * 1024 + y) * 1024 + z;
  const keys = new Float64Array(n);
  for (let k = 0; k < n; k++) { const i = act[k]; keys[k] = keyOf(cell(V[8 * i]), cell(V[8 * i + 1]), cell(V[8 * i + 2])); }
  const order = Uint32Array.from({ length: n }, (_, k) => k).sort((a, b) => keys[a] - keys[b]);
  const grid = new Map();
  for (let q = 0; q < n;) {
    let e = q; const k0 = keys[order[q]];
    while (e < n && keys[order[e]] === k0) e++;
    grid.set(k0, [q, e]); q = e;
  }
  const cw = new Float64Array(K), cm = new Int32Array(K);
  for (let k = 0; k < n; k++) {
    const i = act[k], px = V[8 * i], py = V[8 * i + 1], pz = V[8 * i + 2], nx = V[8 * i + 4], ny = V[8 * i + 5], nz = V[8 * i + 6];
    const cx = cell(px), cy = cell(py), cz = cell(pz);
    let cnt = 0;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const a = grid.get(keyOf(cx + dx, cy + dy, cz + dz));
      if (!a) continue;
      for (let q = a[0]; q < a[1]; q++) {
        const m = order[q], j = act[m];
        const ex = V[8 * j] - px, ey = V[8 * j + 1] - py, ez = V[8 * j + 2] - pz, d2 = ex * ex + ey * ey + ez * ez;
        if (d2 > R * R) continue;
        if (nx * V[8 * j + 4] + ny * V[8 * j + 5] + nz * V[8 * j + 6] < 0.9) continue;
        const w = Math.exp(-0.5 * d2 / (sg * sg));
        let pos = cnt < K ? cnt : K;
        while (pos > 0 && (cw[pos - 1] < w || (cw[pos - 1] === w && cm[pos - 1] > m))) pos--;
        if (pos >= K) continue;
        for (let q2 = Math.min(cnt, K - 1); q2 > pos; q2--) { cw[q2] = cw[q2 - 1]; cm[q2] = cm[q2 - 1]; }
        cw[pos] = w; cm[pos] = m; if (cnt < K) cnt++;
      }
    }
    for (let q = 0; q < cnt; q++) { out[(k * K + q) * 2] = cm[q]; outF[(k * K + q) * 2 + 1] = cw[q]; }
  }
  return out;
}

function interior({ triVerts, triMesh, meshFlags, staticTris, tris, nodes, nodeCount, triCount, mis, K, leafSize }) {
  const keep = [];
  for (let k = 0; k < staticTris; k++) {
    const a = triVerts[3 * k], b = triVerts[3 * k + 1], c = triVerts[3 * k + 2];
    const bench = meshFlags[triMesh[k]] & MESH_FLAG.BENCH;
    if (!bench && mis[a] >= 0.9 * K && mis[b] >= 0.9 * K && mis[c] >= 0.9 * K) continue;
    keep.push(k);
  }
  const n = keep.length;
  if (n === 0 || n === staticTris) return { none: true, n: staticTris };
  const P = new Float32Array(n * 9), I = new Uint32Array(n * 3);
  for (let q = 0; q < n; q++) {
    const o = keep[q] * 16;
    for (let v = 0; v < 3; v++) {
      const base = v === 0 ? [tris[o], tris[o + 1], tris[o + 2]] : [tris[o] + tris[o + 4 * v], tris[o + 1] + tris[o + 4 * v + 1], tris[o + 2] + tris[o + 4 * v + 2]];
      P.set(base, 9 * q + 3 * v); I[3 * q + v] = 3 * q + v;
    }
  }
  const bvh = buildBVH(P, I, { leafSize });
  const nodes2 = new Float32Array((nodeCount + bvh.nodeCount) * 16);
  nodes2.set(nodes); nodes2.set(relocate(bvh.nodes, nodeCount, triCount), nodeCount * 16);
  const tris2 = new Float32Array((triCount + n) * 16);
  tris2.set(tris);
  for (let q = 0; q < n; q++) tris2.set(tris.subarray(keep[bvh.order[q]] * 16, keep[bvh.order[q]] * 16 + 16), (triCount + q) * 16);
  return { nodes: nodes2, tris: tris2, interiorRoot: nodeCount, n, interiorNodes: bvh.nodeCount };
}

const logs = [];
self.onmessage = async (e) => {
  const { id, op, args } = e.data;
  const t0 = performance.now();
  try {
    let out;
    if (op === 'loadPrepared') out = await loadPrepared(new URL('./cache/', import.meta.url).href, cacheKey(args.scene, args.physics, args.opts), (...a) => logs.push(a.join(' ')));
    else if (op === 'ri') {
      const sp = sensorParams(args.physics, args.sensor || {});
      out = Object.fromEntries(args.cams.map((c) => [c.name, relativeIlluminationMap(c.lens, sp, c.size)]));
    } else if (op === 'prepare') out = prepareScene(args.scene, args.meshes, args.physics, args.opts);
    else if (op === 'raster') out = rasterGeometry(args.scene, args.meshes, args.edgeMax);
    else if (op === 'neighbours') out = neighbours(args);
    else if (op === 'interior') out = interior(args);
    else throw new Error(`unknown op ${op}`);
    self.postMessage({ id, ok: true, out, ms: performance.now() - t0, logs: logs.splice(0) }, [...transferables(out)]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.stack || err) });
  }
};
