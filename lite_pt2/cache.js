// Prepared-scene cache of lite_pt2 (2026-11-01). prepareScene (gpu/scene_gpu.js: weld, subdivide, SAH BVHs; ~2.8 s in
// a worker on the M4) depends only on the scene, the meshes, the physics and two options, so tools/build_cache.mjs
// runs it once and writes cache/prepared.{json,bin}. Kept small (the raw object is 191 MB): the two largest arrays
// are stored by what determines them and rebuilt bit-exactly at load (~0.1 s, in the worker):
//   tris  (64 B per triangle): v0, e1, e2 from the vertex positions and the per-slot vertex ids (triVerts), as
//          bvh.js packTriangles computes them, plus the ids, mesh and flags words;
//   nodes (64 B per node): only each child's (ref, count); the boxes are refitted bottom-up as exact min/max of the
//          triangle vertices, which is what bvh.js writes. The adjacency (unused by the tracer) is dropped.
// The builder checks the rebuilt arrays word by word against prepareScene's and stores any differing record as an
// exception, so the loaded scene is identical to the computed one. Any key mismatch falls back to prepareScene.
export const CACHE_VERSION = 3;

export function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, '0');
}

export function cacheKey(scene, physics, opts) {
  return { version: CACHE_VERSION, scene: fnv1a(JSON.stringify(scene)), physics: fnv1a(JSON.stringify(physics || {})), opts };
}

const TA = { Float32Array, Float64Array, Uint32Array, Int32Array, Uint16Array, Int16Array, Uint8Array, Int8Array };

/** obj -> { json (string), bin (Uint8Array) }: typed arrays to the bin (16-byte aligned), the rest to JSON. */
export function pack(obj, key) {
  const parts = []; let off = 0;
  const json = JSON.stringify({ key, data: obj }, (k, v) => {
    if (ArrayBuffer.isView(v) && !(v instanceof DataView)) {
      const o = off; parts.push([o, v]); off = Math.ceil((off + v.byteLength) / 16) * 16;
      return { $ta: v.constructor.name, o, n: v.length };
    }
    if (typeof v === 'number' && !Number.isFinite(v)) return { $num: String(v) };
    return v;
  });
  const bin = new Uint8Array(off);
  for (const [o, v] of parts) bin.set(new Uint8Array(v.buffer, v.byteOffset, v.byteLength), o);
  return { json, bin };
}

export function unpack(jsonText, buf) {
  return JSON.parse(jsonText, (k, v) => {
    if (v && typeof v === 'object') {
      if (v.$ta) return new TA[v.$ta](buf, v.o, v.n).slice();
      if (v.$num) return Number(v.$num);
    }
    return v;
  });
}

function buildTris(C) {
  const n = C.triCount, V = C.verts, TV = C.triVerts, TM = C.triMesh, flags = C.meshTable.map((m) => m.flags);
  const T = new Float32Array(n * 16), U = new Uint32Array(T.buffer);
  for (let k = 0; k < n; k++) {
    const i0 = TV[3 * k], i1 = TV[3 * k + 1], i2 = TV[3 * k + 2], a = 8 * i0, b = 8 * i1, c = 8 * i2, o = 16 * k;
    T[o] = V[a]; T[o + 1] = V[a + 1]; T[o + 2] = V[a + 2];
    T[o + 4] = V[b] - V[a]; T[o + 5] = V[b + 1] - V[a + 1]; T[o + 6] = V[b + 2] - V[a + 2];
    T[o + 8] = V[c] - V[a]; T[o + 9] = V[c + 1] - V[a + 1]; T[o + 10] = V[c + 2] - V[a + 2];
    U[o + 3] = i0; U[o + 7] = i1; U[o + 11] = i2; U[o + 12] = TM[k]; U[o + 13] = flags[TM[k]];
  }
  return T;
}

function buildNodes(C) {
  const nN = C.nodeCount, R = C.nodeRefs, V = C.verts, TV = C.triVerts;
  const N = new Float32Array(nN * 16), U = new Uint32Array(N.buffer);
  for (let i = nN - 1; i >= 0; i--) {
    for (let s = 0; s < 2; s++) {
      const ref = R[4 * i + 2 * s], cnt = R[4 * i + 2 * s + 1], o = 16 * i + 8 * s;
      let x0 = 1e30, y0 = 1e30, z0 = 1e30, x1 = -1e30, y1 = -1e30, z1 = -1e30;
      if (cnt > 0) {
        for (let k = ref; k < ref + cnt; k++) for (let v = 0; v < 3; v++) {
          const p = 8 * TV[3 * k + v], x = V[p], y = V[p + 1], z = V[p + 2];
          if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (z < z0) z0 = z; if (z > z1) z1 = z;
        }
      } else {
        const c = 16 * ref;
        for (let q = 0; q < 2; q++) {
          const b = c + 8 * q;
          x0 = Math.min(x0, N[b]); y0 = Math.min(y0, N[b + 1]); z0 = Math.min(z0, N[b + 2]);
          x1 = Math.max(x1, N[b + 4]); y1 = Math.max(y1, N[b + 5]); z1 = Math.max(z1, N[b + 6]);
        }
      }
      N[o] = x0; N[o + 1] = y0; N[o + 2] = z0; U[o + 3] = ref; N[o + 4] = x1; N[o + 5] = y1; N[o + 6] = z1; U[o + 7] = cnt;
    }
  }
  return N;
}

function exceptions(built, orig) {
  const a = new Uint32Array(built.buffer, built.byteOffset, built.length), b = new Uint32Array(orig.buffer, orig.byteOffset, orig.length);
  const idx = [];
  for (let r = 0; r < a.length / 16; r++) for (let w = 0; w < 16; w++) if (a[16 * r + w] !== b[16 * r + w]) { idx.push(r); break; }
  const words = new Uint32Array(idx.length * 16);
  idx.forEach((r, q) => words.set(b.subarray(16 * r, 16 * r + 16), 16 * q));
  return { idx: Uint32Array.from(idx), words };
}

function applyExceptions(arr, ex) {
  const u = new Uint32Array(arr.buffer, arr.byteOffset, arr.length);
  for (let q = 0; q < ex.idx.length; q++) u.set(ex.words.subarray(16 * q, 16 * q + 16), 16 * ex.idx[q]);
}

/** The prepared scene without tris, node boxes and adjacency, plus the exceptions that make expand() exact. */
export function compactPrepared(S) {
  const nU = new Uint32Array(S.nodes.buffer, S.nodes.byteOffset, S.nodes.length);
  const refs = new Uint32Array(S.nodeCount * 4);
  for (let i = 0; i < S.nodeCount; i++) { const o = 16 * i; refs[4 * i] = nU[o + 3]; refs[4 * i + 1] = nU[o + 7]; refs[4 * i + 2] = nU[o + 11]; refs[4 * i + 3] = nU[o + 15]; }
  const C = { ...S, nodes: null, tris: null, adjacency: null, nodeRefs: refs };
  C.triExc = exceptions(buildTris(C), S.tris);
  C.nodeExc = exceptions(buildNodes(C), S.nodes);
  // narrower storage, restored exactly by expand(): mesh ids as bytes, child counts as bytes, and the static
  // vertices without their words 3 and 7 (= nodeMesh and nodeArea as f32, as prepareScene writes them)
  const small = (a) => (a.every((x) => x < 256) ? Uint8Array.from(a) : a.every((x) => x < 65536) ? Uint16Array.from(a) : a);
  C.triMesh = small(S.triMesh); C.nodeMesh = small(S.nodeMesh);
  const cnt = new Uint32Array(S.nodeCount * 2), ref = new Uint32Array(S.nodeCount * 2);
  for (let i = 0; i < S.nodeCount * 2; i++) { ref[i] = refs[2 * i]; cnt[i] = refs[2 * i + 1]; }
  C.nodeRefs = null; C.nodeRef = ref; C.nodeCnt = small(cnt);
  const nS = S.staticVerts, sv = new Float32Array(nS * 6);
  for (let v = 0; v < nS; v++) for (let w = 0; w < 3; w++) { sv[6 * v + w] = S.verts[8 * v + w]; sv[6 * v + 3 + w] = S.verts[8 * v + 4 + w]; }
  C.verts = null; C.staticVerts6 = sv; C.dynVerts = S.verts.slice(8 * nS);
  return C;
}

function restore(C) {
  const nS = C.staticVerts, verts = new Float32Array(C.vertCount * 8), vu = new Uint32Array(verts.buffer);
  const sv = C.staticVerts6, nodeMesh = Uint32Array.from(C.nodeMesh);
  for (let v = 0; v < nS; v++) {
    const o = 8 * v;
    verts[o] = sv[6 * v]; verts[o + 1] = sv[6 * v + 1]; verts[o + 2] = sv[6 * v + 2];
    verts[o + 4] = sv[6 * v + 3]; verts[o + 5] = sv[6 * v + 4]; verts[o + 6] = sv[6 * v + 5];
    vu[o + 3] = nodeMesh[v]; verts[o + 7] = C.nodeArea[v];
  }
  verts.set(C.dynVerts, 8 * nS);
  const refs = new Uint32Array(C.nodeCount * 4);
  for (let i = 0; i < C.nodeCount * 2; i++) { refs[2 * i] = C.nodeRef[i]; refs[2 * i + 1] = C.nodeCnt[i]; }
  const R = { ...C, verts, nodeMesh, triMesh: Uint32Array.from(C.triMesh), nodeRefs: refs };
  delete R.staticVerts6; delete R.dynVerts; delete R.nodeRef; delete R.nodeCnt;
  return R;
}

export function expandPrepared(C0) {
  const C = restore(C0);
  const tris = buildTris(C), nodes = buildNodes(C);
  applyExceptions(tris, C.triExc); applyExceptions(nodes, C.nodeExc);
  const S = { ...C, tris, nodes, adjacency: { start: new Uint32Array(0), list: new Uint32Array(0) } };
  delete S.nodeRefs; delete S.triExc; delete S.nodeExc;
  S.stats = { ...S.stats, cached: true };
  return S;
}

/** Fetch the cache from `base` when its key matches: the expanded prepared scene, or null (miss / no cache). */
export async function loadPrepared(base, key, log = () => {}) {
  try {
    const r = await fetch(base + 'prepared.json', { cache: 'no-store' });
    if (!r.ok) { log(`no cache (${r.status})`); return null; }
    const text = await r.text();
    const head = JSON.parse(text.slice(0, text.indexOf(',"data":')) + '}');
    if (JSON.stringify(head.key) !== JSON.stringify(key)) { log(`cache key mismatch ${JSON.stringify(head.key)} != ${JSON.stringify(key)}: computing the scene`); return null; }
    const b = await fetch(base + 'prepared.bin', { cache: 'no-store' });
    if (!b.ok) return null;
    return expandPrepared(unpack(text, await b.arrayBuffer()).data);
  } catch (e) {
    log('cache not used:', e.message);
    return null;
  }
}
