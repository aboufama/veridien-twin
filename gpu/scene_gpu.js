// Scene preparation for the GPU tracer: scene.json + meshes/*.bin (docs/WEB_TWIN.md §4) + physics -> packed
// typed arrays. Pure ES module (runs in node for the tests and in the page or a worker).
//
// * Static meshes (dynamic: false) and the bench are welded, then split by conforming edge bisection until every
//   edge is <= nodeSpacing (3 mm). Every static vertex is a radiosity node, so a pixel reads its indirect light
//   by barycentric interpolation of three nodes, and the static geometry is the radiosity mesh (no second copy).
// * One BLAS per posed body ("L/shaft", "L/jaw_a", "scope", "board/<name>"), built in the body frame; the parts of
//   a body (shaft + collar, scope shaft + tip face + light outlet) share it.
// * Materials come from physics.yaml (values unwrapped from {value, unit, status, source}).

import { buildBVH, relocate, packTriangles, NODE_WORDS } from './bvh.js';
import { prepareMaterial, albedoTable } from './bsdf.js';

export const MESH_FLAG = { STATIC: 1, EMITTER: 2, LAYER: 4, BOARD: 8, INSTRUMENT: 16, SCOPE: 32, BENCH: 64, TEXTURED: 128 };
export const ALBEDO_BINS = 32;

export const unwrap = (x) => (x && typeof x === 'object' && !Array.isArray(x) && 'value' in x ? x.value : x);

// ------------------------------------------------------------------ loading

/** Decode one mesh's .bin (ArrayBuffer) using its layout block. Returns {P: Float32Array, N: Float32Array, I: Uint32Array}. */
export function decodeMesh(meta, buf) {
  const lay = meta.layout;
  const nv = meta.vertex_count, nt = meta.triangle_count;
  const vecCount = (c) => (c === nv * 3 && nv * 3 !== nv ? nv : c);       // count may be vertices or scalars
  const idxCount = (c) => (c === nt ? nt * 3 : c);
  const [po, pc] = lay.positions, [no, nc] = lay.normals ?? [null, 0], [io, ic] = lay.indices;
  const P = new Float32Array(buf.slice(po, po + vecCount(pc) * 12));
  const N = no == null ? null : new Float32Array(buf.slice(no, no + vecCount(nc) * 12));
  const I = new Uint32Array(buf.slice(io, io + idxCount(ic) * 4));
  let UV = null;
  if (lay.uvs) { const [uo, uc] = lay.uvs; UV = new Float32Array(buf.slice(uo, uo + (uc === nv * 2 && nv * 2 !== nv ? nv : uc) * 8)); }
  return { P, N: N || computeNormals(P, I), I, UV };
}

/** Area-weighted vertex normals (only when a mesh ships without normals). */
export function computeNormals(P, I) {
  const N = new Float32Array(P.length);
  for (let k = 0; k < I.length; k += 3) {
    const a = I[k] * 3, b = I[k + 1] * 3, c = I[k + 2] * 3;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const v of [a, b, c]) { N[v] += nx; N[v + 1] += ny; N[v + 2] += nz; }
  }
  for (let v = 0; v < N.length; v += 3) {
    const l = Math.hypot(N[v], N[v + 1], N[v + 2]) || 1;
    N[v] /= l; N[v + 1] /= l; N[v + 2] /= l;
  }
  return N;
}

/** Fetch scene.json and every mesh. `base` is the URL (or path, with a node fetch shim) of the scene directory. */
export async function loadScene(base, fetchFn = fetch) {
  const root = base.endsWith('/') ? base : base + '/';
  const scene = await (await fetchFn(root + 'scene.json')).json();
  const meshes = {};
  await Promise.all(scene.meshes.map(async (m) => {
    const buf = await (await fetchFn(root + m.file)).arrayBuffer();
    meshes[m.name] = decodeMesh(m, buf);
  }));
  return { scene, meshes, base: root };
}

// ------------------------------------------------------------------ welding and subdivision

/** Merge vertices with the same position (1e-7 m) and normal (1e-4). */
export function weld(P, N, I) {
  const map = new Map(), P2 = [], N2 = [], remap = new Uint32Array(P.length / 3);
  for (let v = 0; v < P.length / 3; v++) {
    const k = [0, 1, 2].map((c) => Math.round(P[3 * v + c] * 1e7)).join(',') + '|' +
      [0, 1, 2].map((c) => Math.round(N[3 * v + c] * 1e4)).join(',');
    let id = map.get(k);
    if (id === undefined) { id = P2.length / 3; map.set(k, id); P2.push(P[3 * v], P[3 * v + 1], P[3 * v + 2]); N2.push(N[3 * v], N[3 * v + 1], N[3 * v + 2]); }
    remap[v] = id;
  }
  const I2 = new Uint32Array(I.length);
  let w = 0;
  for (let k = 0; k < I.length; k += 3) {
    const a = remap[I[k]], b = remap[I[k + 1]], c = remap[I[k + 2]];
    if (a === b || b === c || a === c) continue;                  // degenerate after welding
    I2[w++] = a; I2[w++] = b; I2[w++] = c;
  }
  return { P: P2, N: N2, I: I2.slice(0, w) };
}

/**
 * Conforming edge bisection: every pass splits all edges longer than h at their midpoints (one shared vertex per
 * edge, so neighbours stay crack-free) and re-triangulates each triangle by its pattern of split edges (1 -> 2,
 * 2 -> 3, 3 -> 4). Positions are exact midpoints (geometry unchanged); normals are the normalised mean.
 * P, N: plain arrays (grown in place); I: Uint32Array. Returns the new index array.
 */
export function subdivide(P, N, I, h, maxPasses = 24) {
  const h2 = h * h;
  const len2 = (a, b) => { const x = P[3 * a] - P[3 * b], y = P[3 * a + 1] - P[3 * b + 1], z = P[3 * a + 2] - P[3 * b + 2]; return x * x + y * y + z * z; };
  let tris = Array.from(I);
  for (let pass = 0; pass < maxPasses; pass++) {
    const mid = new Map();
    let any = false;
    const midpoint = (a, b) => {
      const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
      let m = mid.get(key);
      if (m === undefined) {
        m = P.length / 3;
        P.push(0.5 * (P[3 * a] + P[3 * b]), 0.5 * (P[3 * a + 1] + P[3 * b + 1]), 0.5 * (P[3 * a + 2] + P[3 * b + 2]));
        let nx = N[3 * a] + N[3 * b], ny = N[3 * a + 1] + N[3 * b + 1], nz = N[3 * a + 2] + N[3 * b + 2];
        const l = Math.hypot(nx, ny, nz);
        if (l > 1e-6) { nx /= l; ny /= l; nz /= l; } else { nx = N[3 * a]; ny = N[3 * a + 1]; nz = N[3 * a + 2]; }
        N.push(nx, ny, nz);
        mid.set(key, m);
      }
      return m;
    };
    const out = [];
    for (let k = 0; k < tris.length; k += 3) {
      const v = [tris[k], tris[k + 1], tris[k + 2]];
      const mk = [len2(v[0], v[1]) > h2, len2(v[1], v[2]) > h2, len2(v[2], v[0]) > h2];
      const cnt = mk[0] + mk[1] + mk[2];
      if (!cnt) { out.push(v[0], v[1], v[2]); continue; }
      any = true;
      if (cnt === 3) {
        const a = midpoint(v[0], v[1]), b = midpoint(v[1], v[2]), c = midpoint(v[2], v[0]);
        out.push(v[0], a, c, a, v[1], b, c, b, v[2], a, b, c);
      } else if (cnt === 1) {
        const e = mk.indexOf(true), p = v[e], q = v[(e + 1) % 3], r = v[(e + 2) % 3], m = midpoint(p, q);
        out.push(p, m, r, m, q, r);
      } else {
        const e = mk.indexOf(false);                  // the unsplit edge is (r, p); split edges (p, q) and (q, r)
        const r = v[e], p = v[(e + 1) % 3], q = v[(e + 2) % 3];
        // rotate so the unsplit edge is (v[e], v[e+1]): split edges are (v[e+1], v[e+2]) and (v[e+2], v[e])
        const A = r, B = p, C = q;                    // edge AB unsplit, BC and CA split
        const mBC = midpoint(B, C), mCA = midpoint(C, A);
        out.push(mCA, mBC, C);
        // quad A, B, mBC, mCA: the shorter diagonal
        if (len2(A, mBC) < len2(B, mCA)) out.push(A, B, mBC, A, mBC, mCA);
        else out.push(A, B, mCA, B, mBC, mCA);
      }
    }
    tris = out;
    if (!any) break;
  }
  return Uint32Array.from(tris);
}

/** Bench mesh: a fine grid (<= h) over `fine` = [[x0, y0], [x1, y1]] and coarse quads out to the extent. */
export function benchMesh(z, extent, fine, h) {
  const P = [], N = [], I = [];
  const [[ex0, ey0], [ex1, ey1]] = extent;
  const fx0 = Math.max(ex0, fine[0][0]), fy0 = Math.max(ey0, fine[0][1]), fx1 = Math.min(ex1, fine[1][0]), fy1 = Math.min(ey1, fine[1][1]);
  const nx = Math.max(1, Math.ceil((fx1 - fx0) / (h * 0.999))), ny = Math.max(1, Math.ceil((fy1 - fy0) / (h * 0.999)));
  const v = (x, y) => { P.push(x, y, z); N.push(0, 0, 1); return P.length / 3 - 1; };
  const grid = [];
  for (let j = 0; j <= ny; j++) { const row = []; for (let i = 0; i <= nx; i++) row.push(v(fx0 + (fx1 - fx0) * i / nx, fy0 + (fy1 - fy0) * j / ny)); grid.push(row); }
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const a = grid[j][i], b = grid[j][i + 1], c = grid[j + 1][i], d = grid[j + 1][i + 1];
    if ((i + j) & 1) I.push(a, b, d, a, d, c); else I.push(a, b, c, b, d, c);
  }
  const quad = (x0, y0, x1, y1) => {
    if (!(x1 > x0 && y1 > y0)) return;
    const a = v(x0, y0), b = v(x1, y0), c = v(x0, y1), d = v(x1, y1);
    I.push(a, b, d, a, d, c);
  };
  quad(ex0, ey0, ex1, fy0); quad(ex0, fy1, ex1, ey1); quad(ex0, fy0, fx0, fy1); quad(fx1, fy0, ex1, fy1);
  return { P, N, I: Uint32Array.from(I) };
}

// ------------------------------------------------------------------ materials

/** Material list from physics.yaml (plain object), indexed by name; missing names get a neutral grey. */
export function materialsFromPhysics(physics, names) {
  const src = physics?.materials || {};
  const list = [], index = {};
  for (const name of names) {
    if (name in index) continue;
    const m = src[name] || {};
    const model = unwrap(m.model);
    const metallic = Number(unwrap(m.metallic) ?? (model === 'conductor' ? 1 : 0));
    const ll = m.layer_lines;
    const fa = unwrap(m.fresnel_avg);
    const textured = m.rho_d_paper != null && m.rho_d_ink != null;      // charuco_print: rho_d = mix(ink, paper, mask)
    const mat = model === 'coated_conductor'
      ? prepareMaterial({
        coated: true, f0: unwrap(m.f0), roughness: Number(unwrap(m.roughness) ?? 0.3), coatIor: Number(unwrap(m.coat_ior) ?? 1.5),
        coatRoughness: Number(unwrap(m.coat_roughness) ?? 0.05), tau: [].concat(unwrap(m.coat_transmittance) ?? [1, 1, 1]).map(Number),
      })
      : prepareMaterial({
        metallic, roughness: Number(unwrap(textured ? m.roughness_paper : m.roughness) ?? 0.5), ior: Number(unwrap(m.ior) ?? 1.5),
        rho: unwrap(textured ? m.rho_d_paper : m.rho_d) ?? [0.5, 0.5, 0.5], f0: unwrap(m.f0) ?? [0.56, 0.57, 0.58],
        fbar: fa == null ? undefined : Number(fa),
      });
    if (mat.coated && mat.tau.length === 1) mat.tau = [mat.tau[0], mat.tau[0], mat.tau[0]];
    if (textured) mat.texture = { ink: unwrap(m.rho_d_ink).map(Number), roughInk: Number(unwrap(m.roughness_ink) ?? 0.4) };
    mat.name = name;
    mat.missing = !(name in src);
    mat.emitter = model === 'emitter';
    mat.layer = ll && unwrap(ll.height_m) ? { period: Number(unwrap(ll.height_m)), depth: Number(unwrap(ll.depth_m) ?? 0), profile: unwrap(ll.profile) || 'sine' } : null;
    mat.table = albedoTable(mat, ALBEDO_BINS, 1024);
    index[name] = list.length;
    list.push(mat);
  }
  return { list, index };
}

/** Emitter radiance profile: {theta: rad[], p: []} and L0 (RGB). */
export function emitterFromPhysics(physics) {
  const em = physics?.materials?.emitter || {};
  const L0 = unwrap(em.L0) ?? [1, 1, 1];
  let prof = unwrap(em.profile);
  if (typeof prof === 'string') prof = prof.split('.').reduce((o, k) => o?.[k], physics);
  prof = prof || physics?.light?.profile || {};
  const thDeg = unwrap(prof.theta_deg) ?? [0, 90], p = unwrap(prof.p) ?? [1, 1];
  const axisDeg = Number(unwrap(physics?.scope?.emission_axis?.angle_from_shaft_deg) ?? NaN);
  return { L0: L0.map(Number), theta: thDeg.map((t) => (t * Math.PI) / 180), p: p.map(Number), axisDeg };
}

// ------------------------------------------------------------------ the whole scene

const bodyOfPart = (scene) => {
  const map = {};
  for (const inst of scene.instruments || []) for (const part of inst.parts || []) map[part.mesh] = { key: `${inst.id}/${part.body}`, kind: 'instrument', port: inst.port };
  const sc = scene.scope || {};
  for (const m of sc.mesh_parts || []) map[m] = { key: 'scope', kind: 'scope', port: sc.port };
  if (sc.emitter_mesh) map[sc.emitter_mesh] = { key: 'scope', kind: 'scope', port: sc.port, emitter: true };
  for (const b of scene.boards || []) map[b.mesh] = { key: `board/${b.name}`, kind: 'board', board: b };
  // a camera's own lens front: a fixed body (identity pose) that rays from that camera's centre skip
  for (const m of scene.meshes || []) if (m.owner_camera && !map[m.name]) map[m.name] = { key: `fixed/${m.name}`, kind: 'fixed', owner: m.owner_camera };
  // any other dynamic mesh with a body key (docs/WEB_TWIN.md §13), e.g. "L/cannula": posed by setPoses like the rest
  const ports = new Set((scene.ports || []).map((p) => p.id));
  for (const m of scene.meshes || []) {
    if (map[m.name] || !m.dynamic || !m.body) continue;
    const head = m.body.split('/')[0];
    map[m.name] = { key: m.body, kind: m.body === 'scope' ? 'scope' : head === 'board' ? 'board' : 'fixture', port: m.port ?? (ports.has(head) ? head : undefined) };
  }
  return map;
};

/**
 * Pack everything the GPU needs. Options: nodeSpacing (m, default 3 mm), leafSize.
 * Returns typed arrays + metadata; see renderer.js for how they are bound.
 */
export function prepareScene(scene, meshData, physics, { nodeSpacing = 0.003, leafSize = 4, log = () => {} } = {}) {
  const t0 = now();
  const bodies = bodyOfPart(scene);
  const meshMeta = [...scene.meshes];
  const benchMat = scene.bench?.material || 'bench_white';
  const matNames = [...meshMeta.map((m) => m.material), benchMat];
  const mats = materialsFromPhysics(physics, matNames);
  const emitterProfile = emitterFromPhysics(physics);

  // mesh table (one entry per scene mesh + the bench)
  const meshTable = [];
  const addMesh = (name, material, flags, buildDir) => {
    meshTable.push({ name, mat: mats.index[material], flags, buildDir: buildDir || [0, 0, 1] });
    return meshTable.length - 1;
  };

  // ---------------- static: weld + subdivide each static mesh, then the bench
  const sP = [], sN = [], sI = [], sMesh = [], sVertMesh = [];
  let fine = [[Infinity, Infinity], [-Infinity, -Infinity]];
  const appendStatic = (P, N, I, meshId) => {
    const off = sP.length / 3;
    for (let i = 0; i < P.length; i++) { sP.push(P[i]); sN.push(N[i]); }
    for (let v = 0; v < P.length / 3; v++) sVertMesh.push(meshId);
    for (let k = 0; k < I.length; k += 3) { sI.push(I[k] + off, I[k + 1] + off, I[k + 2] + off); sMesh.push(meshId); }
  };
  let rawStaticTris = 0;
  for (const m of meshMeta) {
    if (m.dynamic || bodies[m.name]) continue;
    const d = meshData[m.name];
    if (!d) throw new Error(`mesh ${m.name} not loaded`);
    rawStaticTris += d.I.length / 3;
    const mat = mats.list[mats.index[m.material]];
    const flags = MESH_FLAG.STATIC | (m.build_dir && mat.layer && mat.layer.depth > 0 ? MESH_FLAG.LAYER : 0);
    const id = addMesh(m.name, m.material, flags, m.build_dir);
    const w = weld(d.P, d.N, d.I);
    const I2 = subdivide(w.P, w.N, w.I, nodeSpacing);
    appendStatic(w.P, w.N, I2, id);
    for (let v = 0; v < w.P.length / 3; v++) {
      fine[0][0] = Math.min(fine[0][0], w.P[3 * v]); fine[0][1] = Math.min(fine[0][1], w.P[3 * v + 1]);
      fine[1][0] = Math.max(fine[1][0], w.P[3 * v]); fine[1][1] = Math.max(fine[1][1], w.P[3 * v + 1]);
    }
  }
  if (scene.bench) {
    const ext = scene.bench.extent || [[-0.5, -0.5], [0.5, 0.5]];
    if (!Number.isFinite(fine[0][0])) fine = ext;
    const margin = 0.006;
    const fb = [[fine[0][0] - margin, fine[0][1] - margin], [fine[1][0] + margin, fine[1][1] + margin]];
    const b = benchMesh(scene.bench.z ?? 0, ext, fb, nodeSpacing);
    const id = addMesh('bench', benchMat, MESH_FLAG.STATIC | MESH_FLAG.BENCH, [0, 0, 1]);
    appendStatic(b.P, b.N, b.I, id);
  }
  const staticVerts = sP.length / 3, staticTris = sI.length / 3;
  log(`static: ${rawStaticTris} CAD triangles -> ${staticTris} triangles, ${staticVerts} nodes (<= ${nodeSpacing * 1e3} mm edges)`);

  // ---------------- dynamic bodies
  const bodyList = [];          // {key, kind, port, meshes: [names]}
  const bodyIndex = {};
  for (const m of meshMeta) {
    const b = bodies[m.name];
    if (!b) continue;
    if (!(b.key in bodyIndex)) { bodyIndex[b.key] = bodyList.length; bodyList.push({ key: b.key, kind: b.kind, port: b.port, board: b.board, owner: b.owner, meshes: [] }); }
    bodyList[bodyIndex[b.key]].meshes.push(m);
  }

  // ---------------- static BVH
  const tb0 = now();
  const sPf = Float32Array.from(sP), sIu = Uint32Array.from(sI);
  const sbvh = buildBVH(sPf, sIu, { leafSize });
  const staticBuildMs = now() - tb0;
  log(`static BVH: ${sbvh.nodeCount} nodes, depth ${sbvh.maxDepth}, ${staticBuildMs.toFixed(0)} ms`);

  // ---------------- per-body BLAS
  const dynP = [], dynN = [], blas = [];
  let dynVertBase = staticVerts;
  const emitTris = [];
  for (const body of bodyList) {
    const P = [], N = [], I = [], triMesh = [], UV = [];
    for (const m of body.meshes) {
      const d = meshData[m.name];
      if (!d) throw new Error(`mesh ${m.name} not loaded`);
      const isEm = bodies[m.name].emitter;
      const mat = isEm ? 'emitter' : m.material;
      if (isEm && !(mat in mats.index)) { mats.index[mat] = mats.list.length; mats.list.push({ ...prepareMaterial({ rho: [0.02, 0.02, 0.02], roughness: 0.05 }), name: mat, emitter: true, table: null }); }
      const texd = !isEm && mats.list[mats.index[m.material]]?.texture && d.UV;
      const flags = (isEm ? MESH_FLAG.EMITTER : 0) | (body.kind === 'board' ? MESH_FLAG.BOARD : 0) | (texd ? MESH_FLAG.TEXTURED : 0) |
        (body.kind === 'instrument' ? MESH_FLAG.INSTRUMENT : 0) | (body.kind === 'scope' ? MESH_FLAG.SCOPE : 0);
      const id = addMesh(m.name, isEm ? 'emitter' : m.material, flags, m.build_dir);
      const off = P.length / 3;
      for (let i = 0; i < d.P.length; i++) { P.push(d.P[i]); N.push(d.N[i]); }
      for (let v = 0; v < d.P.length / 3; v++) UV.push(d.UV ? d.UV[2 * v] : 0, d.UV ? d.UV[2 * v + 1] : 0);
      for (let k = 0; k < d.I.length; k += 3) {
        I.push(d.I[k] + off, d.I[k + 1] + off, d.I[k + 2] + off); triMesh.push(id);
        if (isEm) {
          const a = d.I[k] * 3, b = d.I[k + 1] * 3, c = d.I[k + 2] * 3;
          const v0 = [d.P[a], d.P[a + 1], d.P[a + 2]], e1 = [d.P[b] - v0[0], d.P[b + 1] - v0[1], d.P[b + 2] - v0[2]], e2 = [d.P[c] - v0[0], d.P[c + 1] - v0[1], d.P[c + 2] - v0[2]];
          let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
          const area = 0.5 * Math.hypot(...n);
          const vn = [d.N[a] + d.N[b] + d.N[c], d.N[a + 1] + d.N[b + 1] + d.N[c + 1], d.N[a + 2] + d.N[b + 2] + d.N[c + 2]];
          if (n[0] * vn[0] + n[1] * vn[1] + n[2] * vn[2] < 0) n = n.map((x) => -x);
          const l = Math.hypot(...n) || 1;
          emitTris.push({ v0, e1, e2, n: n.map((x) => x / l), area });
        }
      }
    }
    const Pf = Float32Array.from(P), Iu = Uint32Array.from(I);
    const bv = buildBVH(Pf, Iu, { leafSize });
    blas.push({ body: body.key, bvh: bv, P: Pf, N: Float32Array.from(N), I: Iu, triMesh, vertBase: dynVertBase, UV: Float32Array.from(UV) });
    dynVertBase += Pf.length / 3;
  }

  // ---------------- combined buffers: nodes, triangles (64 B: v0|i0, e1|i1, e2|i2, mesh,flags), vertices (pos, normal)
  let nodeTotal = sbvh.nodeCount, triTotal = staticTris;
  for (const b of blas) { b.nodeOff = nodeTotal; b.triOff = triTotal; nodeTotal += b.bvh.nodeCount; triTotal += b.bvh.triCount; }
  const nodes = new Float32Array(nodeTotal * NODE_WORDS);
  nodes.set(sbvh.nodes, 0);
  for (const b of blas) nodes.set(relocate(b.bvh.nodes, b.nodeOff, b.triOff), b.nodeOff * NODE_WORDS);
  const tris = new Float32Array(triTotal * 16), trisU = new Uint32Array(tris.buffer);
  packTriangles(sPf, sIu, sbvh.order, tris, 0);
  const triVerts = new Uint32Array(triTotal * 3);        // CPU copy of the vertex ids per BVH slot
  const triMeshArr = new Uint32Array(triTotal);
  const putAttr = (slot, i0, i1, i2, mesh) => {
    const o = slot * 16;
    trisU[o + 3] = i0; trisU[o + 7] = i1; trisU[o + 11] = i2; trisU[o + 12] = mesh; trisU[o + 13] = meshTable[mesh].flags;
    triVerts[3 * slot] = i0; triVerts[3 * slot + 1] = i1; triVerts[3 * slot + 2] = i2; triMeshArr[slot] = mesh;
  };
  for (let k = 0; k < staticTris; k++) { const t = sbvh.order[k]; putAttr(k, sIu[3 * t], sIu[3 * t + 1], sIu[3 * t + 2], sMesh[t]); }
  for (const b of blas) {
    packTriangles(b.P, b.I, b.bvh.order, tris, b.triOff);
    for (let k = 0; k < b.bvh.triCount; k++) {
      const t = b.bvh.order[k];
      putAttr(b.triOff + k, b.I[3 * t] + b.vertBase, b.I[3 * t + 1] + b.vertBase, b.I[3 * t + 2] + b.vertBase, b.triMesh[t]);
    }
  }
  const vertTotal = dynVertBase;
  const verts = new Float32Array(vertTotal * 8);
  const nodeArea = new Float64Array(staticVerts);
  for (let k = 0; k < sIu.length; k += 3) {
    const a = sIu[k], b = sIu[k + 1], c = sIu[k + 2];
    const ux = sP[3 * b] - sP[3 * a], uy = sP[3 * b + 1] - sP[3 * a + 1], uz = sP[3 * b + 2] - sP[3 * a + 2];
    const vx = sP[3 * c] - sP[3 * a], vy = sP[3 * c + 1] - sP[3 * a + 1], vz = sP[3 * c + 2] - sP[3 * a + 2];
    const ar = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 3;
    nodeArea[a] += ar; nodeArea[b] += ar; nodeArea[c] += ar;
  }
  const vertsU = new Uint32Array(verts.buffer);
  for (let v = 0; v < staticVerts; v++) {
    verts.set([sP[3 * v], sP[3 * v + 1], sP[3 * v + 2], 0, sN[3 * v], sN[3 * v + 1], sN[3 * v + 2], nodeArea[v]], 8 * v);
    vertsU[8 * v + 3] = sVertMesh[v];
  }
  for (const b of blas) for (let v = 0; v < b.P.length / 3; v++) {
    // dynamic vertices carry their texture coordinates in the two spare words (u in pos.w, v in normal.w)
    const o = 8 * (b.vertBase + v);
    verts.set([b.P[3 * v], b.P[3 * v + 1], b.P[3 * v + 2], b.UV[2 * v], b.N[3 * v], b.N[3 * v + 1], b.N[3 * v + 2], b.UV[2 * v + 1]], o);
  }

  // ---------------- emitter
  let emitArea = 0;
  for (const e of emitTris) emitArea += e.area;
  let acc = 0;
  for (const e of emitTris) { acc += e.area; e.cdf = acc / (emitArea || 1); }
  const meanN = emitTris.reduce((s, e) => [s[0] + e.n[0] * e.area, s[1] + e.n[1] * e.area, s[2] + e.n[2] * e.area], [0, 0, 0]);
  const ml = Math.hypot(...meanN) || 1;
  const emitNormal = meanN.map((x) => x / ml);
  let emitAxis = emitNormal;
  if (Number.isFinite(emitterProfile.axisDeg)) {
    const a = (emitterProfile.axisDeg * Math.PI) / 180;
    emitAxis = [0, Math.sin(a), Math.cos(a)];      // scope +z rotated about +x toward +y (physics.yaml scope.tip_face)
  }
  // Shadow rays end on the emitter from its front side; when no other part of the scope body lies in front of the
  // emitter's plane, the scope cannot occlude them, so they skip its BLAS (exact, and every shadow ray saves it).
  let scopeAhead = -Infinity;
  if (emitTris.length) {
    const c = emitTris.reduce((s, e) => [s[0] + e.v0[0] * e.area, s[1] + e.v0[1] * e.area, s[2] + e.v0[2] * e.area], [0, 0, 0]).map((x) => x / (emitArea || 1));
    for (const b of blas) if (b.body === 'scope') for (let v = 0; v < b.P.length / 3; v++) {
      const d = (b.P[3 * v] - c[0]) * emitNormal[0] + (b.P[3 * v + 1] - c[1]) * emitNormal[1] + (b.P[3 * v + 2] - c[2]) * emitNormal[2];
      scopeAhead = Math.max(scopeAhead, d);
    }
  }
  // centroid and radius of the outlet (scope frame), for the per-point shadow culling test
  const centroid = [0, 0, 0];
  for (const e of emitTris) for (let k = 0; k < 3; k++) centroid[k] += (e.v0[k] + (e.e1[k] + e.e2[k]) / 3) * e.area / (emitArea || 1);
  let radius = 0;
  for (const e of emitTris) for (const q of [e.v0, e.v0.map((x, k) => x + e.e1[k]), e.v0.map((x, k) => x + e.e2[k])]) radius = Math.max(radius, Math.hypot(q[0] - centroid[0], q[1] - centroid[1], q[2] - centroid[2]));
  const axisVsNormalDeg = (Math.acos(Math.max(-1, Math.min(1, emitAxis[0] * emitNormal[0] + emitAxis[1] * emitNormal[1] + emitAxis[2] * emitNormal[2]))) * 180) / Math.PI;

  // ---------------- capsules per body (body frame), for the analytic instrument occlusion of indirect light
  // capsule: bounds the whole BLAS (traversal and shadow culling). occCapsule and rhoEff: only the meshes that sit in
  // the cavity. An overview_only mesh (a trocar housing outside the dome; the Mitsuba reference leaves it out) hides
  // no light inside the dome; fitted in, it set the cannula capsule's radius to the housing's 13 / 17 mm instead of
  // the sheath's 4.25 / 6.4 mm and its albedo into rho_eff (round-2 residual diagnosis, docs/DECISIONS.md).
  const overviewOnly = new Set(meshMeta.filter((m) => m.overview_only).map((m) => m.name));
  for (const b of blas) {
    b.capsule = fitCapsule(b.P);
    const keepV = new Uint8Array(b.P.length / 3);
    let anyDrop = false;
    for (let k = 0; k < b.I.length; k += 3) {
      if (overviewOnly.has(meshTable[b.triMesh[k / 3]].name)) { anyDrop = true; continue; }
      keepV[b.I[k]] = keepV[b.I[k + 1]] = keepV[b.I[k + 2]] = 1;
    }
    if (anyDrop) {
      const Pc = [];
      for (let v = 0; v < keepV.length; v++) if (keepV[v]) Pc.push(b.P[3 * v], b.P[3 * v + 1], b.P[3 * v + 2]);
      b.occCapsule = Pc.length ? fitCapsule(Float32Array.from(Pc)) : { a: [0, 0, 0], b: [0, 0, 0], r: 0 };
    } else b.occCapsule = b.capsule;
    let ar = 0, rho = 0;
    for (let k = 0; k < b.I.length; k += 3) {
      if (anyDrop && overviewOnly.has(meshTable[b.triMesh[k / 3]].name)) continue;
      const a = b.I[k] * 3, c = b.I[k + 1] * 3, d = b.I[k + 2] * 3;
      const ux = b.P[c] - b.P[a], uy = b.P[c + 1] - b.P[a + 1], uz = b.P[c + 2] - b.P[a + 2];
      const vx = b.P[d] - b.P[a], vy = b.P[d + 1] - b.P[a + 1], vz = b.P[d + 2] - b.P[a + 2];
      const A = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
      const mat = mats.list[meshTable[b.triMesh[k / 3]].mat];
      const hh = mat.table ? mat.table.hh : [0, 0, 0];
      ar += A; rho += A * (0.2126 * hh[0] + 0.7152 * hh[1] + 0.0722 * hh[2]);
    }
    b.rhoEff = ar ? rho / ar : 0;
  }

  // ---------------- vertex adjacency of the static nodes (CSR), for optional smoothing
  const adj = adjacency(staticVerts, sIu);

  const out = {
    nodes, nodeCount: nodeTotal, tris, triCount: triTotal, triVerts, triMesh: triMeshArr, verts, vertCount: vertTotal,
    staticVerts, staticTris, staticRoot: 0, staticBounds: [sbvh.bmin, sbvh.bmax],
    blas: blas.map((b, i) => ({ body: b.body, kind: bodyList[i].kind, owner: bodyList[i].owner || null, root: b.nodeOff, triOff: b.triOff, triCount: b.bvh.triCount, bmin: b.bvh.bmin, bmax: b.bvh.bmax, capsule: b.capsule, occCapsule: b.occCapsule, rhoEff: b.rhoEff })),
    bodies: bodyList, meshTable, materials: mats.list, materialIndex: mats.index,
    emitter: { tris: emitTris, area: emitArea, axis: emitAxis, normal: emitNormal, axisVsNormalDeg, L0: emitterProfile.L0, theta: emitterProfile.theta, p: emitterProfile.p, centroid, radius, scopeAheadM: scopeAhead, shadowSkipsScope: scopeAhead <= 1e-6 },
    nodeArea, adjacency: adj, nodeMesh: Uint32Array.from(sVertMesh),
    stats: { rawStaticTris, staticTris, staticVerts, staticBuildMs, totalMs: now() - t0, nodeSpacing, blasTris: blas.reduce((s, b) => s + b.bvh.triCount, 0), staticDepth: sbvh.maxDepth },
  };
  log(`scene prepared in ${out.stats.totalMs.toFixed(0)} ms: ${triTotal} triangles, ${nodeTotal} BVH nodes, ${blas.length} bodies, emitter ${emitTris.length} tris ${(emitArea * 1e6).toFixed(2)} mm^2`);
  return out;
}

/**
 * The capsule of a posed body for the analytic occlusion of indirect light (world frame): only the part of the body
 * that is inside the cavity, and no volume counted twice (the term sums the bodies' occlusions as if disjoint).
 * kind: the BLAS kind; a, b, r: the body's posed occlusion capsule; port: the scene port the body goes through.
 * - a cannula ('fixture' on a port): its segment clipped to s >= 0 (inward of the pivot) and the sheath's radius,
 *   port.cannula_outer_radius, when the scene gives it (the fit bounds the sheath's grip ribs, 4.55 mm over half
 *   its length, against the 4.25 mm sheath).
 * - a body through a port (instrument part, scope): clipped to s >= the cannula's toe, port.cannula_toe_insertion.
 *   Between the pivot and the toe it runs inside the cannula, whose capsule already hides that light.
 * - flush bodies (a camera's lens front, the boards): r = 0, no capsule.
 * Returns {a, b, r}; r = 0 when nothing is left.
 */
export function cavityCapsule(kind, a, b, r, port) {
  if (kind === 'fixed' || kind === 'board') return { a, b, r: 0 };
  if (!port) return { a, b, r };
  const inward = port.axis_out.map((x) => -x);
  const sOf = (X) => (X[0] - port.pivot[0]) * inward[0] + (X[1] - port.pivot[1]) * inward[1] + (X[2] - port.pivot[2]) * inward[2];
  const toe = Number(port.cannula_toe_insertion);
  const fixture = kind === 'fixture';
  const s0 = fixture || !Number.isFinite(toe) ? 0 : toe;
  if (fixture && Number.isFinite(Number(port.cannula_outer_radius))) r = Number(port.cannula_outer_radius);
  const sA = sOf(a), sB = sOf(b);
  if (sA < s0 && sB < s0) return { a, b, r: 0 };
  const lerp = (P, Q, t) => P.map((x, k) => x + (Q[k] - x) * t);
  if (sA < s0) a = lerp(a, b, (s0 - sA) / (sB - sA));
  else if (sB < s0) b = lerp(b, a, (s0 - sB) / (sA - sB));
  return { a, b, r };
}

/** Capsule (segment a-b, radius r) around a vertex set: axis = principal direction, r = max distance from it. */
export function fitCapsule(P) {
  const n = P.length / 3;
  if (!n) return { a: [0, 0, 0], b: [0, 0, 0], r: 0 };
  const c = [0, 0, 0];
  for (let v = 0; v < n; v++) for (let k = 0; k < 3; k++) c[k] += P[3 * v + k] / n;
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let v = 0; v < n; v++) {
    const d = [P[3 * v] - c[0], P[3 * v + 1] - c[1], P[3 * v + 2] - c[2]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += d[i] * d[j];
  }
  let ax = [0.3, 0.5, 0.8];
  for (let it = 0; it < 64; it++) {
    const y = [0, 1, 2].map((i) => C[i][0] * ax[0] + C[i][1] * ax[1] + C[i][2] * ax[2]);
    const l = Math.hypot(...y); if (!(l > 0)) break;
    ax = y.map((x) => x / l);
  }
  let smin = Infinity, smax = -Infinity, r = 0;
  for (let v = 0; v < n; v++) {
    const d = [P[3 * v] - c[0], P[3 * v + 1] - c[1], P[3 * v + 2] - c[2]];
    const s = d[0] * ax[0] + d[1] * ax[1] + d[2] * ax[2];
    smin = Math.min(smin, s); smax = Math.max(smax, s);
    r = Math.max(r, Math.hypot(d[0] - s * ax[0], d[1] - s * ax[1], d[2] - s * ax[2]));
  }
  return { a: c.map((x, k) => x + smin * ax[k]), b: c.map((x, k) => x + smax * ax[k]), r };
}

/** CSR vertex adjacency {start: Uint32Array(n+1), list: Uint32Array}. */
export function adjacency(n, I) {
  const deg = new Uint32Array(n);
  for (let k = 0; k < I.length; k += 3) for (let e = 0; e < 3; e++) { deg[I[k + e]] += 2; }
  const start = new Uint32Array(n + 1);
  for (let v = 0; v < n; v++) start[v + 1] = start[v] + deg[v];
  const tmp = new Uint32Array(start[n]), fill = start.slice(0, n);
  for (let k = 0; k < I.length; k += 3) for (let e = 0; e < 3; e++) {
    const a = I[k + e], b = I[k + (e + 1) % 3], c = I[k + (e + 2) % 3];
    tmp[fill[a]++] = b; tmp[fill[a]++] = c;
  }
  // dedupe per vertex
  const list = [], s2 = new Uint32Array(n + 1);
  for (let v = 0; v < n; v++) {
    const seen = new Set();
    for (let k = start[v]; k < start[v + 1]; k++) seen.add(tmp[k]);
    for (const x of seen) list.push(x);
    s2[v + 1] = list.length;
  }
  return { start: s2, list: Uint32Array.from(list) };
}

function now() { return typeof performance !== 'undefined' ? performance.now() : Date.now(); }
