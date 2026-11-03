// LitePRTRenderer: the lite live renderer of the aimed dome with precomputed indirect light vs scope pose
// (docs/WEB_TWIN.md §18; owner: the lite_prt racer). The §8 API, so the page and window.twinJudge swap it in by name.
//
//   const r = await LitePRTRenderer.create({ canvas?, scene, physics, device?, prt? });
//   r.setPoses(poses0, poses1?); await r.renderCamera(name, { exposure }); r.sensor(name, opts) -> rgba8unorm;
//   await r.readback(name, 'rgb8'|'linear'); r.visible(name, pointsD); r.timings(); r.cameras (SCOPE included);
//   r.present(name, canvasContext)
//
// How a frame is made (no primary ray tracing, no gather table, no Jacobi):
// 1. Primary visibility by raster: every triangle (static + posed bodies) is projected per vertex through the exact
//    KB4 model (or the SCOPE pinhole) into a 4x MSAA r32uint visibility buffer (triangle slot | body). The meshes are
//    <= 3 mm, so straight edges between projected vertices stay sub-pixel.
// 2. One compute pass shades each pixel: the 4 MSAA samples are grouped by triangle; each group casts its exact KB4
//    ray at the samples' mean position against that one triangle (exact hit point and barycentrics), then runs the
//    path tracer's own shading (gpu/wgsl shade.wgsl.js shadeHit: area-emitter light samples with node-culled shadow
//    rays, the §8 BSDF, node indirect x capsule occlusion, posed bodies' BSDF bounce to the node radiosity).
// 3. Indirect light: node irradiance E_free from a PCA basis over the static nodes x coefficients regressed from the
//    scope pose (RBF over sampled poses, precomputed offline with the path tracer's converged node solver). At runtime
//    only the node direct pass runs (shadow rays per node, when something moved), then one tiny pass writes
//    E_free = mean + B c and the node exitance. Without the asset (prt: false) it falls back to the tracer's solver.
import { TwinRenderer } from '../gpu/renderer.js';
import { shaderSource } from '../gpu/wgsl/kernels.wgsl.js';
import { COMMON, defines } from '../gpu/wgsl/common.wgsl.js';
import { LENS } from '../gpu/wgsl/lens.wgsl.js';
import { AutoExposure } from '../gpu/sensor.js';
import { posesFromState, defaultState } from '../control/kinematics.js';

const MS_POS = [[0.375, 0.125], [0.875, 0.375], [0.125, 0.625], [0.625, 0.875]];   // WebGPU/D3D 4x standard pattern

const RASTER = (c) => /* wgsl */ `${defines(c)}${COMMON}${LENS}
@group(0) @binding(0) var<storage, read> tris: array<vec4f>;
@group(0) @binding(1) var<storage, read> triInst: array<u32>;
@group(1) @binding(0) var<uniform> scn: SceneU;
@group(1) @binding(1) var<uniform> cam: CamU;
struct VO { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) id: u32, @location(1) ok: f32 }
@group(0) @binding(2) var<storage, read> sub: array<u32>;        // subdivided long triangles: (slot, u, v) per vertex
fn instOf(ti: u32) -> u32 { let t = ti & 0x7fffffffu; return select(t, MISS, t == 0x7fffffffu); }
fn projectV(p0: vec3f, k: u32, inst: u32) -> VO {
  var p = p0;
  var o: VO;
  o.id = k + 1u;
  o.ok = 1.0;
  if (inst != MISS) {
    let I = scn.inst[inst];
    if ((bitcast<u32>(I.p1.w) & 2u) != 0u || ((cam.flags.w >> inst) & 1u) != 0u) { o.pos = vec4f(0.0, 0.0, 2.0, 1.0); return o; }
    let P = instPose(inst, 0.5);
    p = P.r * p + P.p;
    o.id = o.id | ((inst + 1u) << 27u);
  }
  let rel = p - cam.pos.xyz;
  let X = vec3f(dot(rel, cam.rx.xyz), dot(rel, cam.ry.xyz), dot(rel, cam.rz.xyz));
  let dist = length(X);
  let rxy = length(X.xy);
  let th = atan2(rxy, X.z);
  let W = f32(cam.size.x); let H = f32(cam.size.y);
  var uv: vec2f;
  if (th > 1.2) {
    // beyond every image corner (59.6-65.5 deg, the SCOPE's 45 deg) and short of the KB4 fold: push far outside
    let dir = select(vec2f(1.0, 0.0), X.xy / rxy, rxy > 1e-12);
    uv = cam.lensA.zw + dir * (8.0 * W);
    o.ok = -1e4;          // the fragment drops every triangle with such a vertex (a lens bore spans the axis)
  } else if (cam.pos.w < 0.5) {
    uv = kb4Project(Lens(cam.lensA, cam.lensB, cam.lensC), X);
  } else {
    uv = vec2f(cam.lensA.x * X.x / X.z + cam.lensA.z, cam.lensA.y * X.y / X.z + cam.lensA.w);
  }
  // OpenCV pixel centres at integers: pixel x covers [x - 0.5, x + 0.5]
  o.pos = vec4f(2.0 * (uv.x + 0.5) / W - 1.0, 1.0 - 2.0 * (uv.y + 0.5) / H, clamp(dist * 0.25, 0.0, 1.0), 1.0);
  return o;
}
// draw 1: every triangle short enough (<= 3 mm edges) straight from the BVH slots; long ones are dropped here
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  let k = vi / 3u;
  let c = vi - 3u * k;
  let ti = triInst[k];
  if ((ti & 0x80000000u) != 0u) { var o: VO; o.pos = vec4f(0.0, 0.0, 2.0, 1.0); o.id = 0u; o.ok = -1e4; return o; }
  var p = tris[4u * k].xyz;
  if (c == 1u) { p = p + tris[4u * k + 1u].xyz; } else if (c == 2u) { p = p + tris[4u * k + 2u].xyz; }
  return projectV(p, k, instOf(ti));
}
// draw 2: the long CAD triangles of the posed bodies, bisected to <= 3 mm edges (the KB4 curves edges)
@vertex fn vs2(@builtin(vertex_index) vi: u32) -> VO {
  let k = sub[3u * vi];
  let u = bitcast<f32>(sub[3u * vi + 1u]);
  let v = bitcast<f32>(sub[3u * vi + 2u]);
  let p = tris[4u * k].xyz + tris[4u * k + 1u].xyz * u + tris[4u * k + 2u].xyz * v;
  return projectV(p, k, instOf(triInst[k]));
}
@fragment fn fs(i: VO) -> @location(0) vec2u { if (i.ok < 0.9999) { discard; } return vec2u(i.id & 0xffffu, i.id >> 16u); }   // rg16uint: r32uint has no MSAA
`;

const SHADE_EXTRA = /* wgsl */ `
override LITE_CULL: bool = true;
@group(2) @binding(0) var visTex: texture_multisampled_2d<u32>;
fn msPos(s: u32) -> vec2f {
  switch (s) { case 0u: { return vec2f(0.375, 0.125); } case 1u: { return vec2f(0.875, 0.375); } case 2u: { return vec2f(0.125, 0.625); } default: { return vec2f(0.625, 0.875); } }
}
fn primaryDir(uv: vec2f) -> vec3f {
  var dc: vec3f;
  if (cam.pos.w < 0.5) { dc = kb4Unproject(Lens(cam.lensA, cam.lensB, cam.lensC), uv); }
  else { dc = normalize(vec3f((uv.x - cam.lensA.z) / cam.lensA.x, (uv.y - cam.lensA.w) / cam.lensA.y, 1.0)); }
  return normalize(cam.rx.xyz * dc.x + cam.ry.xyz * dc.y + cam.rz.xyz * dc.z);
}
// the hit of ray (o, d) on the one triangle the raster saw (slot k, body ib - 1): exact when the ray meets it, else
// the nearest point of the triangle (sub-pixel slivers at the projected edges)
fn hitFromId(id: u32, o: vec3f, d: vec3f, time: f32) -> Hit {
  let k = (id & 0x7ffffffu) - 1u;
  let ib = id >> 27u;
  var inst = MISS; var ol = o; var dl = d;
  if (ib != 0u) { inst = ib - 1u; let P = instPose(inst, time); ol = (o - P.p) * P.r; dl = d * P.r; }
  let v0 = tris[4u * k].xyz; let e1 = tris[4u * k + 1u].xyz; let e2 = tris[4u * k + 2u].xyz;
  let pp = cross(dl, e2);
  var det = dot(e1, pp);
  if (abs(det) < 1e-30) { det = 1e-30; }
  let inv = 1.0 / det;
  let s = ol - v0;
  var u = dot(s, pp) * inv;
  let q = cross(s, e1);
  var v = dot(dl, q) * inv;
  var t = dot(e2, q) * inv;
  if (u < 0.0 || v < 0.0 || u + v > 1.0 || t <= 0.0) {
    u = clamp(u, 0.0, 1.0); v = clamp(v, 0.0, 1.0);
    if (u + v > 1.0) { let sc = 1.0 / (u + v); u = u * sc; v = v * sc; }
    t = max(dot(v0 + e1 * u + e2 * v - ol, dl), 1e-6);
  }
  return Hit(t, k, u, v, inst);
}
// Shadow-ray culling for the lite path: the tracer's rule (the hit triangle's three nodes agree that every lit
// emitter sample is visible or hidden, and no posed body comes near the segment to the emitter), but tested against
// the bodies' in-cavity occlusion capsules (sheath radius; the trocar housings outside the dome cannot shadow it) and
// without the scope's own cannula, which lies behind the tip face. The tracer's bounding capsules include the
// housings (13-17 mm), so its test failed for every dome pixel near the scope and every pixel traced 4 shadow rays.
fn liteVis(s: Surf, time: f32) -> f32 {
  if ((s.flags & F_STATIC) == 0u) { return -1.0; }
  let a = nodeState[4u * s.vi.x + 3u].w;
  let b = nodeState[4u * s.vi.y + 3u].w;
  let c = nodeState[4u * s.vi.z + 3u].w;
  if (!(a == b && b == c && (a == 1.0 || a == 0.0))) { return -1.0; }
  let P = instPose(scn.info.y, time);
  let d = (P.r * scn.emitC.xyz + P.p) - s.p;
  let cnt = min(scn.info.x, MAX_INST);
  let skip = bitcast<u32>(scn.axis.w) | bitcast<u32>(cam.aid.y);
  for (var i = 0u; i < cnt; i++) {
    let I = scn.inst[i];
    if ((bitcast<u32>(I.p1.w) & 2u) != 0u || ((skip >> i) & 1u) != 0u || I.capA.w <= 0.0) { continue; }
    if (segNearCapsule(s.p, d, 0.0, 1.0, I.capA.xyz, I.capB.xyz, I.capA.w + scn.emitC.w + scn.eAvg.w)) { return -1.0; }
  }
  return a;
}

// shadeHit (shade.wgsl.js) without the tracker caches, with liteVis
fn liteHit(h: Hit, o: vec3f, d: vec3f, time: f32, ul: vec3f, ub: vec3f, opts: u32, lightIndex: u32, S: PixSeeds) -> vec3f {
  let s = surface(h, o, d, time);
  let wo = -d;
  if ((s.flags & F_EMITTER) != 0u) { return emitted(emitterNormal(h, s, time), emitterAxisAt(time), wo); }
  var M = loadMat(s.mat);
  if ((s.flags & F_TEXTURED) != 0u) {
    let uv = s.b.x * vec2f(verts[2u * s.vi.x].w, verts[2u * s.vi.x + 1u].w) + s.b.y * vec2f(verts[2u * s.vi.y].w, verts[2u * s.vi.y + 1u].w)
           + s.b.z * vec2f(verts[2u * s.vi.z].w, verts[2u * s.vi.z + 1u].w);
    let m = textureSampleLevel(boardTex, boardSmp, uv, 0.0).r;
    M.c = mix(M.x.xyz, M.c, m);
    M.alpha = mix(M.x.w, M.alpha, m);
  }
  let n = shadingNormal(s, opts);
  let isStatic = (s.flags & F_STATIC) != 0u;
  let mis = !isStatic && (opts & O_NO_MIS) == 0u;
  let nl = clamp(bitcast<u32>(cam.lensC.z), 1u, 16u);
  var cached = -1.0;
  if (CULL_SHADOWS) { cached = liteVis(s, time); }
  var L = directLight(s, n, M, wo, time, ul, mis, f32(nl), cached);
  for (var j = 1u; j < nl; j++) {
    let uj = vec3f(kr(lightIndex + j, KL0, S.s3), kr(lightIndex + j, KL1, S.s4), kr(lightIndex + j, KL2, S.s5));
    L = L + directLight(s, n, M, wo, time, uj, mis, f32(nl), cached);
  }
  L = L / f32(nl);
  if (isStatic) {
    if ((opts & (O_NO_INDIRECT | O_DIRECT_ONLY)) == 0u) {
      var E = nodeEfree(s.vi, s.b);
      if ((opts & O_NO_CAPSULE) == 0u) { E = E * capsuleVis(s.p, n); }
      let mu = max(dot(n, wo), 0.0);
      L = L + E * INV_PI * (albedoSpec(s.mat, mu) + albedoDiff(s.mat, mu));
    }
  } else {
    let bs = sampleBSDF(M, n, wo, ub);
    if (bs.w > 0.0 && dot(s.ng, bs.xyz) > 0.0) {
      let l = bs.xyz;
      let thr = evalBSDF(M, n, l, wo) * (dot(n, l) / bs.w);
      L = L + thr * secondary(offsetRay(s.p, s.ng, l), l, time, bs.w, opts, f32(nl));
    }
  }
  return L;
}

@compute @workgroup_size(8, 8)
fn rasterShade(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  gRoot = bitcast<u32>(cam.lensC.y);
  let W = cam.size.x; let H = cam.size.y;
  let px = gid.xy + cam.tile.xy;                        // this dispatch covers one band (docs/WEB_TWIN.md §12)
  if (gid.x >= cam.tile.z || gid.y >= cam.tile.w || px.x >= W || px.y >= H) { return; }
  // SCOPE: a circular image (diameter = cam.aid.x x width), black outside
  if (cam.pos.w > 0.5) {
    let dx = f32(px.x) - cam.lensA.z; let dy = f32(px.y) - cam.lensA.w;
    if (dx * dx + dy * dy > cam.aid.x * cam.aid.x) { textureStore(outTex, vec2i(px), vec4f(0.0, 0.0, 0.0, 0.0)); return; }
  }
  let S = pixSeeds(px.x, px.y);
  var ids: array<u32, 4>;
  for (var s = 0u; s < 4u; s++) { let q = textureLoad(visTex, vec2i(px), i32(s)); ids[s] = q.x | (q.y << 16u); }
  let nl = max(1u, bitcast<u32>(cam.lensC.z));
  var L = vec3f(0.0);
  var done = 0u;
  var anyHit = false;
  for (var s = 0u; s < 4u; s++) {
    if (((done >> s) & 1u) != 0u) { continue; }
    let id = ids[s];
    var cnt = 0u; var ps = vec2f(0.0);
    if ((id >> 27u) != 0u) { done = done | (1u << s); cnt = 1u; ps = msPos(s); }   // posed bodies: one shade per sample
    else { for (var t = s; t < 4u; t++) { if (ids[t] == id) { done = done | (1u << t); cnt++; ps = ps + msPos(t); } } ps = ps / f32(cnt); }
    if (id == 0u) { continue; }
    anyHit = true;
    gKey = 0xffffu;
    let uv = vec2f(f32(px.x) + ps.x - 0.5, f32(px.y) + ps.y - 0.5);
    let d = primaryDir(uv);
    let h = hitFromId(id, cam.pos.xyz, d, 0.5);
    let ul = vec3f(kr(s * nl, KL0, S.s3), kr(s * nl, KL1, S.s4), kr(s * nl, KL2, S.s5));
    let ub = vec3f(kr(s, KB0, S.s6), kr(s, KB1, S.s7), kr(s, KB2, S.s8));
    if (LITE_CULL) { L = L + liteHit(h, cam.pos.xyz, d, 0.5, ul, ub, cam.flags.y, s * nl, S) * f32(cnt); }
    else { L = L + shadeHit(h, cam.pos.xyz, d, 0.5, ul, ub, cam.flags.y, s * nl, S) * f32(cnt); }
  }
  textureStore(outTex, vec2i(px), vec4f(L * (0.25 * cam.expo.z), select(0.0, 1.0, anyHit)));
}
`;

// E_free (node slot 0 xyz) = mean + sum_k c_k B_k, then the node exitance as jacobiUpdate writes it
const PRT_WGSL = (c, K) => /* wgsl */ `${defines(c)}
const INV_PI: f32 = 0.318309886183791;
@group(0) @binding(0) var<storage, read> params: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> nodeState: array<vec4f>;
@group(0) @binding(2) var<storage, read> nodeIdx: array<u32>;        // active node -> static vertex
@group(0) @binding(3) var<storage, read> nodeMat: array<u32>;        // active node -> material
@group(0) @binding(4) var<storage, read> basis: array<u32>;          // (K + 1) x n luminance as f16 pairs; row 0 = mean
@group(0) @binding(5) var<uniform> pu: PrtU;
@group(0) @binding(6) var<storage, read> chroma: array<u32>;         // n x rgb / Y as f16 pairs
struct PrtU { n: u32, slot: u32, k: u32, scale: f32, coef: array<vec4f, ${Math.ceil((K + 1) / 4)}> }
fn bval(row: u32, j: u32) -> f32 {
  let idx = row * pu.n + j;
  let h = unpack2x16float(basis[idx >> 1u]);
  return select(h.x, h.y, (idx & 1u) == 1u);
}
fn cval(j: u32, ch: u32) -> f32 {
  let idx = 3u * j + ch;
  let h = unpack2x16float(chroma[idx >> 1u]);
  return select(h.x, h.y, (idx & 1u) == 1u);
}
fn coef(k: u32) -> f32 { return pu.coef[k >> 2u][k & 3u]; }
@compute @workgroup_size(64)
fn prtApply(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= pu.n) { return; }
  let i = nodeIdx[j];
  var Y = bval(0u, j);
  for (var k = 0u; k < min(pu.k, 64u); k++) { Y = Y + coef(k) * bval(k + 1u, j); }
  let E = max(Y, 0.0) * pu.scale * vec3f(cval(j, 0u), cval(j, 1u), cval(j, 2u));
  let st = nodeState[4u * i];
  let m = nodeMat[j];
  let hh = max(params[MAT_OFF + 5u * m + 2u].xyz, vec3f(1e-4));
  let Ls = (nodeState[4u * i + 3u].xyz + E * st.w * hh) * INV_PI / hh;
  nodeState[4u * i] = vec4f(E, st.w);
  nodeState[4u * i + pu.slot] = vec4f(Ls, 1.0);
}
`;

const m3 = (T) => [[T[0], T[4], T[8]], [T[1], T[5], T[9]], [T[2], T[6], T[10]]];
const mv = (R, v) => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
const nrm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dt = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * The PRT regression input of a scope pose (one implementation, used by the offline generator and at runtime): the
 * emitter centroid relative to the C pivot / 0.03 m (3) and the emission axis (3), both in D. Smooth in every joint,
 * no angle wrap; the 30-degree DOV makes roll move the axis, so roll is captured.
 */
export function scopeFeatures(scene, S, T) {
  const R = m3(T), pv = scene.scope.pivot;
  const c = mv(R, S.emitter.centroid), a = nrm(mv(R, S.emitter.axis));
  return [(c[0] + T[12] - pv[0]) / 0.03, (c[1] + T[13] - pv[1]) / 0.03, (c[2] + T[14] - pv[2]) / 0.03, a[0], a[1], a[2]];
}

export class LitePRTRenderer extends TwinRenderer {
  static async create(opts) {
    const r = new LitePRTRenderer();
    await r._init(opts);
    return r;
  }

  async _init(opts) {
    const t0 = performance.now();
    this.prtMode = opts.prt !== false;
    if (this.prtMode) {
      const base = typeof opts.prt === 'string' ? opts.prt : new URL('./assets/', import.meta.url).href;
      try {
        const meta = await (await fetch(base + 'prt.json')).json();
        const bin = await (await fetch(base + meta.basis_file)).arrayBuffer();
        this.prt = { meta, bin };
      } catch (e) { (opts.log || console.log)('[lite-prt] no PRT asset, falling back to the node solver', e.message); this.prtMode = false; }
    }
    await super._init({ preset: 'judge', ...opts });
    this.isLite = true;
    // SCOPE: pinhole, HFOV from physics (assumed 75 deg), circular image 0.95 of the width, image up toward the tilt
    const sc = this.physics?.scope?.camera || {};
    const unwrap = (x, d) => (x && typeof x === 'object' && 'value' in x ? x.value : (x ?? d));
    const hfov = unwrap(sc.hfov_deg, 75) * Math.PI / 180, circ = unwrap(sc.image_circle_frac, 0.95);
    const w = opts.scopeSize?.[0] ?? 1280, h = opts.scopeSize?.[1] ?? 720;
    const f = (w / 2) / Math.tan(hfov / 2);
    const lens = { fx: f, fy: f, cx: (w - 1) / 2, cy: (h - 1) / 2, k: [0, 0, 0, 0], alpha: 0, size: [w, h] };
    this.cameras.SCOPE = { index: Object.keys(this.cameras).length, meta: { name: 'SCOPE' }, lens, size: [w, h], T: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], pinhole: true, circle: circ * w / 2 };
    this.cameras.SCOPE.ae = new AutoExposure(this.sensorP, { exposureUnits: this.sensorP.exposureUnits0 });
    this.cameraNames = Object.keys(this.cameras);
    await this._buildRaster();
    if (this.prtMode) this._buildPRT();
    this.stats.liteInitMs = performance.now() - t0;
    this.setPoses(this.poses0 || { bodies: {} });
  }

  // ------------------------------------------------------------------ PRT: no gather table at all
  async _buildGatherTable() {
    if (!this.prtMode) return super._buildGatherTable();
    const dev = this.device, b = this.bufs, meta = this.prt.meta;
    if (meta.static_verts !== this.S.staticVerts) throw new Error(`PRT asset is for ${meta.static_verts} nodes, the scene has ${this.S.staticVerts}`);
    const bin = this.prt.bin;
    this.activeList = new Uint32Array(bin, meta.active_offset, meta.n_active);
    this.activeCount = meta.n_active;
    b.active = this._buf(this.activeList, GPUBufferUsage.STORAGE, 'active');
    b.nbr = dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE, label: 'nbr' });
    this.groups.sceneRW = this._sceneRWGroup();
    this.groups.rad = dev.createBindGroup({ layout: this.layouts.radG1, entries: [{ binding: 0, resource: { buffer: b.scn } }, { binding: 1, resource: { buffer: b.radU } }] });
    this.stats.activeNodes = this.activeCount;
    this.stats.gatherBuildMs = 0;
  }

  _buildPRT() {
    const dev = this.device, meta = this.prt.meta, bin = this.prt.bin, n = meta.n_active, K = meta.k;
    const nodeMat = new Uint32Array(n);
    for (let j = 0; j < n; j++) nodeMat[j] = this.S.meshTable[this.S.nodeMesh[this.activeList[j]]].mat;
    this.prtK = K;
    this.prtCenters = new Float32Array(bin, meta.centers_offset, meta.n_centers * meta.n_feat);
    this.prtWeights = new Float32Array(bin, meta.weights_offset, meta.n_centers * K);
    this.prtLin = new Float32Array(bin, meta.lin_offset, (meta.n_feat + 1) * K);
    const basisU32 = new Uint32Array(bin, meta.basis_offset, Math.ceil(((K + 1) * n) / 2));
    const chromaU32 = new Uint32Array(bin, meta.chroma_offset, Math.ceil((3 * n) / 2));
    const mod = dev.createShaderModule({ code: PRT_WGSL(this.consts, K), label: 'prt' });
    this.pipes.prtApply = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'prtApply' } });
    this.prtU = dev.createBuffer({ size: 16 + 16 * Math.ceil((K + 1) / 4), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const ST = GPUBufferUsage.STORAGE;
    this.prtBufs = { idx: this._buf(this.activeList, ST, 'prtIdx'), mat: this._buf(nodeMat, ST, 'prtMat'), basis: this._buf(basisU32, ST, 'prtBasis'), chroma: this._buf(chromaU32, ST, 'prtChroma') };
    this.prtGroup = dev.createBindGroup({ layout: this.pipes.prtApply.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.bufs.params } }, { binding: 1, resource: { buffer: this.bufs.nodeState } },
      { binding: 2, resource: { buffer: this.prtBufs.idx } }, { binding: 3, resource: { buffer: this.prtBufs.mat } },
      { binding: 4, resource: { buffer: this.prtBufs.basis } }, { binding: 5, resource: { buffer: this.prtU } },
      { binding: 6, resource: { buffer: this.prtBufs.chroma } }] });
    this.stats.prt = { k: K, centers: meta.n_centers, basisMB: (basisU32.byteLength + chromaU32.byteLength) / 1e6, heldout: meta.heldout || null };
  }

  _prtFeatures() {
    const T = this.poses0?.bodies?.scope;
    return T ? scopeFeatures(this.scene, this.S, T) : null;
  }

  _prtCoefs() {
    const x = this._prtFeatures(), meta = this.prt.meta, K = this.prtK, F = meta.n_feat, C = this.prtCenters, Wt = this.prtWeights, Ln = this.prtLin;
    const c = new Float32Array(K);
    if (!x) return c;
    for (let k = 0; k < K; k++) { let a = Ln[F * K + k]; for (let f = 0; f < F; f++) a += Ln[f * K + k] * x[f]; c[k] = a; }
    const eps = meta.rbf_eps;
    for (let m = 0; m < meta.n_centers; m++) {
      let d2 = 0;
      for (let f = 0; f < F; f++) { const d = x[f] - C[m * F + f]; d2 += d * d; }
      const r = Math.sqrt(d2);
      const phi = meta.rbf_kernel === 'gaussian' ? Math.exp(-((eps * r) ** 2)) : meta.rbf_kernel === 'cubic' ? r * r * r : r * r * Math.log(r + 1e-12);
      for (let k = 0; k < K; k++) c[k] += phi * Wt[m * K + k];
    }
    return c;
  }

  _prtApply() {
    const K = this.prtK, c = this._prtCoefs();
    const u = new ArrayBuffer(16 + 16 * Math.ceil((K + 1) / 4));
    new Uint32Array(u, 0, 3).set([this.activeCount, this.slot || 1, K]);
    new Float32Array(u, 12, 1)[0] = this.fluxScale ?? 1;
    new Float32Array(u, 16).set(c);
    this.device.queue.writeBuffer(this.prtU, 0, u);
    this._submit('prt.apply', null, 0, (pass) => {
      pass.setPipeline(this.pipes.prtApply); pass.setBindGroup(0, this.prtGroup);
      pass.dispatchWorkgroups(Math.ceil(this.activeCount / 64));
    });
    this.lastCoefs = c;
  }

  _ensureRadiosity(force = false) {
    if (!this.prtMode) return super._ensureRadiosity(force);
    if (!this.activeCount || (!force && this.solvedVersion === this.posesVersion)) return;
    const first = this.solvedVersion < 0;
    const mv = first || force ? { all: true, mask: 0 } : this._movedSinceSolve();
    if (mv.all || mv.mask) {
      this._submitNodeDirect(mv.all ? 0 : mv.mask);
      this._prtApply();
      this._solvedInst = this.instances.map((I) => (I ? { q0: [...I.q0], q1: [...I.q1], p0: [...I.p0], p1: [...I.p1] } : null));
      this._solvedCaps = (this._curCaps || []).map((cc) => cc && [...cc]);
      this.stats.lastRadiosity = mv.all ? 'prt-full' : 'prt-incremental';
    } else this.stats.lastRadiosity = 'reused';
    this.solvedVersion = this.posesVersion;
    this._writeSceneU(this._cut);
    if (mv.all || first) this._refreshEAvg();
  }

  /** Area-weighted first-bounce exitance (luminance) of the active nodes: the light the dome gets from the scope. */
  async _directFlux() {
    const dev = this.device, S = this.S;
    const staging = dev.createBuffer({ size: S.staticVerts * 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.bufs.nodeState, 0, staging, 0, S.staticVerts * 64);
    dev.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const st = new Float32Array(staging.getMappedRange());
    let F = 0;
    for (const i of this.activeList) F += S.nodeArea[i] * (0.2126 * st[16 * i + 12] + 0.7152 * st[16 * i + 13] + 0.0722 * st[16 * i + 14]);
    staging.unmap(); staging.destroy();
    return F;
  }

  /**
   * The PRT was trained with the tools at their default pose; tools elsewhere (deeper, in the beam) absorb a different
   * share of the scope's light, and the dome, an integrating sphere, scales all its indirect light with it. Scale
   * E_free by the first-bounce flux now / with the default tools at this scope pose (two node direct passes).
   */
  async _updateFluxScale() {
    const cur0 = this.poses0, cur1 = this.poses1;
    this._defTools = this._defTools || posesFromState(this.scene, defaultState(this.scene));
    const def = this._defTools;
    const bodies = { ...cur0.bodies };
    for (const [k, v] of Object.entries(def.bodies)) if (k.startsWith('L/') || k.startsWith('R/')) bodies[k] = v;
    TwinRenderer.prototype.setPoses.call(this, { bodies });
    this._submitNodeDirect(0);
    const Fref = await this._directFlux();
    TwinRenderer.prototype.setPoses.call(this, cur0, cur1);
    this._submitNodeDirect(0);
    const Fnow = await this._directFlux();
    this.fluxScale = Fref > 0 ? Fnow / Fref : 1;
    this.stats.fluxScale = this.fluxScale;
  }

  async converge(iters = 200) {
    if (!this.prtMode) return super.converge(iters);
    if (this.opts.fluxScale !== false && this.poses0?.bodies?.scope) await this._updateFluxScale();
    this._ensureRadiosity(true);
    await this.device.queue.onSubmittedWorkDone();
    await this.nodeSummary();
  }

  // ------------------------------------------------------------------ poses: the SCOPE camera follows the scope body
  setPoses(poses0, poses1 = null) {
    super.setPoses(poses0, poses1);
    const T = poses0?.bodies?.scope, cam = this.cameras?.SCOPE;
    if (!T || !cam) return;
    const R = m3(T), sc = this.scene.scope;
    const c = mv(R, sc.objective_centre_K).map((x, k) => x + T[12 + k]);
    const z = nrm(mv(R, sc.view_dir_K)), zK = [T[8], T[9], T[10]];
    let tilt = [z[0] - dt(z, zK) * zK[0], z[1] - dt(z, zK) * zK[1], z[2] - dt(z, zK) * zK[2]];
    if (Math.hypot(...tilt) < 1e-6) tilt = [T[4], T[5], T[6]];
    const up = nrm(tilt.map((x, k) => x - dt(tilt, z) * z[k]));
    const yax = up.map((x) => -x), xax = crs(yax, z);
    cam.T = [[xax[0], yax[0], z[0], c[0]], [xax[1], yax[1], z[1], c[1]], [xax[2], yax[2], z[2], c[2]], [0, 0, 0, 1]];
  }

  // ------------------------------------------------------------------ raster + shade
  async _buildRaster() {
    const dev = this.device, S = this.S;
    const triInst = new Uint32Array(S.triCount).fill(0x7fffffff);
    S.blas.forEach((bl, i) => triInst.fill(i, bl.triOff, bl.triOff + bl.triCount));
    // longest-edge bisection (barycentric) of every triangle with an edge over LMAX: raster-only sub-triangles that
    // keep their BVH slot, so shading still intersects the exact CAD triangle
    const LMAX = 0.003, T = S.tris, subl = [];
    const ln = (k, du, dv) => { const o = 16 * k; const x = du * T[o + 4] + dv * T[o + 8], y = du * T[o + 5] + dv * T[o + 9], z = du * T[o + 6] + dv * T[o + 10]; return Math.hypot(x, y, z); };
    const f2u = new Float32Array(1), u2 = new Uint32Array(f2u.buffer);
    const bits = (x) => { f2u[0] = x; return u2[0]; };
    for (let k = 0; k < S.triCount; k++) {
      if (ln(k, 1, 0) <= 1.05 * LMAX && ln(k, 0, 1) <= 1.05 * LMAX && ln(k, 1, -1) <= 1.05 * LMAX) continue;
      triInst[k] |= 0x80000000;
      const st = [[0, 0, 1, 0, 0, 1]];
      let guard = 0;
      while (st.length && guard++ < 200000) {
        const t = st.pop();
        const la = ln(k, t[2] - t[0], t[3] - t[1]), lb = ln(k, t[4] - t[2], t[5] - t[3]), lc = ln(k, t[0] - t[4], t[1] - t[5]);
        const m = Math.max(la, lb, lc);
        if (m <= LMAX) { for (let c = 0; c < 3; c++) subl.push(k, bits(t[2 * c]), bits(t[2 * c + 1])); continue; }
        if (m === la) { const mu = (t[0] + t[2]) / 2, mv = (t[1] + t[3]) / 2; st.push([t[0], t[1], mu, mv, t[4], t[5]], [mu, mv, t[2], t[3], t[4], t[5]]); }
        else if (m === lb) { const mu = (t[2] + t[4]) / 2, mv = (t[3] + t[5]) / 2; st.push([t[0], t[1], t[2], t[3], mu, mv], [t[0], t[1], mu, mv, t[4], t[5]]); }
        else { const mu = (t[4] + t[0]) / 2, mv = (t[5] + t[1]) / 2; st.push([t[0], t[1], t[2], t[3], mu, mv], [mu, mv, t[2], t[3], t[4], t[5]]); }
      }
    }
    this.subVerts = subl.length / 3;
    this.stats.rasterSubTris = this.subVerts / 3;
    this.rbufs = { triInst: this._buf(triInst, GPUBufferUsage.STORAGE, 'triInst'), sub: this._buf(Uint32Array.from(subl.length ? subl : [0, 0, 0]), GPUBufferUsage.STORAGE, 'rasterSub') };
    const rmod = await this._module(RASTER(this.consts), 'lite raster');
    const V = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const g0 = dev.createBindGroupLayout({ entries: [0, 1, 2].map((binding) => ({ binding, visibility: V, buffer: { type: 'read-only-storage' } })) });
    const g1 = dev.createBindGroupLayout({ entries: [0, 1].map((binding) => ({ binding, visibility: V, buffer: { type: 'uniform' } })) });
    this.rlayout = { g0, g1 };
    const rdesc = (entryPoint) => ({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [g0, g1] }),
      vertex: { module: rmod, entryPoint },
      fragment: { module: rmod, entryPoint: 'fs', targets: [{ format: 'rg16uint' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: 4 },
    });
    this.pipes.raster = await dev.createRenderPipelineAsync(rdesc('vs'));
    this.pipes.raster2 = await dev.createRenderPipelineAsync(rdesc('vs2'));
    this.rgroup0 = dev.createBindGroup({ layout: g0, entries: [{ binding: 0, resource: { buffer: this.bufs.tris } }, { binding: 1, resource: { buffer: this.rbufs.triInst } }, { binding: 2, resource: { buffer: this.rbufs.sub } }] });
    const smod = await this._module(shaderSource('camera', this.consts) + SHADE_EXTRA, 'lite shade');
    const g2 = dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d', multisampled: true } }] });
    this.rlayout.g2 = g2;
    this.pipes.shade = await dev.createComputePipelineAsync({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.layouts.sceneRO, this.layouts.camG1, g2] }),
      compute: { module: smod, entryPoint: 'rasterShade', constants: { CULL_SHADOWS: this.opts.cullShadows ? 1 : 0, LITE_CULL: this.opts.liteCull === false ? 0 : 1 } },
    });
  }

  // the tracer's static first-hit cache is not used: a tiny 1-sample cache keeps its bind group cheap
  _camGroup(cam, spp) { return super._camGroup(cam, 1); }

  /** bits (as f32) of the bodies liteVis ignores: the scope's own cannula */
  _liteSkipF() {
    if (this._liteSkip === undefined) {
      const pc = (this.scene.ports || []).find((p) => p.id === (this.scene.scope?.port ?? 'C'));
      const i = this.S.blas.findIndex((b) => b.body === (pc?.cannula_body || 'C/cannula'));
      const u = new Uint32Array([i < 0 ? 0 : (1 << i) >>> 0]);
      this._liteSkip = new Float32Array(u.buffer)[0];
    }
    return this._liteSkip;
  }

  _scopeBit() { const i = this.S.blas.findIndex((b) => b.body === 'scope'); return i < 0 ? 0 : (1 << i) >>> 0; }

  _rasterTargets(cam) {
    if (cam.r) return cam.r;
    const dev = this.device, [w, h] = cam.size;
    const vis = dev.createTexture({ size: [w, h], format: 'rg16uint', sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, label: `${cam.meta.name}.vis` });
    const depth = dev.createTexture({ size: [w, h], format: 'depth32float', sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: `${cam.meta.name}.depth` });
    const ru = dev.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    cam.r = {
      vis, depth, ru,
      g1: dev.createBindGroup({ layout: this.rlayout.g1, entries: [{ binding: 0, resource: { buffer: this.bufs.scn } }, { binding: 1, resource: { buffer: ru } }] }),
      g2: dev.createBindGroup({ layout: this.rlayout.g2, entries: [{ binding: 0, resource: vis.createView() }] }),
    };
    return cam.r;
  }

  _submitCamera(name, o = {}) {
    const cam = this._camTargets(name), r = this._rasterTargets(cam);
    const [w, h] = cam.size, T = cam.T;
    const R = [[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]];
    const base = {
      R, pos: [T[0][3], T[1][3], T[2][3]], mode: cam.pinhole ? 1 : 0, lens: cam.lens, w, h, spp: 1, seed: o.seed ?? 0,
      camIndex: cam.index, options: this._options(o), t0: 0.5, t1: 0.5, skip: this._skipMask(name) | (cam.pinhole ? this._scopeBit() : 0),
      nLight: o.lightSamples ?? this.opts.liteLightSamples ?? 4, refine: 0, root: this.interiorRoot || 0,
      aid: [cam.circle || 0, this._liteSkipF(), 0, 0],
    };
    this._packCam(cam.ubuf, base);
    this._packCam(r.ru, base);
    const dev = this.device;
    {
      const enc = dev.createCommandEncoder();
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: r.vis.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
        depthStencilAttachment: { view: r.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
        timestampWrites: this.timer.pass(`${name}.raster`, 'raster', w * h, this.frame),
      });
      rp.setPipeline(this.pipes.raster); rp.setBindGroup(0, this.rgroup0); rp.setBindGroup(1, r.g1);
      rp.draw(3 * this.S.triCount);
      if (this.subVerts) { rp.setPipeline(this.pipes.raster2); rp.draw(this.subVerts); }
      rp.end();
      const after = this.timer.flush(enc);
      dev.queue.submit([enc.finish()]); after();
    }
    // shade in row bands sized to ~TARGET_MS (docs/WEB_TWIN.md §12)
    const bandRows = Math.max(8, Math.min(h, Math.floor(this.timer.units('liteShade') / w / 8) * 8));
    for (let y0 = 0; y0 < h; y0 += bandRows) {
      const rows = Math.min(h - y0, bandRows);
      this._packCam(cam.ubuf, { ...base, tile: [0, y0, w, rows] });
      this._submit(`${name}.shade#${y0}`, 'liteShade', w * rows, (pass) => {
        pass.setPipeline(this.pipes.shade);
        pass.setBindGroup(0, this.groups.sceneRO); pass.setBindGroup(1, cam.group); pass.setBindGroup(2, r.g2);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(rows / 8));
      });
    }
    if (o.exposure !== undefined) this._manualExposure(name, o.exposure);
    return cam.tex;
  }

  /** present(name | texture, canvas | GPUCanvasContext): the sensor image on a WebGPU canvas, no CPU readback. */
  present(name, target) {
    const cam = typeof name === 'string' ? this.cameras[name] : null;
    const tex = cam ? (cam.s?.out || this.sensor(name)) : name;
    if (target && typeof target.getCurrentTexture === 'function') {
      const ctx = target, canvas = ctx.canvas;
      if (!canvas._twinCtx) { canvas._twinCtx = ctx; ctx.configure({ device: this.device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' }); }
      return super.present(tex, canvas);
    }
    return super.present(tex, target);
  }

  visible(name, pointsD, o = {}) {
    if (name === 'SCOPE') return pointsD.map(() => true);
    return super.visible(name, pointsD, o);
  }
}

export default LitePRTRenderer;
export { MS_POS };
