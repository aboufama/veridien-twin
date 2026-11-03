// WGSL of the godview_pt candidate (docs/WEB_TWIN.md §19): a progressive path tracer of the god's-eye view and a raster
// pass for while things move, both on the twin's own scene packing (gpu/scene_gpu.js) and its validated WGSL
// (gpu/wgsl: common, BVH traversal, the §8 BSDF word for word). Nothing in gpu/** is edited: TRACE is reused with one
// textual change (the cutaway drops only the materials godview.yaml lists, not every static mesh).

import { COMMON, defines } from '../gpu/wgsl/common.wgsl.js';
import { TRACE } from '../gpu/wgsl/trace.wgsl.js';
import { BSDF } from '../gpu/wgsl/bsdf.wgsl.js';

const CUT_OLD = 'if (dot(scn.cut.xyz, pw) > scn.cut.w) { continue; }';
const CUT_NEW = 'if (dot(scn.cut.xyz, pw) > scn.cut.w && cutMesh(k)) { continue; }';

export function traceSource() {
  if (!TRACE.includes(CUT_OLD)) throw new Error('godview_pt: gpu/wgsl/trace.wgsl.js changed; update the cutaway patch');
  return TRACE.replace(CUT_OLD, CUT_NEW);
}

// emitter functions copied from gpu/wgsl/shade.wgsl.js (profile, sampleEmitter, emitted) - same formula
const EMIT = /* wgsl */ `
fn profile(th: f32) -> f32 {
  let last = params[PROF_OFF + PROF_N - 1u];
  if (th >= last.x) { return select(0.0, last.y, th == last.x); }
  var lo = 0u;
  var hi = PROF_N - 1u;
  for (var it = 0u; it < 40u; it++) {
    if (hi - lo <= 1u) { break; }
    let m = (lo + hi) / 2u;
    if (params[PROF_OFF + m].x <= th) { lo = m; } else { hi = m; }
  }
  let a = params[PROF_OFF + lo];
  let b = params[PROF_OFF + hi];
  let f = clamp((th - a.x) / max(b.x - a.x, 1e-12), 0.0, 1.0);
  return mix(a.y, b.y, f);
}
struct LightSample { p: vec3f, n: vec3f, axis: vec3f }
fn sampleEmitter(u: vec3f, time: f32) -> LightSample {
  var lo = 0u;
  var hi = EMIT_N - 1u;
  for (var it = 0u; it < 40u; it++) {
    if (lo >= hi) { break; }
    let m = (lo + hi) / 2u;
    if (params[EMIT_OFF + 4u * m].w < u.x) { lo = m + 1u; } else { hi = m; }
  }
  let v0 = params[EMIT_OFF + 4u * lo].xyz;
  let e1 = params[EMIT_OFF + 4u * lo + 1u].xyz;
  let e2 = params[EMIT_OFF + 4u * lo + 2u].xyz;
  let nl = params[EMIT_OFF + 4u * lo + 3u].xyz;
  let su = sqrt(u.y);
  let pl = v0 + e1 * (su * (1.0 - u.z)) + e2 * (su * u.z);
  let P = instPose(scn.info.y, time);
  return LightSample(P.r * pl + P.p, P.r * nl, P.r * scn.axis.xyz);
}
fn emitted(n: vec3f, axis: vec3f, w: vec3f) -> vec3f {
  if (dot(w, n) <= 0.0) { return vec3f(0.0); }
  return scn.L0.xyz * profile(acos(clamp(dot(w, axis), -1.0, 1.0)));
}
`;

const GU = /* wgsl */ `
struct GU {
  rx: vec4f, ry: vec4f, rz: vec4f, pos: vec4f,
  lens: vec4f,        // fx, fy, cx, cy (pixel centres at integer coordinates)
  size: vec4u,        // w, h, sample index of this dispatch, max bounces
  tile: vec4u,        // x0, y0, width, height
  keyDir: vec4f,      // unit direction toward the key light, w: 1 = on
  keyE: vec4f,        // key irradiance (rgb)
  env: vec4f,         // constant environment radiance (rgb), w: 1 = on
  opt: vec4u,         // x: cut material mask (bit per material, first 32), y: seed, z: samples per dispatch, w: scope light on
  disp: vec4f,        // exposure, crossfade (0 raster .. 1 traced), denoise strength, display mode
}
`;

export function ptSource(c) {
  return defines(c) + COMMON + GU + /* wgsl */ `
@group(0) @binding(0) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(1) var<storage, read> tris: array<vec4f>;
@group(0) @binding(2) var<storage, read> verts: array<vec4f>;
@group(0) @binding(3) var<storage, read> params: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dbg: array<atomic<u32>, 16>;
@group(1) @binding(0) var<uniform> scn: SceneU;
@group(1) @binding(1) var<uniform> gu: GU;
@group(1) @binding(2) var<storage, read_write> accum: array<vec4f>;   // radiance sum (rgb), sample count
@group(1) @binding(3) var<storage, read_write> gbuf: array<vec4f>;    // first-hit shading normal (xyz), distance (w; 0 = miss)

fn cutMesh(k: u32) -> bool {
  let mesh = bitcast<u32>(tris[4u * k + 3u].x);
  let mat = bitcast<u32>(params[MESH_OFF + 2u * mesh].w);
  return mat < 32u && ((gu.opt.x >> mat) & 1u) != 0u;
}
` + traceSource() + BSDF + EMIT + /* wgsl */ `
fn rnd(st: ptr<function, u32>) -> f32 { *st = pcg(*st); return u01(*st); }
fn emitterNormal(h: Hit, s: Surf) -> vec3f {
  let n = normalize(s.b.x * verts[2u * s.vi.x + 1u].xyz + s.b.y * verts[2u * s.vi.y + 1u].xyz + s.b.z * verts[2u * s.vi.z + 1u].xyz);
  return instPose(h.inst, 0.0).r * n;
}
fn max3(v: vec3f) -> f32 { return max(v.x, max(v.y, v.z)); }

const MAX_BOUNCE_CAP: u32 = 24u;

fn pathSample(px: vec2f, st: ptr<function, u32>, writeG: bool, pix: u32) -> vec3f {
  let dc = vec3f((px.x - gu.lens.z) / gu.lens.x, (px.y - gu.lens.w) / gu.lens.y, 1.0);
  return pathL(gu.pos.xyz, normalize(gu.rx.xyz * dc.x + gu.ry.xyz * dc.y + gu.rz.xyz * dc.z), st, 0u, writeG, pix);
}

// radiance along (o0, d0). mode 0: camera ray; mode 1: a vertex's gather ray for the raster's baked light, whose
// first segment ignores the outlet (the raster adds the outlet's and the key's direct light live, shadow-mapped)
fn pathL(o0: vec3f, d0: vec3f, st: ptr<function, u32>, mode: u32, writeG: bool, pix: u32) -> vec3f {
  var o = o0;
  var d = d0;
  var T = vec3f(1.0);
  var L = vec3f(0.0);
  var pdfPrev = 0.0;
  var fromCamera = mode == 0u;
  let maxB = min(gu.size.w, MAX_BOUNCE_CAP);
  for (var b = 0u; b <= maxB; b++) {
    let h = traceScene(o, d, 0.0, BIG, 0.0, false, 3u);
    if (h.tri == MISS) {
      if (gu.env.w > 0.5) { L += T * gu.env.xyz; }
      if (writeG && b == 0u) { gbuf[pix] = vec4f(0.0); }
      break;
    }
    let s = surface(h, o, d, 0.0);
    if (writeG && b == 0u) { gbuf[pix] = vec4f(s.ns, h.t); }
    if ((s.flags & F_EMITTER) != 0u) {
      if (gu.opt.w != 0u && !(mode == 1u && b == 0u)) {
        let ne = emitterNormal(h, s);
        let axis = instPose(scn.info.y, 0.0).r * scn.axis.xyz;
        let Le = emitted(ne, axis, -d);
        var w = 1.0;
        if (!fromCamera) {
          let cosE = max(dot(-d, ne), 1e-6);
          let pl = h.t * h.t / (cosE * scn.L0.w);
          w = pdfPrev * pdfPrev / (pdfPrev * pdfPrev + pl * pl);
        }
        L += T * Le * w;
      }
      break;
    }
    let M = loadMat(s.mat);
    let n = s.ns;
    let wo = -d;
    // key light (directional, delta): one shadow ray
    if (gu.keyDir.w > 0.5) {
      let l = gu.keyDir.xyz;
      let cl = dot(n, l);
      if (cl > 0.0 && dot(s.ng, l) > 0.0) {
        let f = evalBSDF(M, n, l, wo);
        if (max3(f) > 0.0) {
          let sh = traceScene(offsetRay(s.p, s.ng, l), l, 0.0, BIG, 0.0, true, 3u);
          if (sh.tri == MISS) { L += T * f * gu.keyE.xyz * cl; }
        }
      }
    }
    // scope outlet: area sample, MIS (power) against BSDF sampling
    if (gu.opt.w != 0u) {
      let ls = sampleEmitter(vec3f(rnd(st), rnd(st), rnd(st)), 0.0);
      var wi = ls.p - s.p;
      let d2 = dot(wi, wi);
      let dist = sqrt(d2);
      wi = wi / dist;
      let cosE = dot(-wi, ls.n);
      let cosS = dot(n, wi);
      if (cosE > 0.0 && cosS > 0.0 && dot(s.ng, wi) > 0.0) {
        let Le = emitted(ls.n, ls.axis, -wi);
        if (max3(Le) > 0.0) {
          let f = evalBSDF(M, n, wi, wo);
          if (max3(f) > 0.0) {
            let sh = traceSceneSkip(offsetRay(s.p, s.ng, wi), wi, 0.0, dist - 2e-5, 0.0, true, 3u, bitcast<u32>(scn.axis.w));
            if (sh.tri == MISS) {
              let pdfW = d2 / (cosE * scn.L0.w);
              let pb = pdfBSDF(M, n, wo, wi);
              let w = pdfW * pdfW / (pdfW * pdfW + pb * pb);
              L += T * f * Le * (cosS * w / pdfW);
            }
          }
        }
      }
    }
    if (b == maxB) { break; }
    let bs = sampleBSDF(M, n, wo, vec3f(rnd(st), rnd(st), rnd(st)));
    if (bs.w <= 0.0) { break; }
    let l = bs.xyz;
    if (dot(s.ng, l) <= 0.0) { break; }
    let f = evalBSDF(M, n, l, wo);
    T = T * f * (dot(n, l) / bs.w);
    if (max3(T) <= 0.0) { break; }
    pdfPrev = bs.w;
    fromCamera = false;
    o = offsetRay(s.p, s.ng, l);
    d = l;
    if (b >= 2u) {
      let q = clamp(max3(T), 0.05, 0.95);
      if (rnd(st) > q) { break; }
      T = T / q;
    }
  }
  return L;
}

@group(1) @binding(4) var<storage, read_write> irr: array<vec4f>;      // per bake entry: sum of gathered radiance, count
@group(1) @binding(5) var<storage, read> backList: array<u32>;         // godview_pt2: entry N_STATIC_VERTS + j = back side of vertex backList[j]

@compute @workgroup_size(64)
fn bakeMain(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  let i = gu.tile.x + gid.x;
  if (gid.x >= gu.tile.z || i >= gu.tile.y) { return; }
  // entries < N_STATIC_VERTS: the front (outward normal) of vertex i; above: the back side of a cut-material vertex
  // (the hollow PLA skin's inside, seen through the cut; the path tracer is two-sided, so it is lit from -n)
  var vi = i;
  var sg = 1.0;
  if (i >= N_STATIC_VERTS) { vi = backList[i - N_STATIC_VERTS]; sg = -1.0; }
  let p = verts[2u * vi].xyz;
  let n = sg * normalize(verts[2u * vi + 1u].xyz);
  let t = basisT(n);
  let bb = basisB(n);
  var sum = vec3f(0.0);
  for (var k = 0u; k < gu.opt.z; k++) {
    var st = hash3(i, gu.size.z + k, 0x9e3779b9u ^ gu.opt.y);
    let u1 = rnd(&st);
    let ph = 2.0 * PI * rnd(&st);
    let r = sqrt(u1);
    let l = normalize(t * (r * cos(ph)) + bb * (r * sin(ph)) + n * sqrt(max(0.0, 1.0 - u1)));
    var c = pathL(p + n * 2e-5, l, &st, 1u, false, 0u);
    if (!(c.x == c.x && c.y == c.y && c.z == c.z) || max3(c) > 1e6) { c = vec3f(0.0); }
    sum += c;
  }
  var a = irr[i];
  if (gu.size.z == 0u) { a = vec4f(0.0); }
  irr[i] = a + vec4f(sum, f32(gu.opt.z));
}

@compute @workgroup_size(8, 8)
fn ptMain(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  let x = gu.tile.x + gid.x;
  let y = gu.tile.y + gid.y;
  if (gid.x >= gu.tile.z || gid.y >= gu.tile.w || x >= gu.size.x || y >= gu.size.y) { return; }
  let pix = y * gu.size.x + x;
  var sum = vec3f(0.0);
  let shift = vec2f(u01(hash3(pix, 7u, 0x5bd1e995u)), u01(hash3(pix, 11u, 0x68e31da4u)));
  for (var k = 0u; k < gu.opt.z; k++) {
    let si = gu.size.z + k;
    var st = hash3(pix, si, gu.opt.y);
    // sub-pixel: R2 sequence over the sample index, rotated per pixel; pixel x spans [x - 0.5, x + 0.5]
    let j = vec2f(kr(si, R2A, shift.x), kr(si, R2B, shift.y));
    var c = pathSample(vec2f(f32(x), f32(y)) + j - vec2f(0.5), &st, si == 0u, pix);
    // NaN/inf guard (a degenerate normal must not poison the accumulation)
    if (!(c.x == c.x && c.y == c.y && c.z == c.z) || max3(c) > 1e6) { c = vec3f(0.0); }
    sum += c;
  }
  var a = accum[pix];
  if (gu.size.z == 0u) { a = vec4f(0.0); }
  accum[pix] = a + vec4f(sum, f32(gu.opt.z));
}
`;
}

// ---------------------------------------------------------------- raster (fast pass while the view or a tool moves)
export function rasterSource(c, clip = false) {
  const CL = clip ? '@builtin(clip_distances) clip: array<f32, 1>,' : '';
  return (clip ? 'enable clip_distances;\n' : '') + defines(c) + COMMON + GU + /* wgsl */ `
@group(0) @binding(0) var<storage, read> tris: array<vec4f>;
@group(0) @binding(1) var<storage, read> verts: array<vec4f>;
@group(0) @binding(2) var<storage, read> params: array<vec4f>;
@group(0) @binding(3) var<uniform> scn: SceneU;
@group(0) @binding(4) var<uniform> gu: GU;
@group(0) @binding(5) var<uniform> ru: RU;
@group(0) @binding(6) var shadowMap: texture_depth_2d;
@group(0) @binding(7) var shadowSmp: sampler_comparison;
@group(0) @binding(8) var<storage, read> irr: array<vec4f>;
@group(0) @binding(9) var<storage, read> backSlot: array<u32>;
struct RU { viewProj: mat4x4f, lightProj: mat4x4f, scopeProj: mat4x4f, misc: vec4f, emitN: vec4f }   // misc: shadow texel; emitN: outlet normal (scope frame)
@group(1) @binding(0) var<uniform> draw: vec4u;          // x: instance index (0xffffffff = static), y: triangle offset
@group(1) @binding(1) var scopeShadow: texture_depth_2d;

` + BSDF + EMIT + /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, ${CL} @location(0) p: vec3f, @location(1) n: vec3f, @location(2) @interpolate(flat) mesh: u32, @location(3) amb: vec3f, @location(4) ambB: vec3f };
struct FI { @builtin(position) pos: vec4f, @location(0) p: vec3f, @location(1) n: vec3f, @location(2) @interpolate(flat) mesh: u32, @location(3) amb: vec3f, @location(4) ambB: vec3f };
struct SFI { @builtin(position) pos: vec4f, @location(0) p: vec3f, @location(1) @interpolate(flat) cut: u32 };
fn cutClip(isStatic: bool, mesh: u32, p: vec3f) -> f32 {
  let mat = bitcast<u32>(params[MESH_OFF + 2u * mesh].w);
  if (isStatic && mat < 32u && ((gu.opt.x >> mat) & 1u) != 0u) { return scn.cut.w - dot(scn.cut.xyz, p); }
  return 1.0;
}

fn posed(i: u32) -> Pose { return instPose(i, 0.0); }

@vertex fn vsMain(@builtin(vertex_index) vid: u32) -> VO {
  let k = draw.y + vid / 3u;
  let j = vid % 3u;
  let r0 = tris[4u * k];
  var p = r0.xyz;
  if (j == 1u) { p = p + tris[4u * k + 1u].xyz; }
  if (j == 2u) { p = p + tris[4u * k + 2u].xyz; }
  let vi = bitcast<u32>(tris[4u * k + j].w);
  var n = verts[2u * vi + 1u].xyz;
  var a = gu.env.xyz * gu.env.w;
  var ab = a;
  if (draw.x != 0xffffffffu) {
    let P = posed(draw.x);
    p = P.r * p + P.p;
    n = P.r * n;
  } else if (vi < N_STATIC_VERTS) {
    let e = irr[vi];
    if (e.w > 0.0) { a = e.xyz / e.w; }
    ab = a;
    let sl = backSlot[vi];
    if (sl != 0xffffffffu) { let eb = irr[N_STATIC_VERTS + sl]; if (eb.w > 0.0) { ab = eb.xyz / eb.w; } }
  }
  var o: VO;
  o.pos = ru.viewProj * vec4f(p, 1.0);
  o.p = p;
  o.n = n;
  o.mesh = bitcast<u32>(tris[4u * k + 3u].x);
  o.amb = a;
  o.ambB = ab;
  ${clip ? 'o.clip[0] = cutClip(draw.x == 0xffffffffu, o.mesh, p);' : ''}
  return o;
}

struct SO { @builtin(position) pos: vec4f, ${CL} @location(0) p: vec3f, @location(1) @interpolate(flat) cut: u32 };
@vertex fn vsShadow(@builtin(vertex_index) vid: u32) -> SO {
  let k = draw.y + vid / 3u;
  let j = vid % 3u;
  var p = tris[4u * k].xyz;
  if (j == 1u) { p = p + tris[4u * k + 1u].xyz; }
  if (j == 2u) { p = p + tris[4u * k + 2u].xyz; }
  if (draw.x != 0xffffffffu) { let P = posed(draw.x); p = P.r * p + P.p; }
  let mesh = bitcast<u32>(tris[4u * k + 3u].x);
  let mat = bitcast<u32>(params[MESH_OFF + 2u * mesh].w);
  var o: SO;
  o.pos = ru.lightProj * vec4f(p, 1.0);
  if (draw.z == 1u) { o.pos = ru.scopeProj * vec4f(p, 1.0); }
  o.p = p;
  o.cut = select(0u, 1u, draw.x == 0xffffffffu && mat < 32u && ((gu.opt.x >> mat) & 1u) != 0u);
  ${clip ? 'o.clip[0] = cutClip(draw.x == 0xffffffffu, mesh, p);' : ''}
  return o;
}
// the cutaway removes the PLA for every ray in the traced image, so it removes it from the shadow casters too (per fragment)
@fragment fn fsShadow(v: SFI) {
  if (v.cut == 1u && dot(scn.cut.xyz, v.p) > scn.cut.w) { discard; }
}

fn shadowKey(p: vec3f, ng: vec3f) -> f32 {
  let q = ru.lightProj * vec4f(p + ng * 0.0015, 1.0);
  let uv = vec2f(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || q.z > 1.0) { return 1.0; }
  let t = ru.misc.x;
  var s = 0.0;
  for (var i = -1; i <= 1; i++) { for (var j = -1; j <= 1; j++) {
    s += textureSampleCompareLevel(shadowMap, shadowSmp, uv + vec2f(f32(i), f32(j)) * t, q.z - 0.0005);
  } }
  return s / 9.0;
}
fn shadowScope(p: vec3f, ng: vec3f) -> f32 {
  let q = ru.scopeProj * vec4f(p + ng * 0.002, 1.0);
  if (q.w <= 1e-5) { return 0.0; }
  let c = q.xyz / q.w;
  let uv = vec2f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || c.z > 1.0) { return 0.0; }
  let t = 1.0 / 1024.0;
  var s = 0.0;
  for (var i = -1; i <= 1; i++) { for (var j = -1; j <= 1; j++) {
    s += textureSampleCompareLevel(scopeShadow, shadowSmp, uv + vec2f(f32(i), f32(j)) * t, c.z - 0.00005);
  } }
  return s / 9.0;
}

@fragment fn fsMain(v: FI) -> @location(0) vec4f {
  let ngRaw = cross(dpdx(v.p), dpdy(v.p));
  let mt = params[MESH_OFF + 2u * v.mesh];
  let mat = bitcast<u32>(mt.w);
  let flags = bitcast<u32>(params[MESH_OFF + 2u * v.mesh + 1u].x);
  let isCutMat = mat < 32u && ((gu.opt.x >> mat) & 1u) != 0u && (flags & F_STATIC) != 0u;
  ${clip ? '' : 'if (isCutMat && dot(scn.cut.xyz, v.p) > scn.cut.w) { discard; }'}
  // the hollow skin seen from behind (through the cut): its light is the back-side bake
  let backFace = isCutMat && dot(v.n, gu.pos.xyz - v.p) < 0.0;
  let wo = normalize(gu.pos.xyz - v.p);
  var g = normalize(ngRaw);
  if (!(dot(ngRaw, ngRaw) > 1e-30)) { g = normalize(v.n); }
  if (dot(g, wo) < 0.0) { g = -g; }
  var n = normalize(v.n);
  if (dot(n, g) < 0.0) { n = -n; }
  if (dot(n, wo) < 1e-3) { n = g; }
  if ((flags & F_EMITTER) != 0u) {
    if (gu.opt.w == 0u) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    let axis = posed(scn.info.y).r * scn.axis.xyz;
    return vec4f(scn.L0.xyz * profile(acos(clamp(dot(wo, axis), -1.0, 1.0))), 1.0);
  }
  let M = loadMat(mat);
  var L = vec3f(0.0);
  // key
  if (gu.keyDir.w > 0.5) {
    let l = gu.keyDir.xyz;
    let cl = dot(n, l);
    if (cl > 0.0 && dot(g, l) > 0.0) { L += evalBSDF(M, n, l, wo) * gu.keyE.xyz * cl * shadowKey(v.p, g); }
  }
  // scope outlet as one small area light at its centroid (shadow-mapped from the outlet)
  if (gu.opt.w != 0u) {
    let P = posed(scn.info.y);
    let c = P.r * scn.emitC.xyz + P.p;
    let axis = P.r * scn.axis.xyz;
    var wi = c - v.p;
    let d2 = max(dot(wi, wi), 1e-8);
    wi = wi / sqrt(d2);
    let cosS = dot(n, wi);
    let cosA = dot(-wi, axis);
    let cosN = dot(-wi, normalize(P.r * ru.emitN.xyz));      // one-sided outlet: radiance into the tip-face normal hemisphere
    if (cosS > 0.0 && cosN > 0.0 && dot(g, wi) > 0.0) {
      let Le = scn.L0.xyz * profile(acos(clamp(cosA, -1.0, 1.0)));
      L += evalBSDF(M, n, wi, wo) * Le * (scn.L0.w * cosN * cosS / d2) * shadowScope(v.p, g);
    }
  }
  // ambient: the constant environment through the directional albedo, times a baked ambient occlusion, plus the
  // dome's integrating-sphere fill (ru.misc.y, inside only) for static surfaces
  let mu = clamp(dot(n, wo), 0.0, 1.0);
  let alb = albedoSpec(mat, mu) + albedoDiff(mat, mu);
  L += alb * select(v.amb, v.ambB, backFace);
  return vec4f(L, 1.0);
}
`;
}

// ---------------------------------------------------------------- display: crossfade, edge-aware denoise, tone map
export function displaySource() {
  return GU + /* wgsl */ `
@group(0) @binding(0) var<uniform> gu: GU;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;
@group(0) @binding(2) var<storage, read> gbuf: array<vec4f>;
@group(0) @binding(3) var rasterTex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> outSize: vec4f;
@group(0) @binding(5) var rasSmp: sampler;      // godview_pt2: the live raster may be drawn below canvas size (?gv_scale)

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
fn neutral(c0: vec3f) -> vec3f {
  // Khronos PBR Neutral
  let sc = 0.8 - 0.04;
  let des = 0.15;
  let x = min(c0.r, min(c0.g, c0.b));
  let off = select(0.04, x - 6.25 * x * x, x < 0.08);
  var c = c0 - off;
  let peak = max(c.r, max(c.g, c.b));
  if (peak < sc) { return c; }
  let d = 1.0 - sc;
  let np = 1.0 - d * d / (peak + d - sc);
  c = c * (np / peak);
  let g = 1.0 - 1.0 / (des * (peak - np) + 1.0);
  return mix(c, vec3f(np), g);
}
fn oetf(x: f32) -> f32 { return select(1.055 * pow(max(x, 0.0), 1.0 / 2.4) - 0.055, 12.92 * x, x <= 0.0031308); }
fn traced(q: vec2i) -> vec3f { let a = accum[u32(q.y) * gu.size.x + u32(q.x)]; return a.xyz / max(a.w, 1.0); }

@fragment fn fs(@builtin(position) fp: vec4f) -> @location(0) vec4f {
  let W = i32(gu.size.x); let H = i32(gu.size.y);
  let q = clamp(vec2i(fp.xy * vec2f(f32(W), f32(H)) / outSize.xy), vec2i(0), vec2i(W - 1, H - 1));
  let ras = textureSampleLevel(rasterTex, rasSmp, fp.xy / outSize.xy, 0.0).xyz;
  var col = ras;
  if (gu.disp.y > 0.0) {
    var pt = traced(q);
    let sigma = gu.disp.z;
    if (sigma > 0.0) {
      // edge-aware 5x5 cross-bilateral on the first-hit normal and distance (light: only while samples are few)
      let g0 = gbuf[u32(q.y) * gu.size.x + u32(q.x)];
      var s = vec3f(0.0); var ws = 0.0;
      for (var dy = -2; dy <= 2; dy++) { for (var dx = -2; dx <= 2; dx++) {
        let r = clamp(q + vec2i(dx, dy), vec2i(0), vec2i(W - 1, H - 1));
        let g1 = gbuf[u32(r.y) * gu.size.x + u32(r.x)];
        var w = exp(-f32(dx * dx + dy * dy) / (2.0 * sigma * sigma));
        if ((g0.w > 0.0) != (g1.w > 0.0)) { w = 0.0; }
        if (g0.w > 0.0) {
          w *= pow(max(dot(g0.xyz, g1.xyz), 0.0), 32.0);
          w *= exp(-abs(g1.w - g0.w) / (0.004 * g0.w));
        }
        s += traced(r) * w; ws += w;
      } }
      pt = s / max(ws, 1e-6);
    }
    col = mix(ras, pt, gu.disp.y);
  }
  let t = neutral(max(col * gu.disp.x, vec3f(0.0)));
  return vec4f(oetf(t.r), oetf(t.g), oetf(t.b), 1.0);
}
`;
}

// ---------------------------------------------------------------- bake smoothing (indirect light is smooth; 64 rays per vertex are not)
export const SMOOTH_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> verts: array<vec4f>;
@group(0) @binding(1) var<storage, read> adjStart: array<u32>;
@group(0) @binding(2) var<storage, read> adjList: array<u32>;
@group(0) @binding(3) var<storage, read> src: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(5) var<uniform> su: vec4u;          // x: vertex count, y: back entries
@group(0) @binding(6) var<storage, read> backList: array<u32>;
@group(0) @binding(7) var<storage, read> backSlot: array<u32>;
@compute @workgroup_size(64)
fn smoothMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= su.x + su.y) { return; }
  let isB = i >= su.x;
  var vi = i;
  if (isB) { vi = backList[i - su.x]; }
  let ni = normalize(verts[2u * vi + 1u].xyz);
  var acc = vec3f(0.0);
  var w = 0.0;
  let a = src[i];
  if (a.w > 0.0) { acc = a.xyz / a.w; w = 1.0; }
  let k0 = adjStart[vi];
  let k1 = min(adjStart[vi + 1u], k0 + 32u);
  for (var k = k0; k < k1; k++) {
    let j = adjList[k];
    var e = j;
    if (isB) { let sl = backSlot[j]; if (sl == 0xffffffffu) { continue; } e = su.x + sl; }
    let b = src[e];
    if (b.w <= 0.0) { continue; }
    let c = max(dot(ni, normalize(verts[2u * j + 1u].xyz)), 0.0);
    let wj = c * c * c * c * c * c * c * c;
    acc += wj * b.xyz / b.w;
    w += wj;
  }
  if (w > 0.0) { dst[i] = vec4f(acc / w, 1.0); } else { dst[i] = vec4f(0.0); }
}
`;
