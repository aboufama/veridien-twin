// lite_nodes: a WebGPU RASTER renderer for the live twin that keeps the validated path tracer's NODE RADIOSITY
// (gpu/renderer.js: nodeDirect + Jacobi on the static vertices, capsule occlusion) and drops its per-pixel camera
// ray tracing. docs/WEB_TWIN.md §18 (lite API, SCOPE camera), §8 (API, BSDF), §9 (sensor, display look only).
//
// Light transport per fragment:
//   * direct: the scope outlet (area emitter, L0 p(theta)) integrated with NS fixed stratified area samples, the §8
//     BSDF (gpu/wgsl/bsdf.wgsl.js, the very same WGSL) per sample; soft shadows from K shadow maps, one per outlet
//     sector (equidistant projection about the emission axis, 2x2-bilinear PCF), static and dynamic casters in
//     separate maps (the static map is re-rendered only when the scope moves);
//   * indirect on static surfaces: the path tracer's node irradiance interpolated per vertex with its active-node
//     weights (static vertex = node), times the analytic capsule occlusion of the posed bodies, times the
//     directional albedo -- exactly the path tracer's static-surface term;
//   * indirect on posed bodies: a uniform environment of the dome's mean exitance (eAvg x albedo / pi) times the
//     body's directional albedo.
// Cameras: CAM1-3 with per-vertex KB4 (all meshes subdivided to <= 3 mm), SCOPE pinhole (75 deg HFOV, 0.95 circle).
import { TwinRenderer } from '../gpu/renderer.js';
import { COMMON, defines } from '../gpu/wgsl/common.wgsl.js';
import { BSDF } from '../gpu/wgsl/bsdf.wgsl.js';
import { shaderSource } from '../gpu/wgsl/kernels.wgsl.js';
import { sensorParams } from '../gpu/sensor.js';

/**
 * The path tracer reduced to what lite_nodes uses: its node radiosity (gather table, nodeDirect, Jacobi) and its CPU
 * visible(). Its camera, overview, display and sensor kernels are never compiled (they were most of the shader
 * compile time), and with `skipInterior` the second (interior-only) static BVH is not built (2 s of CPU at load;
 * the node rays then traverse the full static tree, which is exact, only slower).
 */
class NodeTracer extends TwinRenderer {
  static async createNodes(opts) { const r = new NodeTracer(); await r._init(opts); return r; }

  async _buildPipelines() {
    const dev = this.device;
    const ro = { type: 'read-only-storage' }, rw = { type: 'storage' }, un = { type: 'uniform' };
    const C = GPUShaderStage.COMPUTE;
    const entries = (types) => types.map((buffer, binding) => ({ binding, visibility: C, buffer }));
    this.layouts = {
      sceneRO: dev.createBindGroupLayout({ entries: [...entries([ro, ro, ro, ro, ro, rw]), { binding: 6, visibility: C, texture: { sampleType: 'float' } }, { binding: 7, visibility: C, sampler: { type: 'filtering' } }] }),
      sceneRW: dev.createBindGroupLayout({ entries: entries([ro, ro, ro, ro, rw, rw, ro, rw, ro, rw]) }),
      radG1: dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: un }, { binding: 1, visibility: C, buffer: un }] }),
    };
    const radMod = await this._module(shaderSource('radiosity', this.consts), 'radiosity');
    const radLayout = dev.createPipelineLayout({ bindGroupLayouts: [this.layouts.sceneRW, this.layouts.radG1] });
    const mk = (entryPoint) => dev.createComputePipelineAsync({ layout: radLayout, compute: { module: radMod, entryPoint } });
    const names = ['gatherBuild', 'gatherCount', 'gatherRemap', 'nodeDirect', 'jacobiGather', 'jacobiUpdate'];
    const ps = await Promise.all(names.map(mk));
    this.pipes = Object.fromEntries(names.map((n, i) => [n, ps[i]]));
    this.groups = { sceneRO: this._sceneROGroup() };
  }

  async _buildSensor() { this.sensorP = sensorParams(this.physics, this.opts.sensor || {}); }

  _buildInteriorBVH() {
    if (this.opts.skipInterior) { this.interiorRoot = 0; this.stats.interiorTris = this.S.staticTris; this.stats.interiorSkipped = true; return; }
    super._buildInteriorBVH();
  }
}

const K_SECT = 4;            // outlet sectors = shadow maps
const NS_DYN = 96;           // emitter samples for posed bodies (glossy steel)
const NS_STATIC = 24;        // emitter samples for static surfaces (first NS_STATIC of the same stratified set)
const SM_RES = 2048;
const SM_THETA_MAX = (62 * Math.PI) / 180;   // the profile is 0 past 55 deg
const FAR = 1.0;
const SUBDIV_H = 0.003;
const SUBDIV_CANNULA = 0.01;      // the cannulas only show at the ports' edges
const MSAA = 4;
const STATIC_INST = 0xffff;

const unwrap = (x) => (x && typeof x === 'object' && !Array.isArray(x) && 'value' in x ? x.value : x);
const rowsOf = (T) => (Array.isArray(T[0]) ? T : [[T[0], T[4], T[8], T[12]], [T[1], T[5], T[9], T[13]], [T[2], T[6], T[10], T[14]], [T[3], T[7], T[11], T[15]]]);
const mulRv = (R, v) => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

// ------------------------------------------------------------------------------------------------ WGSL
const VIEW_U = /* wgsl */ `
struct ViewU {
  rx: vec4f, ry: vec4f, rz: vec4f,   // camera axes in D (x right, y down, z forward)
  pos: vec4f,                        // centre; w = mode (0 kb4, 1 pinhole, 2 shadow-equidistant)
  lensA: vec4f,                      // fx fy cx cy
  lensB: vec4f,                      // k1..k4
  size: vec4f,                       // w h thetaCut outScale
  misc: vec4u,                       // skip mask, nStatic samples, nDyn samples, casters (0 all, 1 static, 2 dynamic)
  env: vec4f,                        // uniform environment radiance for posed bodies (rgb), time
  axisW: vec4f,                      // emission axis (world), nSect
  lt: vec4f, lb: vec4f,              // shadow frame tangent / bitangent (world); lt.w = thetaMax, lb.w = far
  lo: array<vec4f, ${K_SECT}>,       // shadow origins (world); w = depth bias (m)
}
`;

const SHADER = (c) => /* wgsl */ `
${defines(c)}
${COMMON}
${BSDF}
${VIEW_U}
@group(0) @binding(0) var<storage, read> params: array<vec4f>;
@group(0) @binding(1) var<storage, read> nodeState: array<vec4f>;
@group(0) @binding(2) var<uniform> scn: SceneU;
@group(0) @binding(3) var<uniform> view: ViewU;
@group(0) @binding(4) var<storage, read> es: array<vec4f>;
@group(0) @binding(5) var smStatic: texture_depth_2d_array;
@group(0) @binding(6) var smDyn: texture_depth_2d_array;
@group(0) @binding(7) var cmpSmp: sampler_comparison;
@group(0) @binding(8) var boardTex: texture_2d<f32>;
@group(0) @binding(9) var boardSmp: sampler;

const K_SECT: u32 = ${K_SECT}u;

struct VIn { @location(0) p: vec3f, @location(1) bits: u32, @location(2) n: vec3f, @location(3) node: u32, @location(4) uv: vec2f }
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) p: vec3f,
  @location(1) n: vec3f,
  @location(2) ew: vec4f,
  @location(3) @interpolate(flat) mi: u32,
  @location(4) uv: vec2f,
}

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

fn capsuleOcc(p: vec3f, n: vec3f, a: vec3f, bb: vec3f, r: f32) -> f32 {
  let ab = bb - a;
  let L = length(ab);
  if (L < 1e-9 || r <= 0.0) { return 0.0; }
  let e = ab / L;
  let ap = p - a;
  let s0 = dot(ap, e);
  let perp = ap - s0 * e;
  let h = sqrt(dot(perp, perp) + r * r);
  var ep = -perp / max(length(perp), 1e-12);
  if (length(perp) < 1e-12) { ep = normalize(cross(e, basisT(e))); }
  let np = dot(n, ep);
  let na = dot(n, e);
  var f1 = atan((0.0 - s0) / h);
  var f2 = atan((L - s0) / h);
  let f0 = atan2(na, np);
  f1 = max(f1, f0 - 0.5 * PI);
  f2 = min(f2, f0 + 0.5 * PI);
  if (f2 <= f1) { return 0.0; }
  let I = np * (0.5 * (f2 - f1) + 0.25 * (sin(2.0 * f2) - sin(2.0 * f1))) + na * 0.5 * (sin(f2) * sin(f2) - sin(f1) * sin(f1));
  return clamp(INV_PI * (2.0 * r / h) * I, 0.0, 1.0);
}

fn capsuleVis(p: vec3f, n: vec3f) -> f32 {
  var v = 1.0;
  let cnt = min(scn.info.x, MAX_INST);
  for (var i = 0u; i < cnt; i++) {
    let I = scn.inst[i];
    if ((bitcast<u32>(I.p1.w) & 2u) != 0u) { continue; }
    let o = capsuleOcc(p, n, I.capA.xyz, I.capB.xyz, I.capA.w);
    v = v - o * (1.0 - I.capB.w);
  }
  return clamp(v, 0.0, 1.0);
}

// equidistant projection about the emission axis from shadow origin k: (ndc.xy, dist)
fn smProject(pw: vec3f, k: u32) -> vec3f {
  let d = pw - view.lo[k].xyz;
  let x = dot(d, view.lt.xyz);
  let y = dot(d, view.lb.xyz);
  let z = dot(d, view.axisW.xyz);
  let rho = length(vec2f(x, y));
  let th = atan2(rho, z);
  let s = th / (view.lt.w * max(rho, 1e-12));
  return vec3f(x * s, y * s, length(d));
}

struct World { p: vec3f, n: vec3f, ok: bool }

fn toWorld(v: VIn) -> World {
  let inst = v.bits >> 16u;
  if (inst == ${STATIC_INST}u) { return World(v.p, v.n, true); }
  let I = scn.inst[inst];
  if ((bitcast<u32>(I.p1.w) & 2u) != 0u || ((view.misc.x >> inst) & 1u) != 0u) { return World(vec3f(0.0), vec3f(0.0), false); }
  let P = instPose(inst, view.env.w);
  return World(P.r * v.p + P.p, P.r * v.n, true);
}

fn projectCam(p: vec3f) -> vec4f {
  let d = p - view.pos.xyz;
  let x = dot(d, view.rx.xyz);
  let y = dot(d, view.ry.xyz);
  let z = dot(d, view.rz.xyz);
  let W = view.size.x;
  let H = view.size.y;
  if (view.pos.w > 0.5) {         // pinhole
    let nearZ = 1e-3;
    return vec4f((2.0 * view.lensA.x * x + (2.0 * (view.lensA.z + 0.5) - W) * z) / W,
                 ((H - 2.0 * (view.lensA.w + 0.5)) * z - 2.0 * view.lensA.y * y) / H,
                 (z - nearZ) * ${FAR.toFixed(3)} / (${FAR.toFixed(3)} - nearZ), z);
  }
  let r = length(vec2f(x, y));
  let th = atan2(r, z);
  if (th > view.size.z) {
    // past the cut (behind the lens or beyond the fold): a point at infinity along its azimuth (w < 0), so a
    // triangle wholly past it is clipped away and one crossing it is clipped at w = 0, extending outward
    let e = normalize(vec2f(view.lensA.x * x / W, -view.lensA.y * y / H) + vec2f(1e-12, 0.0));
    return vec4f(e, -1e-3, -1e-3);
  }
  let t2 = th * th;
  let thd = th * (1.0 + t2 * (view.lensB.x + t2 * (view.lensB.y + t2 * (view.lensB.z + t2 * view.lensB.w))));
  let s = thd / max(r, 1e-12);
  let u = view.lensA.x * x * s + view.lensA.z + 0.5;      // OpenCV pixel centres (integer u) -> raster centres (i + 0.5)
  let vv = view.lensA.y * y * s + view.lensA.w + 0.5;
  return vec4f(2.0 * u / W - 1.0, 1.0 - 2.0 * vv / H, length(d) / ${FAR.toFixed(3)}, 1.0);
}

@vertex fn vsCam(v: VIn) -> VOut {
  var o: VOut;
  let w = toWorld(v);
  if (!w.ok) { o.pos = vec4f(0.0, 0.0, -2.0, 1.0); return o; }
  o.pos = projectCam(w.p);
  o.p = w.p;
  o.n = w.n;
  o.mi = v.bits;
  o.uv = v.uv;
  let ni = v.node;
  if (ni != 0xffffffffu) {
    let slot = scn.info.z;
    let wa = nodeState[4u * ni + slot].w;
    o.ew = vec4f(nodeState[4u * ni].xyz * wa, wa);
  } else { o.ew = vec4f(0.0); }
  return o;
}

struct SOut { @builtin(position) pos: vec4f }
@vertex fn vsShadow(v: VIn, @builtin(instance_index) k: u32) -> SOut {
  var o: SOut;
  let w = toWorld(v);
  let bits = v.bits;
  let inst = bits >> 16u;
  let isStatic = inst == ${STATIC_INST}u;
  let mesh = bits & 0xffffu;
  let flags = bitcast<u32>(params[MESH_OFF + 2u * mesh + 1u].x);
  var keep = w.ok && (flags & (F_EMITTER | F_SCOPE)) == 0u;
  if (view.misc.w == 1u && !isStatic) { keep = false; }
  if (view.misc.w == 2u && isStatic) { keep = false; }
  if (!keep) { o.pos = vec4f(0.0, 0.0, -2.0, 1.0); return o; }
  let q = smProject(w.p, k);
  if (length(q.xy) > 1.0) {
    // outside the map's cone (the profile is 0 there): a point at infinity along its azimuth (w < 0), so that a
    // triangle next to the outlet spanning a large angle (the C cannula's toe rim) is clipped instead of cutting
    // straight across the map
    o.pos = vec4f(normalize(q.xy + vec2f(1e-12, 0.0)), -1e-3, -1e-3);
    return o;
  }
  o.pos = vec4f(q.x, q.y, q.z / view.lb.w, 1.0);
  return o;
}

fn smVis(pw: vec3f, k: u32, biasM: f32) -> f32 {
  let q = smProject(pw, k);
  if (abs(q.x) >= 1.0 || abs(q.y) >= 1.0) { return 1.0; }
  let uv = vec2f(0.5 + 0.5 * q.x, 0.5 - 0.5 * q.y);
  let dref = (q.z - biasM) / view.lb.w;
  let t = 1.0 / f32(${SM_RES});
  var s = 0.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let o = uv + vec2f(f32(i), f32(j)) * t;
      let a = textureSampleCompareLevel(smStatic, cmpSmp, o, k, dref);
      let b = textureSampleCompareLevel(smDyn, cmpSmp, o, k, dref);
      s = s + min(a, b);
    }
  }
  return s / 9.0;
}

@fragment fn fsCam(in: VOut) -> @location(0) vec4f {
  let dpx = dpdx(in.p);
  let dpy = dpdy(in.p);
  let mesh = in.mi & 0xffffu;
  let inst = in.mi >> 16u;
  let row0 = params[MESH_OFF + 2u * mesh];
  let flags = bitcast<u32>(params[MESH_OFF + 2u * mesh + 1u].x);
  let mat = bitcast<u32>(row0.w);
  let p = in.p;
  let wo = normalize(view.pos.xyz - p);
  var ng = cross(dpx, dpy);
  if (dot(ng, ng) < 1e-30) { ng = in.n; }
  ng = normalize(ng);
  if (dot(ng, wo) < 0.0) { ng = -ng; }
  var ns = normalize(in.n);
  if ((flags & F_EMITTER) != 0u) {
    if (dot(wo, ns) <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    let Le = scn.L0.xyz * profile(acos(clamp(dot(wo, view.axisW.xyz), -1.0, 1.0)));
    return vec4f(min(Le * view.size.w, vec3f(60000.0)), 1.0);
  }
  if (dot(ns, ng) < 0.0) { ns = -ns; }
  if (dot(ns, wo) < 1e-4) { ns = ng; }
  var M = loadMat(mat);
  if ((flags & F_TEXTURED) != 0u) {
    let m = textureSampleLevel(boardTex, boardSmp, in.uv, 0.0).r;
    M.c = mix(M.x.xyz, M.c, m);
    M.alpha = mix(M.x.w, M.alpha, m);
  }
  let isStatic = inst == ${STATIC_INST}u;
  let nS = select(view.misc.z, view.misc.y, isStatic);
  var acc: array<vec3f, ${K_SECT}>;
  for (var k = 0u; k < K_SECT; k++) { acc[k] = vec3f(0.0); }
  for (var j = 0u; j < nS; j++) {
    let a = es[2u * j];
    let b = es[2u * j + 1u];
    var wi = a.xyz - p;
    let d2 = dot(wi, wi);
    wi = wi * inverseSqrt(d2);
    let cosE = dot(-wi, b.xyz);
    if (cosE <= 0.0) { continue; }
    let cosS = dot(ns, wi);
    if (cosS <= 0.0 || dot(ng, wi) <= 0.0) { continue; }
    let pr = profile(acos(clamp(dot(-wi, view.axisW.xyz), -1.0, 1.0)));
    if (pr <= 0.0) { continue; }
    let f = evalBSDF(M, ns, wi, wo);
    let k = u32(a.w);
    acc[k] = acc[k] + f * (pr * cosS * cosE * b.w / d2);
  }
  var L = vec3f(0.0);
  // receiver offset along the normal; posed bodies (5 mm shafts, steel) self-shadow in the dynamic map without a
  // larger one (r2: shafts -24 %, steel -33 %, highlights -47 %)
  let off = p + ng * select(6e-4, 1.5e-4, isStatic);
  let dbg = u32(view.axisW.w);
  var vsum = 0.0;
  for (var k = 0u; k < K_SECT; k++) {
    var vk = 1.0;
    if (dbg != 1u) { vk = smVis(off, k, select(6e-4, view.lo[k].w, isStatic)); }
    vsum = vsum + vk;
    if (max(acc[k].x, max(acc[k].y, acc[k].z)) <= 0.0) { continue; }
    L = L + acc[k] * vk;
  }
  if (dbg == 2u) { return vec4f(vec3f(vsum / f32(K_SECT)), 1.0); }
  L = L * scn.L0.xyz / f32(nS);
  let mu = max(dot(ns, wo), 0.0);
  let A = albedoSpec(mat, mu) + albedoDiff(mat, mu);
  if (isStatic) {
    var E = vec3f(0.0);
    if (in.ew.w > 1e-6) { E = in.ew.xyz / in.ew.w; }
    E = E * capsuleVis(p, ns);
    L = L + E * INV_PI * A;
  } else {
    L = L + view.env.xyz * A;
  }
  return vec4f(L * view.size.w, 1.0);
}
`;

// resolve (rgba16float, scaled) -> rgba32float radiance, SCOPE circle mask
const FINISH = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u: vec4f;      // 1/outScale, circle radius px (0 = none), w, h
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3u) {
  let d = textureDimensions(src);
  if (g.x >= d.x || g.y >= d.y) { return; }
  var c = textureLoad(src, vec2i(g.xy), 0).xyz * u.x;
  if (u.y > 0.0) {
    let q = vec2f(g.xy) + 0.5 - 0.5 * vec2f(d);
    if (length(q) > u.y) { c = vec3f(0.0); }
  }
  textureStore(dst, vec2i(g.xy), vec4f(c, 1.0));
}
`;

// display look (§18.1): exposure (AE from the previous frame's mean), relative illumination (none on SCOPE),
// veiling glare (global-mean approximation of the sigma = 0.25 diag Gaussian), white balance, sRGB + S-curve.
const DISPLAY = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
struct DU { gain: vec4f, wb: vec4f, tone: vec4f, lens: vec4f, lensB: vec4f }  // gain.x exposure, gain.y glare k, gain.zw -, tone: c, black, mode, -
@group(0) @binding(2) var<uniform> du: DU;
@group(0) @binding(3) var<storage, read_write> stats: array<atomic<u32>, 4>;
@group(0) @binding(4) var<storage, read> prev: array<u32, 4>;
fn oetf(x: f32) -> f32 { if (x <= 0.0031308) { return 12.92 * x; } return 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
fn scurve(u: f32, c: f32) -> f32 { let a = pow(clamp(u, 0.0, 1.0), c); return a / (a + pow(clamp(1.0 - u, 0.0, 1.0), c)); }
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3u) {
  let d = textureDimensions(src);
  if (g.x >= d.x || g.y >= d.y) { return; }
  let L = textureLoad(src, vec2i(g.xy), 0).xyz;
  if ((g.x & 3u) == 0u && (g.y & 3u) == 0u) {
    let y = dot(L, vec3f(0.2126, 0.7152, 0.0722));
    atomicAdd(&stats[0], u32(clamp(y * du.gain.w, 0.0, 65535.0)));
    atomicAdd(&stats[1], 1u);
  }
  let n = max(f32(prev[1]), 1.0);
  let mean = f32(prev[0]) / max(du.gain.w, 1e-30) / n;
  var E = L;
  // relative illumination for the KB4 cameras (lens-mapping form: (theta_d/ r) * dtheta_d/dtheta ... approximated by cos(theta))
  if (du.lens.x > 0.0) {
    let q = (vec2f(g.xy) + 0.5 - du.lens.zw) / du.lens.xy;
    let thd = length(q);
    E = E * mix(1.0, 0.57, clamp(thd / 1.1, 0.0, 1.0) * clamp(thd / 1.1, 0.0, 1.0));
  }
  E = (1.0 - du.gain.y) * E + du.gain.y * vec3f(mean);
  var x = E * du.gain.x * du.wb.xyz;
  let b = du.tone.y;
  var o = vec3f(0.0);
  for (var c = 0; c < 3; c++) {
    let v = clamp((x[c] - b) / (1.0 - b), 0.0, 1.0);
    var t = oetf(v);
    if (du.tone.x > 0.0) { t = scurve(t, du.tone.x); }
    o[c] = t;
  }
  textureStore(dst, vec2i(g.xy), vec4f(o, 1.0));
}
`;

const BLIT = /* wgsl */ `
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var<uniform> sz: vec4f;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); return vec4f(p[i], 0, 1); }
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let d = vec2f(textureDimensions(t));
  let s = min(sz.x / d.x, sz.y / d.y);
  let o = 0.5 * (sz.xy - d * s);
  let q = (p.xy - o) / s;
  if (q.x < 0.0 || q.y < 0.0 || q.x >= d.x || q.y >= d.y) { return vec4f(0.04, 0.04, 0.045, 1.0); }
  return textureLoad(t, vec2i(q), 0);
}
`;

// ------------------------------------------------------------------------------------------------ geometry
/**
 * Slice a mesh by the planes x = k h, y = k h, z = k h (body frame) so that no triangle spans more than h along any
 * axis (edges <= sqrt(3) h). Long slivers (shafts, the scope) become 2 triangles per slab instead of the O(n^2) a
 * longest-edge bisection makes of them. Edge crossings are computed from canonically ordered endpoints, so the two
 * triangles sharing an edge get bit-identical new vertices (watertight raster). Output is unindexed.
 */
function sliceTris(P, N, UV, I, h) {
  let tris = [];
  for (let t = 0; t < I.length; t += 3) {
    tris.push([0, 1, 2].map((j) => { const v = I[t + j]; return { p: [P[3 * v], P[3 * v + 1], P[3 * v + 2]], n: [N[3 * v], N[3 * v + 1], N[3 * v + 2]], uv: [UV[2 * v], UV[2 * v + 1]] }; }));
  }
  const less = (a, b) => (a.p[0] !== b.p[0] ? a.p[0] < b.p[0] : a.p[1] !== b.p[1] ? a.p[1] < b.p[1] : a.p[2] < b.p[2]);
  const cut = (A, B, ax, c) => {
    const [a, b] = less(A, B) ? [A, B] : [B, A];
    const f = (c - a.p[ax]) / (b.p[ax] - a.p[ax]);
    const p = [0, 1, 2].map((k) => a.p[k] + f * (b.p[k] - a.p[k]));
    p[ax] = c;
    return { p, n: norm3([0, 1, 2].map((k) => a.n[k] + f * (b.n[k] - a.n[k]))), uv: [a.uv[0] + f * (b.uv[0] - a.uv[0]), a.uv[1] + f * (b.uv[1] - a.uv[1])] };
  };
  for (let ax = 0; ax < 3; ax++) {
    const out = [];
    for (const tri of tris) {
      const vals = tri.map((v) => v.p[ax]);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const k0 = Math.floor(lo / h) + 1, k1 = Math.ceil(hi / h) - 1;
      if (k1 < k0) { out.push(tri); continue; }
      let poly = tri;
      for (let k = k0; k <= k1 + 1; k++) {
        if (k > k1) { for (let j = 1; j + 1 < poly.length; j++) out.push([poly[0], poly[j], poly[j + 1]]); break; }
        const c = k * h;
        const below = [], above = [];
        for (let j = 0; j < poly.length; j++) {
          const A = poly[j], B = poly[(j + 1) % poly.length];
          const sa = A.p[ax] - c, sb = B.p[ax] - c;
          if (sa <= 0) below.push(A);
          if (sa >= 0) above.push(A);
          if ((sa < 0 && sb > 0) || (sa > 0 && sb < 0)) { const X = cut(A, B, ax, c); below.push(X); above.push(X); }
        }
        if (below.length >= 3) for (let j = 1; j + 1 < below.length; j++) out.push([below[0], below[j], below[j + 1]]);
        if (above.length < 3) break;
        poly = above;
      }
    }
    tris = out;
  }
  const n = tris.length * 3, P2 = new Array(3 * n), N2 = new Array(3 * n), UV2 = new Array(2 * n), I2 = new Array(n);
  let o = 0;
  for (const tri of tris) for (const v of tri) { P2[3 * o] = v.p[0]; P2[3 * o + 1] = v.p[1]; P2[3 * o + 2] = v.p[2]; N2[3 * o] = v.n[0]; N2[3 * o + 1] = v.n[1]; N2[3 * o + 2] = v.n[2]; UV2[2 * o] = v.uv[0]; UV2[2 * o + 1] = v.uv[1]; I2[o] = o; o++; }
  return { P: P2, N: N2, UV: UV2, I: I2 };
}

// ------------------------------------------------------------------------------------------------ renderer
export class LiteNodesRenderer {
  /** opts: {canvas?, scene ({scene, meshes} bundle or scene dir URL), physics, device?, tracerOpts?} */
  static async create(opts) {
    const r = new LiteNodesRenderer();
    await r._init(opts);
    return r;
  }

  async _init(opts) {
    const t0 = performance.now();
    this.log = opts.log || ((...a) => console.log('[lite-nodes]', ...a));
    this.debug = opts.debug || 0;          // 1: no shadows, 2: shadow visibility as grey
    let physics = opts.physics;
    if (typeof physics === 'string') physics = await (await fetch(physics)).json();
    this.physics = physics || {};
    // the validated tracer, used only for its node radiosity (and visible())
    // fast: coarser nodes (6 mm, 4x fewer: 4x less static raster, BVH and gather-table work), 32 gather rays, no node
    // smoothing (its neighbour search was 1.1 s), no interior BVH. Judged against Mitsuba like the 3 mm default.
    const fast = opts.fast ? { nodeSpacing: 0.006, gatherK: 32, smoothSigma: 0, skipInterior: true } : {};
    this.tr = await NodeTracer.createNodes({ scene: opts.scene, physics: this.physics, device: opts.device, preset: 'live', log: this.log, ...fast, ...(opts.tracerOpts || {}) });
    this.device = this.tr.device;
    this.S = this.tr.S;
    this.scene = this.tr.scene;
    this.stats = { tracerMs: performance.now() - t0 };
    this._buildGeometry();
    this._buildEmitterSamples();
    await this._buildPipelines();
    this._setupCameras();
    this._timerSetup();
    this.posesVersion = 0;
    this._smScopeKey = null;
    this.setPoses({ bodies: {} });
    this.stats.loadMs = performance.now() - t0;
    this.log(`lite_nodes ready in ${this.stats.loadMs.toFixed(0)} ms (tracer ${this.stats.tracerMs.toFixed(0)} ms), ${this.indexCount / 3} raster triangles`);
  }

  get cameras() { return [...this.scene.cameras.map((c) => c.name), 'SCOPE']; }

  _buildGeometry() {
    const S = this.S, dev = this.device;
    const overviewOnly = new Set(this.scene.meshes.filter((m) => m.overview_only).map((m) => m.name));
    const verts = [];   // 10 floats each: px py pz bits | nx ny nz node | u v
    const F = [], U = [];
    // static part: the tracer's own static vertices (node i = vertex i)
    const nSt = S.staticVerts;
    const vb = new ArrayBuffer((nSt) * 40);
    const vf = new Float32Array(vb), vu = new Uint32Array(vb);
    const SV = S.verts, SVU = new Uint32Array(SV.buffer, SV.byteOffset, SV.length);
    for (let i = 0; i < nSt; i++) {
      const o = 10 * i, s = 8 * i;
      vf[o] = SV[s]; vf[o + 1] = SV[s + 1]; vf[o + 2] = SV[s + 2];
      vu[o + 3] = (SVU[s + 3] & 0xffff) | (STATIC_INST << 16);
      vf[o + 4] = SV[s + 4]; vf[o + 5] = SV[s + 5]; vf[o + 6] = SV[s + 6];
      vu[o + 7] = i;
    }
    const idx = [];
    for (let k = 0; k < S.staticTris; k++) { idx.push(S.triVerts[3 * k], S.triVerts[3 * k + 1], S.triVerts[3 * k + 2]); }
    // dynamic bodies: subdivided to <= 3 mm in the body frame (per mesh)
    const dyn = [];
    let nDyn = 0;
    S.blas.forEach((bl, bi) => {
      const byMesh = new Map();
      for (let k = 0; k < bl.triCount; k++) {
        const s = bl.triOff + k, mesh = S.triMesh[s];
        if (overviewOnly.has(S.meshTable[mesh].name)) continue;
        if (!byMesh.has(mesh)) byMesh.set(mesh, []);
        byMesh.get(mesh).push(S.triVerts[3 * s], S.triVerts[3 * s + 1], S.triVerts[3 * s + 2]);
      }
      for (const [mesh, gI] of byMesh) {
        const remap = new Map(), P = [], N = [], UV = [], I = [];
        for (const g of gI) {
          let l = remap.get(g);
          if (l === undefined) {
            l = P.length / 3; remap.set(g, l);
            P.push(SV[8 * g], SV[8 * g + 1], SV[8 * g + 2]); N.push(SV[8 * g + 4], SV[8 * g + 5], SV[8 * g + 6]); UV.push(SV[8 * g + 3], SV[8 * g + 7]);
          }
          I.push(l);
        }
        const sd = sliceTris(P, N, UV, I, bl.kind === 'fixture' ? SUBDIV_CANNULA : SUBDIV_H);
        dyn.push({ mesh, inst: bi, ...sd });
        nDyn += sd.P.length / 3;
      }
    });
    const db = new ArrayBuffer(nDyn * 40), df = new Float32Array(db), du = new Uint32Array(db);
    let base = nSt, o = 0;
    for (const d of dyn) {
      const n = d.P.length / 3;
      for (let v = 0; v < n; v++, o += 10) {
        df[o] = d.P[3 * v]; df[o + 1] = d.P[3 * v + 1]; df[o + 2] = d.P[3 * v + 2];
        du[o + 3] = (d.mesh & 0xffff) | (d.inst << 16);
        df[o + 4] = d.N[3 * v]; df[o + 5] = d.N[3 * v + 1]; df[o + 6] = d.N[3 * v + 2];
        du[o + 7] = 0xffffffff;
        df[o + 8] = d.UV[2 * v]; df[o + 9] = d.UV[2 * v + 1];
      }
      for (const i of d.I) idx.push(i + base);
      base += n;
    }
    this.staticIndexCount = S.staticTris * 3;
    this.indexCount = idx.length;
    const total = nSt + nDyn;
    this.vbuf = dev.createBuffer({ size: total * 40, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: 'lite.verts' });
    dev.queue.writeBuffer(this.vbuf, 0, vb);
    dev.queue.writeBuffer(this.vbuf, nSt * 40, db);
    this.ibuf = dev.createBuffer({ size: Math.ceil(idx.length * 4 / 16) * 16, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, label: 'lite.index' });
    dev.queue.writeBuffer(this.ibuf, 0, Uint32Array.from(idx));
    this.stats.rasterTris = idx.length / 3; this.stats.dynVerts = nDyn;
  }

  /** NS_DYN stratified area samples on the outlet (scope frame), grouped into K_SECT sectors about its centroid. */
  _buildEmitterSamples() {
    const em = this.S.emitter, tris = em.tris;
    const c = em.centroid, nrm = em.normal;
    const t = norm3(cross3(Math.abs(nrm[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], nrm)), b = cross3(nrm, t);
    const smp = [];
    const g = 1.22074408460575947536, a1 = 1 / g, a2 = 1 / (g * g), a3 = 1 / (g * g * g);
    for (let i = 0; i < NS_DYN; i++) {
      const u1 = (0.5 + a1 * i) % 1, u2 = (0.5 + a2 * i) % 1, u3 = (0.5 + a3 * i) % 1;
      let lo = 0, hi = tris.length - 1;
      while (lo < hi) { const m = (lo + hi) >> 1; if (tris[m].cdf < u1) lo = m + 1; else hi = m; }
      const T = tris[lo], su = Math.sqrt(u2);
      const p = [0, 1, 2].map((k) => T.v0[k] + T.e1[k] * su * (1 - u3) + T.e2[k] * su * u3);
      const d = sub3(p, c);
      const ang = Math.atan2(dot3(d, b), dot3(d, t));
      const sect = Math.min(K_SECT - 1, Math.floor(((ang + Math.PI) / (2 * Math.PI)) * K_SECT));
      smp.push({ p, n: T.n, sect });
    }
    this.emSamples = smp;
    // sector centroids (scope frame)
    this.sectC = [...Array(K_SECT)].map((_, k) => {
      const s = smp.filter((x) => x.sect === k);
      if (!s.length) return c;
      return [0, 1, 2].map((j) => s.reduce((acc, x) => acc + x.p[j], 0) / s.length);
    });
    this.emArea = em.area;
    this.esBuf = this.device.createBuffer({ size: NS_DYN * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'lite.es' });
  }

  async _buildPipelines() {
    const dev = this.device;
    const code = SHADER(this.tr.consts);
    const mod = dev.createShaderModule({ code, label: 'lite_nodes' });
    const info = await mod.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error('lite_nodes WGSL: ' + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    const vbl = [{ arrayStride: 40, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'uint32' }, { shaderLocation: 2, offset: 16, format: 'float32x3' }, { shaderLocation: 3, offset: 28, format: 'uint32' }, { shaderLocation: 4, offset: 32, format: 'float32x2' }] }];
    const vs = (entry) => ({ module: mod, entryPoint: entry, buffers: vbl });
    this.bgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    const layout = dev.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    this.pipeCam = await dev.createRenderPipelineAsync({
      layout, vertex: vs('vsCam'),
      fragment: { module: mod, entryPoint: 'fsCam', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: MSAA },
    });
    this.pipeShadow = await dev.createRenderPipelineAsync({
      layout, vertex: vs('vsShadow'),
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 0, depthBiasSlopeScale: 0 },
    });
    // shadow maps: static and dynamic casters, one layer per outlet sector
    const smTex = (label) => dev.createTexture({ size: [SM_RES, SM_RES, K_SECT], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, label });
    this.smStatic = smTex('lite.smStatic'); this.smDyn = smTex('lite.smDyn');
    this.cmpSmp = dev.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    this.shadowViewU = dev.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const fin = dev.createShaderModule({ code: FINISH, label: 'lite.finish' });
    this.pipeFinish = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: fin, entryPoint: 'main' } });
    const dsp = dev.createShaderModule({ code: DISPLAY, label: 'lite.display' });
    this.pipeDisplay = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: dsp, entryPoint: 'main' } });
    const bl = dev.createShaderModule({ code: BLIT, label: 'lite.blit' });
    this._blitFormat = navigator.gpu.getPreferredCanvasFormat();
    this.pipeBlit = await dev.createRenderPipelineAsync({ layout: 'auto', vertex: { module: bl, entryPoint: 'vs' }, fragment: { module: bl, entryPoint: 'fs', targets: [{ format: this._blitFormat }] } });
    // the shadow passes bind 1-texel dummies in place of the shadow maps they render into
    this.smDummy = dev.createTexture({ size: [1, 1, K_SECT], format: 'depth32float', usage: GPUTextureUsage.TEXTURE_BINDING, label: 'lite.smDummy' });
    this._mkGroup = (viewBuf, forShadow = false) => dev.createBindGroup({
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.tr.bufs.params } },
        { binding: 1, resource: { buffer: this.tr.bufs.nodeState } },
        { binding: 2, resource: { buffer: this.tr.bufs.scn } },
        { binding: 3, resource: { buffer: viewBuf } },
        { binding: 4, resource: { buffer: this.esBuf } },
        { binding: 5, resource: (forShadow ? this.smDummy : this.smStatic).createView({ dimension: '2d-array' }) },
        { binding: 6, resource: (forShadow ? this.smDummy : this.smDyn).createView({ dimension: '2d-array' }) },
        { binding: 7, resource: this.cmpSmp },
        { binding: 8, resource: this.tr.boardTex.createView() },
        { binding: 9, resource: this.tr.boardSmp },
      ],
    });
    // hemispherical albedo of the dome (environment radiance for posed bodies)
    const dome = this.S.materials.find((m) => m.name === 'pla_matte_white') || this.S.materials[0];
    this.domeHH = dome.table ? dome.table.hh : [0.9, 0.9, 0.9];
  }

  _setupCameras() {
    this.cams = {};
    for (const c of this.scene.cameras) {
      const R = rowsOf(c.T_D_C);
      const K = c.lens.K, D = c.lens.D || [0, 0, 0, 0];
      this.cams[c.name] = {
        name: c.name, mode: 0, size: c.size || c.lens.size, fx: K[0][0], fy: K[1][1], cx: K[0][2], cy: K[1][2], D,
        R: [[R[0][0], R[0][1], R[0][2]], [R[1][0], R[1][1], R[1][2]], [R[2][0], R[2][1], R[2][2]]], pos: [R[0][3], R[1][3], R[2][3]],
        thetaCut: Math.min(1.45, (c.theta_fold_rad || 1.5) * 0.97), skip: this.tr._skipMask(c.name), circle: 0,
      };
    }
    const sc = this.physics.scope?.camera || {};
    const hfov = ((unwrap(sc.hfov_deg) ?? 75) * Math.PI) / 180;      // physics.yaml scope.camera (assumed); 75 deg default
    const frac = unwrap(sc.image_circle_frac) ?? 0.95;
    const W = 1280, H = 720, f = W / 2 / Math.tan(hfov / 2);
    this.cams.SCOPE = { name: 'SCOPE', mode: 1, size: [W, H], fx: f, fy: f, cx: W / 2, cy: H / 2, D: [0, 0, 0, 0], R: null, pos: null, thetaCut: 1.5, skip: 0, circle: (frac * W) / 2, hfov, frac };
    const scopeI = this.S.blas.findIndex((b) => b.body === 'scope');
    if (scopeI >= 0) this.cams.SCOPE.skip = (1 << scopeI) >>> 0;   // the scope body lies behind its own objective
  }

  _timerSetup() {
    const dev = this.device;
    this.ts = dev.features.has('timestamp-query');
    this.gpuMs = {};
    this.hist = [];
    if (this.ts) {
      this.qs = dev.createQuerySet({ type: 'timestamp', count: 16 });
      this.qres = dev.createBuffer({ size: 16 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.qpool = [];
    }
  }

  // -------------------------------------------------------------------------- poses
  setPoses(poses0, poses1 = null) {
    this.tr.setPoses(poses0, poses1);
    this.poses0 = poses0; this.poses1 = poses1 || poses0;
    this.posesVersion++;
    // scope pose (mid-exposure: the scope rarely moves within an exposure; poses0 is used)
    const T = poses0?.bodies?.scope;
    if (T) {
      const M = rowsOf(T);
      this.scopeR = [[M[0][0], M[0][1], M[0][2]], [M[1][0], M[1][1], M[1][2]], [M[2][0], M[2][1], M[2][2]]];
      this.scopeP = [M[0][3], M[1][3], M[2][3]];
    } else { this.scopeR = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; this.scopeP = [0, 0, 0]; }
    const R = this.scopeR, P = this.scopeP;
    const W = (v) => mulRv(R, v).map((x, k) => x + P[k]);
    // emitter samples (world)
    const es = new Float32Array(NS_DYN * 8);
    const wArea = this.emArea / 1;          // per-sample weight is A; the shader divides by the sample count
    this.emSamples.forEach((s, i) => { es.set([...W(s.p), s.sect, ...mulRv(R, s.n), wArea], 8 * i); });
    this.device.queue.writeBuffer(this.esBuf, 0, es);
    this.axisW = norm3(mulRv(R, this.S.emitter.axis));
    const t = norm3(cross3(Math.abs(this.axisW[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], this.axisW));
    this.lt = t; this.lb = cross3(this.axisW, t);
    this.lo = this.sectC.map((c) => W(c));
    // the scope camera follows the scope body
    const sc = this.scene.scope;
    if (sc) {
      const z = norm3(mulRv(R, sc.view_dir_K || [0, 0.5, 0.866]));
      const upK = mulRv(R, [0, 1, 0]);                 // the tilt side (+y_K) is image up
      let up = sub3(upK, z.map((x) => x * dot3(upK, z)));
      up = norm3(up);
      const y = up.map((x) => -x), x = cross3(y, z);
      this.cams.SCOPE.R = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
      this.cams.SCOPE.pos = W(sc.objective_centre_K || [0, 0, 0]);
    }
    const key = JSON.stringify([this.scopeR, this.scopeP]);
    if (key !== this._smScopeKey) { this._smScopeKey = key; this._smStaticDirty = true; }
  }

  _viewData(cam, casters = 0) {
    const f = new Float32Array(128), u = new Uint32Array(f.buffer);
    const R = cam.R || [[1, 0, 0], [0, 1, 0], [0, 0, 1]], p = cam.pos || [0, 0, 0];
    f.set([R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0, R[0][2], R[1][2], R[2][2], 0], 0);
    f.set([...p, cam.mode], 12);
    f.set([cam.fx, cam.fy, cam.cx, cam.cy], 16);
    f.set(cam.D.slice(0, 4), 20);
    f.set([cam.size[0], cam.size[1], cam.thetaCut, this._outScale()], 24);
    u[28] = cam.skip >>> 0; u[29] = NS_STATIC; u[30] = NS_DYN; u[31] = casters;
    const eA = this.tr.eAvg || [0, 0, 0];
    f.set([eA[0] * this.domeHH[0] / Math.PI, eA[1] * this.domeHH[1] / Math.PI, eA[2] * this.domeHH[2] / Math.PI, 0.5], 32);
    f.set([...this.axisW, this.debug || 0], 36);
    f.set([...this.lt, SM_THETA_MAX], 40);
    f.set([...this.lb, FAR], 44);
    this.lo.forEach((o, k) => f.set([...o, 2e-4], 48 + 4 * k));
    return f;
  }

  _outScale() { const L0 = this.S.emitter.L0; return 1000 / Math.max(L0[0], L0[1], L0[2], 1e-30); }

  _targets(name) {
    const cam = this.cams[name];
    if (cam.t) return cam.t;
    const [w, h] = cam.size, dev = this.device;
    const t = {
      msaa: dev.createTexture({ size: [w, h], sampleCount: MSAA, format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT }),
      depth: dev.createTexture({ size: [w, h], sampleCount: MSAA, format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT }),
      res: dev.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
      tex: dev.createTexture({ size: [w, h], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC, label: `${name}.radiance` }),
      out: dev.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC, label: `${name}.display` }),
      viewU: dev.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      finU: dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      dispU: dev.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      stats: [0, 1].map(() => dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })),
      flip: 0,
    };
    t.group = this._mkGroup(t.viewU);
    t.finG = dev.createBindGroup({ layout: this.pipeFinish.getBindGroupLayout(0), entries: [{ binding: 0, resource: t.res.createView() }, { binding: 1, resource: t.tex.createView() }, { binding: 2, resource: { buffer: t.finU } }] });
    cam.t = t;
    return t;
  }

  /** GPU ms of the K shadow passes (which: 1 static casters, 2 dynamic), from timestamp queries. */
  async shadowGpuMs(which = 2) {
    if (!this.ts) return null;
    const dev = this.device;
    const qs = (this._sqs ||= dev.createQuerySet({ type: 'timestamp', count: 2 }));
    const rs = (this._srs ||= dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }));
    const rb = dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    this._shadowPasses(enc, which, qs);
    enc.resolveQuerySet(qs, 0, 2, rs, 0); enc.copyBufferToBuffer(rs, 0, rb, 0, 16);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const q = new BigUint64Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    return Number(q[1] - q[0]) / 1e6;
  }

  _shadowPasses(enc, which, qs = null) {
    const cam = { R: null, pos: null, mode: 2, fx: 0, fy: 0, cx: 0, cy: 0, D: [0, 0, 0, 0], size: [SM_RES, SM_RES], thetaCut: 1.5, skip: 0 };
    const tex = which === 1 ? this.smStatic : this.smDyn;
    const buf = which === 1 ? (this._svU1 ||= this.device.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })) : (this._svU2 ||= this.device.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    const grp = which === 1 ? (this._svG1 ||= this._mkGroup(buf, true)) : (this._svG2 ||= this._mkGroup(buf, true));
    this.device.queue.writeBuffer(buf, 0, this._viewData(cam, which));
    for (let k = 0; k < K_SECT; k++) {
      const tw = qs ? { querySet: qs, ...(k === 0 ? { beginningOfPassWriteIndex: 0 } : {}), ...(k === K_SECT - 1 ? { endOfPassWriteIndex: 1 } : {}) } : undefined;
      const rp = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: tex.createView({ dimension: '2d', baseArrayLayer: k, arrayLayerCount: 1 }), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' }, ...(tw ? { timestampWrites: tw } : {}) });
      rp.setPipeline(this.pipeShadow); rp.setBindGroup(0, grp);
      rp.setVertexBuffer(0, this.vbuf); rp.setIndexBuffer(this.ibuf, 'uint32');
      if (which === 1) rp.drawIndexed(this.staticIndexCount, 1, 0, 0, k);
      else rp.drawIndexed(this.indexCount - this.staticIndexCount, 1, this.staticIndexCount, 0, k);
      rp.end();
    }
  }

  _tsWrites(i) { return this.ts ? { querySet: this.qs, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 } : undefined; }

  /** scene-referred linear radiance -> rgba32float texture (the §8 contract). */
  async renderCamera(name, opts = {}) {
    const cam = this.cams[name];
    if (!cam) throw new Error(`unknown camera ${name}`);
    const dev = this.device, t = this._targets(name);
    // node radiosity for the current poses (incremental, the tracer's own scheduling)
    this.tr._ensureRadiosity();
    const enc = dev.createCommandEncoder();
    const marks = [];
    if (this._smStaticDirty) { this._shadowPasses(enc, 1); this._smStaticDirty = false; marks.push('smStatic'); }
    if (this._smDynVersion !== this.posesVersion) { this._shadowPasses(enc, 2); this._smDynVersion = this.posesVersion; marks.push('smDyn'); }
    dev.queue.writeBuffer(t.viewU, 0, this._viewData(cam, 0));
    const [w, h] = cam.size;
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: t.msaa.createView(), resolveTarget: t.res.createView(), loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 1] }],
      depthStencilAttachment: { view: t.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      timestampWrites: this._tsWrites(0),
    });
    rp.setPipeline(this.pipeCam); rp.setBindGroup(0, t.group);
    rp.setVertexBuffer(0, this.vbuf); rp.setIndexBuffer(this.ibuf, 'uint32');
    rp.drawIndexed(this.indexCount);
    rp.end();
    dev.queue.writeBuffer(t.finU, 0, new Float32Array([1 / this._outScale(), cam.circle, w, h]));
    const cp = enc.beginComputePass({ timestampWrites: this._tsWrites(2) });
    cp.setPipeline(this.pipeFinish); cp.setBindGroup(0, t.finG); cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); cp.end();
    this._tsResolve(enc, name, 4);
    dev.queue.submit([enc.finish()]);
    this.lastCamera = name;
    if (opts.wait !== false) await dev.queue.onSubmittedWorkDone();
    return t.tex;
  }

  _tsResolve(enc, name, n) {
    if (!this.ts) return;
    const rb = this.qpool.pop() || this.device.createBuffer({ size: 16 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.resolveQuerySet(this.qs, 0, n, this.qres, 0);
    enc.copyBufferToBuffer(this.qres, 0, rb, 0, n * 8);
    const sub = () => rb.mapAsync(GPUMapMode.READ).then(() => {
      const q = new BigUint64Array(rb.getMappedRange().slice(0));
      rb.unmap(); this.qpool.push(rb);
      const main = Number(q[1] - q[0]) / 1e6, fin = Number(q[3] - q[2]) / 1e6, tot = Number(q[3] - q[0]) / 1e6;
      const rec = { camera: name, main_ms: main, finish_ms: fin, total_ms: tot };
      this.gpuMs[name] = rec; this.hist.push(rec); if (this.hist.length > 400) this.hist.shift();
    }).catch(() => {});
    queueMicrotask(() => {});
    this._pendingTs = (this._pendingTs || Promise.resolve()).then(() => new Promise((res) => setTimeout(res, 0))).then(sub);
  }

  /** Display look (§18.1) -> rgba8unorm GPUTexture. */
  sensor(name, opts = {}) {
    const cam = this.cams[name], t = this._targets(name), dev = this.device;
    const sp = this.physics.sensor || {}, isp = this.physics.isp || {};
    const c = unwrap(isp.s_curve_c) ?? 1.39, b = unwrap(isp.black_offset_linear) ?? 0.014;
    const k = unwrap(sp.lens?.veiling_glare?.fraction) ?? 0.0;
    const target = unwrap(sp.auto_exposure?.target_mean) ?? 0.18;
    // auto exposure from the previous frame's mean (UVC-loop-like, one frame late)
    const prev = t.stats[t.flip], cur = t.stats[1 - t.flip];
    let gain = opts.gain;
    if (gain == null) {
      const m = t.meanY || 0;
      gain = m > 0 ? target / m : 1 / (this._outScale() > 0 ? 5e-3 * Math.max(...this.S.emitter.L0) : 1);
      if (opts.exposure_s != null && this._refExposure) gain *= opts.exposure_s / this._refExposure;
    }
    const wb = unwrap(isp.white_balance_gains) || [1, 1, 1];
    const lens = cam.mode === 0 ? [cam.fx, cam.fy, cam.cx, cam.cy] : [0, 0, 0, 0];
    const scaleY = 4096 / Math.max(1e-30, (t.meanY || 5e-3 * Math.max(...this.S.emitter.L0)) * 4);
    dev.queue.writeBuffer(t.dispU, 0, new Float32Array([gain, k, 0, scaleY, ...wb.slice(0, 3), 0, c, b, 0, 0, ...lens, 0, 0, 0, 0]));
    dev.queue.writeBuffer(cur, 0, new Uint32Array([0, 0, 0, 0]));
    const g = dev.createBindGroup({ layout: this.pipeDisplay.getBindGroupLayout(0), entries: [{ binding: 0, resource: t.tex.createView() }, { binding: 1, resource: t.out.createView() }, { binding: 2, resource: { buffer: t.dispU } }, { binding: 3, resource: { buffer: cur } }, { binding: 4, resource: { buffer: prev } }] });
    const enc = dev.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(this.pipeDisplay); cp.setBindGroup(0, g);
    const [w, h] = cam.size;
    cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); cp.end();
    // read the mean back asynchronously for the next frame's AE
    const rb = dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(cur, 0, rb, 0, 16);
    dev.queue.submit([enc.finish()]);
    rb.mapAsync(GPUMapMode.READ).then(() => { const v = new Uint32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy(); if (v[1]) t.meanY = (v[0] / v[1]) / scaleY; }).catch(() => {});
    t.flip = 1 - t.flip;
    return t.out;
  }

  /** Draw a camera's display texture into a WebGPU canvas context (letterboxed), no CPU readback. */
  present(name, ctx) {
    const dev = this.device, t = this._targets(name);
    if (!ctx._liteConfigured) { ctx.configure({ device: dev, format: this._blitFormat, alphaMode: 'opaque' }); ctx._liteConfigured = true; }
    const cv = ctx.canvas;
    const u = (this._blitU ||= dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    dev.queue.writeBuffer(u, 0, new Float32Array([cv.width, cv.height, 0, 0]));
    const g = dev.createBindGroup({ layout: this.pipeBlit.getBindGroupLayout(0), entries: [{ binding: 0, resource: t.out.createView() }, { binding: 1, resource: { buffer: u } }] });
    const enc = dev.createCommandEncoder();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0.04, 0.04, 0.045, 1] }] });
    rp.setPipeline(this.pipeBlit); rp.setBindGroup(0, g); rp.draw(3); rp.end();
    dev.queue.submit([enc.finish()]);
  }

  async readback(name, kind = 'rgb8') {
    if (kind === 'yuv420' || kind === 'raw') return null;     // no ISP planes here: app.js then encodes the JPEG from rgb8
    const cam = this.cams[name], t = this._targets(name), dev = this.device;
    const [w, h] = cam.size;
    const tex = kind === 'linear' ? t.tex : t.out, bpp = kind === 'linear' ? 16 : 4;
    const bpr = Math.ceil((w * bpp) / 256) * 256;
    const st = dev.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: st, bytesPerRow: bpr }, [w, h]);
    dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
    if (kind === 'linear') {
      const out = new Float32Array(w * h * 4);
      for (let y = 0; y < h; y++) out.set(new Float32Array(data.buffer, y * bpr, w * 4), y * w * 4);
      return out;
    }
    const ch = kind === 'rgba8' ? 4 : 3;
    const out = new Uint8Array(w * h * ch);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * bpr + 4 * x, j = ch * (y * w + x);
      out[j] = data[i]; out[j + 1] = data[i + 1]; out[j + 2] = data[i + 2]; if (ch === 4) out[j + 3] = 255;
    }
    return out;
  }

  visible(name, pointsD, o) {
    if (name !== 'SCOPE') return this.tr.visible(name, pointsD, o);
    const c = this.cams.SCOPE;
    return pointsD.map((p) => {
      const d = sub3(p, c.pos), z = c.R[0][2] * d[0] + c.R[1][2] * d[1] + c.R[2][2] * d[2];
      if (z <= 0) return false;
      const x = c.R[0][0] * d[0] + c.R[1][0] * d[1] + c.R[2][0] * d[2], y = c.R[0][1] * d[0] + c.R[1][1] * d[1] + c.R[2][1] * d[2];
      const u = c.fx * x / z, v = c.fy * y / z;
      return Math.hypot(u, v) < c.circle;
    });
  }

  /** Full radiosity solve for the current poses (judge mode). */
  async converge(iters = 200) { return this.tr.converge(iters); }

  timings() {
    const per = {};
    for (const r of this.hist) (per[r.camera] ||= []).push(r.total_ms);
    const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
    return { last: this.gpuMs, median_total_ms: Object.fromEntries(Object.entries(per).map(([k, v]) => [k, med(v)])), n: this.hist.length, load_ms: this.stats.loadMs, stats: this.stats, tracer: this.tr.timings?.() };
  }

  async idle() { await this.device.queue.onSubmittedWorkDone(); await this._pendingTs; }
}

export default LiteNodesRenderer;
