// lite_rt: hybrid raster + ray-traced direct light (docs/WEB_TWIN.md §18.1 candidate; owner: the lite_rt agent).
//
// Per visible view, one submit with two passes:
//   1. raster: every triangle of the path tracer's own GPU triangle buffer (static scene + each posed body BLAS, posed
//      by the same instance transforms), projected per vertex with the camera's exact KB4 lens (CAM1-3) or the scope
//      pinhole (SCOPE), into a visibility buffer (triangle id, instance id) + depth;
//   2. shade (compute): per pixel, the exact KB4/pinhole ray is intersected with the rastered triangle's plane (so
//      shading positions are exact, raster only decides coverage) and handed to the path tracer's own shadeHit
//      (gpu/wgsl/shade.wgsl.js, imported read-only): area-emitter L0 p(theta) with 1-2 shadow rays against the BVH,
//      the §8 BSDF, static indirect from the node radiosity (static vertices = nodes) with the capsule occlusion, and
//      for the instruments one BSDF-sampled ray that reads the node radiosity (MIS with the light).
// Temporal accumulation while nothing moves (poses + camera unchanged): a QMC sequence over frames for the light
// samples, the sub-pixel jitter (box-integrated pixels) and the shutter time, in a running mean capped at ACC_CAP.
// The node radiosity is the path tracer's own solve (a TwinRenderer instance used for its scene buffers, BVH and
// node transport only; it never traces a camera here).
import { TwinRenderer } from '../gpu/renderer.js';
import { shaderSource } from '../gpu/wgsl/kernels.wgsl.js';
import { defines, COMMON } from '../gpu/wgsl/common.wgsl.js';
import { LENS } from '../gpu/wgsl/lens.wgsl.js';
import { MESH_FLAG } from '../gpu/scene_gpu.js';

const MISS = 0xffffffff;
const R2A = 0.7548776662466927, R2B = 0.5698402909980532, GOLD = 0.6180339887498949;
const fract = (x) => x - Math.floor(x);
const rowMajor4 = (T) => (Array.isArray(T[0]) ? T : [[T[0], T[4], T[8], T[12]], [T[1], T[5], T[9], T[13]], [T[2], T[6], T[10], T[14]], [T[3], T[7], T[11], T[15]]]);
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const mulRv = (T, v) => [T[0][0] * v[0] + T[0][1] * v[1] + T[0][2] * v[2], T[1][0] * v[0] + T[1][1] * v[1] + T[1][2] * v[2], T[2][0] * v[0] + T[2][1] * v[1] + T[2][2] * v[2]];

const RASTER_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> scn: SceneU;
@group(0) @binding(1) var<storage, read> tris: array<vec4f>;
@group(0) @binding(2) var<uniform> rc: RasterU;
@group(0) @binding(3) var<storage, read> subv: array<vec4f>;   // sub-triangles of long triangles: body-frame xyz, parent tri (bits)
struct RasterU {
  rx: vec4f, ry: vec4f, rz: vec4f,
  pos: vec4f,            // centre; w = mode (0 kb4, 1 pinhole)
  lensA: vec4f, lensB: vec4f,
  lensC: vec4f,          // alpha, theta max (kb4), far (m), -
  sz: vec4f,             // W, H, jitter x, jitter y (pixels)
  t: vec4f,              // time, skip mask (bits), long-edge limit for bodies (m), for static triangles (m)
}
struct VO { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) id: vec2u }

@vertex
fn vsMain(@builtin(vertex_index) vid: u32, @builtin(instance_index) ii: u32) -> VO {
  let k = vid / 3u;
  let c = vid % 3u;
  let e1 = tris[4u * k + 1u].xyz;
  let e2 = tris[4u * k + 2u].xyz;
  let e3 = e2 - e1;
  let lim = select(rc.t.z, rc.t.w, ii == 0u);      // static triangles: their own limit (the scene is already <= 3 mm)
  let L2 = lim * lim;
  if (max(dot(e1, e1), max(dot(e2, e2), dot(e3, e3))) > L2) {
    var o: VO; o.pos = vec4f(4.0, 4.0, 0.5, 1.0); o.id = vec2u(0u, 0u); return o;    // drawn by vsSub instead
  }
  var p = tris[4u * k].xyz;
  if (c == 1u) { p = p + e1; } else if (c == 2u) { p = p + e2; }
  return project(p, k, ii);
}

@vertex
fn vsSub(@builtin(vertex_index) vid: u32, @builtin(instance_index) ii: u32) -> VO {
  let s = subv[vid];
  return project(s.xyz, bitcast<u32>(s.w), ii);
}

fn project(p0: vec3f, k: u32, ii: u32) -> VO {
  var out: VO;
  var p = p0;
  var inst = 0xffffffffu;
  if (ii > 0u) {
    inst = ii - 1u;
    let I = scn.inst[inst];
    if ((bitcast<u32>(I.p1.w) & 2u) != 0u || ((bitcast<u32>(rc.t.y) >> inst) & 1u) != 0u) {
      out.pos = vec4f(4.0, 4.0, 0.5, 1.0); out.id = vec2u(0u, 0u); return out;
    }
    let P = instPose(inst, rc.t.x);
    p = P.r * p + P.p;
  }
  let q = p - rc.pos.xyz;
  let X = vec3f(dot(q, rc.rx.xyz), dot(q, rc.ry.xyz), dot(q, rc.rz.xyz));
  let dist = length(X);
  let off = vec2f(0.5) - rc.sz.zw;
  if (rc.pos.w < 0.5) {
    let th = atan2(length(X.xy), X.z);
    let uv = kb4Project(Lens(rc.lensA, rc.lensB, rc.lensC), X) + off;
    let ok = th < rc.lensC.y;
    out.pos = vec4f(2.0 * uv.x / rc.sz.x - 1.0, 1.0 - 2.0 * uv.y / rc.sz.y, select(-1.0, min(dist / rc.lensC.z, 1.0), ok), 1.0);
  } else {
    // homogeneous pinhole, so triangles that cross the camera plane clip correctly
    let w = X.z;
    let fx = rc.lensA.x * X.x + (rc.lensA.z + off.x) * w;
    let fy = rc.lensA.y * X.y + (rc.lensA.w + off.y) * w;
    out.pos = vec4f(2.0 * fx / rc.sz.x - w, w - 2.0 * fy / rc.sz.y, min(dist / rc.lensC.z, 1.0) * w, w);
  }
  out.id = vec2u(k + 1u, inst);
  return out;
}

@fragment
fn fsMain(v: VO) -> @location(0) vec2u { return v.id; }
`;

const SHADE_MAIN = /* wgsl */ `
struct LtU {
  jit: vec4f,            // jitter x, y (pixels), image-circle radius (px, 0 = none), shutter time
  info: vec4u,           // mode (0 still, 1 reset, 2 tools moved), sample index, accumulation cap, history cap when tools move
}
@group(2) @binding(0) var visTex: texture_2d<u32>;
@group(2) @binding(1) var<uniform> lt: LtU;
@group(2) @binding(2) var<storage, read_write> pid: array<vec2u>;   // last frame's (tri + 1, instance) per pixel

@compute @workgroup_size(8, 8)
fn ltShade(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  gRoot = bitcast<u32>(cam.lensC.y);
  let W = cam.size.x;
  let H = cam.size.y;
  let px = gid.xy;
  if (px.x >= W || px.y >= H) { return; }
  let idx = px.y * W + px.x;
  let vis = textureLoad(visTex, vec2i(px), 0).xy;
  let uv = vec2f(px) + lt.jit.xy;
  var inside = true;
  if (lt.jit.z > 0.0) { let dd = uv - cam.lensA.zw; inside = dot(dd, dd) <= lt.jit.z * lt.jit.z; }
  var c = vec3f(0.0);
  if (vis.x != 0u && inside) {
    var dc: vec3f;
    if (cam.pos.w < 0.5) { dc = kb4Unproject(Lens(cam.lensA, cam.lensB, cam.lensC), uv); }
    else { dc = normalize(vec3f((uv.x - cam.lensA.z) / cam.lensA.x, (uv.y - cam.lensA.w) / cam.lensA.y, 1.0)); }
    let d = normalize(cam.rx.xyz * dc.x + cam.ry.xyz * dc.y + cam.rz.xyz * dc.z);
    let o = cam.pos.xyz;
    let k = vis.x - 1u;
    let inst = vis.y;
    let time = lt.jit.w;
    var ol = o;
    var dl = d;
    if (inst != MISS) { let P = instPose(inst, time); ol = (o - P.p) * P.r; dl = d * P.r; }
    // the pixel ray against the rastered triangle's plane; barycentrics clamped to the triangle (coverage is raster's)
    let v0 = tris[4u * k].xyz;
    let e1 = tris[4u * k + 1u].xyz;
    let e2 = tris[4u * k + 2u].xyz;
    let pp = cross(dl, e2);
    let det = dot(e1, pp);
    let inv = 1.0 / select(det, 1e-30, abs(det) < 1e-30);
    let s = ol - v0;
    var u = dot(s, pp) * inv;
    let qq = cross(s, e1);
    var v = dot(dl, qq) * inv;
    var t = dot(e2, qq) * inv;
    u = clamp(u, 0.0, 1.0);
    v = clamp(v, 0.0, 1.0 - u);
    if (!(t > 0.0) || abs(det) < 1e-30) {
      // grazing: the clamped barycentric point instead
      let pw = v0 + e1 * u + e2 * v;
      t = max(dot(pw - ol, dl), 1e-6);
    }
    let h = Hit(t, k, u, v, inst);
    let S = pixSeeds(px.x, px.y);
    let nl = max(1u, bitcast<u32>(cam.lensC.z));
    let i = lt.info.y;
    let ul = vec3f(kr(i * nl, KL0, S.s3), kr(i * nl, KL1, S.s4), kr(i * nl, KL2, S.s5));
    let ub = vec3f(kr(i, KB0, S.s6), kr(i, KB1, S.s7), kr(i, KB2, S.s8));
    c = shadeHit(h, o, d, time, ul, ub, cam.flags.y, i * nl, S);
    if (!(c.x >= 0.0 && c.y >= 0.0 && c.z >= 0.0 && c.x < 1e30 && c.y < 1e30 && c.z < 1e30)) { c = vec3f(0.0); }
  }
  var a = acc1[idx];
  // history: dropped on a full reset, when the pixel shows another surface than last frame, and on posed bodies
  // while tools move; static pixels keep at most info.w frames of history while tools move (soft shadows settle
  // in a few frames, a moving shadow lags by about that much)
  let prev = pid[idx];
  pid[idx] = vis;
  if (lt.info.x == 1u) { a = vec4f(0.0); }
  else if (lt.info.x == 2u) {
    if (vis.y != MISS || any(prev != vis)) { a = vec4f(0.0); } else { a.w = min(a.w, f32(max(lt.info.w, 1u)) - 1.0); }
  }
  let n = min(a.w + 1.0, f32(max(lt.info.z, 1u)));
  a = vec4f(a.xyz + (c - a.xyz) / n, n);
  acc1[idx] = a;
  textureStore(outTex, vec2i(px), vec4f(a.xyz * cam.expo.z, select(0.0, 1.0, inside)));
}
`;

// display: linear radiance -> screen (§18.1 light sensor stage, no Bayer/demosaic)
const METER_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> meter: array<vec4f, 2>;   // [0]: mean rgb (glare), n; [1]: gain, target, damping, init
var<workgroup> wsum: array<vec4f, 256>;
@compute @workgroup_size(256)
fn meterMain(@builtin(local_invocation_index) lid: u32) {
  let sz = textureDimensions(src);
  var s = vec4f(0.0);
  // 64 x 36 grid, 9 samples per invocation
  for (var j = 0u; j < 9u; j++) {
    let g = lid * 9u + j;
    let gx = g % 64u;
    let gy = g / 64u;
    let p = vec2u((gx * sz.x) / 64u + sz.x / 128u, (gy * sz.y) / 36u + sz.y / 72u);
    let c = textureLoad(src, vec2i(p), 0);
    if (c.w > 0.5) { s = s + vec4f(c.xyz, 1.0); }
  }
  wsum[lid] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st / 2u) {
    if (lid < st) { wsum[lid] = wsum[lid] + wsum[lid + st]; }
    workgroupBarrier();
  }
  if (lid == 0u) {
    let m = wsum[0].xyz / max(wsum[0].w, 1.0);
    let y = max(dot(m, vec3f(0.2126, 0.7152, 0.0722)), 1e-12);
    meter[0] = vec4f(m, wsum[0].w);
    let want = meter[1].y / y;
    var gain = meter[1].x;
    if (meter[1].w < 0.5 || !(gain > 0.0)) { gain = want; } else { gain = gain * pow(want / gain, meter[1].z); }
    meter[1] = vec4f(gain, meter[1].y, meter[1].z, 1.0);
  }
}
`;

const PRESENT_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> meter: array<vec4f, 2>;
@group(0) @binding(2) var<uniform> pu: PU;
@group(0) @binding(3) var ri: texture_2d<f32>;
struct PU { rect: vec4f, tone: vec4f, wb: vec4f, bg: vec4f }   // rect: x0 y0 w h (canvas px); tone: c, black, glare k, srgb flag
struct VO { @builtin(position) pos: vec4f }
@vertex fn vsFull(@builtin(vertex_index) i: u32) -> VO {
  let p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return VO(vec4f(p[i], 0.0, 1.0));
}
fn oetf(x: f32) -> f32 { return select(1.055 * pow(x, 1.0 / 2.4) - 0.055, 12.92 * x, x <= 0.0031308); }
fn scurve(u: f32, c: f32) -> f32 { let a = pow(u, c); return a / (a + pow(1.0 - u, c)); }
@fragment fn fsPresent(v: VO) -> @location(0) vec4f {
  let q = (v.pos.xy - pu.rect.xy) / pu.rect.zw;
  if (any(q < vec2f(0.0)) || any(q >= vec2f(1.0))) { return pu.bg; }
  let sz = textureDimensions(src);
  let p = vec2i(q * vec2f(sz));
  let c = textureLoad(src, p, 0);
  if (c.w < 0.5) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let r = textureLoad(ri, vec2i(q * vec2f(textureDimensions(ri))), 0).x;
  let k = pu.tone.z;
  let E = ((1.0 - k) * c.xyz + k * meter[0].xyz) * r * meter[1].x * pu.wb.xyz;
  let b = pu.tone.y;
  var o = vec3f(0.0);
  for (var j = 0; j < 3; j++) {
    let u = clamp((E[j] - b) / (1.0 - b), 0.0, 1.0);
    o[j] = scurve(oetf(u), pu.tone.x);
  }
  return vec4f(o, 1.0);
}
`;

const pval = (x, d) => {
  if (x === undefined || x === null) return d;
  if (typeof x === 'object' && 'value' in x) return x.value;
  return x;
};

export class LiteRTRenderer {
  /** opts: {scene, physics, device?, log?, lightSamples (1-2), accumulate (bool), accCap, cullShadows} */
  static async create(opts) {
    const r = new LiteRTRenderer();
    await r._init(opts);
    return r;
  }

  async _init(opts) {
    const t0 = performance.now();
    this.log = opts.log || ((...a) => console.log('[lite_rt]', ...a));
    this.opts = { lightSamples: 1, accumulate: true, accCap: 64, settleIters: 2, settleFrames: 24, ...opts };
    // the path tracer, for its scene buffers, BVH, materials, emitter and node radiosity (no camera traced by it)
    this.tr = await TwinRenderer.create({ ...opts, log: this.log, layerLines: false, preset: opts.preset === 'judge' ? undefined : opts.preset });
    this.device = this.tr.device;
    this.scene = this.tr.scene;
    this.physics = this.tr.physics;
    this.adapterInfo = this.tr.adapterInfo;
    this.stats = { trInitMs: performance.now() - t0 };
    this._addScope();
    this.cameras = {};
    for (const [n, c] of Object.entries(this.tr.cameras)) this.cameras[n] = c;
    this.views = {};
    await this._buildPipelines();
    this.frameNo = 1000000;
    this.stats.initMs = performance.now() - t0;
    this.log(`ready in ${this.stats.initMs.toFixed(0)} ms (path-tracer scene + node solve setup ${this.stats.trInitMs.toFixed(0)} ms)`);
  }

  _addScope() {
    const sc = this.scene.scope || {};
    const ph = (this.physics.scope && this.physics.scope.camera) || {};
    const hfov = Number(pval(ph.hfov_deg, 75));
    const circ = Number(pval(ph.image_circle_frac, 0.95));
    const W = 1280, H = 720;
    const fx = (W / 2) / Math.tan((hfov * Math.PI) / 360);
    this.scopeCam = { hfov, circ, W, H, fx, body: sc.body || 'scope', c: sc.objective_centre_K || [0, 0, 0], v: sc.view_dir_K || [0, 0.5, 0.866] };
    const T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    this.tr.cameras.SCOPE = {
      index: Object.keys(this.tr.cameras).length, meta: { name: 'SCOPE', size: [W, H] },
      lens: { fx, fy: fx, cx: (W - 1) / 2, cy: (H - 1) / 2, k: [0, 0, 0, 0], alpha: 0 }, size: [W, H], T, pinhole: true,
      circleR: 0.5 * circ * W,
    };
  }

  /** SCOPE follows the scope body: centre objective_centre_K, axis view_dir_K, image up toward the 30° tilt. */
  _updateScope() {
    const Tk = this.tr.poses0?.bodies?.[this.scopeCam.body];
    if (!Tk) return;
    const T = rowMajor4(Tk);
    const sc = this.scopeCam;
    const c = mulRv(T, sc.c).map((x, i) => x + T[i][3]);
    const z = norm3(mulRv(T, sc.v));
    // shaft axis +z_K; the view tilts toward +y_K: image up = the tilt side
    const yk = mulRv(T, [0, 1, 0]);
    const upw = norm3(yk.map((x, i) => x - dot3(yk, z) * z[i]));
    const y = upw.map((x) => -x);              // OpenCV y is down
    const x = cross3(y, z);
    this.tr.cameras.SCOPE.T = [[x[0], y[0], z[0], c[0]], [x[1], y[1], z[1], c[1]], [x[2], y[2], z[2], c[2]], [0, 0, 0, 1]];
  }

  async _buildPipelines() {
    const dev = this.device, tr = this.tr;
    const shadeCode = shaderSource('camera', tr.consts) + SHADE_MAIN;
    const shadeMod = await tr._module(shadeCode, 'lite_rt.shade');
    this.layoutG2 = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'uint' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
    const shadeLayout = dev.createPipelineLayout({ bindGroupLayouts: [tr.layouts.sceneRO, tr.layouts.camG1, this.layoutG2] });
    const cull = this.opts.cullShadows ?? tr.opts.cullShadows;
    this.shadePipe = await dev.createComputePipelineAsync({ layout: shadeLayout, compute: { module: shadeMod, entryPoint: 'ltShade', constants: { CULL_SHADOWS: cull ? 1 : 0 } } });

    const rasterCode = defines(tr.consts) + COMMON + LENS + RASTER_WGSL;
    const rasterMod = await tr._module(rasterCode, 'lite_rt.raster');
    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    this.layoutR = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 3, visibility: VF, buffer: { type: 'read-only-storage' } }] });
    const rdesc = (entryPoint) => ({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.layoutR] }),
      vertex: { module: rasterMod, entryPoint },
      fragment: { module: rasterMod, entryPoint: 'fsMain', targets: [{ format: 'rg32uint' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.rasterPipe = await dev.createRenderPipelineAsync(rdesc('vsMain'));
    this.rasterSubPipe = await dev.createRenderPipelineAsync(rdesc('vsSub'));
    this._buildSubdiv();
    const meterMod = await tr._module(METER_WGSL, 'lite_rt.meter');
    this.meterPipe = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: meterMod, entryPoint: 'meterMain' } });
    this.presentMod = await tr._module(PRESENT_WGSL, 'lite_rt.present');
    this.presentPipes = {};
    // draw list: static triangles (instance 0) and each body BLAS (instance i + 1)
    const S = tr.S;
    this.draws = [tr.interiorRoot && tr.stats.interiorTris ? { first: S.triCount, count: tr.stats.interiorTris, inst: 0 } : { first: 0, count: S.staticTris, inst: 0 }];
    S.blas.forEach((b, i) => { if (b.triCount > 0) this.draws.push({ first: b.triOff, count: b.triCount, inst: i + 1, body: b.body }); });
    this.stats.draws = this.draws.length;
    this.stats.triangles = S.triCount;
  }

  /**
   * Long triangles (instrument shafts, cannulae: edges up to ~0.3 m) cannot be rastered with per-vertex KB4 (the
   * fisheye bends straight edges). They are split by longest-edge bisection until every edge is <= longEdge, in their
   * own body frame, keeping the parent triangle id: the shade pass still intersects the exact parent triangle.
   */
  _buildSubdiv() {
    const t0 = performance.now();
    const S = this.tr.S, T = S.tris;
    const L = this.opts.longEdge ?? 0.003;
    this.longEdge = L;
    const L2b = (L * 0.995) ** 2;
    // the static scene is pre-split to <= 3 mm (scene_gpu nodeSpacing): a hair above that, nothing static is split again
    const Ls = this.opts.longEdgeStatic ?? Math.max(L, 1.2 * (this.tr.opts.nodeSpacing || 0.003));
    this.longEdgeStatic = Ls;
    let buf = new Float32Array(1 << 18), bu = new Uint32Array(buf.buffer), n = 0;
    const emit = (a, id) => {
      if (n + 4 > buf.length) { const nb = new Float32Array(buf.length * 2); nb.set(buf); buf = nb; bu = new Uint32Array(buf.buffer); }
      buf[n] = a[0]; buf[n + 1] = a[1]; buf[n + 2] = a[2]; bu[n + 3] = id; n += 4;
    };
    const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
    const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
    const ranges = [{ first: 0, count: S.staticTris, inst: 0 }];
    // the tracer's interior criterion (gpu/renderer.js _buildInteriorBVH): exterior static triangles never reach a feed
    const tr = this.tr, mis = tr.nodeMisses, K = tr.opts.gatherK;
    const interior = (k) => {
      if (!tr.interiorRoot || !mis) return true;
      if (S.meshTable[S.triMesh[k]].flags & MESH_FLAG.BENCH) return true;
      const a = S.triVerts[3 * k], b = S.triVerts[3 * k + 1], c = S.triVerts[3 * k + 2];
      return !(mis[a] >= 0.9 * K && mis[b] >= 0.9 * K && mis[c] >= 0.9 * K);
    };
    let nStaticLong = 0;
    S.blas.forEach((b, i) => { if (b.triCount > 0) ranges.push({ first: b.triOff, count: b.triCount, inst: i + 1 }); });
    this.subDraws = [];
    let nLong = 0;
    for (const r of ranges) {
      const v0 = n / 4;
      const L2 = r.inst === 0 ? (Ls * 0.995) ** 2 : L2b;
      for (let k = r.first; k < r.first + r.count; k++) {
        const o = 16 * k;
        const ex = T[o + 4], ey = T[o + 5], ez = T[o + 6], fx = T[o + 8], fy = T[o + 9], fz = T[o + 10];
        const l1 = ex * ex + ey * ey + ez * ez, l2 = fx * fx + fy * fy + fz * fz, l3 = (fx - ex) ** 2 + (fy - ey) ** 2 + (fz - ez) ** 2;
        if (Math.max(l1, l2, l3) <= L2) continue;
        if (r.inst === 0) { if (!interior(k)) continue; nStaticLong++; }
        nLong++;
        const a = [T[o], T[o + 1], T[o + 2]];
        const stack = [[a, [a[0] + ex, a[1] + ey, a[2] + ez], [a[0] + fx, a[1] + fy, a[2] + fz]]];
        let guard = 0;
        while (stack.length && guard++ < 200000) {
          const [p, q, w] = stack.pop();
          const ab = d2(p, q), bc = d2(q, w), ca = d2(w, p);
          const m = Math.max(ab, bc, ca);
          if (m <= L2) { emit(p, k); emit(q, k); emit(w, k); continue; }
          if (m === ab) { const mm = mid(p, q); stack.push([p, mm, w], [mm, q, w]); }
          else if (m === bc) { const mm = mid(q, w); stack.push([p, q, mm], [p, mm, w]); }
          else { const mm = mid(w, p); stack.push([p, q, mm], [mm, q, w]); }
        }
      }
      const cnt = n / 4 - v0;
      if (cnt > 0) this.subDraws.push({ first: v0, count: cnt, inst: r.inst });
      (this.stats.subByBody ||= {})[r.inst === 0 ? 'static' : S.blas[r.inst - 1].body] = cnt / 3;
    }
    if (n === 0) emit([0, 0, 0], 0);
    this.subBuf = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'lite_rt.subv' });
    this.device.queue.writeBuffer(this.subBuf, 0, buf, 0, n);
    this.stats.longTris = nLong; this.stats.staticLongTris = nStaticLong; this.stats.subTris = n / 12; this.stats.subdivMs = performance.now() - t0;
    this.log(`subdivided ${nLong} long triangles into ${(n / 12) | 0} raster triangles (edge <= ${L * 1e3} mm) in ${this.stats.subdivMs.toFixed(0)} ms`);
  }

  _view(name) {
    let v = this.views[name];
    if (v) return v;
    const dev = this.device, tr = this.tr;
    const cam = tr._camTargets(name);
    const [w, h] = cam.size;
    v = { name, w, h, cam };
    v.vis = dev.createTexture({ size: [w, h], format: 'rg32uint', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, label: `${name}.vis` });
    v.depth = dev.createTexture({ size: [w, h], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT, label: `${name}.depth` });
    v.rbuf = dev.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    v.lbuf = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    v.rgroup = dev.createBindGroup({ layout: this.layoutR, entries: [{ binding: 0, resource: { buffer: tr.bufs.scn } }, { binding: 1, resource: { buffer: tr.bufs.tris } }, { binding: 2, resource: { buffer: v.rbuf } }, { binding: 3, resource: { buffer: this.subBuf } }] });
    v.pid = dev.createBuffer({ size: w * h * 8, usage: GPUBufferUsage.STORAGE, label: `${name}.pid` });
    v.g2 = dev.createBindGroup({ layout: this.layoutG2, entries: [{ binding: 0, resource: v.vis.createView() }, { binding: 1, resource: { buffer: v.lbuf } }, { binding: 2, resource: { buffer: v.pid } }] });
    v.key = null; v.n = 0;
    // skip mask: the camera's own lens front (owner) for CAM*, the scope body for SCOPE
    let skip = tr._skipMask(name);
    if (cam.pinhole) tr.S.blas.forEach((b, i) => { if (b.body === this.scopeCam.body) skip |= 1 << i; });
    v.skip = skip >>> 0;
    // theta max for KB4 (beyond it the mapping folds)
    v.thetaMax = cam.meta?.theta_fold_rad ? Math.min(cam.meta.theta_fold_rad, 1.6) : 1.55;
    this.views[name] = v;
    return v;
  }

  setPoses(poses0, poses1 = null) {
    this._settledThisFrame = false;
    this.tr.setPoses(poses0, poses1);
    this._updateScope();
  }

  _poseKey(name) {
    const I = this.tr.instances || [];
    const cam = this.tr.cameras[name];
    return JSON.stringify([I.map((x) => (x ? [x.q0, x.q1, x.p0, x.p1] : 0)), cam.T, this.opts.lightSamples]);
  }

  /** The camera and the light (the scope): when either moves, every pixel's history is dropped. */
  _viewKey(name) {
    const S = this.tr.S, I = this.tr.instances || [];
    const si = S.blas.findIndex((b) => b.body === this.scopeCam.body);
    const x = si >= 0 ? I[si] : null;
    return JSON.stringify([x ? [x.q0, x.q1, x.p0, x.p1] : 0, this.tr.cameras[name].T, this.opts.lightSamples]);
  }

  /** Render one view's scene-referred linear radiance (rgba32float, alpha 0 outside the image circle). */
  async renderCamera(name, o = {}) {
    return this._submitView(name, o);
  }

  _submitView(name, o = {}) {
    const tr = this.tr, dev = this.device;
    if (name === 'SCOPE') this._updateScope();
    const v = this._view(name);
    const cam = v.cam;
    this.frameNo++;
    tr.frame = this.frameNo;
    tr._ensureRadiosity();
    const key = this._poseKey(name), vkey = this._viewKey(name);
    const accumulate = o.accumulate ?? this.opts.accumulate;
    const reset = !accumulate || vkey !== v.vkey;              // camera or light moved: drop all history
    const toolsMoved = !reset && key !== v.key;                 // only tools moved: per-pixel short history
    if (reset || toolsMoved) { v.key = key; v.vkey = vkey; }
    if (reset) { v.n = 0; v.seq = 0; }
    if (toolsMoved) v.n = 0;
    // still: converge the node radiosity further (warm-started Jacobi) for the first settleFrames frames, then, once the
    // running mean holds accCap frames, stop rendering this view at all (a still scene costs nothing)
    if (!reset && accumulate) {
      if (!toolsMoved && v.n < this.opts.settleFrames && this.opts.settleIters > 0 && tr.activeCount && !this._settledThisFrame) {
        tr._submitJacobi(this.opts.settleIters); tr._writeSceneU(tr._cut); this._settledThisFrame = true;
      }
      if (!toolsMoved && v.n >= (o.accCap ?? this.opts.accCap) && !o.force) { this.stats.skipped = (this.stats.skipped || 0) + 1; return cam.tex; }
    }
    v.seq = (v.seq || 0);
    const i = reset ? 0 : v.seq;                    // QMC index over frames since the last full reset
    const moving = tr.instances.some((I) => I && (I.q0.some((x, k) => x !== I.q1[k]) || I.p0.some((x, k) => x !== I.p1[k])));
    // sub-pixel jitter (box-integrated pixels) only while everything is still: with tools moving the raster stays on
    // pixel centres, so a static pixel keeps its surface id and its short history
    const jit = (i === 0 || toolsMoved) ? [0, 0] : [fract(0.5 + i * R2A) - 0.5, fract(0.5 + i * R2B) - 0.5];
    const time = moving ? fract(0.5 + i * GOLD) : 0.5;
    const T = cam.T;
    const R = [[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]];
    const pos = [T[0][3], T[1][3], T[2][3]];
    const L = cam.lens;
    // raster uniform
    const rf = new Float32Array(36), ru = new Uint32Array(rf.buffer);
    rf.set([R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0, R[0][2], R[1][2], R[2][2], 0, pos[0], pos[1], pos[2], cam.pinhole ? 1 : 0], 0);
    rf.set([L.fx, L.fy, L.cx, L.cy, ...(L.k || [0, 0, 0, 0]), L.alpha || 0, v.thetaMax, 4.0, 0], 16);
    rf.set([v.w, v.h, jit[0], jit[1], time], 28);
    ru[33] = v.skip;
    rf[34] = cam.pinhole ? 1e9 : this.longEdge;      // a pinhole keeps straight edges straight: no sub-triangles
    rf[35] = cam.pinhole ? 1e9 : this.longEdgeStatic;
    dev.queue.writeBuffer(v.rbuf, 0, rf);
    // camera uniform (the tracer's CamU), shade uniform
    tr._packCam(cam.ubuf, {
      R, pos, mode: cam.pinhole ? 1 : 0, lens: L, w: v.w, h: v.h, spp: 1, seed: (o.seed ?? 0) >>> 0, camIndex: cam.index,
      options: tr._options(o), t0: time, t1: time, scale: 1, skip: v.skip, nLight: o.lightSamples ?? this.opts.lightSamples,
      refine: 0, root: tr.interiorRoot,
    });
    const lf = new Float32Array(8), lu = new Uint32Array(lf.buffer);
    lf.set([jit[0], jit[1], cam.pinhole ? cam.circleR : 0, time], 0);
    lu.set([reset ? 1 : (toolsMoved ? 2 : 0), i, o.accCap ?? this.opts.accCap, this.opts.historyMoving ?? 8], 4);
    dev.queue.writeBuffer(v.lbuf, 0, lf);
    if (tr.timer.tripped) throw new Error(`GPU budget tripped (${tr.timer.tripped})`);
    const enc = dev.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: v.vis.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      depthStencilAttachment: { view: v.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
      timestampWrites: tr.timer.pass(`${name}.raster`, 'lt_raster', v.w * v.h, this.frameNo),
    });
    rp.setPipeline(this.rasterPipe);
    rp.setBindGroup(0, v.rgroup);
    for (const d of this.draws) {
      if (d.inst > 0 && ((v.skip >>> (d.inst - 1)) & 1)) continue;
      rp.draw(3 * d.count, 1, 3 * d.first, d.inst);
    }
    rp.setPipeline(this.rasterSubPipe);
    for (const d of (cam.pinhole ? [] : this.subDraws)) {
      if (d.inst > 0 && ((v.skip >>> (d.inst - 1)) & 1)) continue;
      rp.draw(d.count, 1, d.first, d.inst);
    }
    rp.end();
    const cp = enc.beginComputePass({ timestampWrites: tr.timer.pass(`${name}.shade`, 'lt_shade', v.w * v.h, this.frameNo) });
    cp.setPipeline(this.shadePipe);
    cp.setBindGroup(0, tr.groups.sceneRO); cp.setBindGroup(1, cam.group); cp.setBindGroup(2, v.g2);
    cp.dispatchWorkgroups(Math.ceil(v.w / 8), Math.ceil(v.h / 8));
    cp.end();
    const after = tr.timer.flush(enc);
    dev.queue.submit([enc.finish()]); after();
    v.n = v.n + 1; v.seq = i + 1;
    if (o.exposure !== undefined) tr._manualExposure?.(name, o.exposure);
    return cam.tex;
  }

  renderFrame(opts = {}) {
    const names = opts.cameras || ['SCOPE'];
    const out = {};
    for (const n of names) out[n] = { linear: this._submitView(n, opts) };
    out.done = this.device.queue.onSubmittedWorkDone();
    return out;
  }

  /** 8-bit camera RGB through the path tracer's full §9 chain (Bayer, demosaic, JPEG-ready) for CAM1-3. */
  sensor(name, opts = {}) {
    if (name === 'SCOPE') return null;
    return this.tr.sensor(name, opts);
  }

  async readback(name, kind = 'rgb8') {
    if (name === 'SCOPE' && kind !== 'linear') {
      // display-chain rgb8 of the scope view (CPU, for judges only)
      const lin = await this.tr.readback(name, 'linear');
      return this._cpuDisplay(lin);
    }
    return this.tr.readback(name, kind);
  }

  _cpuDisplay(lin) {
    const n = lin.length / 4, out = new Uint8Array(n * 3);
    let s = 0, c = 0;
    for (let i = 0; i < n; i += 37) if (lin[4 * i + 3] > 0.5) { s += 0.2126 * lin[4 * i] + 0.7152 * lin[4 * i + 1] + 0.0722 * lin[4 * i + 2]; c++; }
    const g = c ? 0.18 / (s / c) : 1;
    const oetf = (x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055);
    const sc = (u, k) => { const a = u ** k; return a / (a + (1 - u) ** k); };
    for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) out[3 * i + j] = Math.round(255 * sc(oetf(Math.min(1, Math.max(0, lin[4 * i + j] * g))), 1.39));
    return out;
  }

  visible(name, pointsD, o) { return this.tr.visible(name, pointsD, o); }
  project(name, pointsD) { return this.tr.project(name, pointsD); }

  timings() {
    const t = this.tr.timings();
    return { ...t, renderer: 'lite_rt' };
  }

  async idle() { return this.tr.idle(); }

  // ---------------------------------------------------------------- display
  _displayParams() {
    if (this._dp) return this._dp;
    const s = this.physics.sensor || {};
    const isp = s.isp || s;
    const c = Number(pval(s.s_curve_c ?? isp.s_curve_c, 1.39));
    const b = Number(pval(s.black_offset_linear ?? isp.black_offset_linear, 0.014));
    const k = Number(pval(s.lens?.veiling_glare?.fraction, 0.10));
    const wb = pval(s.white_balance_gains ?? isp.white_balance_gains, [1, 1, 1]);
    const q = Number(pval(s.lens?.relative_illumination?.pupil_obliquity_exponent, 1.0));
    this._dp = { c, b, k, wb: Array.isArray(wb) ? wb.map(Number) : [1, 1, 1], q, target: 0.18 };
    return this._dp;
  }

  /** Relative illumination map (§9 kb4_mapping: cos^q(θ) sin θ / (θ_d dθ_d/dθ)); 1 for the pinhole scope. */
  _riTexture(name) {
    const v = this._view(name);
    if (v.ri) return v.ri;
    const cam = v.cam, dev = this.device;
    const rw = 160, rh = 90, data = new Float32Array(rw * rh * 4);
    const { q } = this._displayParams();
    const L = cam.lens, k = L.k || [0, 0, 0, 0];
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
      let r = 1;
      if (!cam.pinhole) {
        const u = ((x + 0.5) / rw) * v.w - 0.5, vv = ((y + 0.5) / rh) * v.h - 0.5;
        const yd = (vv - L.cy) / L.fy, xd = (u - L.cx) / L.fx - (L.alpha || 0) * yd;
        const thd = Math.hypot(xd, yd);
        let th = thd;
        for (let it = 0; it < 10; it++) {
          const t2 = th * th;
          const f = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - thd;
          const fp = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
          th -= f / fp;
        }
        if (thd > 1e-6) {
          const t2 = th * th;
          const dthd = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
          r = Math.max(0, Math.cos(th) ** q * Math.sin(th) / (thd * dthd));
        }
      }
      data.set([r, r, r, 1], 4 * (y * rw + x));
    }
    v.ri = dev.createTexture({ size: [rw, rh], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    dev.queue.writeTexture({ texture: v.ri }, data, { bytesPerRow: rw * 16 }, [rw, rh]);
    return v.ri;
  }

  /**
   * Draw a view into a WebGPU canvas context (configured here on first use): relative illumination, veiling glare
   * (k x the frame mean: the σ = 0.25-diagonal Gaussian is nearly flat over the frame), auto-exposure (damped, metered
   * on the GPU, no read-back), white balance, sRGB + S-curve; letterboxed on near-black. No Bayer/demosaic.
   */
  present(name, ctx, o = {}) {
    const dev = this.device;
    const v = this._view(name);
    if (!ctx.__ltConfigured) {
      const fmt = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({ device: dev, format: fmt, alphaMode: 'opaque' });
      ctx.__ltConfigured = fmt;
    }
    const fmt = ctx.__ltConfigured;
    if (!this.presentPipes[fmt]) {
      this.presentPipes[fmt] = dev.createRenderPipeline({ layout: 'auto', vertex: { module: this.presentMod, entryPoint: 'vsFull' }, fragment: { module: this.presentMod, entryPoint: 'fsPresent', targets: [{ format: fmt }] }, primitive: { topology: 'triangle-list' } });
    }
    const pipe = this.presentPipes[fmt];
    if (!v.meter) {
      v.meter = dev.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(v.meter, 0, new Float32Array([0, 0, 0, 0, 0, this._displayParams().target, 0.15, 0]));
      v.pubuf = dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      v.mgroup = dev.createBindGroup({ layout: this.meterPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: v.cam.tex.createView() }, { binding: 1, resource: { buffer: v.meter } }] });
      v.pgroup = new Map();
    }
    const tex = ctx.getCurrentTexture();
    const W = tex.width, H = tex.height;
    const s = Math.min(W / v.w, H / v.h), dw = v.w * s, dh = v.h * s;
    const dp = this._displayParams();
    const ae = o.exposure !== undefined ? o.exposure : null;
    dev.queue.writeBuffer(v.pubuf, 0, new Float32Array([(W - dw) / 2, (H - dh) / 2, dw, dh, dp.c, dp.b, dp.k, 1, ...dp.wb, 1, ...(o.background || [0.04, 0.04, 0.045, 1])]));
    if (ae !== null) dev.queue.writeBuffer(v.meter, 16, new Float32Array([ae, dp.target, 0, 1]));
    let pg = v.pgroup.get(pipe);
    if (!pg) {
      pg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: v.cam.tex.createView() }, { binding: 1, resource: { buffer: v.meter } }, { binding: 2, resource: { buffer: v.pubuf } }, { binding: 3, resource: this._riTexture(name).createView() }] });
      v.pgroup.set(pipe, pg);
    }
    const enc = dev.createCommandEncoder();
    if (ae === null) {
      const mp = enc.beginComputePass();
      mp.setPipeline(this.meterPipe); mp.setBindGroup(0, v.mgroup); mp.dispatchWorkgroups(1); mp.end();
    }
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.04, g: 0.04, b: 0.045, a: 1 } }] });
    rp.setPipeline(pipe); rp.setBindGroup(0, pg); rp.draw(3); rp.end();
    dev.queue.submit([enc.finish()]);
  }

  destroy() {
    for (const v of Object.values(this.views)) { v.pid?.destroy(); v.vis?.destroy(); v.depth?.destroy(); v.rbuf?.destroy(); v.lbuf?.destroy(); v.ri?.destroy(); v.meter?.destroy(); }
    this.tr.destroy();
  }
}

export { LiteRTRenderer as LiteRenderer };
export default LiteRTRenderer;
