// LitePTRenderer: the validated path tracer (gpu/renderer.js, TwinRenderer) at a fraction of the pixels, upsampled to
// full resolution by a mesh-id-guided filter over a full-resolution 4x MSAA raster G-buffer (per-vertex KB4), then the
// sensor stage at full resolution. docs/WEB_TWIN.md §18.1 API (lite_pt race candidate, 2026-10-31).
//
//   const r = await LitePTRenderer.create({ canvas?, scene, physics, device?, factor?: 3, preset?: 'live' });
//   r.setPoses(poses0, poses1?);  await r.renderCamera(name, { exposure, seed });  r.sensor(name, opts) -> rgba8unorm
//   await r.readback(name, 'rgb8' | 'linear');  r.visible(name, pointsD);  r.timings();  r.present(name, ctx | canvas)
//   r.cameras -> ['SCOPE', 'CAM1', 'CAM2', 'CAM3']
//
// Light transport is untouched: the inner TwinRenderer traces cameras named '<cam>@lo' whose KB4 intrinsics are the
// full camera's scaled by 1/factor (pixel centres at integer coordinates: c' = (c + 0.5)/F - 0.5). Each low-res pixel
// also carries the mesh its ray hit (the tracer's own pass-1 key, cam.acc), so the upsampler only blends low-res
// samples of the same mesh as each full-res MSAA sub-sample: silhouettes come from the raster at full resolution.
// The full-res cameras '<cam>' exist in the inner renderer only as targets of the upsampled radiance and its sensor.
// When nothing moves, successive frames average with fresh seeds (temporal accumulation up to `accumulate` frames).

import { TwinRenderer, PRESETS } from '../gpu/renderer.js';
import { packSensorUniform } from '../gpu/sensor.js';
import { loadScene, weld, subdivide } from '../gpu/scene_gpu.js';

const SCOPE_HFOV_DEG = 75;       // physics.yaml scope.camera.hfov_deg is absent (2026-10-31): §18.1 default, assumed
const SCOPE_CIRCLE_FRAC = 0.95;  // physics.yaml scope.camera.image_circle_frac is absent: §18.1 default, assumed
const SCOPE_SIZE = [1280, 720];
// pinhole as KB4: theta_d = tan(theta) = theta (1 + t2/3 + 2 t2^2/15 + 17 t2^3/315 + 62 t2^4/2835) (+ O(theta^11),
// < 0.05 px inside the 36 deg image circle)
const TAN_KB4 = [1 / 3, 2 / 15, 17 / 315, 62 / 2835];

const MSAA = 4;
const GB_FORMAT = 'rgba16float';

const RASTER_WGSL = /* wgsl */`
struct View { T_C_D: mat4x4f, K: vec4f, D: vec4f, size: vec4f, mode: vec4f };
struct Obj { M: mat4x4f, info: vec4f };     // info.x: mesh id + 1, info.y: 1 = hidden
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> objs: array<Obj>;
struct VOut { @builtin(position) pos: vec4f, @location(0) n: vec3f, @location(1) dist: f32, @location(2) @interpolate(flat) id: f32 };
@vertex fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @builtin(instance_index) oi: u32) -> VOut {
  let o = objs[oi];
  let pD = (o.M * vec4f(p, 1.0)).xyz;
  let pc = (V.T_C_D * vec4f(pD, 1.0)).xyz;
  var out: VOut;
  let rho = length(pc.xy);
  let th = atan2(rho, pc.z);
  let t2 = th * th;
  let thd = th * (1.0 + t2 * (V.D.x + t2 * (V.D.y + t2 * (V.D.z + t2 * V.D.w))));
  var uv = V.K.zw;
  if (rho > 1e-12) { uv = vec2f(V.K.x * thd * pc.x / rho, V.K.y * thd * pc.y / rho) + V.K.zw; }
  let ndc = vec2f((uv.x + 0.5) / V.size.x * 2.0 - 1.0, 1.0 - (uv.y + 0.5) / V.size.y * 2.0);
  let dist = length(pc);
  var z = clamp((dist - V.size.z) / (V.size.w - V.size.z), 0.0, 1.0);
  if (th > V.mode.y || o.info.y > 0.5) { z = 2.0; }
  out.pos = vec4f(ndc, z, 1.0);
  if (V.mode.x > 0.5) {
    // pinhole (SCOPE): exact perspective with homogeneous clipping, so long straight edges need no subdivision
    let cxp = V.K.z + 0.5; let cyp = V.K.w + 0.5;
    let n = V.size.z; let f = V.size.w;
    out.pos = vec4f((V.K.x * pc.x + cxp * pc.z) * 2.0 / V.size.x - pc.z, pc.z - (V.K.y * pc.y + cyp * pc.z) * 2.0 / V.size.y,
                    (pc.z - n) * f / (f - n), pc.z);
    if (o.info.y > 0.5) { out.pos = vec4f(0.0, 0.0, -1.0, 1.0); }
  }
  out.n = (V.T_C_D * vec4f((o.M * vec4f(n, 0.0)).xyz, 0.0)).xyz;
  out.dist = dist;
  out.id = o.info.x;
  return out;
}
@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let n = normalize(v.n);
  return vec4f(v.id, v.dist, n.x, n.y);
}`;

const UPSAMPLE_WGSL = /* wgsl */`
struct U { full: vec4f, lo: vec4f, circ: vec4f, opt: vec4f };   // full: W H F 1/F; lo: W H n_acc sigma; circ: cx cy r^2 on; opt: x accumulate weight
@group(0) @binding(0) var loTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> loAcc: array<vec4f>;
@group(0) @binding(2) var gbuf: texture_multisampled_2d<f32>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> u: U;
@group(0) @binding(5) var<storage, read_write> hist: array<vec4f>;     // low-res accumulation (sum, n)
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>>;

const OFF = array<vec2f, 4>(vec2f(-0.125, -0.375), vec2f(0.375, -0.125), vec2f(-0.375, 0.125), vec2f(0.125, 0.375));

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) gid: vec3u) {
  let W = u32(u.lo.x); let H = u32(u.lo.y);
  if (gid.x >= W || gid.y >= H) { return; }
  let i = gid.y * W + gid.x;
  let c = textureLoad(loTex, vec2i(gid.xy), 0).xyz;
  if (u.opt.x < 0.5) { hist[i] = vec4f(c, 1.0); } else { let h = hist[i]; hist[i] = vec4f(h.xyz + c, h.w + 1.0); }
}

@compute @workgroup_size(8, 8)
fn upsample(@builtin(global_invocation_id) gid: vec3u) {
  let W = u32(u.full.x); let H = u32(u.full.y);
  if (gid.x >= W || gid.y >= H) { return; }
  let LW = i32(u.lo.x); let LH = i32(u.lo.y);
  let F = u.full.z; let iF = u.full.w;
  let sig2 = u.lo.w * u.lo.w;
  // the low-res 4x4 neighbourhood of this full-res pixel: radiance (accumulated mean) and first-hit mesh id
  let pc = (vec2f(gid.xy) + 0.5) * iF - 0.5;
  let b = vec2i(floor(pc)) - vec2i(1, 1);
  var rad: array<vec3f, 16>;
  var mid: array<u32, 16>;
  var pos: array<vec2f, 16>;
  var posed = false;
  for (var k = 0; k < 16; k++) {
    let q = b + vec2i(k % 4, k / 4);
    let qc = clamp(q, vec2i(0), vec2i(LW - 1, LH - 1));
    let i = u32(qc.y * LW + qc.x);
    let h = hist[i];
    rad[k] = h.xyz / max(h.w, 1.0);
    let key = bitcast<u32>(loAcc[i].w);
    mid[k] = key & 0xffffu;
    if ((key & 0x10000u) != 0u) { posed = true; }
    if (q.x != qc.x || q.y != qc.y) { mid[k] = 0xfff0u; }
    pos[k] = vec2f(q);
  }
  var acc = vec3f(0.0);
  var fb = 0u;
  for (var s = 0; s < 4; s++) {
    let g = textureLoad(gbuf, vec2i(gid.xy), s);
    let fp = vec2f(gid.xy) + OFF[s];
    if (u.circ.w > 0.5) {
      let d = fp - u.circ.xy;
      if (dot(d, d) > u.circ.z) { continue; }
    }
    if (g.x < 0.5) { continue; }                  // nothing hit (the dome is closed: rare)
    let id = u32(g.x + 0.5) - 1u;
    let p = (fp + 0.5) * iF - 0.5;
    var sum = vec3f(0.0); var ws = 0.0;
    var sumA = vec3f(0.0); var wa = 0.0;
    for (var k = 0; k < 16; k++) {
      let d = pos[k] - p;
      let w = exp(-0.5 * dot(d, d) / sig2);
      if (mid[k] == id) { sum += w * rad[k]; ws += w; }
      if (mid[k] != 0xfff0u) { sumA += w * rad[k]; wa += w; }
    }
    // motion blur (u.opt.y): near a moving body the silhouette is a time-integrated ramp, so the time-sampled
    // low-res radiance is blended across meshes instead of being cut at the mid-exposure raster edge
    if (u.opt.y > 0.5 && posed) { acc += sumA / max(wa, 1e-6); }
    else if (ws > 1e-4) { acc += sum / ws; } else { acc += sumA / max(wa, 1e-6); fb += 1u; }
  }
  textureStore(outTex, vec2i(gid.xy), vec4f(acc * 0.25, 1.0));
  if (fb > 0u) { atomicAdd(&stats[0], fb); }
}`;


// Display sensor (§18.1): the §9 chain without the Bayer mosaic, demosaic and noise, in one pass after the inner
// renderer's own glare pyramid: RI, ESF blur (3x3 of the pixel-integrated Gaussian), veiling glare, the response
// matrix, exposure x gain, full well, ADC range, black level, WB, CCM, sRGB + S-curve; luma and RGB sums for AE / AWB.
const DISPLAY_WGSL = /* wgsl */`
struct SensU {
  size: vec4u, optics: vec4f, m0: vec4f, m1: vec4f, m2: vec4f, noise: vec4f, adc: vec4f, wb: vec4f,
  c0: vec4f, c1: vec4f, c2: vec4f, tone: vec4f, flags: vec4u, glare: vec4f, tone2: vec4f,
}
@group(0) @binding(0) var<uniform> su: SensU;
@group(0) @binding(1) var<storage, read> riMap: array<f32>;
@group(0) @binding(2) var srcTex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(6) var dstTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(9) var glareIn: texture_2d<f32>;
fn erfA(z: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * abs(z));
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return select(-y, y, z >= 0.0);
}
fn tapW(k: i32, sg: f32) -> f32 {
  if (sg <= 0.0) { return select(0.0, 1.0, k == 0); }
  let s2 = sg * 1.41421356;
  return 0.5 * (erfA((f32(k) + 0.5) / s2) - erfA((f32(k) - 0.5) / s2));
}
fn glareAt(x: i32, y: i32) -> vec3f {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let F = su.glare.z;
  let gw = (W + i32(F) - 1) / i32(F); let gh = (H + i32(F) - 1) / i32(F);
  let p = vec2f((f32(x) + 0.5) / F - 0.5, (f32(y) + 0.5) / F - 0.5);
  let p0 = vec2i(floor(p));
  let f = p - floor(p);
  let mx = vec2i(gw - 1, gh - 1);
  let a = textureLoad(glareIn, clamp(p0, vec2i(0), mx), 0).xyz;
  let b = textureLoad(glareIn, clamp(p0 + vec2i(1, 0), vec2i(0), mx), 0).xyz;
  let c = textureLoad(glareIn, clamp(p0 + vec2i(0, 1), vec2i(0), mx), 0).xyz;
  let d = textureLoad(glareIn, clamp(p0 + vec2i(1, 1), vec2i(0), mx), 0).xyz;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
fn toneC(x: f32) -> f32 {
  let m = su.tone.x;
  if (m > 2.5) {
    let b = su.tone2.y;
    let v = clamp((x - b) / (1.0 - b), 0.0, 1.0);
    let u = select(1.055 * pow(v, 1.0 / 2.4) - 0.055, 12.92 * v, v <= 0.0031308);
    let a = pow(u, su.tone2.x);
    return a / max(a + pow(1.0 - u, su.tone2.x), 1e-12);
  }
  let c = clamp(x, 0.0, 1.0);
  if (m < 0.5) { return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308); }
  if (m < 1.5) { return pow(c, 1.0 / max(su.tone.y, 1e-3)); }
  return c;
}
var<workgroup> wsum: array<vec4u, 64>;
@compute @workgroup_size(8, 8)
fn display(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let x = i32(gid.x); let y = i32(gid.y);
  var st = vec4u(0u);
  if (x < W && y < H) {
    let sg = su.optics.y;
    let w0 = tapW(0, sg); let w1 = tapW(1, sg);
    var E = vec3f(0.0); var ws = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let xx = clamp(x + dx, 0, W - 1); let yy = clamp(y + dy, 0, H - 1);
        let w = select(w1, w0, dx == 0) * select(w1, w0, dy == 0);
        E += w * textureLoad(srcTex, vec2i(xx, yy), 0).xyz * riMap[u32(yy * W + xx)];
        ws += w;
      }
    }
    E = E * (su.optics.x / ws);
    if (su.glare.x > 0.0) { E = (1.0 - su.glare.x) * E + su.glare.x * glareAt(x, y); }
    let texp = su.optics.z;
    var e = vec3f(dot(su.m0.xyz, E), dot(su.m1.xyz, E), dot(su.m2.xyz, E));
    e = min(max(e, vec3f(0.0)) * texp + vec3f(su.noise.z * texp), vec3f(su.noise.y));
    let dn = clamp(e * su.noise.w * su.optics.w + vec3f(su.adc.y), vec3f(0.0), vec3f(su.adc.x));
    let full = max(su.adc.x - su.adc.y, 1.0);
    var c = (dn - vec3f(su.adc.y)) / full;
    let q = vec3u(round(clamp(c, vec3f(0.0), vec3f(1.0)) * 1023.0));
    c = c * su.wb.xyz;
    c = vec3f(dot(su.c0.xyz, c), dot(su.c1.xyz, c), dot(su.c2.xyz, c));
    let o = round(clamp(vec3f(toneC(c.x), toneC(c.y), toneC(c.z)), vec3f(0.0), vec3f(1.0)) * 255.0);
    textureStore(dstTex, vec2i(x, y), vec4f(o / 255.0, 1.0));
    st = vec4u(u32(clamp(round(0.299 * o.x + 0.587 * o.y + 0.114 * o.z), 0.0, 255.0)), q);
  }
  wsum[li] = st;
  workgroupBarrier();
  if (li == 0u) {
    var t = vec4u(0u);
    for (var k = 0u; k < 64u; k++) { t += wsum[k]; }
    atomicAdd(&stats[0], t.x); atomicAdd(&stats[1], t.y); atomicAdd(&stats[2], t.z); atomicAdd(&stats[3], t.w);
  }
}`;

const BLIT_WGSL = /* wgsl */`
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var<uniform> sz: vec4f;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); return vec4f(p[i], 0, 1); }
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  // letterbox the image into the canvas (near-black outside), nearest texel
  let d = vec2f(textureDimensions(t));
  let s = min(sz.x / d.x, sz.y / d.y);
  let o = (sz.xy - d * s) * 0.5;
  let q = (p.xy - o) / s;
  if (q.x < 0.0 || q.y < 0.0 || q.x >= d.x || q.y >= d.y) { return vec4f(0.02, 0.02, 0.02, 1.0); }
  return textureLoad(t, vec2i(q), 0);
}`;

// ------------------------------------------------------------------ small math
const rowMajor4 = (T) => (Array.isArray(T[0]) ? T : [[T[0], T[4], T[8], T[12]], [T[1], T[5], T[9], T[13]], [T[2], T[6], T[10], T[14]], [T[3], T[7], T[11], T[15]]]);
const mulRv = (R, v) => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const poseSig = (p) => { const b = p?.bodies || {}; return Object.keys(b).sort().map((k) => `${k}:${b[k] ? Array.from(b[k]).join(',') : 'null'}`).join(';'); };
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** T_D_C (row-major 4x4) of the scope camera from the scope body pose T_D_K (column-major 16). */
export function scopeCameraT(scene, T_D_K) {
  const T = rowMajor4(T_D_K);
  const R = [[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]];
  const sc = scene.scope || {};
  const oK = sc.objective_centre_K || [0, 0, 0];
  const vK = sc.view_dir_K || [0, Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)];
  const c = mulRv(R, oK).map((x, i) => x + T[i][3]);
  const z = norm3(mulRv(R, vK));
  // image up = the side the oblique view tilts toward (+y_K, the light post side on the tower monitor)
  const yK = mulRv(R, [0, 1, 0]);
  const up = norm3(yK.map((x, i) => x - dot3(yK, z) * z[i]));
  const y = up.map((x) => -x);                    // OpenCV y is down
  const x = cross3(y, z);
  return [[x[0], y[0], z[0], c[0]], [x[1], y[1], z[1], c[1]], [x[2], y[2], z[2], c[2]], [0, 0, 0, 1]];
}

/** Scene lens block of a pinhole SCOPE camera as KB4 (tan series). */
export function scopeLens(size = SCOPE_SIZE, hfovDeg = SCOPE_HFOV_DEG) {
  const [w, h] = size;
  const f = (w / 2) / Math.tan((hfovDeg * Math.PI) / 360);
  return { model: 'kb4', K: [[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1]], D: [...TAN_KB4], size: [w, h] };
}

/** The same lens at 1/F of the pixels (pixel centres at integer coordinates). */
export function scaleLens(lens, F) {
  const K = lens.K, [w, h] = lens.size;
  const W = Math.round(w / F), H = Math.round(h / F);
  const sx = W / w, sy = H / h;
  return { ...lens, K: [[K[0][0] * sx, K[0][1] * sx, (K[0][2] + 0.5) * sx - 0.5], [0, K[1][1] * sy, (K[1][2] + 0.5) * sy - 0.5], [0, 0, 1]], size: [W, H] };
}

export class LitePTRenderer {
  /** opts: {canvas?, scene (URL | {scene, meshes, base}), physics (object | URL), device?, factor: 3, preset: 'live', accumulate: 16, sigma: 1.0, edgeMax: 0.006, ...TwinRenderer opts} */
  static async create(opts) {
    const r = new LitePTRenderer();
    await r._init(opts);
    return r;
  }

  get isLitePT() { return true; }

  async _init(opts) {
    const t0 = performance.now();
    this.log = opts.log || ((...a) => console.log('[lite-pt]', ...a));
    let src = opts.scene;
    if (typeof src === 'string') src = await loadScene(src);
    let physics = opts.physics;
    if (typeof physics === 'string') physics = await (await fetch(physics)).json();
    this.physics = physics || {};
    this.F = opts.factor ?? 3;
    this.accN = opts.accumulate ?? 16;
    this.sigma = opts.sigma ?? 1.0;
    this.edgeMax = opts.edgeMax ?? 0.006;
    this.traceOpts = { spp: opts.spp ?? 1, refine: opts.refine ?? 0, lightSamples: opts.lightSamples ?? 1 };
    const scene = src.scene;
    this.srcScene = scene;
    this.meshData = src.meshes;
    // cameras: the tracking cameras from the scene plus SCOPE; each at full res (targets) and at 1/F (traced)
    const camMeta = scene.cameras.map((c) => ({ ...c }));
    const scopeCam = { name: 'SCOPE', size: [...SCOPE_SIZE], lens: scopeLens(SCOPE_SIZE), T_D_C: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], scope: true };
    if (scene.scope) camMeta.unshift(scopeCam);
    this.cameraNames = camMeta.map((c) => c.name);
    const cams = [];
    for (const c of camMeta) {
      const size = c.size || c.lens.size;
      const lens = { ...c.lens, size: [...size] };
      cams.push({ ...c, size, lens });
      const lo = scaleLens(lens, this.F);
      cams.push({ ...c, name: `${c.name}@lo`, size: lo.size, lens: lo });
    }
    const scene2 = { ...scene, cameras: cams };
    // live preset, radiosity refreshed every 4th pose change (one view, ~15 Hz indirect at 60 fps; judged states converge)
    const preset = { ...(PRESETS[opts.preset || 'live'] || {}), radiosityEvery: 4 };
    const { canvas, scene: _s, physics: _p, factor: _f, accumulate: _a, ...rest } = opts;
    this.inner = await TwinRenderer.create({ ...preset, ...rest, preset: undefined, scene: { scene: scene2, meshes: src.meshes, base: src.base }, physics: this.physics, log: this.log });
    const R = this.inner;
    this.device = R.device;
    this.canvas = canvas || null;
    // own lens front skipped by '<cam>@lo' too; SCOPE rays skip the scope body (they start at its objective)
    const skip0 = R._skipMask.bind(R);
    const scopeBit = (() => { const i = R.S.blas.findIndex((b) => b.body === 'scope'); return i >= 0 ? (1 << i) >>> 0 : 0; })();
    R._skipMask = (name) => {
      const base = name.split('@')[0];
      return (skip0(base) | (base === 'SCOPE' ? scopeBit : 0)) >>> 0;
    };
    const tr = performance.now();
    this._buildRaster(scene);
    this.rasterBuildMs = performance.now() - tr;
    this._buildUpsample();
    this.state = {};
    for (const n of this.cameraNames) this.state[n] = { gKey: null, accKey: null, n: 0 };
    this.posesVersion = 0;
    this.stats = { createMs: performance.now() - t0, factor: this.F, inner: R.stats, fallbackSamples: 0, rasterTris: this.rasterTris, rasterTrisPinhole: this.rasterTrisPinhole, rasterBuildMs: this.rasterBuildMs };
    this.setPoses({ bodies: {} });
    this.log(`lite-pt ready in ${(performance.now() - t0).toFixed(0)} ms: factor ${this.F}, cameras ${this.cameraNames.join(' ')}`);
  }

  /** Camera names (SCOPE first). */
  get cameras() { return this.cameraNames; }
  cameraSize(name) { return this.inner.cameras[name].size; }

  // ---------------------------------------------------------------- raster G-buffer
  _buildRaster(scene) {
    const dev = this.device, R = this.inner;
    const meshIndex = new Map(R.S.meshTable.map((m, i) => [m.name, i]));
    this.draws = [];
    this.edgeMax = this.edgeMax || 0.003;
    let nTris = 0;
    const geo = (P, N, I) => {
      const vb = dev.createBuffer({ size: Math.max(16, P.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(vb, 0, P);
      const nb = dev.createBuffer({ size: Math.max(16, N.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(nb, 0, N);
      const ib = dev.createBuffer({ size: Math.max(16, I.byteLength), usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(ib, 0, I);
      return { vb, nb, ib, count: I.length };
    };
    let nTris0 = 0;
    // sub: edges <= edgeMax (fisheye cameras, per-vertex KB4); orig: the exported mesh (pinhole SCOPE)
    const add = (name, P, N, I, body, owner, scopePart, orig = null) => {
      const id = meshIndex.get(name);
      if (id === undefined) return;
      const sub = geo(P, N, I);
      const o = orig ? geo(orig.P, orig.N, orig.I) : sub;
      this.draws.push({ name, sub, orig: o, id, body, owner, scopePart, obj: this.draws.length });
      nTris += I.length / 3; nTris0 += o.count / 3;
    };
    const scopeMeshes = new Set([...(scene.scope?.mesh_parts || []), scene.scope?.emitter_mesh].filter(Boolean));
    for (const m of scene.meshes) {
      const md = this.meshData[m.name];
      if (!md || m.overview_only) continue;
      const body = m.dynamic ? (m.body || (scopeMeshes.has(m.name) ? 'scope' : null)) : null;
      // per-vertex KB4 bends only at vertices: split every edge to <= edgeMax (the exported meshes have edges up to
      // 312 mm: the shafts' long facets would project as chords, pixels off the true curved silhouette)
      const w = weld(md.P, md.N, md.I);
      const I2 = subdivide(w.P, w.N, w.I, this.edgeMax);
      add(m.name, Float32Array.from(w.P), Float32Array.from(w.N), Uint32Array.from(I2), body, m.owner_camera || null, scopeMeshes.has(m.name) || body === 'scope',
        { P: new Float32Array(md.P), N: new Float32Array(md.N), I: new Uint32Array(md.I) });
    }
    if (scene.bench) {
      // the bench as a fine grid (per-vertex KB4 needs short edges; one big quad bends wrongly and clips)
      const bz = scene.bench.z ?? 0, ex = scene.bench.extent || [[-0.5, -0.5], [0.5, 0.5]];
      const nx = Math.min(400, Math.ceil((ex[1][0] - ex[0][0]) / 0.004)), ny = Math.min(400, Math.ceil((ex[1][1] - ex[0][1]) / 0.004));
      const P = new Float32Array((nx + 1) * (ny + 1) * 3), N = new Float32Array((nx + 1) * (ny + 1) * 3), I = new Uint32Array(nx * ny * 6);
      for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
        const k = j * (nx + 1) + i;
        P.set([ex[0][0] + (ex[1][0] - ex[0][0]) * i / nx, ex[0][1] + (ex[1][1] - ex[0][1]) * j / ny, bz], 3 * k); N.set([0, 0, 1], 3 * k);
      }
      let q = 0;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
        I.set([a, b, d, a, d, c], q); q += 6;
      }
      add('bench', P, N, I, null, null, false);
    }
    this.rasterTris = nTris; this.rasterTrisPinhole = nTris0;
    this.OBJ_F = 20;
    this.objData = new Float32Array(this.draws.length * this.OBJ_F);
    this.objBuf = dev.createBuffer({ size: this.objData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const mod = dev.createShaderModule({ code: RASTER_WGSL });
    this.rasterBGL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }] });
    this.rasterPipe = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.rasterBGL] }),
      vertex: { module: mod, entryPoint: 'vs', buffers: [
        { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
        { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] }] },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: GB_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: MSAA },
    });
  }

  _gbufTargets(name) {
    const st = this.state[name];
    if (st.gb) return st;
    const dev = this.device, [w, h] = this.inner.cameras[name].size;
    st.gb = dev.createTexture({ size: [w, h], format: GB_FORMAT, sampleCount: MSAA, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, label: `${name}.gbuf` });
    st.depth = dev.createTexture({ size: [w, h], format: 'depth32float', sampleCount: MSAA, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: `${name}.gdepth` });
    st.viewBuf = dev.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    st.rasterGroup = dev.createBindGroup({ layout: this.rasterBGL, entries: [{ binding: 0, resource: { buffer: st.viewBuf } }, { binding: 1, resource: { buffer: this.objBuf } }] });
    return st;
  }

  _writeObjs() {
    const B = (this.poses0 || {}).bodies || {}, od = this.objData, F = this.OBJ_F;
    // mid-exposure pose of every posed body (the tracer's own slerp/lerp over the shutter): a motion-blurred edge
    // crosses half its contrast at the mid pose
    const R = this.inner, mid = {};
    R.S.blas.forEach((b, i) => {
      const P = R._poseAt(i, 0.5);
      if (b.kind === 'fixed') return;
      mid[b.body] = P ? [P.R[0][0], P.R[1][0], P.R[2][0], 0, P.R[0][1], P.R[1][1], P.R[2][1], 0, P.R[0][2], P.R[1][2], P.R[2][2], 0, P.p[0], P.p[1], P.p[2], 1] : null;
    });
    for (const d of this.draws) {
      const o = d.obj * F;
      let T = d.body ? (d.body in mid ? mid[d.body] : B[d.body]) : null, hidden = 0;
      if (d.body && !T) hidden = 1;
      if (!T) T = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
      // mid-exposure pose for the G-buffer (the tracer integrates the shutter; the silhouette is the mid pose)
      for (let k = 0; k < 16; k++) od[o + k] = T[k];
      od[o + 16] = d.id + 1; od[o + 17] = hidden; od[o + 18] = 0; od[o + 19] = 0;
    }
    this.device.queue.writeBuffer(this.objBuf, 0, od);
    this._objVersion = this.posesVersion;
  }

  _rasterGBuffer(name) {
    const st = this._gbufTargets(name), cam = this.inner.cameras[name], dev = this.device;
    const T = cam.T;   // row-major T_D_C
    // T_C_D = [R^T | -R^T c], column-major for WGSL
    const Rt = [[T[0][0], T[1][0], T[2][0]], [T[0][1], T[1][1], T[2][1]], [T[0][2], T[1][2], T[2][2]]];
    const c = [T[0][3], T[1][3], T[2][3]];
    const t = mulRv(Rt, c).map((x) => -x);
    const M = [Rt[0][0], Rt[1][0], Rt[2][0], 0, Rt[0][1], Rt[1][1], Rt[2][1], 0, Rt[0][2], Rt[1][2], Rt[2][2], 0, t[0], t[1], t[2], 1];
    const L = cam.lens, [w, h] = cam.size;
    const thMax = cam.meta.theta_fold_rad ? Math.min(cam.meta.theta_fold_rad, 1.45) : (cam.meta.scope ? 1.3 : 1.45);
    const buf = new Float32Array(32);
    buf.set(M, 0); buf.set([L.fx, L.fy, L.cx, L.cy], 16); buf.set(L.k, 20); buf.set([w, h, 1e-4, 1.5], 24); buf.set([cam.meta.scope ? 1 : 0, thMax, 0, 0], 28);
    dev.queue.writeBuffer(st.viewBuf, 0, buf);
    if (this._objVersion !== this.posesVersion) this._writeObjs();
    const base = name;
    const enc = dev.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: st.gb.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
      depthStencilAttachment: { view: st.depth.createView(), depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'discard' },
      timestampWrites: this.inner.timer.pass(`${name}.gbuffer`, null, 0, this.inner.frame),
    });
    rp.setPipeline(this.rasterPipe); rp.setBindGroup(0, st.rasterGroup);
    for (const d of this.draws) {
      if (d.owner && d.owner === base) continue;            // a camera's own lens front
      if (base === 'SCOPE' && d.scopePart) continue;         // the scope's own body
      const g = cam.meta.scope ? d.orig : d.sub;
      rp.setVertexBuffer(0, g.vb); rp.setVertexBuffer(1, g.nb); rp.setIndexBuffer(g.ib, 'uint32');
      rp.drawIndexed(g.count, 1, 0, 0, d.obj);
    }
    rp.end();
    const after = this.inner.timer.flush(enc);
    dev.queue.submit([enc.finish()]); after();
  }

  // ---------------------------------------------------------------- upsampling
  _buildUpsample() {
    const dev = this.device;
    const mod = dev.createShaderModule({ code: UPSAMPLE_WGSL });
    this.upBGL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', multisampled: true } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
    const layout = dev.createPipelineLayout({ bindGroupLayouts: [this.upBGL] });
    this.upPipe = dev.createComputePipeline({ layout, compute: { module: mod, entryPoint: 'upsample' } });
    this.accPipe = dev.createComputePipeline({ layout, compute: { module: mod, entryPoint: 'accumulate' } });
    this.upStats = dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  _upTargets(name) {
    const st = this.state[name];
    if (st.upGroup) return st;
    const R = this.inner, dev = this.device;
    const full = R._camTargets(name), lo = R._camTargets(`${name}@lo`);
    const [lw, lh] = lo.size;
    st.hist = dev.createBuffer({ size: lw * lh * 16, usage: GPUBufferUsage.STORAGE, label: `${name}.hist` });
    st.upU = dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._gbufTargets(name);
    st.upGroup = dev.createBindGroup({ layout: this.upBGL, entries: [
      { binding: 0, resource: lo.tex.createView() }, { binding: 1, resource: { buffer: lo.acc } },
      { binding: 2, resource: st.gb.createView() }, { binding: 3, resource: full.tex.createView() },
      { binding: 4, resource: { buffer: st.upU } }, { binding: 5, resource: { buffer: st.hist } }, { binding: 6, resource: { buffer: this.upStats } }] });
    return st;
  }

  // ---------------------------------------------------------------- API
  setPoses(poses0, poses1 = null) {
    // the same poses again (a still scene): keep the radiosity, the G-buffers and the accumulated history
    const sig = poseSig(poses0) + '|' + poseSig(poses1 || poses0);
    if (sig === this._poseSig) return;
    this._poseSig = sig;
    this.inner.setPoses(poses0, poses1);
    this.poses0 = poses0; this.poses1 = poses1 || poses0;
    this._moving = !!poses1 && poseSig(poses1) !== poseSig(poses0);
    this.posesVersion++;
    const T = poses0?.bodies?.scope;
    if (T && this.inner.cameras.SCOPE) {
      const Tc = scopeCameraT(this.srcScene, T);
      this.inner.cameras.SCOPE.T = Tc; this.inner.cameras['SCOPE@lo'].T = Tc;
    }
  }

  async converge(iters = 200) { return this.inner.converge ? this.inner.converge(iters) : null; }

  /**
   * Trace `<name>@lo`, accumulate it while nothing moves, upsample into `<name>`'s radiance. opts: {seed, exposure,
   * spp, refine, lightSamples, accumulate: false (one frame, no history)}. Returns the full-res rgba32float texture.
   */
  async renderCamera(name, opts = {}) {
    const R = this.inner, st = this._upTargets(name), cam = R.cameras[name];
    const lo = `${name}@lo`;
    // the history is valid while poses, camera and trace settings hold; a fixed seed (judge) renders one frame
    const key = `${this.posesVersion}|${cam.T.flat().join(',')}`;
    const accumulate = opts.accumulate !== false && opts.seed === undefined && this.accN > 1;
    let n = (accumulate && st.accKey === key) ? st.n : 0;
    if (n >= this.accN) {
      // converged: nothing to trace; the radiance texture already holds the mean
      if (opts.exposure !== undefined) R._manualExposure(name, opts.exposure);
      this.stats.lastSkipped = true;
      return R.cameras[name].tex;
    }
    this.stats.lastSkipped = false;
    const seed = opts.seed ?? (n === 0 ? 0 : 1000 + n);
    await R.renderCamera(lo, { ...this.traceOpts, ...(opts.spp ? { spp: opts.spp } : {}), seed, ...(opts.refine != null ? { refine: opts.refine } : {}), ...(opts.lightSamples ? { lightSamples: opts.lightSamples } : {}) });
    if (st.gKey !== key) { this._rasterGBuffer(name); st.gKey = key; }
    const [W, H] = cam.size, [lw, lh] = R.cameras[lo].size;
    const circ = cam.meta.scope ? [W / 2 - 0.5, H / 2 - 0.5, (SCOPE_CIRCLE_FRAC * W / 2) ** 2, 1] : [0, 0, 0, 0];
    this.device.queue.writeBuffer(st.upU, 0, new Float32Array([W, H, this.F, lw / W, lw, lh, n + 1, this.sigma, ...circ, n > 0 ? 1 : 0, this._moving ? 1 : 0, 0, 0]));
    R._submit(`${name}.accumulate`, null, 0, (pass) => { pass.setPipeline(this.accPipe); pass.setBindGroup(0, st.upGroup); pass.dispatchWorkgroups(Math.ceil(lw / 8), Math.ceil(lh / 8)); });
    R._submit(`${name}.upsample`, null, 0, (pass) => { pass.setPipeline(this.upPipe); pass.setBindGroup(0, st.upGroup); pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8)); });
    st.n = n + 1; st.accKey = key;
    if (opts.exposure !== undefined) R._manualExposure(name, opts.exposure);
    return R.cameras[name].tex;
  }

  /**
   * Radiance -> 8-bit display RGB. Default: the display sensor (no mosaic, demosaic or noise; §18.1). opts.full = true:
   * the validated §9 camera chain of the inner renderer (Bayer, demosaic, NR, noise unless noise: false).
   */
  sensor(name, opts = {}) {
    if (opts.full || this.fullSensor) return this.inner.sensor(name, { noise: false, ...opts });
    const R = this.inner, dev = this.device;
    const cam = R._camTargets(name), s = R._sensorTargets(name), [w, h] = cam.size;
    if (!this._dispPipe) {
      const m = dev.createShaderModule({ code: DISPLAY_WGSL });
      this._dispPipe = dev.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'display' } });
    }
    if (!s.dispGroup) {
      s.dispGroup = dev.createBindGroup({ layout: this._dispPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: s.u } }, { binding: 1, resource: { buffer: s.ri } }, { binding: 2, resource: cam.tex.createView() },
        { binding: 5, resource: { buffer: s.stats } }, { binding: 6, resource: s.out.createView() }, { binding: 9, resource: s.GA.createView() }] });
    }
    const ex = R.exposureOf(name);
    const p = { ...R.sensorP, ...(opts.params || {}) };
    if (p.awb && cam.awb && !(opts.params && opts.params.wb)) p.wb = p.wb.map((g, c) => g * cam.awb[c]);
    const ub = packSensorUniform(p, { width: w, height: h, frame: opts.frame ?? R.frame, seed: opts.seed ?? 0, exposure: opts.exposure_s ?? ex.time_s, gain: opts.gain ?? ex.gain, camIndex: cam.index });
    dev.queue.writeBuffer(s.u, 0, ub);
    dev.queue.writeBuffer(s.stats, 0, new Uint32Array(4));
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass({ timestampWrites: R.timer.pass(`${name}.display`, null, 0, R.frame) });
    for (const [pipe, group, W, H] of s.groups.slice(1, 4)) { pass.setPipeline(pipe); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8)); }
    pass.setPipeline(this._dispPipe); pass.setBindGroup(0, s.dispGroup); pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    pass.end();
    const rd = s.statsRead.pop() || dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(s.stats, 0, rd, 0, 16);
    s.pendingStats = rd;
    const after = R.timer.flush(enc);
    dev.queue.submit([enc.finish()]); after();
    R._afterSensor(name);
    return s.out;
  }

  exposureOf(name) { return this.inner.exposureOf(name); }
  sensorInfo() { return this.inner.sensorInfo(); }

  async readback(name, kind = 'rgb8') { return this.inner.readback(name, kind); }
  visible(name, pointsD, o) { return this.inner.visible(name, pointsD, o); }
  project(name, pointsD) { return this.inner.project(name, pointsD); }

  /** Draw a camera's last sensor output to a WebGPU canvas (or its context) on this device, letterboxed. */
  present(name, target) {
    const dev = this.device, s = this.inner.cameras[name]?.s;
    if (!s) return;
    if (!this._blit) {
      const m = dev.createShaderModule({ code: BLIT_WGSL });
      this._blitFormat = navigator.gpu.getPreferredCanvasFormat();
      this._blit = dev.createRenderPipeline({ layout: 'auto', vertex: { module: m, entryPoint: 'vs' }, fragment: { module: m, entryPoint: 'fs', targets: [{ format: this._blitFormat }] } });
      this._blitU = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    let ctx = target;
    if (target && typeof target.getContext === 'function') {
      if (!target._litePtCtx) { target._litePtCtx = target.getContext('webgpu'); target._litePtCtx.configure({ device: dev, format: this._blitFormat, alphaMode: 'opaque' }); }
      ctx = target._litePtCtx;
    }
    const tex = ctx.getCurrentTexture();
    dev.queue.writeBuffer(this._blitU, 0, new Float32Array([tex.width, tex.height, 0, 0]));
    const g = dev.createBindGroup({ layout: this._blit.getBindGroupLayout(0), entries: [{ binding: 0, resource: s.out.createView() }, { binding: 1, resource: { buffer: this._blitU } }] });
    const enc = dev.createCommandEncoder();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0.02, 0.02, 0.02, 1] }] });
    rp.setPipeline(this._blit); rp.setBindGroup(0, g); rp.draw(3); rp.end();
    dev.queue.submit([enc.finish()]);
  }

  async fallbackCount() {
    const dev = this.device;
    const st = dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(this.upStats, 0, st, 0, 16); enc.clearBuffer(this.upStats); dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ); const v = new Uint32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
    return v[0];
  }

  timings() { const t = this.inner.timings(); t.litePT = { ...this.stats }; return t; }
  async idle() { return this.inner.idle(); }
  destroy() { this.inner.destroy(); }
}

export default LitePTRenderer;
