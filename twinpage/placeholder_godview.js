// The placeholder god's-eye renderer (docs/WEB_TWIN.md §19): a plain WebGPU raster of the whole rig so the page is
// usable and testable before the real god-view renderers (godview_three/, godview_pt/) land. Same API as theirs:
//
//   static async create({ canvas, scene, physics, godview, device? })   scene: loadScene() bundle {scene, meshes, base}
//                                                                       or a parsed scene.json (meshes fetched from base)
//   setPoses(poses) · setCamera({eye, target, up, vfov_deg}) · setCutaway(on) · render() · async captureLinear(w, h)
//   timings() · destroy()
//
// Light (godview.yaml lights, studio units): a constant white environment (hemispherical, no occlusion), the key
// directional light with a 2048² PCF shadow map, and the scope LED as a spot (cos^m profile, no shadow). The §8 BSDF is
// reduced to diffuse rho_d + one GGX lobe (F0 from the IOR, or the conductor's f0). Display: exposure, Khronos PBR
// Neutral, sRGB. Honest limits: no indirect light, no interior occlusion of the environment, no textures.
import { loadScene } from '../gpu/scene_gpu.js';

const U = (x) => (x && typeof x === 'object' && !Array.isArray(x) && 'value' in x ? x.value : x);
const DEG = Math.PI / 180;
const HDR = 'rgba16float';
const MSAA = 4;
const SHADOW = 2048;
const DRAW_STRIDE = 256;

const WGSL = /* wgsl */`
struct Frame {
  viewProj : mat4x4f,
  lightVP  : mat4x4f,
  eye      : vec4f,
  cut      : vec4f,     // xyz unit normal toward the eye, w: discard when dot(p, n) > w
  keyDir   : vec4f,     // xyz direction TO the light, w: shadow texel (uv)
  keyE     : vec4f,     // rgb irradiance, w: cutaway on (1) / off (0)
  env      : vec4f,     // rgb radiance
  scopeP   : vec4f,     // xyz LED position, w: profile exponent
  scopeA   : vec4f,     // xyz emission axis
  scopeI   : vec4f,     // rgb axial intensity (studio units / sr)
  misc     : vec4f,     // x exposure
};
struct Draw {
  model : mat4x4f,
  base  : vec4f,        // rgb diffuse rho_d, w: flags (1 cuttable, 2 emissive, 4 metal, 8 ghost)
  spec  : vec4f,        // rgb F0, w roughness
  emit  : vec4f,        // rgb emitted radiance, w alpha (ghosts)
};
@group(0) @binding(0) var<uniform> F : Frame;
@group(0) @binding(1) var shadowMap : texture_depth_2d;
@group(0) @binding(2) var shadowSmp : sampler_comparison;
@group(1) @binding(0) var<uniform> D : Draw;

struct VOut { @builtin(position) pos : vec4f, @location(0) wp : vec3f, @location(1) n : vec3f };

@vertex fn vsMain(@location(0) p : vec3f, @location(1) n : vec3f) -> VOut {
  var o : VOut;
  let w = D.model * vec4f(p, 1.0);
  o.wp = w.xyz;
  o.n = (D.model * vec4f(n, 0.0)).xyz;
  o.pos = F.viewProj * w;
  return o;
}

struct SOut { @builtin(position) pos : vec4f, @location(0) wp : vec3f };
@vertex fn vsShadow(@location(0) p : vec3f) -> SOut {
  var o : SOut;
  let w = D.model * vec4f(p, 1.0);
  o.wp = w.xyz;
  o.pos = F.lightVP * w;
  return o;
}

fn cutAway(wp : vec3f) -> bool {
  let flags = u32(D.base.w + 0.5);
  return (flags & 1u) != 0u && F.keyE.w > 0.5 && dot(wp, F.cut.xyz) > F.cut.w;
}

@fragment fn fsShadow(i : SOut) { if (cutAway(i.wp)) { discard; } }

const PI = 3.14159265;

fn shadowAt(wp : vec3f, ndl : f32) -> f32 {
  let c = F.lightVP * vec4f(wp, 1.0);
  let uv = vec2f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }
  let bias = 0.0006 + 0.002 * (1.0 - ndl);
  let z = c.z - bias;
  let t = F.keyDir.w;
  var s = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      s += textureSampleCompareLevel(shadowMap, shadowSmp, uv + vec2f(f32(x), f32(y)) * t, z);
    }
  }
  return s / 9.0;
}

fn ggx(n : vec3f, v : vec3f, l : vec3f, a : f32) -> f32 {
  let h = normalize(v + l);
  let nh = max(dot(n, h), 0.0);
  let nl = max(dot(n, l), 1e-4);
  let nv = max(dot(n, v), 1e-4);
  let a2 = a * a;
  let d = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
  let k = a * 0.5;
  let g = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
  return d * g / (4.0 * nl * nv);
}

fn schlick(f0 : vec3f, c : f32) -> vec3f { return f0 + (vec3f(1.0) - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0); }

@fragment fn fsMain(i : VOut, @builtin(front_facing) ff : bool) -> @location(0) vec4f {
  if (cutAway(i.wp)) { discard; }
  let flags = u32(D.base.w + 0.5);
  if ((flags & 2u) != 0u) { return vec4f(D.emit.rgb, 1.0); }
  var n = normalize(i.n);
  let v = normalize(F.eye.xyz - i.wp);
  if (dot(n, v) < 0.0) { n = -n; }
  let rough = max(D.spec.w, 0.05);
  let a = rough * rough;
  let f0 = D.spec.rgb;
  let nv = max(dot(n, v), 1e-3);
  let metal = (flags & 4u) != 0u;
  // environment: a white room above, the white bench below (hemispherical; no occlusion)
  let hemi = 0.8 * (0.62 + 0.38 * n.z);     // ~ the sky the rig and the bench leave a surface (no AO in the placeholder)
  let r = reflect(-v, n);
  let envSpec = F.env.rgb * (0.55 + 0.45 * clamp(r.z, -1.0, 1.0));
  let fe = schlick(f0, nv) * (1.0 - 0.6 * a);
  var col = vec3f(0.0);
  let kd = select(vec3f(1.0) - fe, vec3f(0.0), metal);
  col += D.base.rgb * kd * F.env.rgb * hemi + fe * envSpec;
  // key light
  let l = F.keyDir.xyz;
  let ndl = dot(n, l);
  if (ndl > 0.0) {
    let sh = shadowAt(i.wp, ndl);
    let h = normalize(v + l);
    let fk = schlick(f0, dot(v, h));
    let spec = fk * ggx(n, v, l, a);
    col += sh * F.keyE.rgb * ndl * (D.base.rgb * select(vec3f(1.0) - fk, vec3f(0.0), metal) / PI + spec);
  }
  // scope LED (spot, inverse square)
  let dl = F.scopeP.xyz - i.wp;
  let d2 = max(dot(dl, dl), 1e-6);
  let ls = dl * inverseSqrt(d2);
  let ct = dot(-ls, normalize(F.scopeA.xyz));
  let nls = dot(n, ls);
  if (ct > 0.0 && nls > 0.0) {
    let I = F.scopeI.rgb * pow(ct, F.scopeP.w);
    let E = I * nls / d2;
    let fs = schlick(f0, dot(v, normalize(v + ls)));
    col += E * (D.base.rgb * select(vec3f(1.0) - fs, vec3f(0.0), metal) / PI + fs * ggx(n, v, ls, a));
  }
  if ((flags & 8u) != 0u) { return vec4f(mix(col, D.emit.rgb, 0.5), D.emit.w); }
  return vec4f(col, 1.0);
}

// ---- display: exposure, Khronos PBR Neutral, sRGB
@group(0) @binding(0) var hdr : texture_2d<f32>;
@group(0) @binding(1) var<uniform> T : vec4f;    // x exposure

@vertex fn vsFull(@builtin(vertex_index) k : u32) -> @builtin(position) vec4f {
  var p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[k], 0.0, 1.0);
}
fn neutral(c0 : vec3f) -> vec3f {
  let sc = 0.8 - 0.04;
  let des = 0.15;
  let x = min(c0.r, min(c0.g, c0.b));
  let off = select(0.04, x - 6.25 * x * x, x < 0.08);
  var c = c0 - off;
  let peak = max(c.r, max(c.g, c.b));
  if (peak < sc) { return c; }
  let d = 1.0 - sc;
  let np = 1.0 - d * d / (peak + d - sc);
  c *= np / peak;
  let g = 1.0 - 1.0 / (des * (peak - np) + 1.0);
  return mix(c, vec3f(np), g);
}
fn srgb(c : vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}
@fragment fn fsTone(@builtin(position) p : vec4f) -> @location(0) vec4f {
  let c = textureLoad(hdr, vec2i(p.xy), 0).rgb * T.x;
  return vec4f(srgb(clamp(neutral(max(c, vec3f(0.0))), vec3f(0.0), vec3f(1.0))), 1.0);
}
`;

// --------------------------------------------------------------------------- small math (column-major 4x4)
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
function mul4(A, B) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += A[k * 4 + r] * B[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
/** View matrix (right-handed, camera looks down -z), column-major. */
export function lookAt(eye, target, up) {
  const f = nrm(sub(target, eye));
  let s = cross(f, up);
  if (Math.hypot(...s) < 1e-9) s = cross(f, [0, 1, 0]);
  s = nrm(s);
  const u = cross(s, f);
  return new Float32Array([s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0, -dot(s, eye), -dot(u, eye), dot(f, eye), 1]);
}
/** WebGPU perspective (depth 0..1), column-major. */
export function perspective(vfovRad, aspect, near, far) {
  const f = 1 / Math.tan(vfovRad / 2);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0]);
}
function ortho(l, r, b, t, n, f) {
  return new Float32Array([2 / (r - l), 0, 0, 0, 0, 2 / (t - b), 0, 0, 0, 0, 1 / (n - f), 0, -(r + l) / (r - l), -(t + b) / (t - b), n / (n - f), 1]);
}
const I4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 5.960464477539063e-8;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}
const HALF = (() => { const t = new Float32Array(65536); for (let i = 0; i < 65536; i++) t[i] = halfToFloat(i); return t; })();

// --------------------------------------------------------------------------- materials
function material(physics, name) {
  const m = (physics?.materials || {})[name] || {};
  const v = (k, d) => (m[k] != null ? U(m[k]) : d);
  const ior = Number(v('ior', 1.5));
  const f0d = ((ior - 1) / (ior + 1)) ** 2;
  if (name === 'emitter' || v('model') === 'emitter') return { base: [0, 0, 0], f0: [0.04, 0.04, 0.04], rough: 0.3, flags: 2 };
  if (v('model') === 'conductor' || v('model') === 'coated_conductor') {
    const f0 = v('f0', [0.56, 0.57, 0.58]);
    return { base: [0, 0, 0], f0, rough: Number(v('coat_roughness', v('roughness', 0.3))), flags: 4 };
  }
  const rho = v('rho_d', v('albedo', v('rho_d_paper', [0.5, 0.5, 0.5])));
  return { base: rho, f0: [f0d, f0d, f0d], rough: Number(v('roughness', v('roughness_paper', 0.5))), flags: 0 };
}

// --------------------------------------------------------------------------- the renderer
export class GodView {
  static async create(opts) { const g = new GodView(); await g._init(opts); return g; }

  get kind() { return 'placeholder'; }
  get label() { return 'placeholder raster'; }

  async _init({ canvas, scene, physics, godview, device, log }) {
    const t0 = performance.now();
    this.log = log || (() => {});
    this.canvas = canvas;
    this.physics = physics || {};
    this.gv = godview || {};
    let bundle = scene;
    if (!bundle?.meshes) bundle = await loadScene(scene?.base || 'assets/');
    this.scene = bundle.scene; this.meshes = bundle.meshes;
    if (device) this.device = device;
    else {
      if (!navigator.gpu) throw new Error('WebGPU is not available');
      const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!ad) throw new Error('no WebGPU adapter');
      const feats = ['timestamp-query'].filter((f) => ad.features.has(f));
      this.device = await ad.requestDevice({ requiredFeatures: feats });
      this.ownDevice = true;
    }
    const dev = this.device;
    dev.lost?.then((i) => { this.lostInfo = i?.message || 'device lost'; });
    this.ctx = canvas ? canvas.getContext('webgpu') : null;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx?.configure({ device: dev, format: this.format, alphaMode: 'opaque' });
    this.module = dev.createShaderModule({ code: WGSL, label: 'placeholder godview' });
    this._layouts();
    this._geometry();
    this._shadowTarget();
    this.cam = { eye: [0.6, 0.2, 0.4], target: [-0.05, 0.027, 0.1], up: [0, 0, 1], vfov_deg: Number(this.gv.vfov_deg) || 35 };
    this.cutaway = (this.gv.cutaway?.default ?? 'on') !== 'off' && this.gv.cutaway?.default !== false;
    this.exposure = Number(U(this.gv.display?.exposure)) || 0.5;
    this.frames = 0; this.gpuMs = []; this.cpuMs = [];
    this._timer();
    this.loadMs = performance.now() - t0;
    this.log(`placeholder god view ready in ${this.loadMs.toFixed(0)} ms, ${this.draws.length} draws`);
  }

  _layouts() {
    const dev = this.device;
    this.frameBuf = dev.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bgl0 = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } }] });
    this.bgl1 = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true } }] });
    this.bglS = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }] });
    const vbufs = [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
      { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] }];
    const main = (blend, depthWrite) => dev.createRenderPipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.bgl0, this.bgl1] }),
      vertex: { module: this.module, entryPoint: 'vsMain', buffers: vbufs },
      fragment: { module: this.module, entryPoint: 'fsMain', targets: [{ format: HDR, ...(blend ? { blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } } : {}) }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: depthWrite, depthCompare: 'less' },
      multisample: { count: MSAA },
    });
    this.pMain = main(false, true);
    this.pGhost = main(true, false);
    this.pShadow = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.bglS, this.bgl1] }),
      vertex: { module: this.module, entryPoint: 'vsShadow', buffers: [vbufs[0]] },
      fragment: { module: this.module, entryPoint: 'fsShadow', targets: [] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2 },
    });
    this.toneL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }] });
    this.toneBuf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pTone = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.toneL] }),
      vertex: { module: this.module, entryPoint: 'vsFull' },
      fragment: { module: this.module, entryPoint: 'fsTone', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.cmpSampler = dev.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
  }

  _geometry() {
    const dev = this.device, sc = this.scene;
    const buf = (data, usage) => { const b = dev.createBuffer({ size: Math.max(16, (data.byteLength + 3) & ~3), usage: usage | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(b, 0, data); return b; };
    this.draws = [];
    const add = (name, P, N, I, mat, body, flagsExtra = 0) => {
      const m = material(this.physics, mat);
      this.draws.push({ name, mat, body, count: I.length, pos: buf(P, GPUBufferUsage.VERTEX), nrm: buf(N, GPUBufferUsage.VERTEX), idx: buf(I, GPUBufferUsage.INDEX),
        m, flags: m.flags | flagsExtra, visible: !body });
    };
    for (const meta of sc.meshes) {
      const g = this.meshes[meta.name];
      if (!g) continue;
      const cut = (this.gv.cutaway?.materials || ['pla_matte_white']).includes(meta.material) ? 1 : 0;
      add(meta.name, g.P, g.N, g.I, meta.material, meta.body || null, cut);
    }
    // the bench: a big plane (godview.yaml bench.extent) at the scene's bench z
    const z = Number(sc.bench?.z ?? 0), ex = this.gv.bench?.extent || [[-1, -1], [1, 1]];
    const [[x0, y0], [x1, y1]] = ex;
    add('bench', new Float32Array([x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z]), new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      new Uint32Array([0, 1, 2, 0, 2, 3]), sc.bench?.material || 'bench_white', null);
    // per-draw uniforms (dynamic offsets)
    this.drawBuf = dev.createBuffer({ size: DRAW_STRIDE * (this.draws.length + 8), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.drawBG = dev.createBindGroup({ layout: this.bgl1, entries: [{ binding: 0, resource: { buffer: this.drawBuf, size: 112 } }] });
    this.draws.forEach((d, k) => { d.slot = k; this._writeDraw(d, I4); });
    // scope LED in the scope body frame K
    const s = sc.scope || {};
    this.ledK = s.objective_centre_K ? [0, -0.0015, -0.003] : [0, 0, 0];
    this.ledAxisK = s.emission_axis_K || s.view_dir_K || [0, 0.5, 0.866];
  }

  _writeDraw(d, model, ghost = null) {
    const f = new Float32Array(28);
    f.set(model, 0);
    const m = d.m;
    f.set([...m.base, d.flags | (ghost ? 8 : 0)], 16);
    f.set([...m.f0, m.rough], 20);
    const em = d.flags & 2 ? [40, 40, 40] : ghost ? ghost.color : [0, 0, 0];
    f.set([...em, ghost ? ghost.alpha : 1], 24);
    this.device.queue.writeBuffer(this.drawBuf, (ghost ? ghost.slot : d.slot) * DRAW_STRIDE, f);
  }

  _shadowTarget() {
    const dev = this.device;
    this.shadowTex = dev.createTexture({ size: [SHADOW, SHADOW], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.shadowFrameBuf = dev.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bg0 = dev.createBindGroup({ layout: this.bgl0, entries: [
      { binding: 0, resource: { buffer: this.frameBuf } }, { binding: 1, resource: this.shadowTex.createView() }, { binding: 2, resource: this.cmpSampler }] });
    this.bgS = dev.createBindGroup({ layout: this.bglS, entries: [{ binding: 0, resource: { buffer: this.frameBuf } }] });
  }

  _timer() {
    const dev = this.device;
    this.ts = dev.features.has('timestamp-query');
    if (!this.ts) return;
    this.qs = dev.createQuerySet({ type: 'timestamp', count: 2 });
    this.qResolve = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.qRead = [0, 1].map(() => dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    this.qFree = [...this.qRead];
  }

  // ------------------------------------------------------------------- API
  setPoses(poses) {
    const B = poses?.bodies || {};
    this.bodies = B;
    for (const d of this.draws) {
      if (!d.body) continue;
      const T = B[d.body];
      d.visible = !!T;
      if (T) this._writeDraw(d, Float32Array.from(T));
    }
    const T = B.scope;
    if (T) {
      const ap = (p) => [T[0] * p[0] + T[4] * p[1] + T[8] * p[2] + T[12], T[1] * p[0] + T[5] * p[1] + T[9] * p[2] + T[13], T[2] * p[0] + T[6] * p[1] + T[10] * p[2] + T[14]];
      const av = (p) => [T[0] * p[0] + T[4] * p[1] + T[8] * p[2], T[1] * p[0] + T[5] * p[1] + T[9] * p[2], T[2] * p[0] + T[6] * p[1] + T[10] * p[2]];
      this.led = ap(this.ledK); this.ledAxis = nrm(av(this.ledAxisK));
    }
    this.dirty = true;
  }

  /** Optional (page extension): predicted tools as translucent ghosts. ghosts: {bodies} or null. */
  setGhosts(poses) {
    this.ghosts = [];
    if (!poses?.bodies) return;
    let slot = this.draws.length;
    for (const d of this.draws) {
      const T = d.body && poses.bodies[d.body];
      if (!T || !/^(L|R)\//.test(d.body) || /cannula/.test(d.body) || slot >= this.draws.length + 8) continue;
      const g = { d, slot: slot++, color: [0.18, 0.36, 0.83], alpha: 0.45 };
      this._writeDraw(d, Float32Array.from(T), g);
      this.ghosts.push(g);
    }
  }

  setCamera({ eye, target, up, vfov_deg }) {
    this.cam = { eye: [...eye], target: [...target], up: up ? [...up] : [0, 0, 1], vfov_deg: vfov_deg ?? this.cam.vfov_deg };
  }

  setCutaway(on) { this.cutaway = !!on; }
  setExposure(e) { if (Number.isFinite(e) && e > 0) this.exposure = e; }

  _frameUniform(aspect) {
    const c = this.cam, gv = this.gv;
    const near = 0.02, far = 6.0;
    const V = lookAt(c.eye, c.target, c.up);
    const P = perspective((c.vfov_deg || 35) * DEG, aspect, near, far);
    const L = gv.lights || {};
    const kaz = (L.key?.from_az_deg ?? 150) * DEG, kel = (L.key?.from_el_deg ?? 65) * DEG;
    const kd = [Math.cos(kel) * Math.cos(kaz), Math.cos(kel) * Math.sin(kaz), Math.sin(kel)];
    const centre = gv.centre || [-0.05, 0.027, 0.1];
    const lc = [centre[0], centre[1], Number(this.scene.bench?.z ?? 0)];
    const lv = lookAt([lc[0] + kd[0] * 1.5, lc[1] + kd[1] * 1.5, lc[2] + kd[2] * 1.5], lc, Math.abs(kd[2]) > 0.99 ? [0, 1, 0] : [0, 0, 1]);
    const R = 0.42;
    const lp = ortho(-R, R, -R, R, 0.5, 2.6);
    const n = nrm(sub(c.eye, centre));
    const off = Number(gv.cutaway?.offset ?? 0.02);
    const keyE = L.key?.irradiance || [2, 2, 2];
    const env = L.environment?.radiance || [1, 1, 1];
    // the scope LED: physics L0 x area x gain (studio units); without a gain, the axial intensity that makes the floor
    // under the scope read about 3x the key-lit exterior PLA (the godview.yaml rule), at ~0.1 m
    const gain = U(L.scope?.gain);
    const em = (this.physics.materials || {}).emitter || {};
    const L0 = U(em.L0) || [10091.9, 9193.2, 8830.7];
    const area = Number(this.scene.scope?.emitter_area_m2) || 4.79e-5;
    const I0 = gain != null && Number.isFinite(Number(gain)) ? L0.map((x) => x * area * Number(gain)) : [0.16, 0.155, 0.15];
    const f = new Float32Array(128);
    f.set(mul4(P, V), 0);
    f.set(mul4(lp, lv), 16);
    f.set([...c.eye, 1], 32);
    f.set([...n, dot(n, centre) + off], 36);
    f.set([...kd, 1 / SHADOW], 40);
    f.set([...keyE, this.cutaway ? 1 : 0], 44);
    f.set([...env, 0], 48);
    f.set([...(this.led || [0, 0, -10]), 1.6], 52);
    f.set([...(this.ledAxis || [0, 0, -1]), 0], 56);
    f.set([...I0, 0], 60);
    f.set([this.exposure, 0, 0, 0], 64);
    return f;
  }

  _targets(w, h) {
    const key = `${w}x${h}`;
    if (this.tg?.key === key) return this.tg;
    for (const t of ['msaa', 'depth', 'hdr']) this.tg?.[t]?.destroy();
    const dev = this.device;
    const tg = { key, w, h,
      msaa: dev.createTexture({ size: [w, h], format: HDR, sampleCount: MSAA, usage: GPUTextureUsage.RENDER_ATTACHMENT }),
      depth: dev.createTexture({ size: [w, h], format: 'depth32float', sampleCount: MSAA, usage: GPUTextureUsage.RENDER_ATTACHMENT }),
      hdr: dev.createTexture({ size: [w, h], format: HDR, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC }) };
    tg.toneBG = dev.createBindGroup({ layout: this.toneL, entries: [{ binding: 0, resource: tg.hdr.createView() }, { binding: 1, resource: { buffer: this.toneBuf } }] });
    this.tg = tg;
    return tg;
  }

  _encodeScene(enc, tg, timed) {
    const dev = this.device;
    dev.queue.writeBuffer(this.frameBuf, 0, this._frameUniform(tg.w / tg.h));
    const sp = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: this.shadowTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      ...(timed ? { timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: 0 } } : {}) });
    sp.setPipeline(this.pShadow); sp.setBindGroup(0, this.bgS);
    for (const d of this.draws) {
      if (!d.visible || d.flags & 2 || d.name === 'bench') continue;
      sp.setBindGroup(1, this.drawBG, [d.slot * DRAW_STRIDE]);
      sp.setVertexBuffer(0, d.pos); sp.setIndexBuffer(d.idx, 'uint32'); sp.drawIndexed(d.count);
    }
    sp.end();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: tg.msaa.createView(), resolveTarget: tg.hdr.createView(), loadOp: 'clear', storeOp: 'discard', clearValue: { r: 0.9, g: 0.9, b: 0.9, a: 1 } }],
      depthStencilAttachment: { view: tg.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    rp.setPipeline(this.pMain); rp.setBindGroup(0, this.bg0);
    for (const d of this.draws) {
      if (!d.visible) continue;
      rp.setBindGroup(1, this.drawBG, [d.slot * DRAW_STRIDE]);
      rp.setVertexBuffer(0, d.pos); rp.setVertexBuffer(1, d.nrm); rp.setIndexBuffer(d.idx, 'uint32'); rp.drawIndexed(d.count);
    }
    if (this.ghosts?.length) {
      rp.setPipeline(this.pGhost);
      for (const g of this.ghosts) {
        rp.setBindGroup(1, this.drawBG, [g.slot * DRAW_STRIDE]);
        rp.setVertexBuffer(0, g.d.pos); rp.setVertexBuffer(1, g.d.nrm); rp.setIndexBuffer(g.d.idx, 'uint32'); rp.drawIndexed(g.d.count);
      }
    }
    rp.end();
  }

  render() {
    if (!this.ctx || this.lostInfo) return;
    const cv = this.canvas, dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    if (cv.clientWidth && cv.clientHeight) {
      const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    }
    if (!cv.width || !cv.height) return;
    const t0 = performance.now();
    const tg = this._targets(this.canvas.width, this.canvas.height);
    const dev = this.device;
    const timed = this.ts && this.qFree.length > 0;
    const enc = dev.createCommandEncoder({ label: 'godview' });
    this._encodeScene(enc, tg, timed);
    dev.queue.writeBuffer(this.toneBuf, 0, new Float32Array([this.exposure, 0, 0, 0]));
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 1, g: 1, b: 1, a: 1 } }],
      ...(timed ? { timestampWrites: { querySet: this.qs, endOfPassWriteIndex: 1 } } : {}) });
    tp.setPipeline(this.pTone); tp.setBindGroup(0, tg.toneBG); tp.draw(3); tp.end();
    let rd = null;
    if (timed) { rd = this.qFree.pop(); enc.resolveQuerySet(this.qs, 0, 2, this.qResolve, 0); enc.copyBufferToBuffer(this.qResolve, 0, rd, 0, 16); }
    dev.queue.submit([enc.finish()]);
    if (rd) {
      rd.mapAsync(GPUMapMode.READ).then(() => {
        const q = new BigUint64Array(rd.getMappedRange().slice(0));
        rd.unmap(); this.qFree.push(rd);
        const ms = Number(q[1] - q[0]) / 1e6;
        if (ms > 0 && ms < 1000) { this.gpuMs.push(ms); if (this.gpuMs.length > 240) this.gpuMs.shift(); }
      }).catch(() => { this.qFree.push(rd); });
    }
    this.frames++;
    this.cpuMs.push(performance.now() - t0); if (this.cpuMs.length > 240) this.cpuMs.shift();
  }

  async captureLinear(width, height) {
    const dev = this.device;
    const tg = this._targets(width, height);
    const enc = dev.createCommandEncoder({ label: 'godview capture' });
    this._encodeScene(enc, tg, false);
    const bpr = Math.ceil(width * 8 / 256) * 256;
    const rb = dev.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: tg.hdr }, { buffer: rb, bytesPerRow: bpr }, [width, height]);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const u16 = new Uint16Array(rb.getMappedRange());
    const out = new Float32Array(width * height * 3);
    const row = bpr / 2;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const s = y * row + x * 4, o = (y * width + x) * 3;
      out[o] = HALF[u16[s]]; out[o + 1] = HALF[u16[s + 1]]; out[o + 2] = HALF[u16[s + 2]];
    }
    rb.unmap(); rb.destroy();
    this.tg = null;      // the display re-allocates its own size
    for (const t of ['msaa', 'depth', 'hdr']) tg[t].destroy();
    return out;
  }

  timings() {
    const s = [...this.gpuMs].sort((a, b) => a - b);
    return { gpu_ms_last: this.gpuMs.at(-1) ?? null, gpu_ms_median: s.length ? s[s.length >> 1] : null, frames: this.frames,
      cpu_ms_median: [...this.cpuMs].sort((a, b) => a - b)[this.cpuMs.length >> 1] ?? null, load_ms: this.loadMs, renderer: 'placeholder' };
  }

  destroy() {
    for (const d of this.draws || []) { d.pos.destroy(); d.nrm.destroy(); d.idx.destroy(); }
    for (const t of ['msaa', 'depth', 'hdr']) this.tg?.[t]?.destroy();
    this.shadowTex?.destroy();
    if (this.ownDevice) this.device.destroy();
  }
}

export default GodView;
