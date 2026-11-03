// The simulated camera views of the twin page's side column (docs/WEB_TWIN.md §21, Sim mode), drawn by the god view's
// own raster (godview_pt3) on its device: no path tracing, no second renderer. Each view:
//   1. the scope-LED-only shadow map (no cutaway: the dome is closed), its static casters cached per scope pose;
//   2. the main raster pipeline from the camera's pose into a symmetric pinhole that covers the image (camview.js
//      coveringPinhole), with only the scope LED: no environment, no key, no cutaway. Its indirect light is a SECOND
//      vertex bake (the same bake kernel and smoothing, scope light only, the inside of the dome only), filled
//      progressively in idle time (stepBake) and swapped in whole when done; moving parts (instruments) get the bake's
//      mean as their ambient (the dome is an integrating sphere: its inside radiance is nearly uniform);
//   3. the display: KB4 per pixel for CAM1-CAM3 (inverse by Newton, sampled from the pinhole), the pinhole + image circle
//      for SCOPE, a per-view auto-exposure (camview.js exposureFrom, from a 64 x 36 luminance readback), Khronos PBR
//      Neutral and sRGB as the main view.
// GPU cost is measured with its own timestamp queries (timings()).
import { ptSource, rasterSource } from './shaders.js';
import { cameraModel, coveringPinhole, pinholeSize, exposureFrom, settleExposure, interiorVertices } from './camview.js';

const DEG = Math.PI / 180;
const AE_W = 64, AE_H = 36;
const GU_FLOATS = 52;                                 // godview.js GU_BYTES / 4
const RU_FLOATS = 64;                                 // godview.js RU_BYTES / 4
const SM_SIZE = 2048;                                 // the scope light's map for the camera views (the main view's: 1024)
const BAKE_MIN = 256;   // the main bake's 4096 floor ran 160 ms per chunk on the M4 with the stand's 1.58 M triangles

const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const mulRv = (R, v) => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
function mul4(A, B) {
  const C = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += A[k * 4 + r] * B[c * 4 + k]; C[c * 4 + r] = s; }
  return C;
}
/** column-major view-projection (GL view looking down -Z, depth [0, 1]) with half extents tx, ty (tan) */
function viewProj(eye, x, y, z, tx, ty, near, far) {
  const V = [x[0], y[0], -z[0], 0, x[1], y[1], -z[1], 0, x[2], y[2], -z[2], 0, -dot3(x, eye), -dot3(y, eye), dot3(z, eye), 1];
  const P = [1 / tx, 0, 0, 0, 0, 1 / ty, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
  return mul4(P, V);
}

// the bake kernel's twin for a vertex LIST (backList binding = the dome's inside vertices; entry j bakes vertex
// list[j]'s front into irr[vertex]); appended to ptSource, so pathL is the bake's exactly (mode 1: the outlet's direct
// light is left to the raster, shadow-mapped)
const CAM_BAKE_WGSL = /* wgsl */ `
@compute @workgroup_size(64)
fn camBakeMain(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  let j = gu.tile.x + gid.x;
  if (gid.x >= gu.tile.z || j >= gu.tile.y) { return; }
  let vi = backList[j];
  let p = verts[2u * vi].xyz;
  let n = normalize(verts[2u * vi + 1u].xyz);
  let t = basisT(n);
  let bb = basisB(n);
  var sum = vec3f(0.0);
  for (var k = 0u; k < gu.opt.z; k++) {
    var st = hash3(vi, gu.size.z + k, 0x9e3779b9u ^ gu.opt.y);
    let u1 = rnd(&st);
    let ph = 2.0 * PI * rnd(&st);
    let r = sqrt(u1);
    let l = normalize(t * (r * cos(ph)) + bb * (r * sin(ph)) + n * sqrt(max(0.0, 1.0 - u1)));
    var c = pathL(p + n * 2e-5, l, &st, 1u, false, 0u);
    if (!(c.x == c.x && c.y == c.y && c.z == c.z) || max3(c) > 1e6) { c = vec3f(0.0); }
    // fireflies (the LED's caustics off the glossy floor and steel, found by BSDF sampling) clamped to gu.disp.x: the
    // smoothing spread them into bright blotches on the dome walls (pod check 2026-11-01)
    let cm = max3(c);
    if (gu.disp.x > 0.0 && cm > gu.disp.x) { c = c * (gu.disp.x / cm); }
    sum += c;
  }
  var a = irr[vi];
  if (gu.size.z == 0u) { a = vec4f(0.0); }
  irr[vi] = a + vec4f(sum, f32(gu.opt.z));
}
`;

const DISPLAY_WGSL = /* wgsl */ `
struct CU { osz: vec4f, img: vec4f, K: vec4f, D: vec4f, nat: vec4f, pin: vec4f }
// osz: canvas w, h; img: image rect (x, y, w, h, canvas px); K: fx fy cx cy; D: k1..k4; nat: native w, h, thetaMax,
// model (0 pinhole, 1 kb4); pin: pinhole tan half extents tx, ty, exposure, image-circle radius (native px, 0 none)
@group(0) @binding(0) var<uniform> cu: CU;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
fn neutral(c0: vec3f) -> vec3f {
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
@fragment fn fs(@builtin(position) fp: vec4f) -> @location(0) vec4f {
  let frame = vec4f(0.0047, 0.0050, 0.0047, 1.0);          // rgb(12, 13, 12) linear: the tile's frame
  let q = (fp.xy - cu.img.xy) / cu.img.zw;
  if (q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0) { return vec4f(oetf(frame.r), oetf(frame.g), oetf(frame.b), 1.0); }
  let u = q * cu.nat.xy - 0.5;
  if (cu.pin.w > 0.0 && length(u - (cu.nat.xy * 0.5 - 0.5)) > cu.pin.w) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let m = (u - cu.K.zw) / cu.K.xy;
  var d = vec3f(m, 1.0);
  if (cu.nat.w > 0.5) {
    let td = length(m);
    var th = td;
    for (var i = 0; i < 6; i++) {
      let t2 = th * th;
      let f = th * (1.0 + t2 * (cu.D.x + t2 * (cu.D.y + t2 * (cu.D.z + t2 * cu.D.w)))) - td;
      let df = 1.0 + t2 * (3.0 * cu.D.x + t2 * (5.0 * cu.D.y + t2 * (7.0 * cu.D.z + t2 * 9.0 * cu.D.w)));
      th = th - f / max(df, 1e-4);
    }
    if (th > cu.nat.z) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    let s = select(1.0, sin(th) / td, td > 1e-6);
    d = vec3f(m * s, cos(th));
  }
  if (d.z <= 1e-4) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let uv = vec2f(0.5 + 0.5 * d.x / d.z / cu.pin.x, 0.5 + 0.5 * d.y / d.z / cu.pin.y);
  let c = textureSampleLevel(tex, smp, uv, 0.0).rgb;
  let t = neutral(max(c * cu.pin.z, vec3f(0.0)));
  return vec4f(oetf(t.r), oetf(t.g), oetf(t.b), 1.0);
}
`;

const AE_WGSL = /* wgsl */ `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outL: array<f32>;
@compute @workgroup_size(8, 4)
fn aeMain(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= ${AE_W}u || gid.y >= ${AE_H}u) { return; }
  let sz = vec2f(textureDimensions(tex));
  let p = vec2i((vec2f(gid.xy) + 0.5) / vec2f(${AE_W}.0, ${AE_H}.0) * sz);
  let c = textureLoad(tex, p, 0).rgb;
  outL[gid.y * ${AE_W}u + gid.x] = dot(c, vec3f(0.2126, 0.7152, 0.0722));
}
`;

/**
 * Runs {off, count} of the static triangles (S.tris: 16 floats each, v0 | e1 | e2 | mesh) whose centroid lies inside the
 * dome's ellipsoid grown by `grow` (or the floor under it), gaps between runs merged until at most maxRuns remain (a few
 * outside triangles drawn are harmless; each run is one draw).
 */
export function insideRuns(tris, nStatic, ell, { grow = 1.15, below = 0.03, maxRuns = 48 } = {}) {
  if (!ell || !tris || !nStatic) return [{ off: 0, count: nStatic }];
  const [c, h] = ell, runs = [];
  let start = -1;
  for (let k = 0; k <= nStatic; k++) {
    let inside = false;
    if (k < nStatic) {
      const o = 16 * k;
      const x = tris[o] + (tris[o + 4] + tris[o + 8]) / 3, y = tris[o + 1] + (tris[o + 5] + tris[o + 9]) / 3, z = tris[o + 2] + (tris[o + 6] + tris[o + 10]) / 3;
      if (z >= c[2] - below) {
        const qx = (x - c[0]) / h[0], qy = (y - c[1]) / h[1], qz = Math.max(0, z - c[2]) / h[2];
        inside = qx * qx + qy * qy + qz * qz <= grow * grow;
      }
    }
    if (inside && start < 0) start = k;
    if (!inside && start >= 0) { runs.push({ off: start, count: k - start }); start = -1; }
  }
  if (runs.length > maxRuns) {
    const gaps = runs.slice(1).map((r, i) => r.off - (runs[i].off + runs[i].count)).sort((a, b) => a - b);
    const cut = gaps[runs.length - maxRuns - 1];
    const out = [{ ...runs[0] }];
    for (let i = 1; i < runs.length; i++) {
      const r = runs[i], last = out[out.length - 1], gap = r.off - (last.off + last.count);
      if (gap <= cut) last.count = r.off + r.count - last.off; else out.push({ ...r });
    }
    return out;
  }
  return runs;
}

export class CameraViews {
  /** g: the GodView (godview_pt3/godview.js), already initialised. */
  constructor(g, opts = {}) {
    this.g = g; this.dev = g.device;
    this.opts = { bakeRays: 64, bakeMs: 3, smoothPasses: 6, tileMaxPx: 600000, largeMaxPx: 2400000, ...opts };
    this.ready = null; this.views = {}; this.targets = {}; this.ev = {}; this.gpuMs = []; this.bakeGpu = [];
    this.bake = { sample: 0, pos: 0, chunk: BAKE_MIN, msPerRay: null, stale: true, key: null, done: false, version: 0, inFlight: false };
    this.fill = [0, 0, 0];
    this.frames = 0;
    this.smKey = null; this.smVer = -1;
  }

  /** Compile the camera bake and display pipelines (async, once); render() draws nothing until they exist. */
  init() {
    if (this.ready) return this.ready;
    this.ready = this._init().catch((e) => { this.error = String(e?.message || e); this.g.log('camera views unavailable:', this.error); throw e; });
    return this.ready;
  }

  async _init() {
    const g = this.g, dev = this.dev, S = g.S, ST = GPUBufferUsage.STORAGE;
    try { performance.mark('gv:camviews:start'); } catch { /* old browser */ }
    // the inside of the dome: the only static vertices a dome camera (or the scope) sees
    this.list = interiorVertices(S.verts, S.staticVerts, g.domeEll);
    // the static triangles a dome camera can see (and the scope light can shadow): the inside of the dome, as runs of the
    // static draw; the bench, the stand and everything else outside the closed dome are not drawn (pod check 2026-11-01:
    // the stand's 1.58 M triangles made a tile 3 ms)
    this.runs = insideRuns(S.tris, S.staticTris ?? g.draws[0].count, g.domeEll);
    this.runBufs = this.runs.map((r) => [0, 1].map((kind) => g._buf(new Uint32Array([0xffffffff, r.off, kind, 0]), GPUBufferUsage.UNIFORM, 'cam run')));
    const n = g.bakeTotal;
    this.irrAcc = dev.createBuffer({ size: Math.max(16, n * 16), usage: ST | GPUBufferUsage.COPY_DST, label: 'cam irr (bake sums)' });
    this.irrTmp = dev.createBuffer({ size: Math.max(16, n * 16), usage: ST, label: 'cam irr (smoothing)' });
    this.irr = dev.createBuffer({ size: Math.max(16, n * 16), usage: ST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'cam irr (raster)' });
    this.listBuf = g._buf(this.list.length ? this.list : new Uint32Array([0]), ST, 'cam bake list');
    this.bakeGu = dev.createBuffer({ size: GU_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.gu = dev.createBuffer({ size: GU_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.ru = dev.createBuffer({ size: RU_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // bake: the godview bake's module + one entry point over a vertex list
    const bMod = await g._module(ptSource(g.consts) + CAM_BAKE_WGSL, 'godview_pt3 camera bake');
    this.bakePipe = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: bMod, entryPoint: 'camBakeMain', constants: { CUTAWAY: 1 } } });
    this.bakeG0 = dev.createBindGroup({ layout: this.bakePipe.getBindGroupLayout(0), entries: ['nodes', 'tris', 'verts', 'params', 'dbg'].map((k, i) => ({ binding: i, resource: { buffer: g.bufs[k] } })) });
    this.bakeG1 = dev.createBindGroup({ layout: this.bakePipe.getBindGroupLayout(1), entries: [
      { binding: 0, resource: { buffer: g.bufs.scn } }, { binding: 1, resource: { buffer: this.bakeGu } },
      { binding: 3, resource: { buffer: dev.createBuffer({ size: 16, usage: ST }) } },
      { binding: 4, resource: { buffer: this.irrAcc } }, { binding: 5, resource: { buffer: this.listBuf } }] });
    const smG = (src, dst) => dev.createBindGroup({ layout: g.smoothPipe.getBindGroupLayout(0), entries: [g.bufs.verts, g.bufs.adjStart, g.bufs.adjList, src, dst, g.bufs.smoothU, g.bufs.backList, g.bufs.backSlot].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    this.smoothGroups = [smG(this.irrAcc, this.irrTmp), smG(this.irrTmp, this.irr), smG(this.irr, this.irrTmp)];
    // raster: the god view's pipelines; group 0 with this view's uniforms and bake, no key map (key off)
    const smp = dev.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    this.rG0 = dev.createBindGroup({ layout: g.rasterPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: g.bufs.tris } }, { binding: 1, resource: { buffer: g.bufs.verts } }, { binding: 2, resource: { buffer: g.bufs.params } },
      { binding: 3, resource: { buffer: g.bufs.scn } }, { binding: 4, resource: { buffer: this.gu } }, { binding: 5, resource: { buffer: this.ru } },
      { binding: 6, resource: g.dummyDepth.createView() }, { binding: 7, resource: smp }, { binding: 8, resource: { buffer: this.irr } }, { binding: 9, resource: { buffer: g.bufs.backSlot } }] });
    // the camera views' own scope shadow pipeline: the main one's with a larger slope-scaled bias (the inside walls meet
    // the LED's rays at grazing angles; with the main view's bias they showed radial acne streaks, pod check 2026-11-01)
    const rMod = await g._module(rasterSource(g.consts, g.clip), 'godview_pt3 camera shadow');
    this.shadowPipe = await dev.createRenderPipelineAsync({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [g.rasterPipe.getBindGroupLayout(0), g.rasterPipe.getBindGroupLayout(1)] }),
      vertex: { module: rMod, entryPoint: 'vsShadow' },
      ...(g.clip ? {} : { fragment: { module: rMod, entryPoint: 'fsShadow', targets: [] } }),
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 8, depthBiasSlopeScale: 4 },
    });
    const SMU = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
    this.smStatic = dev.createTexture({ size: [SM_SIZE, SM_SIZE], format: 'depth32float', usage: SMU });
    this.sm = dev.createTexture({ size: [SM_SIZE, SM_SIZE], format: 'depth32float', usage: SMU });
    const l1 = g.rasterPipe.getBindGroupLayout(1);
    this.rG1 = g.drawBufs.map(([b0]) => dev.createBindGroup({ layout: l1, entries: [{ binding: 0, resource: { buffer: b0 } }, { binding: 1, resource: this.sm.createView() }] }));
    this.runG1 = this.runBufs.map(([b0]) => dev.createBindGroup({ layout: l1, entries: [{ binding: 0, resource: { buffer: b0 } }, { binding: 1, resource: this.sm.createView() }] }));
    this.runG1shadow = this.runBufs.map(([, b1]) => dev.createBindGroup({ layout: l1, entries: [{ binding: 0, resource: { buffer: b1 } }, { binding: 1, resource: g.dummyDepth.createView() }] }));
    // display + auto-exposure
    const dMod = await g._module(DISPLAY_WGSL, 'godview_pt3 camera display');
    this.dispPipe = await dev.createRenderPipelineAsync({ layout: 'auto', vertex: { module: dMod, entryPoint: 'vs' }, fragment: { module: dMod, entryPoint: 'fs', targets: [{ format: g.canvasFormat }] }, primitive: { topology: 'triangle-list' } });
    this.linSmp = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const aMod = await g._module(AE_WGSL, 'godview_pt3 camera AE');
    this.aePipe = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: aMod, entryPoint: 'aeMain' } });
    this.aeBuf = dev.createBuffer({ size: AE_W * AE_H * 4, usage: ST | GPUBufferUsage.COPY_SRC });
    this.aeRead = dev.createBuffer({ size: AE_W * AE_H * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.aeBusy = false;
    this.cuBufs = [0, 1, 2, 3].map(() => dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    if (g.hasTS) {
      this.ts = dev.createQuerySet({ type: 'timestamp', count: 4 });
      this.tsRes = dev.createBuffer({ size: 512, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.tsRead = [0, 1].map(() => dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
      this.tsBusy = [false, false];
    }
    try { performance.mark('gv:camviews:ready'); } catch { /* old browser */ }
    this.ok = true;
    this.g.log(`camera views ready: ${this.list.length} inside vertices of ${S.staticVerts} baked for the scope light`);
  }

  // ---------------------------------------------------------------- poses, lights
  _scopePose() {
    const I = this.g.instances?.[this.g.scopeIdx];
    return I ? { R: I.R || [[1, 0, 0], [0, 1, 0], [0, 0, 1]], p: I.p, q: I.q } : null;
  }

  _scopeKey() { const P = this._scopePose(); return P ? [...P.p, ...P.q] : null; }

  /** The scope light's shadow projection (godview.js _writeRU, the same perspective from the outlet). */
  _scopeProj() {
    const g = this.g, I = g.instances?.[g.scopeIdx];
    if (!I) return new Array(16).fill(0);
    const R = I.R, pc = mulRv(R, g.S.emitter.centroid).map((x, k) => x + I.p[k]), ax = norm3(mulRv(R, g.S.emitter.axis));
    const eye = pc.map((x, k) => x + ax[k] * 1e-4);
    const up = Math.abs(ax[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
    const z = ax, x = norm3(cross3(z, up)), y = cross3(x, z), t = Math.tan(60 * DEG);
    return viewProj(eye, x, y, z, t, t, 0.0015, 1.0);
  }

  _writeRU(VP) {
    const F = new Float32Array(RU_FLOATS);
    F.set(VP, 0); F.set(this._scopeProj(), 32);
    F.set([1.5 / 2048, 0, 0, 0], 48);
    F.set([...(this.g.S.emitter.normal || this.g.S.emitter.axis), 0], 52);
    // dome / domeH zero: no outside-the-dome ground term (domeH.w = 0)
    this.dev.queue.writeBuffer(this.ru, 0, F);
  }

  /** GU for the camera raster: the eye (for wo), no key, the fill as the "environment" of unbaked parts, no cut. */
  _writeGU(cam) {
    const F = new Float32Array(GU_FLOATS), U = new Uint32Array(F.buffer);
    F.set([...cam.axes.x, 0, ...cam.axes.y, 0, ...cam.axes.z, 0, ...cam.centre, 0], 0);
    F.set([...this.fill, 1], 36);                                // env: the dome's mean inside radiance (moving parts)
    U.set([0, 0, 1, this.g.scopeOn ? 1 : 0], 40);                 // no cut mask, caps hidden; scope light
    this.dev.queue.writeBuffer(this.gu, 0, F);
  }

  // ---------------------------------------------------------------- the scope-only bake (idle time)
  /** One chunk of the scope-only bake (<= opts.bakeMs GPU), when none is in flight. Returns true while work is left. */
  stepBake() {
    if (!this.bakePipe || this.bake.inFlight) return !this.bake.done;
    const g = this.g, B = this.bake, n = this.list.length;
    const key = this._scopeKey();
    if (!key || !n) return false;
    const moved = !B.key || key.some((v, i) => Math.abs(v - B.key[i]) > (i < 3 ? 1e-3 : 4e-3));
    if (moved) { B.key = key; B.sample = 0; B.pos = 0; B.done = false; }
    if (B.done) return false;
    const rays = 1;
    const count = Math.max(64, Math.min(n - B.pos, B.chunk));
    const F = new Float32Array(GU_FLOATS), U = new Uint32Array(F.buffer);
    U.set([1, 1, B.sample, g.opts.maxBounces], 20);
    U.set([B.pos, n, count, 1], 24);
    U.set([0, 0x7a11, rays, g.scopeOn ? 1 : 0], 40);
    // the firefly clamp: 10x the last bake's mean inside radiance, 3 before the first (that mean was 0.29 on the pod)
    const fy = 0.2126 * this.fill[0] + 0.7152 * this.fill[1] + 0.0722 * this.fill[2];
    F[44] = fy > 0 ? 10 * fy : 3.0;
    this.dev.queue.writeBuffer(this.bakeGu, 0, F);
    const enc = this.dev.createCommandEncoder();
    const tsK = this.ts && !this.tsBusy[1];
    const p = enc.beginComputePass(tsK ? { timestampWrites: { querySet: this.ts, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } } : {});
    p.setPipeline(this.bakePipe); p.setBindGroup(0, this.bakeG0); p.setBindGroup(1, this.bakeG1);
    p.dispatchWorkgroups(Math.ceil(count / 64)); p.end();
    const work = count * rays;
    // the chunk follows the measured GPU ms (x2 below half the budget, /2 above 1.5x): a small chunk's time is the latency
    // of its longest path inside the dome (up to 16 bounces), not count x a per-ray cost, so the per-ray model kept a 256
    // floor at 256 (pod check 2026-11-01: 2.7 ms for 256 vertices, the bake never finished)
    const adapt = (ms) => {
      B.msPerRay = ms / work;
      if (ms < 0.5 * this.opts.bakeMs) B.chunk = Math.min(2 * B.chunk, 1 << 20);
      else if (ms > 1.5 * this.opts.bakeMs) B.chunk = Math.max(BAKE_MIN, Math.floor(B.chunk / 2 / 64) * 64);
    };
    if (tsK) this._ts(enc, 1, (ms) => { this.bakeGpu.push(ms); if (this.bakeGpu.length > 100) this.bakeGpu.shift(); adapt(ms); });
    this.dev.queue.submit([enc.finish()]);
    this._tsFlush();
    B.inFlight = true;
    const t0 = performance.now();
    this.dev.queue.onSubmittedWorkDone().then(() => { B.inFlight = false; if (!this.ts) adapt(performance.now() - t0); });
    B.pos += count;
    if (B.pos >= n) {
      B.pos = 0; B.sample += rays;
      if (B.sample >= this.opts.bakeRays) { this._smooth(); B.done = true; }
    }
    return !B.done;
  }

  _smooth() {
    const g = this.g, dev = this.dev;
    const enc = dev.createCommandEncoder();
    const seq = [0, 1];
    for (let k = 2; k < this.opts.smoothPasses; k += 2) seq.push(2, 1);
    dev.queue.writeBuffer(g.bufs.smoothU, 8, new Uint32Array([0]));
    for (const gi of seq) {
      const p = enc.beginComputePass();
      p.setPipeline(g.smoothPipe); p.setBindGroup(0, this.smoothGroups[gi]); p.dispatchWorkgroups(Math.ceil(g.bakeTotal / 64)); p.end();
    }
    // the fill for moving parts: the mean of the smoothed inside radiance (read back once per bake, never waited on)
    const rb = dev.createBuffer({ size: g.bakeTotal * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(this.irr, 0, rb, 0, g.bakeTotal * 16);
    dev.queue.submit([enc.finish()]);
    this.bake.version++;
    rb.mapAsync(GPUMapMode.READ).then(() => {
      const A = new Float32Array(rb.getMappedRange());
      const s = [0, 0, 0]; let m = 0;
      for (const vi of this.list) { const w = A[4 * vi + 3]; if (w > 0) { s[0] += A[4 * vi] / w; s[1] += A[4 * vi + 1] / w; s[2] += A[4 * vi + 2] / w; m++; } }
      rb.unmap(); rb.destroy();
      if (m) this.fill = s.map((x) => x / m);
    }).catch(() => rb.destroy());
  }

  // ---------------------------------------------------------------- render
  _target(key, w, h) {
    const t = this.targets[key];
    if (t && t.w === w && t.h === h) return t;
    for (const k of ['ms', 'tex', 'depth']) t?.[k]?.destroy?.();
    const dev = this.dev, msaa = this.g.msaa;
    const n = { w, h };
    n.tex = dev.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    if (msaa > 1) n.ms = dev.createTexture({ size: [w, h], format: 'rgba16float', sampleCount: msaa, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    n.depth = dev.createTexture({ size: [w, h], format: 'depth32float', sampleCount: msaa, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    n.aeG = dev.createBindGroup({ layout: this.aePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: n.tex.createView() }, { binding: 1, resource: { buffer: this.aeBuf } }] });
    n.dispG = this.cuBufs.map((b) => dev.createBindGroup({ layout: this.dispPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: b } }, { binding: 1, resource: n.tex.createView() }, { binding: 2, resource: this.linSmp }] }));
    this.targets[key] = n;
    return n;
  }

  /** The scope light's shadow map without the cutaway: static casters per scope pose (all slices at once), the moving
   *  parts once per pose version; shared by every camera view. */
  _encodeShadow(enc, tsW) {
    const g = this.g;
    if (!g.scopeOn) return;
    const key = JSON.stringify(this._scopeKey());
    if (key !== this.smKey) {
      const sp = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: this.smStatic.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' }, ...tsW() });
      sp.setPipeline(this.shadowPipe); sp.setBindGroup(0, this.rG0);
      this.runs.forEach((r, i) => { sp.setBindGroup(1, this.runG1shadow[i]); sp.draw(3 * r.count); });
      sp.end();
      this.smKey = key; this.smVer = -1;
    }
    if (this.smVer === g.poseVer) return;
    enc.copyTextureToTexture({ texture: this.smStatic }, { texture: this.sm }, [SM_SIZE, SM_SIZE]);
    const sp = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: this.sm.createView(), depthLoadOp: 'load', depthStoreOp: 'store' }, ...tsW() });
    sp.setPipeline(this.shadowPipe); sp.setBindGroup(0, this.rG0);
    g.draws.forEach((d, i) => {
      if (d.inst === 0xffffffff || !g.instances[d.inst] || d.inst === g.scopeIdx) return;
      sp.setBindGroup(1, g.rG1shadow[i][1]); sp.draw(3 * d.count);
    });
    sp.end();
    this.smVer = g.poseVer;
  }

  /**
   * Draw camera `name` into each target {ctx, w, h} (WebGPU contexts configured on this device, canvas format); the
   * first target sets the render size (large: opts.large). Never waits on the GPU. Returns false until ready.
   */
  render(targets, name, { large = false } = {}) {
    if (!this.ok) { this.init(); return false; }          // every pipeline compiled (_init sets ok last)
    const g = this.g;
    const cam = cameraModel(g.scene, name, this._scopePose());
    if (!cam) return false;
    const V = this.views[name] ??= {};
    if (!V.pin || V.model !== cam.model) { V.pin = cam.model === 'kb4' ? coveringPinhole(cam) : { tx: (cam.size[0] / 2) / cam.K[0] * 1.002, ty: (cam.size[1] / 2) / cam.K[1] * 1.002 }; V.model = cam.model; }
    const pin = V.pin;
    const outW = Math.max(...targets.map((t) => Math.min(t.w, t.h * cam.size[0] / cam.size[1])));
    const [pw, ph] = pinholeSize(cam, pin, outW, large ? this.opts.largeMaxPx : this.opts.tileMaxPx);
    const T = this._target(`${name}:${large ? 'large' : 'tile'}`, pw, ph);   // one per view: no reallocation round-robin
    // the pinhole: OpenGL-style view with screen up = -y_C, so screen x = x_C; half extents tx, ty
    const eye = cam.centre, z = cam.axes.z, x = cam.axes.x, y = cam.axes.y.map((v) => -v);
    this._writeRU(viewProj(eye, x, y, z, pin.tx, pin.ty, cam.near, 3.0));
    this._writeGU(cam);
    const enc = this.dev.createCommandEncoder();
    const tsF = this.ts && !this.tsBusy[0];
    let first = true;
    const tsW = () => { if (!(tsF && first)) return {}; first = false; return { timestampWrites: { querySet: this.ts, beginningOfPassWriteIndex: 0 } }; };
    this._encodeShadow(enc, tsW);
    const msaa = g.msaa > 1;
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: (msaa ? T.ms : T.tex).createView(), ...(msaa ? { resolveTarget: T.tex.createView(), storeOp: 'discard' } : { storeOp: 'store' }), loadOp: 'clear', clearValue: [0, 0, 0, 1] }],
      depthStencilAttachment: { view: T.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      ...tsW(),
    });
    rp.setPipeline(g.rasterPipe); rp.setBindGroup(0, this.rG0);
    this.runs.forEach((r, i) => { rp.setBindGroup(1, this.runG1[i]); rp.draw(3 * r.count); });   // the static inside
    g.draws.forEach((d, i) => {
      if (d.inst === 0xffffffff || !g.instances[d.inst]) return;
      rp.setBindGroup(1, this.rG1[i]); rp.draw(3 * d.count);
    });
    rp.end();
    // auto-exposure: 64 x 36 luminance samples of this frame, read back without waiting; used from the next frame
    const ae = !this.aeBusy;
    if (ae) {
      const cp = enc.beginComputePass(); cp.setPipeline(this.aePipe); cp.setBindGroup(0, T.aeG); cp.dispatchWorkgroups(AE_W / 8, Math.ceil(AE_H / 4)); cp.end();
      enc.copyBufferToBuffer(this.aeBuf, 0, this.aeRead, 0, AE_W * AE_H * 4);
    }
    const ev = this.ev[name] ?? 1;
    targets.forEach((t, k) => {
      const ar = cam.size[0] / cam.size[1];
      const iw = Math.min(t.w, t.h * ar), ih = iw / ar;
      const cu = new Float32Array(24);
      cu.set([t.w, t.h, 0, 0, (t.w - iw) / 2, (t.h - ih) / 2, iw, ih, ...cam.K, ...cam.D, cam.size[0], cam.size[1], cam.thetaMax, cam.model === 'kb4' ? 1 : 0, pin.tx, pin.ty, ev, cam.maskR]);
      const slot = (large ? 2 : 0) + Math.min(1, k);
      this.dev.queue.writeBuffer(this.cuBufs[slot], 0, cu);
      const last = k === targets.length - 1;
      const dp = enc.beginRenderPass({ colorAttachments: [{ view: t.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
        ...(tsF && last ? { timestampWrites: { querySet: this.ts, endOfPassWriteIndex: 1 } } : {}) });
      dp.setPipeline(this.dispPipe); dp.setBindGroup(0, T.dispG[slot]); dp.draw(3); dp.end();
    });
    if (tsF) this._ts(enc, 0, (ms) => { this.gpuMs.push({ ms, large, px: pw * ph }); if (this.gpuMs.length > 200) this.gpuMs.shift(); });
    this.dev.queue.submit([enc.finish()]);
    this._tsFlush();
    if (ae) {
      this.aeBusy = true;
      this.aeRead.mapAsync(GPUMapMode.READ).then(() => {
        const L = new Float32Array(this.aeRead.getMappedRange().slice(0));
        this.aeRead.unmap(); this.aeBusy = false;
        const target = exposureFrom(L);
        if (target) this.ev[name] = settleExposure(this.ev[name], target, 0.5);
      }).catch(() => { this.aeBusy = false; });
    }
    this.frames++;
    return true;
  }

  _ts(enc, k, onMs) {
    if (!this.ts || this.tsBusy[k]) return;
    enc.resolveQuerySet(this.ts, 2 * k, 2, this.tsRes, 256 * k);
    enc.copyBufferToBuffer(this.tsRes, 256 * k, this.tsRead[k], 0, 16);
    this.tsBusy[k] = true;
    (this._tsPending ??= []).push(() => this.tsRead[k].mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigUint64Array(this.tsRead[k].getMappedRange().slice(0, 16));
      this.tsRead[k].unmap(); this.tsBusy[k] = false;
      const ms = Number(t[1] - t[0]) / 1e6;
      if (ms >= 0 && ms < 1e4) onMs(ms);
    }).catch(() => { this.tsBusy[k] = false; }));
  }
  _tsFlush() { const p = this._tsPending || []; this._tsPending = []; for (const f of p) f(); }

  timings() {
    const med = (a) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
    const tile = this.gpuMs.filter((x) => !x.large).map((x) => x.ms), big = this.gpuMs.filter((x) => x.large).map((x) => x.ms);
    return { ready: !!this.ok, error: this.error ?? null, frames: this.frames, tile_gpu_ms_median: med(tile), tile_gpu_ms_max: tile.length ? Math.max(...tile) : null,
      large_gpu_ms_median: med(big), tile_px: this.gpuMs.filter((x) => !x.large).at(-1)?.px ?? null,
      bake: { done: this.bake.done, sample: this.bake.sample, rays: this.opts.bakeRays, version: this.bake.version, inside_vertices: this.list?.length ?? 0, chunk: this.bake.chunk, gpu_ms_median: med(this.bakeGpu) },
      fill: this.fill, exposure: { ...this.ev }, timestamp_query: !!this.ts,
      static_drawn: this.runs ? this.runs.reduce((a, r) => a + r.count, 0) : null, static_runs: this.runs?.length ?? null };
  }
}
