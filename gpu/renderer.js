// TwinRenderer: the live WebGPU ray tracer of the aimed dome (docs/WEB_TWIN.md §8, ADR-142). WebGPU + WGSL only.
//
//   const r = await TwinRenderer.create({ canvas?, scene, physics, device? });
//   r.setPoses(poses0, poses1?);            // exposure start / end, poses.bodies[key] = column-major T_D_body
//   await r.renderCamera(name, opts);       // linear RGB radiance (W sr^-1 m^-2) -> GPUTexture rgba32float
//   r.sensor(name, opts);                   // radiance -> 8-bit camera RGB (§9) -> GPUTexture rgba8unorm
//   r.renderOverview(view, opts);           // pinhole orbit view, cutaway, viewing-aid light -> rgba8unorm
//   await r.readback(name, 'rgb8' | 'linear' | 'yuv420' | 'raw');
//   r.visible(name, pointsD) -> bool[];     // occlusion from the camera centre, in the same BVH (CPU traversal)
//   r.timings();                            // GPU timestamp ms per pass and per submit (CPU fallback without the feature)
//
// Per frame: direct light at every radiosity node (shadow rays, posed tools and scope), a few warm-started Jacobi
// steps of the node radiosity, then per camera a pass that box-integrates each pixel with `spp` stratified sub-samples
// (each its own KB4 ray, time in the exposure, light samples and BSDF sample) and a refinement pass that adds
// `refine` sub-samples on edges, posed bodies and penumbrae. Every pass is split into submits of about 8 ms of GPU
// time (docs/WEB_TWIN.md §12: macOS cannot preempt long GPU work), and every WGSL loop has a hard cap.

import { prepareScene, loadScene, MESH_FLAG, ALBEDO_BINS, cavityCapsule } from './scene_gpu.js';
import { traverse, buildBVH, relocate, NODE_WORDS } from './bvh.js';
import { lensParams, project } from './lens.js';
import { MAX_INST } from './wgsl/common.wgsl.js';
import { shaderSource } from './wgsl/kernels.wgsl.js';
import { SENSOR } from './wgsl/sensor.wgsl.js';
import { sensorParams, relativeIlluminationMap, AutoExposure, packSensorUniform, describe, updateAWB } from './sensor.js';

const O = { CUT: 1, AID: 2, NO_INDIRECT: 4, DIRECT_ONLY: 8, LAYER: 16, NO_CAPSULE: 32, NORMALS: 64, NO_MIS: 128, PACKET: 512, REFINE_SHADOW: 1024, CACHE_WRITE: 2048, CACHE_READ: 4096, SHADOW_CACHE: 8192 };
export const OPTION_BITS = O;
const SCENE_U_BYTES = 96 + MAX_INST * 192;
const CAM_U_BYTES = 192;

export const DEFAULTS = Object.freeze({
  spp: 1,                 // pass-1 sub-samples per pixel (each: a KB4 ray, a time, lightSamples light samples, a BSDF sample)
  refine: 6,              // pass-2 extra sub-samples on edges, posed bodies and pixels whose shadow rays were traced
  lightSamples: 4,        // emitter samples per pass-1 sub-sample (shadow rays mostly culled, so cheap)
  refineLightSamples: 1,  // emitter samples per pass-2 sub-sample (these pixels trace their shadow rays)
  radiosityEvery: 1,      // update the node radiosity every n-th setPoses (live: 2 halves its cost, 1 frame of lag)
  nodeSpacing: 0.003,     // radiosity node spacing (m): every static edge <= this
  gatherK: 64,            // gather rays per node
  nodeSamples: 4,         // emitter samples per node per frame
  itersPerFrame: 4,       // warm-started Jacobi steps per frame
  itersInitial: 96,       // on the first solve
  smoothSigma: 0.003,     // Gaussian sigma (m) of the node irradiance smoothing; 0 = off
  smoothRadius: 0.006,    // neighbours within this distance (m), same-facing (n.n > 0.9), at most nbrK
  nbrK: 16,
  packet: false,          // workgroup packet traversal of primary rays: exact, but measured 2x slower here (barriers)
  refineShadows: true,    // also refine pixels whose shadow rays were traced (penumbrae, near the tools)
  hitCache: true,         // fixed cameras: each sub-sample's static first hit is traced once and kept (exact)
  shadowCache: true,      // still scope: each sub-sample's static occlusion per light sample is kept (exact)
  cullShadows: true,      // skip shadow rays whose answer the three nodes and the bodies' capsules already give
  layerLines: false,      // FDM layer lines are off in v1 (coordinator, 2026-10-31); physics carries their numbers
  leafSize: 4,
});

/**
 * Operating points measured with tools/gpu_bench.py (headless Brave, M4 Air, 3 x 1280x720, overview hidden, tools moving,
 * scope still, 2026-10-31; static first-hit and static-occlusion caches on):
 *   live      ~45 ms GPU / ~50 ms wall per frame; render-noise spread median 0.66x sensor noise (p95 4.2x)
 *   balanced  ~84 ms;  median 0.50x (p95 3.1x)
 *   judge     ~190-245 ms; median 0.23x (p95 1.3x)
 * With a fixed seed (the default) a static frame renders identically every time: its temporal render noise is 0;
 * the spread above is across seeds, i.e. the estimator error that shows as flicker only when things move.
 */
export const PRESETS = Object.freeze({
  judge: {},
  balanced: { spp: 1, lightSamples: 2, refine: 2, refineLightSamples: 1, refineShadows: false, radiosityEvery: 2, itersPerFrame: 1 },
  live: { spp: 1, lightSamples: 1, refine: 0, radiosityEvery: 2, itersPerFrame: 1 },
});

// ------------------------------------------------------------------ small math
const quatFromMat = (m) => {          // m: column-major 4x4 (or 3x3 in the top-left)
  const r00 = m[0], r10 = m[1], r20 = m[2], r01 = m[4], r11 = m[5], r21 = m[6], r02 = m[8], r12 = m[9], r22 = m[10];
  const tr = r00 + r11 + r22;
  let x, y, z, w;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (r21 - r12) / s; y = (r02 - r20) / s; z = (r10 - r01) / s; }
  else if (r00 > r11 && r00 > r22) { const s = Math.sqrt(1 + r00 - r11 - r22) * 2; w = (r21 - r12) / s; x = 0.25 * s; y = (r01 + r10) / s; z = (r02 + r20) / s; }
  else if (r11 > r22) { const s = Math.sqrt(1 + r11 - r00 - r22) * 2; w = (r02 - r20) / s; x = (r01 + r10) / s; y = 0.25 * s; z = (r12 + r21) / s; }
  else { const s = Math.sqrt(1 + r22 - r00 - r11) * 2; w = (r10 - r01) / s; x = (r02 + r20) / s; y = (r12 + r21) / s; z = 0.25 * s; }
  const l = Math.hypot(x, y, z, w);
  return [x / l, y / l, z / l, w / l];
};
const quatMat = (q) => {
  const [x, y, z, w] = q;
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
};
const slerpJS = (a, b, t) => {
  let c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (c < 0) { bb = b.map((x) => -x); c = -c; }
  if (c > 0.9995) { const r = a.map((x, i) => x + (bb[i] - x) * t); const l = Math.hypot(...r); return r.map((x) => x / l); }
  const th = Math.acos(c), s = Math.sin(th);
  return a.map((x, i) => (x * Math.sin((1 - t) * th) + bb[i] * Math.sin(t * th)) / s);
};
const mulRv = (R, v) => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
const mulRtv = (R, v) => [R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2], R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2], R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2]];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const rowMajor4 = (T) => (Array.isArray(T[0]) ? T : [[T[0], T[4], T[8], T[12]], [T[1], T[5], T[9], T[13]], [T[2], T[6], T[10], T[14]], [T[3], T[7], T[11], T[15]]]);

// ------------------------------------------------------------------ GPU timing and the submit budget
// docs/WEB_TWIN.md §12: every per-pixel or per-node pass is split so one queue.submit stays near TARGET_MS of GPU
// time (macOS cannot preempt a long GPU command buffer; WindowServer starves). Each submit carries one timed
// compute pass; its measured ms per work unit sets the next chunk size of that kind. Any submit over WARN_MS is logged.
// A submit over TRIP_MS is an overrun: logged, and its kind's measured rate shrinks the next chunk at once. GPU work
// stops only on a runaway budget: one submit over HARD_TRIP_MS, or TRIP_STREAK overruns in a row. A single 60-120 ms
// submit (a busy compositor, a cold chunk) is a hitch, far from the ~40 s of starvation that reset the Mac (ADR-142);
// stopping on it left the live app dead mid-session (2026-10-31 motion runs: 4 of 6 tripped).
export const TARGET_MS = 8, WARN_MS = 16, TRIP_MS = 64, HARD_TRIP_MS = 1000, TRIP_STREAK = 8;
const CHUNK = {           // [initial units, min, max, target ms] per kind; units: pixels x sub-samples, nodes x rays, ...
  // kinds whose cost per unit varies a lot across the scene (gather rays, refinement) aim lower than TARGET_MS
  camera: [32768, 4096, 524288, 6], cameraCached: [65536, 4096, 524288, 6], refine: [32768, 2048, 524288, 4], overview: [32768, 4096, 524288, 6],
  // 1-D kinds dispatch ceil(units / 64) workgroups (nodeDirect: / NODE_SAMPLES first), so their max stays at or under
  // 65535 x 64 = 4194240: WebGPU's maxComputeWorkgroupsPerDimension is 65535, and a dispatch over it is dropped with
  // only a validation error (the gather table lost a 4194304-ray chunk on every fast load; round-2 diagnosis)
  nodeDirect: [65536, 4096, 4194240, 6], nodeDirectInc: [131072, 4096, 4194240, 6], gather: [65536, 4096, 4194240, 5], remap: [65536, 4096, 4194240, 5],
  // the Jacobi gather (K rays per node) costs far more per node than the update: one shared rate sized the gather
  // from the update's and overran (66-72 ms jacobiGather0 submits on the M4), so each pass has its own
  jacobiGather: [8192, 1024, 1048576, 6], jacobiUpdate: [65536, 8192, 1048576, TARGET_MS],
  sensor: [1, 1, 1, TARGET_MS],
};

class Timer {
  constructor(device, log) {
    this.device = device; this.log = log;
    this.gpu = device.features.has('timestamp-query');
    this.labels = []; this.pending = 0;
    this.rates = {}; this.frames = new Map(); this.history = [];
    this.maxSubmit = { ms: 0, label: null }; this.over = []; this.submits = 0;
    if (this.gpu) {
      this.qs = device.createQuerySet({ type: 'timestamp', count: 64 });
      this.resolve = device.createBuffer({ size: 64 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.pool = [];
    }
  }
  /** timestampWrites for a pass of this submit; kind/units feed the adaptive chunk size. */
  pass(label, kind = null, units = 0, frame = 0) {
    const i = this.labels.length;
    this.labels.push({ label, kind, units, frame });
    if (!this.gpu || i >= 31) return undefined;
    return { querySet: this.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
  }
  flush(encoder) {
    const labels = this.labels;
    this.labels = [];
    this.submits++;
    if (!labels.length) return () => {};
    if (!this.gpu) {
      const t0 = performance.now();
      return () => { this.pending++; this.device.queue.onSubmittedWorkDone().then(() => { this.pending--; this._record(labels, labels.map(() => (performance.now() - t0) / labels.length)); }); };
    }
    const n = Math.min(labels.length, 31) * 2;
    const staging = this.pool.pop() || this.device.createBuffer({ size: 64 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.resolveQuerySet(this.qs, 0, n, this.resolve, 0);
    encoder.copyBufferToBuffer(this.resolve, 0, staging, 0, n * 8);
    return () => {
      this.pending++;
      staging.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigUint64Array(staging.getMappedRange().slice(0, n * 8));
        staging.unmap(); this.pool.push(staging); this.pending--;
        this._record(labels, labels.map((l, i) => (2 * i + 1 < n ? Number(t[2 * i + 1] - t[2 * i]) / 1e6 : 0)));
      }).catch(() => { this.pending--; });
    };
  }
  _record(labels, ms) {
    const total = ms.reduce((a, b) => a + b, 0);
    if (total > this.maxSubmit.ms) this.maxSubmit = { ms: total, label: labels.map((l) => l.label).join('+') };
    if (total > WARN_MS) { this.over.push({ ms: total, label: labels[0].label }); if (this.over.length > 256) this.over.shift(); this.log(`submit over ${WARN_MS} ms: ${labels[0].label} ${total.toFixed(1)} ms`); }
    this.streak = total > TRIP_MS ? (this.streak || 0) + 1 : 0;
    if (total > TRIP_MS) this.overruns = (this.overruns || 0) + 1;
    if ((total > HARD_TRIP_MS || this.streak >= TRIP_STREAK) && !this.tripped) {
      this.tripped = `${labels[0].label} took ${total.toFixed(0)} ms${this.streak >= TRIP_STREAK ? ` (${this.streak} overruns in a row)` : ''}`;
      this.log(`GPU budget tripped: ${this.tripped}; no further submits`);
    }
    labels.forEach((l, i) => {
      if (l.kind && l.units > 0 && ms[i] > 0) {
        const r = ms[i] / l.units, old = this.rates[l.kind];
        this.rates[l.kind] = old ? Math.max(0.7 * old + 0.3 * r, r * 0.5) : r;       // EMA, but react fast to slow-downs
        if (r > (old || 0)) this.rates[l.kind] = Math.max(this.rates[l.kind], r);
      }
      let f = this.frames.get(l.frame);
      if (!f) { f = { frame: l.frame, submits: 0 }; this.frames.set(l.frame, f); }
      const key = l.label.replace(/#\d+$/, '');
      f[key] = (f[key] || 0) + ms[i];
      f.submits++;
      f.maxSubmit = Math.max(f.maxSubmit || 0, total);
    });
    while (this.frames.size > 64) { const k = this.frames.keys().next().value; this.history.push(this.frames.get(k)); this.frames.delete(k); if (this.history.length > 512) this.history.shift(); }
  }
  /** Work units per submit for a kind, so one submit is about TARGET_MS. */
  units(kind) {
    const [init, lo, hi, target] = CHUNK[kind] || [65536, 1024, 524288, TARGET_MS];
    const r = this.rates[kind];
    return Math.max(lo, Math.min(hi, r ? Math.floor(target / r) : init));
  }
  frameRecords() { return [...this.history, ...this.frames.values()]; }
}

// ------------------------------------------------------------------ the renderer
export class TwinRenderer {
  /** opts: {canvas?, scene (URL of the scene dir | {scene, meshes} | prepared), physics (object | URL), device?, preset?: 'live'|'judge', ...DEFAULTS} */
  static async create(opts) {
    const r = new TwinRenderer();
    await r._init(opts);
    return r;
  }

  async _init(opts) {
    this.opts = { ...DEFAULTS, ...(PRESETS[opts.preset] || {}), ...opts };
    this.log = opts.log || ((...a) => console.log('[twin-gpu]', ...a));
    let physics = opts.physics;
    if (typeof physics === 'string') physics = await (await fetch(physics)).json();
    this.physics = physics || {};
    let src = opts.scene;
    if (typeof src === 'string') src = await loadScene(src);
    if (!src.meshes) throw new Error('scene: pass a scene directory URL or {scene, meshes}');
    this.scene = src.scene;
    const t0 = performance.now();
    this.S = src.prepared || prepareScene(src.scene, src.meshes, this.physics, { nodeSpacing: this.opts.nodeSpacing, leafSize: this.opts.leafSize, log: this.log });
    this.stats = { prepareMs: performance.now() - t0, ...this.S.stats };

    // device
    if (opts.device) this.device = opts.device;
    else {
      if (!navigator.gpu) throw new Error('WebGPU is not available');
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('no WebGPU adapter');
      this.adapterInfo = adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : {};
      const features = ['timestamp-query'].filter((f) => adapter.features.has(f));
      const lim = adapter.limits;
      this.device = await adapter.requestDevice({
        requiredFeatures: features,
        requiredLimits: {
          maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize,
          maxStorageBuffersPerShaderStage: Math.min(10, lim.maxStorageBuffersPerShaderStage),
          maxComputeWorkgroupStorageSize: Math.min(32768, lim.maxComputeWorkgroupStorageSize),
        },
      });
    }
    const dev = this.device;
    dev.lost?.then((i) => { this.lostInfo = i.message || String(i.reason); this.log('device lost', i.reason, i.message); });
    dev.addEventListener?.('uncapturederror', (e) => { this.lastError = e.error?.message; this.log('GPU error', e.error?.message); });
    this.timer = new Timer(dev, this.log);

    // cameras
    this.cameras = {};
    (this.scene.cameras || []).forEach((c, i) => {
      const L = lensParams(c.lens);
      this.cameras[c.name] = { index: i, meta: c, lens: L, size: c.size || c.lens.size, T: rowMajor4(c.T_D_C) };
    });

    this._buildBuffers();
    await this._loadBoardTexture(src.base);
    await this._buildPipelines();
    await this._buildSensor();
    await this._buildGatherTable();
    this.posesVersion = 0; this.solvedVersion = -1; this.frame = 0;
    this.interiorRoot = this.interiorRoot ?? 0;
    this.setPoses({ bodies: {} });
    this.log(`ready: ${this.S.triCount} triangles, ${this.S.staticVerts} nodes (${this.activeCount} active), gather table ${(this.gatherBytes / 1e6).toFixed(1)} MB, timestamp-query ${this.timer.gpu}`);
  }

  // ---------------------------------------------------------------- buffers
  _buf(data, usage, label) {
    const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    const b = this.device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST, label });
    this.device.queue.writeBuffer(b, 0, data.buffer ? data.buffer : data, data.byteOffset || 0, data.byteLength);
    return b;
  }

  _buildBuffers() {
    const S = this.S, dev = this.device, ST = GPUBufferUsage.STORAGE;
    // params: mesh table | materials | albedo tables | profile | emitter triangles
    const nMesh = S.meshTable.length, nMat = S.materials.length;
    const em = S.emitter, emitN = Math.max(1, em.tris.length), profN = Math.max(2, em.theta.length);
    const meshOff = 0, matOff = meshOff + 2 * nMesh, albOff = matOff + 5 * nMat, profOff = albOff + nMat * ALBEDO_BINS * 2, emitOff = profOff + profN;
    const nVec = emitOff + 4 * emitN;
    const P = new Float32Array(nVec * 4), PU = new Uint32Array(P.buffer);
    S.meshTable.forEach((m, i) => {
      const o = (meshOff + 2 * i) * 4;
      const mat = S.materials[m.mat];
      P.set(m.buildDir, o); PU[o + 3] = m.mat;
      PU[o + 4] = m.flags;
      P[o + 5] = mat.layer ? mat.layer.period : 0; P[o + 6] = mat.layer ? mat.layer.depth : 0;
    });
    S.materials.forEach((m, i) => {
      const o = (matOff + 5 * i) * 4;
      const c = m.metallic ? m.f0 : m.rho;
      P.set([c[0], c[1], c[2], m.alpha, m.ior, m.metallic, m.fbar, m.roughness], o);
      const hh = m.table ? m.table.hh : [0, 0, 0], hs = m.table ? m.table.hhSpec : [0, 0, 0];
      P.set([hh[0], hh[1], hh[2], 0, hs[0], hs[1], hs[2], m.emitter ? 1 : 0], o + 8);
      if (m.coated) P.set([...m.tau, m.alphaC], o + 16);
      else if (m.texture) P.set([...m.texture.ink, m.texture.roughInk * m.texture.roughInk], o + 16);
      if (m.table) for (let b = 0; b < ALBEDO_BINS; b++) {
        const q = (albOff + (i * ALBEDO_BINS + b) * 2) * 4;
        P.set([m.table.spec[3 * b], m.table.spec[3 * b + 1], m.table.spec[3 * b + 2], 0, m.table.diff[3 * b], m.table.diff[3 * b + 1], m.table.diff[3 * b + 2], 0], q);
      }
    });
    for (let k = 0; k < profN; k++) {
      const th = em.theta[Math.min(k, em.theta.length - 1)] ?? (k ? Math.PI / 2 : 0), p = em.p[Math.min(k, em.p.length - 1)] ?? 1;
      P.set([th, p, 0, 0], (profOff + k) * 4);
    }
    em.tris.forEach((t, k) => {
      const o = (emitOff + 4 * k) * 4;
      P.set([...t.v0, t.cdf, ...t.e1, 0, ...t.e2, 0, ...t.n, t.area], o);
    });
    this.consts = {
      nMesh, nMat, meshOff, matOff, albOff, albBins: ALBEDO_BINS, profOff, profN, emitOff, emitN,
      nStaticVerts: S.staticVerts, gatherK: this.opts.gatherK, nodeSamples: this.opts.nodeSamples, nbrK: this.opts.nbrK,
    };
    this.bufs = {
      nodes: this._buf(S.nodes, ST, 'nodes'),
      tris: this._buf(S.tris, ST, 'tris'),
      verts: this._buf(S.verts, ST, 'verts'),
      params: this._buf(P, ST, 'params'),
      nodeState: dev.createBuffer({ size: Math.max(64, S.staticVerts * 64), usage: ST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'nodeState' }),
      gtable: dev.createBuffer({ size: Math.max(16, S.staticVerts * this.opts.gatherK * 4), usage: ST | GPUBufferUsage.COPY_SRC, label: 'gtable' }),
      aux: dev.createBuffer({ size: Math.max(16, S.staticVerts * 16), usage: ST | GPUBufferUsage.COPY_SRC, label: 'aux' }),
      scn: dev.createBuffer({ size: SCENE_U_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'sceneU' }),
      radU: dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      dbg: dev.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'dbg' }),
    };
    // node state starts with visibility code -1 (unknown: pixels trace their shadow rays), so inactive nodes never
    // claim a pixel is lit or shadowed
    const st0 = new Float32Array(Math.max(16, S.staticVerts * 16));
    for (let i = 0; i < S.staticVerts; i++) { st0[16 * i + 3] = 1; st0[16 * i + 15] = -1; }
    dev.queue.writeBuffer(this.bufs.nodeState, 0, st0);
    this.gatherBytes = S.staticVerts * this.opts.gatherK * 4;
    this.sceneU = new ArrayBuffer(SCENE_U_BYTES);
    this.eAvg = [0, 0, 0];
  }

  async _module(code, label) {
    const m = this.device.createShaderModule({ code, label });
    const info = await m.getCompilationInfo();
    const errs = info.messages.filter((x) => x.type === 'error');
    if (errs.length) {
      const lines = code.split('\n');
      throw new Error(`${label}: ` + errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message} | ${lines[e.lineNum - 1]?.trim()}`).join('\n'));
    }
    return m;
  }

  async _buildPipelines() {
    const dev = this.device;
    const ro = { type: 'read-only-storage' }, rw = { type: 'storage' }, un = { type: 'uniform' };
    const C = GPUShaderStage.COMPUTE;
    const entries = (types) => types.map((buffer, binding) => ({ binding, visibility: C, buffer }));
    this.layouts = {
      sceneRO: dev.createBindGroupLayout({ entries: [...entries([ro, ro, ro, ro, ro, rw]), { binding: 6, visibility: C, texture: { sampleType: 'float' } }, { binding: 7, visibility: C, sampler: { type: 'filtering' } }] }),
      sceneRW: dev.createBindGroupLayout({ entries: entries([ro, ro, ro, ro, rw, rw, ro, rw, ro, rw]) }),
      camG1: dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: un }, { binding: 1, visibility: C, buffer: un }, { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'rgba32float' } }, { binding: 3, visibility: C, buffer: rw }, { binding: 4, visibility: C, buffer: rw }, { binding: 5, visibility: C, buffer: rw }] }),
      radG1: dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: un }, { binding: 1, visibility: C, buffer: un }] }),
      testG1: dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: un }, { binding: 1, visibility: C, buffer: un }, { binding: 2, visibility: C, buffer: ro }, { binding: 3, visibility: C, buffer: rw }] }),
    };
    const camMod = await this._module(shaderSource('camera', this.consts), 'camera');
    const radMod = await this._module(shaderSource('radiosity', this.consts), 'radiosity');
    const camLayout = dev.createPipelineLayout({ bindGroupLayouts: [this.layouts.sceneRO, this.layouts.camG1] });
    const radLayout = dev.createPipelineLayout({ bindGroupLayouts: [this.layouts.sceneRW, this.layouts.radG1] });
    this.pipes = {
      camera: await dev.createComputePipelineAsync({ layout: camLayout, compute: { module: camMod, entryPoint: 'camMain', constants: { CULL_SHADOWS: this.opts.cullShadows ? 1 : 0 } } }),
      cameraExact: await dev.createComputePipelineAsync({ layout: camLayout, compute: { module: camMod, entryPoint: 'camMain', constants: { CULL_SHADOWS: 0 } } }),
      refine: await dev.createComputePipelineAsync({ layout: camLayout, compute: { module: camMod, entryPoint: 'camRefine', constants: { CULL_SHADOWS: this.opts.cullShadows ? 1 : 0 } } }),
      refineExact: await dev.createComputePipelineAsync({ layout: camLayout, compute: { module: camMod, entryPoint: 'camRefine', constants: { CULL_SHADOWS: 0 } } }),
      overview: await dev.createComputePipelineAsync({ layout: camLayout, compute: { module: camMod, entryPoint: 'camMain', constants: { CUTAWAY: 1, CULL_SHADOWS: this.opts.cullShadows ? 1 : 0 } } }),
      gatherBuild: await dev.createComputePipelineAsync({ layout: radLayout, compute: { module: radMod, entryPoint: 'gatherBuild' } }),
      gatherCount: await dev.createComputePipelineAsync({ layout: radLayout, compute: { module: radMod, entryPoint: 'gatherCount' } }),
      gatherRemap: await dev.createComputePipelineAsync({ layout: radLayout, compute: { module: radMod, entryPoint: 'gatherRemap' } }),
      nodeDirect: await dev.createComputePipelineAsync({ layout: radLayout, compute: { module: radMod, entryPoint: 'nodeDirect' } }),
      jacobiGather: await dev.createComputePipelineAsync({ layout: radLayout, compute: { module: radMod, entryPoint: 'jacobiGather' } }),
      jacobiUpdate: await dev.createComputePipelineAsync({ layout: radLayout, compute: { module: radMod, entryPoint: 'jacobiUpdate' } }),
    };
    const dispMod = await this._module(shaderSource('display', this.consts), 'display');
    this.pipes.display = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: dispMod, entryPoint: 'displayMain' } });
    const b = this.bufs;
    this.groups = {
      sceneRO: this._sceneROGroup(),
    };
  }

  _sceneRWGroup() {
    const b = this.bufs;
    return this.device.createBindGroup({ layout: this.layouts.sceneRW, entries: [b.nodes, b.tris, b.verts, b.params, b.nodeState, b.gtable, b.active, b.aux, b.nbr, b.dbg].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }

  _sceneROGroup() {
    const b = this.bufs;
    return this.device.createBindGroup({
      layout: this.layouts.sceneRO,
      entries: [...[b.nodes, b.tris, b.verts, b.params, b.nodeState, b.dbg].map((buffer, binding) => ({ binding, resource: { buffer } })),
        { binding: 6, resource: this.boardTex.createView() }, { binding: 7, resource: this.boardSmp }],
    });
  }

  /** The first board's printed texture (scene.boards[0].texture, PNG mask, 1 = paper); 1x1 white without one. */
  async _loadBoardTexture(base) {
    const dev = this.device;
    this.boardSmp = dev.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    const b = (this.scene.boards || [])[0];
    if (b?.texture && base) {
      try {
        const blob = await (await fetch(base + b.texture)).blob();
        const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
        this.boardTex = dev.createTexture({ size: [bmp.width, bmp.height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
        dev.queue.copyExternalImageToTexture({ source: bmp }, { texture: this.boardTex }, [bmp.width, bmp.height]);
        this.stats.boardTexture = [bmp.width, bmp.height];
        return;
      } catch (e) { this.log('board texture not loaded:', String(e)); }
    }
    this.boardTex = dev.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    dev.queue.writeTexture({ texture: this.boardTex }, new Uint8Array([255, 255, 255, 255]), { bytesPerRow: 4 }, [1, 1]);
  }

  /** One queue.submit holding one timed compute pass (docs/WEB_TWIN.md §12). */
  /** Workgroups of 64 for a 1-D dispatch of n invocations; throws rather than let WebGPU drop an oversized dispatch. */
  _wg1(n) {
    const w = Math.ceil(n / 64), lim = this.device.limits?.maxComputeWorkgroupsPerDimension || 65535;
    if (w > lim) throw new Error(`dispatch of ${w} workgroups exceeds maxComputeWorkgroupsPerDimension ${lim}`);
    return w;
  }

  /** Largest 1-D invocation count one dispatch can take (64-wide workgroups). */
  get _max1D() { return 64 * (this.device.limits?.maxComputeWorkgroupsPerDimension || 65535); }

  _submit(label, kind, units, encode) {
    if (this.timer.tripped) throw new Error(`GPU budget tripped (${this.timer.tripped}); create a new renderer after fixing the cause`);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass({ timestampWrites: this.timer.pass(label, kind, units, this.frame) });
    encode(pass);
    pass.end();
    const after = this.timer.flush(enc);
    this.device.queue.submit([enc.finish()]); after();
  }

  /** In long async loops: wait for the queue and leave the GPU idle 50 ms at least every 1.5 s of work. */
  async _pace(force = false) {
    const now = performance.now();
    this._paceT0 = this._paceT0 ?? now;
    if (force || now - this._paceT0 > 1500) {
      await this.device.queue.onSubmittedWorkDone();
      await new Promise((res) => setTimeout(res, 50));
      this._paceT0 = performance.now();
    }
  }

  /** Loop-cap counters (traversal visits, leaf size); all zero unless a cap was hit. */
  async debugFlags() {
    const st = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(this.bufs.dbg, 0, st, 0, 64); this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const v = Array.from(new Uint32Array(st.getMappedRange().slice(0)));
    st.unmap(); st.destroy();
    return { traversalCap: v[0], leafCap: v[1], refined: v[2], refinedBody: v[3], refinedShadow: v[4], refinedSubsample: v[5], corruptTree: v[6], packetCap: v[7] };
  }

  async _buildGatherTable() {
    const dev = this.device, S = this.S, b = this.bufs;
    b.active = dev.createBuffer({ size: Math.max(16, S.staticVerts * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'active' });
    b.nbr = dev.createBuffer({ size: Math.max(16, S.staticVerts * this.opts.nbrK * 8), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'nbr' });
    this.groups.sceneRW = this._sceneRWGroup();
    this.groups.rad = dev.createBindGroup({ layout: this.layouts.radG1, entries: [{ binding: 0, resource: { buffer: b.scn } }, { binding: 1, resource: { buffer: b.radU } }] });
    this._writeSceneU();           // instances hidden: the table is built against the static BVH only
    const t0 = performance.now();
    // a dropped submit must fail the load, not leave table entries at 0 (= node 0): validation errors of the whole
    // build (table, count, remap) are caught in an error scope and thrown below
    dev.pushErrorScope?.('validation');
    // chunks of nodes, each its own submit of about TARGET_MS (K rays per node)
    const K = this.opts.gatherK, rays = S.staticVerts * K, max1D = this._max1D;
    let first = true, buildChunk = 4096;
    for (let start = 0; start < rays;) {
      // the first chunk is a small warm-up (first touch of the scene buffers); then ~4 ms chunks from the measured rate
      const count = Math.min(rays - start, max1D, first ? 4096 : Math.max(4096, Math.floor(this.timer.units('gather') / 64) * 64));
      first = false;
      buildChunk = Math.max(buildChunk, count);
      dev.queue.writeBuffer(b.radU, 0, new Uint32Array([0, 1, 2, S.staticVerts, start, count, 0, 0]));
      this._submit(`gather table #${start}`, 'gather', count, (pass) => {
        pass.setPipeline(this.pipes.gatherBuild);
        pass.setBindGroup(0, this.groups.sceneRW); pass.setBindGroup(1, this.groups.rad);
        pass.dispatchWorkgroups(this._wg1(count));
      });
      start += count;
      await this.idle();                               // one submit deep, and the measured rate is current
      await this._pace();
    }
    for (let start = 0; start < S.staticVerts; start += 65536) {
      const count = Math.min(65536, S.staticVerts - start);
      dev.queue.writeBuffer(b.radU, 0, new Uint32Array([0, 1, 2, S.staticVerts, start, count, 0, 0]));
      this._submit(`gather count #${start}`, null, 0, (pass) => {
        pass.setPipeline(this.pipes.gatherCount);
        pass.setBindGroup(0, this.groups.sceneRW); pass.setBindGroup(1, this.groups.rad);
        pass.dispatchWorkgroups(this._wg1(count));
      });
    }
    // entries pointing at an inactive node -> the hit triangle's nearest active vertex (re-traced; most entries
    // return at once). Chunks no larger than the build's, and sized by their own measured rate: the re-traced share
    // varies along the table, and one fixed build-sized chunk overran (79 ms on the M4, which killed the setup).
    for (let start = 0; start < rays;) {
      const count = Math.min(rays - start, buildChunk, max1D, Math.max(64, Math.floor(this.timer.units('remap') / 64) * 64));
      dev.queue.writeBuffer(b.radU, 0, new Uint32Array([0, 1, 2, S.staticVerts, start, count, 0, 0]));
      this._submit(`gather remap #${start}`, 'remap', count, (pass) => {
        pass.setPipeline(this.pipes.gatherRemap);
        pass.setBindGroup(0, this.groups.sceneRW); pass.setBindGroup(1, this.groups.rad);
        pass.dispatchWorkgroups(this._wg1(count));
      });
      start += count;
      await this.idle();
      await this._pace();
    }
    const err = await dev.popErrorScope?.();
    if (err) throw new Error(`gather table build failed (a submit was rejected): ${err.message}`);
    const enc = dev.createCommandEncoder();
    const staging = dev.createBuffer({ size: Math.max(16, S.staticVerts * 16), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(b.aux, 0, staging, 0, Math.max(16, S.staticVerts * 16));
    dev.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const aux = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    const act = [];
    this.nodeMisses = new Float32Array(S.staticVerts);
    for (let i = 0; i < S.staticVerts; i++) { this.nodeMisses[i] = aux[4 * i]; if (aux[4 * i] <= K / 2) act.push(i); }
    this.activeList = Uint32Array.from(act);
    this.activeCount = act.length;
    dev.queue.writeBuffer(b.active, 0, this.activeList);
    this.stats.gatherBuildMs = performance.now() - t0;
    const t1 = performance.now();
    const nb = this._neighbours();
    dev.queue.writeBuffer(b.nbr, 0, nb);
    this.stats.neighbourMs = performance.now() - t1;
    this._buildInteriorBVH();
    this.stats.activeNodes = this.activeCount;
  }

  /**
   * A second static BVH without the exterior: triangles whose three nodes see the sky in >= 90 % of their gather
   * rays (the shell's outer skin). Rays from the cameras' centres, the nodes and the cavity never reach those, so
   * camera feeds and lighting traverse the smaller tree; the overview (which looks at the outside) keeps the full one.
   */
  _buildInteriorBVH() {
    const S = this.S, K = this.opts.gatherK, mis = this.nodeMisses, t0 = performance.now();
    const keep = [];
    for (let k = 0; k < S.staticTris; k++) {
      const a = S.triVerts[3 * k], b = S.triVerts[3 * k + 1], c = S.triVerts[3 * k + 2];
      const bench = S.meshTable[S.triMesh[k]].flags & MESH_FLAG.BENCH;
      if (!bench && mis[a] >= 0.9 * K && mis[b] >= 0.9 * K && mis[c] >= 0.9 * K) continue;
      keep.push(k);
    }
    const n = keep.length;
    if (n === 0 || n === S.staticTris) { this.interiorRoot = 0; this.stats.interiorTris = S.staticTris; return; }   // nothing to drop
    const P = new Float32Array(n * 9), I = new Uint32Array(n * 3);
    for (let q = 0; q < n; q++) {
      const o = keep[q] * 16;
      for (let v = 0; v < 3; v++) {
        const base = v === 0 ? [S.tris[o], S.tris[o + 1], S.tris[o + 2]] : [S.tris[o] + S.tris[o + 4 * v], S.tris[o + 1] + S.tris[o + 4 * v + 1], S.tris[o + 2] + S.tris[o + 4 * v + 2]];
        P.set(base, 9 * q + 3 * v); I[3 * q + v] = 3 * q + v;
      }
    }
    const bvh = buildBVH(P, I, { leafSize: this.opts.leafSize });
    const nodeOff = S.nodeCount, triOff = S.triCount;
    const nodes = new Float32Array((S.nodeCount + bvh.nodeCount) * 16);
    nodes.set(S.nodes); nodes.set(relocate(bvh.nodes, nodeOff, triOff), nodeOff * 16);
    const tris = new Float32Array((S.triCount + n) * 16);
    tris.set(S.tris);
    for (let q = 0; q < n; q++) tris.set(S.tris.subarray(keep[bvh.order[q]] * 16, keep[bvh.order[q]] * 16 + 16), (triOff + q) * 16);
    this.interiorRoot = nodeOff;
    const b = this.bufs, ST = GPUBufferUsage.STORAGE;
    b.nodes.destroy(); b.tris.destroy();
    b.nodes = this._buf(nodes, ST, 'nodes'); b.tris = this._buf(tris, ST, 'tris');
    this.groups.sceneRO = this._sceneROGroup();
    this.groups.sceneRW = this._sceneRWGroup();
    this.stats.interiorTris = n; this.stats.interiorNodes = bvh.nodeCount; this.stats.interiorBuildMs = performance.now() - t0;
  }

  /** Smoothing neighbours of every active node: (active index, Gaussian weight) x nbrK, same-facing, within radius. */
  _neighbours() {
    const S = this.S, V = S.verts, K = this.opts.nbrK, R = this.opts.smoothRadius, sg = this.opts.smoothSigma;
    const act = this.activeList, n = act.length;
    const out = new Uint32Array(n * K * 2), outF = new Float32Array(out.buffer);
    if (!(sg > 0)) { for (let k = 0; k < n; k++) { out[k * K * 2] = k; outF[k * K * 2 + 1] = 1; } return out; }
    const cell = (x) => Math.floor(x / R) + 512;              // the dome spans < 1024 cells of 6 mm per axis
    const keyOf = (x, y, z) => (x * 1024 + y) * 1024 + z;
    // counting sort of active nodes into cells (numeric keys; a Map of arrays is 10x slower here)
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
          // keep the K largest weights (insertion into a sorted list; ties by index for determinism)
          let pos = cnt < K ? cnt : K;
          while (pos > 0 && (cw[pos - 1] < w || (cw[pos - 1] === w && cm[pos - 1] > m))) pos--;
          if (pos >= K) continue;
          for (let q = Math.min(cnt, K - 1); q > pos; q--) { cw[q] = cw[q - 1]; cm[q] = cm[q - 1]; }
          cw[pos] = w; cm[pos] = m; if (cnt < K) cnt++;
        }
      }
      for (let q = 0; q < cnt; q++) { out[(k * K + q) * 2] = cm[q]; outF[(k * K + q) * 2 + 1] = cw[q]; }
    }
    return out;
  }

  // ---------------------------------------------------------------- poses
  /** poses = {bodies: {key: T (16 col-major) | null}}; poses1 = exposure end (default poses0). */
  setPoses(poses0, poses1 = null) {
    const S = this.S;
    const b0 = poses0?.bodies || {}, b1 = (poses1 || poses0)?.bodies || {};
    this.instances = S.blas.map((bl) => {
      if (bl.kind === 'fixed') return { q0: [0, 0, 0, 1], q1: [0, 0, 0, 1], p0: [0, 0, 0], p1: [0, 0, 0] };
      const T0 = b0[bl.body], T1 = b1[bl.body] ?? T0;
      if (!T0) return null;
      const q0 = quatFromMat(T0), q1 = quatFromMat(T1);
      return { q0, q1, p0: [T0[12], T0[13], T0[14]], p1: [T1[12], T1[13], T1[14]] };
    });
    this.poses0 = poses0; this.poses1 = poses1 || poses0;
    this.posesVersion++;
    this._writeSceneU();
  }

  _poseAt(i, t) {
    const I = this.instances[i];
    if (!I) return null;
    const q = slerpJS(I.q0, I.q1, t);
    return { R: quatMat(q), p: I.p0.map((x, k) => x + (I.p1[k] - x) * t) };
  }

  _writeSceneU(cut = null) {
    const S = this.S, f = new Float32Array(this.sceneU), u = new Uint32Array(this.sceneU);
    f.fill(0);
    const L0 = S.emitter.L0;
    f.set([L0[0], L0[1], L0[2], S.emitter.area || 1e-12], 0);
    f.set([...S.emitter.axis, 0], 4);
    const scopeI = S.blas.findIndex((b) => b.body === 'scope');
    u[7] = S.emitter.shadowSkipsScope && scopeI >= 0 ? (1 << scopeI) >>> 0 : 0;     // shadow-ray skip mask
    f.set([...this.eAvg, 5e-4], 8);         // w: margin (m) of the shadow-culling capsule test
    f.set(cut ? [...cut.n, cut.d] : [0, 0, 0, 1e30], 12);
    const scopeIdx = S.blas.findIndex((b) => b.body === 'scope');
    u[16] = S.blas.length; u[17] = scopeIdx < 0 ? 0 : scopeIdx; u[18] = this.slot || 1; u[19] = this.frame || 0;
    f.set([...S.emitter.centroid, S.emitter.radius], 20);
    if (S.blas.length > MAX_INST) throw new Error(`too many bodies (${S.blas.length} > ${MAX_INST})`);
    S.blas.forEach((bl, i) => {
      const o = 24 + 48 * i;
      const I = this.instances?.[i];
      if (!I) { u[o + 15] = 2; u[o + 11] = bl.root; return; }
      f.set(I.q0, o); f.set(I.q1, o + 4);
      f.set(I.p0, o + 8); u[o + 11] = bl.root;
      f.set(I.p1, o + 12);
      const moving = I.q0.some((x, k) => x !== I.q1[k]) || I.p0.some((x, k) => x !== I.p1[k]);
      u[o + 15] = moving ? 1 : 0;
      // world box swept over the exposure
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      const ts = moving ? [0, 0.25, 0.5, 0.75, 1] : [0];
      for (const t of ts) {
        const P = this._poseAt(i, t);
        for (let c = 0; c < 8; c++) {
          const v = [c & 1 ? bl.bmax[0] : bl.bmin[0], c & 2 ? bl.bmax[1] : bl.bmin[1], c & 4 ? bl.bmax[2] : bl.bmin[2]];
          const w = mulRv(P.R, v);
          for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], w[k] + P.p[k]); hi[k] = Math.max(hi[k], w[k] + P.p[k]); }
        }
      }
      const pad = moving ? 1e-3 : 1e-6;
      f.set([lo[0] - pad, lo[1] - pad, lo[2] - pad, 0, hi[0] + pad, hi[1] + pad, hi[2] + pad, 0], o + 16);
      // occlusion capsule at mid-exposure: the body's in-cavity meshes, clipped so that only the part inside the
      // cavity counts and nothing is counted twice (a cannula: its sheath, pivot to toe; a tool or the scope: from the
      // cannula's toe on; flush lens fronts and plates: none). scene_gpu.js cavityCapsule.
      const P = this._poseAt(i, 0.5);
      const oc = bl.occCapsule || bl.capsule;
      const body = S.bodies.find((x) => x.key === bl.body);
      const port = body?.port && (this.scene.ports || []).find((p) => p.id === body.port);
      const cap = cavityCapsule(bl.kind, mulRv(P.R, oc.a).map((x, k) => x + P.p[k]), mulRv(P.R, oc.b).map((x, k) => x + P.p[k]), oc.r, port);
      f.set([...cap.a, cap.r, ...cap.b, bl.rhoEff], o + 24);
      // bounding capsule of the whole BLAS (culls traversals), padded by the largest motion over the exposure
      const ca = mulRv(P.R, bl.capsule.a).map((x, k) => x + P.p[k]), cb = mulRv(P.R, bl.capsule.b).map((x, k) => x + P.p[k]);
      let motion = 0;
      if (moving) {
        const P0 = this._poseAt(i, 0), P1 = this._poseAt(i, 1);
        for (const v of [bl.capsule.a, bl.capsule.b]) for (const Q of [P0, P1]) {
          const w = mulRv(Q.R, v).map((x, k) => x + Q.p[k]), m = mulRv(P.R, v).map((x, k) => x + P.p[k]);
          motion = Math.max(motion, Math.hypot(w[0] - m[0], w[1] - m[1], w[2] - m[2]));
        }
        motion *= 1.5;
      }
      const bc = [...ca, bl.capsule.r * 1.0001 + 1e-6 + motion, ...cb, 0];
      f.set(bc, o + 32);
      this._curCaps = this._curCaps || [];
      this._curCaps[i] = bc;
      f.set(this._solvedCaps?.[i] || bc, o + 40);
    });
    this.device.queue.writeBuffer(this.bufs.scn, 0, this.sceneU);
  }

  // ---------------------------------------------------------------- radiosity (chunked submits)
  _radU(start, count, slotIn = 1, slotOut = 2, moved = 0) {
    this.device.queue.writeBuffer(this.bufs.radU, 0, new Uint32Array([this.activeCount, slotIn, slotOut, this.S.staticVerts, start, count, this.interiorRoot >>> 0, moved >>> 0]));
  }

  /** Direct light at every active node (or, with a moved-body mask, only near those bodies), in ~TARGET_MS chunks. */
  _submitNodeDirect(moved = 0) {
    const n = this.activeCount, NS = this.opts.nodeSamples;
    const kind = moved ? 'nodeDirectInc' : 'nodeDirect';
    for (let start = 0; start < n;) {
      const count = Math.min(n - start, Math.max(64, Math.floor(this.timer.units(kind) / NS / 64) * 64));
      this._radU(start, count, 1, 2, moved);
      this._submit(`radiosity.direct#${start}`, kind, count * NS, (pass) => {
        pass.setPipeline(this.pipes.nodeDirect);
        pass.setBindGroup(0, this.groups.sceneRW); pass.setBindGroup(1, this.groups.rad);
        pass.dispatchWorkgroups(this._wg1(count));
      });
      start += count;
    }
  }

  /** `iters` Jacobi steps; each step gathers all chunks, then updates all chunks (ping-pong L slots). */
  _submitJacobi(iters) {
    const n = this.activeCount;
    let slot = this.slot || 1;
    for (let k = 0; k < iters; k++) {
      const out = slot === 1 ? 2 : 1;
      for (const [entry, kind] of [['jacobiGather', 'jacobiGather'], ['jacobiUpdate', 'jacobiUpdate']]) {
        for (let start = 0; start < n;) {
          const count = Math.min(n - start, Math.max(64, Math.floor(this.timer.units(kind) / 64) * 64));
          this._radU(start, count, slot, out);
          this._submit(`radiosity.jacobi#${entry}${start}`, kind, count, (pass) => {
            pass.setPipeline(this.pipes[entry]);
            pass.setBindGroup(0, this.groups.sceneRW); pass.setBindGroup(1, this.groups.rad);
            pass.dispatchWorkgroups(this._wg1(count));
          });
          start += count;
        }
      }
      slot = out;
    }
    this.slot = slot;
  }

  /** Make sure the node radiosity matches the current poses (once per setPoses). */
  /**
   * Bodies whose pose differs from the last node update (bit mask), and whether the scope (the light) moved.
   * Nothing moved: no radiosity work. Only tools moved: an incremental node update (nodes whose path to the light
   * passes near a moved body's capsule, then or now) and itersPerFrame Jacobi steps. The scope moved: everything.
   */
  _movedSinceSolve() {
    const last = this._solvedInst;
    if (!last) return { all: true, mask: 0 };
    let mask = 0, scope = false;
    const scopeI = this.S.blas.findIndex((b) => b.body === 'scope');
    this.instances.forEach((I, i) => {
      const L = last[i];
      const same = (!I && !L) || (I && L && I.q0.every((x, k) => Math.abs(x - L.q0[k]) < 1e-9) && I.q1.every((x, k) => Math.abs(x - L.q1[k]) < 1e-9)
        && I.p0.every((x, k) => Math.abs(x - L.p0[k]) < 1e-10) && I.p1.every((x, k) => Math.abs(x - L.p1[k]) < 1e-10));
      if (!same) { mask |= 1 << i; if (i === scopeI) scope = true; }
    });
    return { all: scope, mask: mask >>> 0 };
  }

  _ensureRadiosity(force = false) {
    if (!this.activeCount || (!force && this.solvedVersion === this.posesVersion)) return;
    if (!force && this.solvedVersion >= 0 && this.posesVersion - this.solvedVersion < (this.opts.radiosityEvery || 1)) return;
    const first = this.solvedVersion < 0;
    const mv = first || force ? { all: true, mask: 0 } : this._movedSinceSolve();
    if (mv.all || mv.mask) {
      this._submitNodeDirect(mv.all ? 0 : mv.mask);
      this._submitJacobi(first ? Math.min(this.opts.itersInitial, 8) : this.opts.itersPerFrame);
      this._solvedInst = this.instances.map((I) => (I ? { q0: [...I.q0], q1: [...I.q1], p0: [...I.p0], p1: [...I.p1] } : null));
      this._solvedCaps = (this._curCaps || []).map((c) => c && [...c]);
      this.stats.lastRadiosity = mv.all ? 'full' : 'incremental';
    } else this.stats.lastRadiosity = 'reused';
    if (first) this._pendingIters = this.opts.itersInitial - 8;      // the rest of the first solve, spread over frames
    else if (this._pendingIters > 0) { this._submitJacobi(Math.min(8, this._pendingIters)); this._pendingIters -= 8; }
    this.solvedVersion = this.posesVersion;
    this._writeSceneU(this._cut);
    // the mean irradiance changes much only when the light moves (a full update) or a solve finishes; tool motion
    // alone (incremental updates) does not trigger the read-back
    if (mv.all || first || (this._pendingIters !== undefined && this._pendingIters <= 0 && !this._eAvgSettled)) {
      if (this._pendingIters <= 0) this._eAvgSettled = true;
      this._refreshEAvg();
    }
  }

  /** Live path: refresh eAvg (mean irradiance for the tool-to-tool bounce) in the background, at most every 2 s. */
  _refreshEAvg() {
    const now = performance.now();
    if (this._eAvgBusy || (this._eAvgT !== undefined && now - this._eAvgT < 2000)) return;
    this._eAvgBusy = true; this._eAvgT = now;
    this.nodeSummary().catch(() => {}).finally(() => { this._eAvgBusy = false; });
  }

  /** Run Jacobi steps to convergence (judge mode), paced: one step per submit chunk, idle gaps every 1.5 s. */
  async converge(iters = 200) {
    // callers pass a flag as well as a count (judges.py capture_twin: converge = (k == 0) through app.js
    // r.converge(opts.converge ?? 200)): true means a full solve, false none. A boolean must not become 1 or 0 steps.
    if (typeof iters !== 'number' || !Number.isFinite(iters)) iters = iters === false ? 0 : 200;
    if (this.solvedVersion !== this.posesVersion) {
      this._submitNodeDirect(); this.solvedVersion = this.posesVersion;
      this._solvedInst = this.instances.map((I) => (I ? { q0: [...I.q0], q1: [...I.q1], p0: [...I.p0], p1: [...I.p1] } : null));
      this._solvedCaps = (this._curCaps || []).map((c) => c && [...c]);
    }
    for (let k = 0; k < iters; k++) {
      this._submitJacobi(1);
      if (k % 4 === 3) await this.device.queue.onSubmittedWorkDone();
      await this._pace();
    }
    this._pendingIters = 0;
    this._writeSceneU(this._cut);
    await this.device.queue.onSubmittedWorkDone();
    await this.nodeSummary();          // eAvg: the mean irradiance a posed body's BSDF ray reads (was left at 0)
  }

  /** Area-weighted mean irradiance over active nodes (also refreshes eAvg). Reads the node state back. */
  async nodeSummary() {
    const dev = this.device, S = this.S;
    const staging = dev.createBuffer({ size: S.staticVerts * 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.bufs.nodeState, 0, staging, 0, S.staticVerts * 64);
    dev.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const st = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    const slot = this.slot || 1;
    let A = 0;
    const E = [0, 0, 0], Ed = [0, 0, 0], Lo = [0, 0, 0];
    for (const i of this.activeList) {
      const a = S.nodeArea[i], V = st[16 * i + 3];
      A += a;
      for (let c = 0; c < 3; c++) {
        E[c] += a * st[16 * i + c] * V;
      }
      const m = this.S.materials[this.S.meshTable[this.S.nodeMesh[i]].mat];
      // slots 1/2 hold the separable source Ls; exitance / pi = A_hh Ls (kernels.wgsl.js jacobiUpdate)
      for (let c = 0; c < 3; c++) Lo[c] += a * st[16 * i + 4 * slot + c] * Math.max(1e-4, m.table ? m.table.hh[c] : 0);
      for (let c = 0; c < 3; c++) Ed[c] += a * st[16 * i + 12 + c] / Math.max(1e-3, m.table ? m.table.hh[c] : 1);
    }
    const mean = E.map((x) => x / A), eAvg = mean.map((x, c) => x + Ed[c] / A);
    this.eAvg = eAvg;
    this._writeSceneU(this._cut);
    return { area: A, meanIndirect: mean, meanDirect: Ed.map((x) => x / A), meanLout: Lo.map((x) => x / A), eAvg, state: st };
  }

  // ---------------------------------------------------------------- cameras
  _camTargets(name) {
    const cam = this.cameras[name];
    if (!cam) throw new Error(`unknown camera ${name}`);
    if (!cam.tex) {
      const [w, h] = cam.size, dev = this.device;
      cam.tex = dev.createTexture({ size: [w, h], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST, label: `${name}.radiance` });
      cam.ubuf = dev.createBuffer({ size: CAM_U_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      cam.acc = dev.createBuffer({ size: w * h * 16, usage: GPUBufferUsage.STORAGE });
      this._camGroup(cam, 1);
    }
    return cam;
  }

  /** (Re)build a camera's group-1 bind group with a static-hit cache for `spp` pass-1 sub-samples. */
  _camGroup(cam, spp) {
    const [w, h] = cam.size;
    if (cam.hitCache && cam.cacheSpp >= spp) return;
    cam.hitCache?.destroy(); cam.shadowCache?.destroy();
    cam.cacheSpp = spp; cam.cacheKey = null; cam.shadowKey = null;
    cam.hitCache = this.device.createBuffer({ size: w * h * 16 * spp, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `${cam.meta.name}.hitCache` });
    cam.shadowCache = this.device.createBuffer({ size: w * h * 4 * spp, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `${cam.meta.name}.shadowCache` });
    cam.group = this.device.createBindGroup({ layout: this.layouts.camG1, entries: [{ binding: 0, resource: { buffer: this.bufs.scn } }, { binding: 1, resource: { buffer: cam.ubuf } }, { binding: 2, resource: cam.tex.createView() }, { binding: 3, resource: { buffer: cam.acc } }, { binding: 4, resource: { buffer: cam.hitCache } }, { binding: 5, resource: { buffer: cam.shadowCache } }] });
  }

  _clearBuf(buf) { const enc = this.device.createCommandEncoder(); enc.clearBuffer(buf); this.device.queue.submit([enc.finish()]); }

  /** Bits of the bodies a camera's own rays ignore (its lens front, scene.meshes[].owner_camera). */
  _skipMask(name) {
    let m = 0;
    this.S.blas.forEach((b, i) => { if (b.owner && b.owner === name) m |= 1 << i; });
    return m >>> 0;
  }

  _packCam(buf, { R, pos, mode, lens, w, h, spp, seed, camIndex, options, aid = [0, 0, 0, 0], t0 = 0, t1 = 1, scale = 1, skip = 0, nLight = 1, refine = 0, root = 0, tile = [0, 0, w, h] }) {
    const f = new Float32Array(CAM_U_BYTES / 4), u = new Uint32Array(f.buffer);
    f.set([R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0, R[0][2], R[1][2], R[2][2], 0, pos[0], pos[1], pos[2], mode], 0);
    f.set([lens.fx, lens.fy, lens.cx, lens.cy, ...(lens.k || [0, 0, 0, 0]), lens.alpha || 0, 0, 0, 0], 16);
    u[25] = root >>> 0; u[26] = nLight >>> 0;
    u.set([w, h, spp, seed >>> 0, camIndex, options, this.frame >>> 0, skip >>> 0], 28);
    f.set(aid, 36); f.set([t0, t1, scale, refine], 40);
    u.set(tile, 44);
    this.device.queue.writeBuffer(buf, 0, f);
    return f;
  }

  /**
   * Trace a camera (or the overview) in row bands, one submit each, sized from the measured cost so each submit
   * stays near TARGET_MS (<= 0.5 M pixel samples). Pass 1 over every band, then the refinement pass over every band.
   */
  _submitTiles(label, kind, ubuf, group, base, pipe1, pipe2) {
    const { w, h, spp, refine } = base;
    const bandRows = (k, per) => Math.max(1, Math.min(h, Math.floor(this.timer.units(k) / (w * per))));
    for (const [pipe, k, per] of [[pipe1, kind, Math.max(1, spp)], [pipe2, 'refine', Math.max(1, refine)]]) {
      if (!pipe) continue;
      for (let y0 = 0; y0 < h;) {
        const rows = Math.min(h - y0, bandRows(k, per));
        this._packCam(ubuf, { ...base, tile: [0, y0, w, rows], nLight: pipe === pipe2 ? (base.nLightRefine ?? base.nLight) : base.nLight });
        this._submit(`${label}${pipe === pipe2 ? '.refine' : ''}#${y0}`, k, w * rows * per, (pass) => {
          pass.setPipeline(pipe);
          pass.setBindGroup(0, this.groups.sceneRO); pass.setBindGroup(1, group);
          pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(rows / 8));
        });
        y0 += rows;
      }
    }
  }

  _options(o = {}) {
    let bits = 0;
    if (o.layerLines ?? this.opts.layerLines) bits |= O.LAYER;
    if (o.noIndirect) bits |= O.NO_INDIRECT;
    if (o.directOnly) bits |= O.DIRECT_ONLY;
    if (o.noCapsule) bits |= O.NO_CAPSULE;
    if (o.normals) bits |= O.NORMALS;
    if (o.noMIS) bits |= O.NO_MIS;
    if (o.packet ?? this.opts.packet) bits |= O.PACKET;
    if (o.refineShadows ?? this.opts.refineShadows) bits |= O.REFINE_SHADOW;
    return bits;
  }

  _submitCamera(name, o = {}) {
    const cam = this._camTargets(name);
    const [w, h] = cam.size;
    const T = cam.T;
    const R = [[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]];
    const refine = o.refine ?? this.opts.refine;
    // static first-hit cache, one entry per sub-sample index (pass 1 and refinement): valid while the sample
    // pattern (seed), the BVH root and the camera hold; cleared to empty (t = 0) when they change
    const spp = o.spp ?? this.opts.spp;
    this._camGroup(cam, spp + refine);
    const key = `${o.seed ?? 0}|${o.fullBVH ? 0 : this.interiorRoot}|${T.flat().join(',')}|${cam.lens.fx},${cam.lens.cx},${cam.lens.cy}`;
    let cacheBits = 0;
    if (this.opts.hitCache !== false && o.hitCache !== false && !o.packet) {
      if (cam.cacheKey !== key) this._clearBuf(cam.hitCache);
      cacheBits = O.CACHE_WRITE | (cam.cacheKey === key ? O.CACHE_READ : 0);
    }
    // static-occlusion cache: also keyed on the scope (the light) over the exposure and the light-sample count
    if (cacheBits && this.opts.shadowCache !== false && o.shadowCache !== false && !o.exactShadows) {
      const si = this.S.blas.findIndex((b) => b.body === 'scope'), I = this.instances[si];
      const skey = `${key}|${o.lightSamples ?? this.opts.lightSamples}|${I ? [...I.q0, ...I.q1, ...I.p0, ...I.p1].join(',') : 'none'}`;
      if (cam.shadowKey !== skey) { this._clearBuf(cam.shadowCache); cam.shadowKey = skey; }
      cacheBits |= O.SHADOW_CACHE;
    }
    const base = {
      R, pos: [T[0][3], T[1][3], T[2][3]], mode: 0, lens: cam.lens, w, h, spp: o.spp ?? this.opts.spp, seed: o.seed ?? 0,
      camIndex: cam.index, options: this._options(o) | cacheBits, t0: o.shutter?.[0] ?? 0, t1: o.shutter?.[1] ?? 1, skip: this._skipMask(name),
      nLight: o.lightSamples ?? this.opts.lightSamples, nLightRefine: o.refineLightSamples ?? this.opts.refineLightSamples,
      refine, root: o.fullBVH ? 0 : this.interiorRoot,
    };
    this._submitTiles(name, (cacheBits & O.CACHE_READ) ? 'cameraCached' : 'camera', cam.ubuf, cam.group, base,
      o.exactShadows ? this.pipes.cameraExact : this.pipes.camera,
      refine > 0 ? (o.exactShadows ? this.pipes.refineExact : this.pipes.refine) : null);
    if (cacheBits) cam.cacheKey = key;
    this.stats.hitCache = (cacheBits & O.CACHE_READ) ? 'read' : (cacheBits & O.CACHE_WRITE) ? 'write' : 'off';
    this.stats.shadowCache = (cacheBits & O.SHADOW_CACHE) ? 'on' : 'off';
    if (o.exposure !== undefined) this._manualExposure(name, o.exposure);
    return cam.tex;
  }

  /** Render one tracking camera's scene-referred radiance (several budgeted submits). opts: {spp, refine, lightSamples, seed, exposure?, shutter?, exactShadows, ...debug}. */
  async renderCamera(name, opts = {}) {
    this._ensureRadiosity();
    return this._submitCamera(name, opts);
  }

  /**
   * A whole frame as a sequence of budgeted submits: radiosity update, every listed camera, their sensor stage and the
   * overview. opts: {cameras: [names], sensor: bool, overview: view|null, overviewOpts, sensorOpts, ...camera opts}.
   * Returns the textures and `done`, a promise for the GPU finishing the frame: live callers should await it before
   * the next frame, so submits never pile up (docs/WEB_TWIN.md §12).
   */
  renderFrame(opts = {}) {
    const names = opts.cameras || Object.keys(this.cameras);
    this._ensureRadiosity();
    const out = {};
    for (const n of names) out[n] = { linear: this._submitCamera(n, opts) };
    if (opts.sensor) for (const n of names) out[n].rgb8 = this.sensor(n, opts.sensorOpts || {});
    if (opts.overview) out.overview = this.renderOverview(opts.overview, { present: false, ...(opts.overviewOpts || {}) });
    this.frame++;
    out.done = this.device.queue.onSubmittedWorkDone();
    return out;
  }

  // ---------------------------------------------------------------- sensor
  async _buildSensor() {
    this.sensorP = sensorParams(this.physics, this.opts.sensor || {});
    this.sensorModule = await this._module(SENSOR, 'sensor');
    const mk = (entryPoint) => this.device.createComputePipeline({ layout: 'auto', compute: { module: this.sensorModule, entryPoint } });
    for (const e of ['sensorOptics', 'glareDown', 'glareBlurH', 'glareBlurV', 'sensorExpose', 'demosaicG', 'demosaicRB', 'sensorNR', 'sensorOutput']) this.pipes[e] = mk(e);
    for (const [name, cam] of Object.entries(this.cameras)) cam.ae = new AutoExposure(this.sensorP, { exposureUnits: this.sensorP.exposureUnits0 });
  }

  _sensorTargets(name) {
    const cam = this._camTargets(name);
    if (cam.s) return cam.s;
    const dev = this.device, [w, h] = cam.size;
    const t32 = (label, W = w, H = h) => dev.createTexture({ size: [W, H], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC, label });
    const F = this.sensorP.glareLevel || 16, gw = Math.ceil(w / F), gh = Math.ceil(h / F);
    const s = {
      u: dev.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      ri: this._buf(relativeIlluminationMap(cam.lens, this.sensorP, [w, h]), GPUBufferUsage.STORAGE, `${name}.ri`),
      A: t32(`${name}.irr`), B: t32(`${name}.isp`), GA: t32(`${name}.glareA`, gw, gh), GB: t32(`${name}.glareB`, gw, gh), gw, gh,
      raw: dev.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      gpl: dev.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.STORAGE }),
      stats: dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
      statsRead: [],
      yuv: dev.createBuffer({ size: (w * h + 2 * Math.ceil(w / 2) * Math.ceil(h / 2)) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      out: dev.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC, label: `${name}.rgb8` }),
    };
    const P = this.pipes;
    const U = { binding: 0, resource: { buffer: s.u } }, RI = { binding: 1, resource: { buffer: s.ri } };
    const tex = (binding, t) => ({ binding, resource: t.createView() }), buf = (binding, b) => ({ binding, resource: { buffer: b } });
    const G = (pipe, entries) => dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    s.groups = [
      [P.sensorOptics, G(P.sensorOptics, [U, RI, tex(2, cam.tex), tex(3, s.A)]), w, h],
      [P.glareDown, G(P.glareDown, [U, RI, tex(2, cam.tex), tex(10, s.GA)]), gw, gh],
      [P.glareBlurH, G(P.glareBlurH, [U, tex(9, s.GA), tex(10, s.GB)]), gw, gh],
      [P.glareBlurV, G(P.glareBlurV, [U, tex(9, s.GB), tex(10, s.GA)]), gw, gh],
      [P.sensorExpose, G(P.sensorExpose, [U, tex(2, s.A), tex(9, s.GA), buf(4, s.raw)]), w, h],
      [P.demosaicG, G(P.demosaicG, [U, buf(4, s.raw), buf(8, s.gpl)]), w, h],
      [P.demosaicRB, G(P.demosaicRB, [U, buf(4, s.raw), buf(8, s.gpl), tex(3, s.B), buf(5, s.stats)]), w, h],
      [P.sensorNR, G(P.sensorNR, [U, tex(2, s.B), tex(3, s.A)]), w, h],
      [P.sensorOutput, G(P.sensorOutput, [U, tex(2, s.A), buf(5, s.stats), tex(6, s.out), buf(7, s.yuv)]), Math.ceil(w / 2), Math.ceil(h / 2)],
    ];
    cam.s = s;
    return s;
  }

  _manualExposure(name, e) {
    const cam = this.cameras[name];
    if (e === null || e === 'auto') { cam.manual = null; return; }
    cam.manual = typeof e === 'number' ? { time_s: e, gain: 1 } : { time_s: e.time_s ?? e.exposure_s ?? e.time, gain: e.gain ?? 1 };
  }

  /** The exposure the sensor stage will use next for this camera (AE or manual). */
  exposureOf(name) {
    const cam = this.cameras[name];
    if (cam.manual) return { ...cam.manual, auto: false };
    return { time_s: cam.ae.exposure, gain: cam.ae.gain, gainUnits: cam.ae.gainUnits, units: cam.ae.units, auto: true };
  }

  _encodeSensor(enc, name, o = {}) {
    const cam = this._camTargets(name), s = this._sensorTargets(name), [w, h] = cam.size;
    const ex = this.exposureOf(name);
    const p = { ...this.sensorP, ...(o.params || {}) };
    if (p.awb && cam.awb && !(o.params && o.params.wb)) p.wb = p.wb.map((g, c) => g * cam.awb[c]);
    if (o.noise === false) p.noise = false;
    const ub = packSensorUniform(p, { width: w, height: h, frame: o.frame ?? this.frame, seed: o.seed ?? 0, exposure: o.exposure_s ?? ex.time_s, gain: o.gain ?? ex.gain, camIndex: cam.index });
    this.device.queue.writeBuffer(s.u, 0, ub);
    this.device.queue.writeBuffer(s.stats, 0, new Uint32Array(4));
    const pass = enc.beginComputePass({ timestampWrites: this.timer.pass(`${name}.sensor`, 'sensor', 1, this.frame) });
    for (const [pipe, group, W, H] of s.groups) { pass.setPipeline(pipe); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8)); }
    pass.end();
    // AE statistics, read back without blocking (UVC AE also acts on the previous frame)
    const rd = s.statsRead.pop() || this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(s.stats, 0, rd, 0, 16);
    s.pendingStats = rd;
    return s.out;
  }

  _afterSensor(name) {
    const cam = this.cameras[name], s = cam.s, rd = s.pendingStats;
    if (!rd) return;
    s.pendingStats = null;
    const [w, h] = cam.size;
    rd.mapAsync(GPUMapMode.READ).then(() => {
      const st = new Uint32Array(rd.getMappedRange().slice(0));
      const sum = st[0];
      cam.meanLuma = sum / (w * h * 255);
      cam.meanRGB = [st[1], st[2], st[3]].map((x) => x / (w * h * 1023));
      if (!cam.manual) cam.ae.update(cam.meanLuma);
      if (this.sensorP.awb) cam.awb = updateAWB(cam.awb, cam.meanRGB, this.sensorP.awbDamping);
      rd.unmap(); s.statsRead.push(rd);
    }).catch(() => {});
  }

  /** Radiance -> 8-bit camera RGB for `name` (uses the radiance from its last renderCamera). opts: {seed, frame, noise, exposure_s, gain, params} */
  sensor(name, opts = {}) {
    const enc = this.device.createCommandEncoder();
    const tex = this._encodeSensor(enc, name, opts);
    const after = this.timer.flush(enc);
    this.device.queue.submit([enc.finish()]); after();
    this._afterSensor(name);
    return tex;
  }

  sensorInfo() { return { params: this.sensorP, ...describe(this.sensorP) }; }

  // ---------------------------------------------------------------- overview
  _overviewTargets(w, h) {
    if (this.ov && this.ov.w === w && this.ov.h === h) return this.ov;
    const dev = this.device;
    const ov = { w, h };
    ov.tex = dev.createTexture({ size: [w, h], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    ov.out = dev.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    ov.ubuf = dev.createBuffer({ size: CAM_U_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    ov.dbuf = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    ov.acc = dev.createBuffer({ size: w * h * 16, usage: GPUBufferUsage.STORAGE });
    ov.cache = dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });        // the overview moves: no hit cache
    ov.cache2 = dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    ov.group = dev.createBindGroup({ layout: this.layouts.camG1, entries: [{ binding: 0, resource: { buffer: this.bufs.scn } }, { binding: 1, resource: { buffer: ov.ubuf } }, { binding: 2, resource: ov.tex.createView() }, { binding: 3, resource: { buffer: ov.acc } }, { binding: 4, resource: { buffer: ov.cache } }, { binding: 5, resource: { buffer: ov.cache2 } }] });
    ov.dgroup = dev.createBindGroup({ layout: this.pipes.display.getBindGroupLayout(0), entries: [{ binding: 0, resource: ov.tex.createView() }, { binding: 1, resource: ov.out.createView() }, { binding: 2, resource: { buffer: ov.dbuf } }] });
    this.ov = ov;
    return ov;
  }

  /** Pinhole camera axes and intrinsics for an orbit view {eye, target, up?, fovy_deg?, size?}. */
  overviewCamera(view) {
    const [w, h] = view.size || [960, 540];
    const z = norm3(sub3(view.target, view.eye));
    const x = norm3(cross3(z, view.up || [0, 0, 1])), y = cross3(z, x);
    const f = (h / 2) / Math.tan(((view.fovy_deg || 50) * Math.PI) / 360);
    return { R: [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]], pos: view.eye, lens: { fx: f, fy: f, cx: w / 2 - 0.5, cy: h / 2 - 0.5 }, w, h };
  }

  /** Project a D-frame point into the overview (for overlays). */
  overviewProject(view, p) {
    const c = this.overviewCamera(view);
    const q = mulRtv(c.R, sub3(p, c.pos));
    return q[2] > 0 ? [c.lens.fx * q[0] / q[2] + c.lens.cx, c.lens.fy * q[1] / q[2] + c.lens.cy, q[2]] : null;
  }

  /** opts: {cut: {n, d} | null, aid: false | {intensity, dir?}, spp, displayScale, background: [r,g,b] (0..1, display-encoded) | null}. Returns rgba8unorm. */
  renderOverview(view, o = {}) {
    this._ensureRadiosity();
    const c = this.overviewCamera(view);
    const ov = this._overviewTargets(c.w, c.h);
    let bits = this._options(o);
    if (o.cut) bits |= O.CUT;
    const aid = o.aid === false || o.aid == null ? null : { intensity: 1, ...(typeof o.aid === 'object' ? o.aid : {}) };
    if (aid) bits |= O.AID;
    const fwd = c.R.map((r) => r[2]);
    this._cut = o.cut ? { n: norm3(o.cut.n), d: o.cut.d } : null;
    this._writeSceneU(this._cut);
    const base = {
      R: c.R, pos: c.pos, mode: 1, lens: c.lens, w: c.w, h: c.h, spp: o.spp ?? 1, seed: o.seed ?? 0, camIndex: 15, options: bits,
      aid: aid ? [...(aid.dir || fwd.map((x) => -x)), aid.intensity * (aid.scale ?? 20)] : [0, 0, 0, 0],
      nLight: o.lightSamples ?? 1, refine: 0, root: 0,
    };
    this._submitTiles('overview', 'overview', ov.ubuf, ov.group, base, this.pipes.overview, null);
    // background of the overview's misses (paper #f3f3ef by default, `background: null` for black); camera feeds stay dark
    const bg = o.background === undefined ? (this.opts.overviewBackground ?? [0xf3 / 255, 0xf3 / 255, 0xef / 255]) : o.background;
    this.device.queue.writeBuffer(ov.dbuf, 0, new Float32Array([o.displayScale ?? 0.1, 0, 0, 0, ...(bg ? [bg[0], bg[1], bg[2], 1] : [0, 0, 0, 0])]));
    this._submit('overview.display', null, 0, (pass) => {
      pass.setPipeline(this.pipes.display);
      pass.setBindGroup(0, ov.dgroup);
      pass.dispatchWorkgroups(Math.ceil(c.w / 8), Math.ceil(c.h / 8));
    });
    if (this.opts.canvas && o.present !== false) this.present(ov.out, this.opts.canvas);
    return ov.out;
  }

  /** Draw an rgba8 texture onto a canvas (WebGPU context). */
  present(tex, canvas) {
    const dev = this.device;
    if (!this._blit) {
      const code = `@group(0) @binding(0) var t: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); return vec4f(p[i], 0, 1); }
@group(0) @binding(1) var<uniform> sz: vec4f;
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f { let d = vec2f(textureDimensions(t)); let q = vec2i(p.xy * d / sz.xy); return textureLoad(t, q, 0); }`;
      const m = dev.createShaderModule({ code });
      this._blitFormat = navigator.gpu.getPreferredCanvasFormat();
      this._blit = dev.createRenderPipeline({ layout: 'auto', vertex: { module: m, entryPoint: 'vs' }, fragment: { module: m, entryPoint: 'fs', targets: [{ format: this._blitFormat }] } });
      this._blitU = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (!canvas._twinCtx) { canvas._twinCtx = canvas.getContext('webgpu'); canvas._twinCtx.configure({ device: dev, format: this._blitFormat, alphaMode: 'opaque' }); }
    dev.queue.writeBuffer(this._blitU, 0, new Float32Array([canvas.width, canvas.height, 0, 0]));
    const g = dev.createBindGroup({ layout: this._blit.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: { buffer: this._blitU } }] });
    const enc = dev.createCommandEncoder();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: canvas._twinCtx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    rp.setPipeline(this._blit); rp.setBindGroup(0, g); rp.draw(3); rp.end();
    dev.queue.submit([enc.finish()]);
  }

  // ---------------------------------------------------------------- readback
  /** CPU copy. kind: 'linear' (Float32Array w*h*4), 'rgb8' (Uint8Array w*h*3), 'rgba8', 'yuv420' ({y, cb, cr}), 'raw' (Float32Array DN). */
  async readback(name, kind = 'rgb8') {
    const dev = this.device;
    let tex, bpp, w, h;
    if (name === 'overview') { tex = kind === 'linear' ? this.ov.tex : this.ov.out; w = this.ov.w; h = this.ov.h; }
    else {
      const cam = this.cameras[name];
      [w, h] = cam.size;
      if (kind === 'linear') tex = cam.tex;
      else if (kind === 'raw' || kind === 'yuv420') {
        const src = kind === 'raw' ? cam.s.raw : cam.s.yuv;
        const size = src.size;
        const st = dev.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(src, 0, st, 0, size); dev.queue.submit([enc.finish()]);
        await st.mapAsync(GPUMapMode.READ);
        const data = st.getMappedRange().slice(0); st.unmap(); st.destroy();
        if (kind === 'raw') return new Float32Array(data);
        const u = new Uint32Array(data), cw = Math.ceil(w / 2), ch = Math.ceil(h / 2);
        return { width: w, height: h, y: Uint8Array.from(u.subarray(0, w * h)), cb: Uint8Array.from(u.subarray(w * h, w * h + cw * ch)), cr: Uint8Array.from(u.subarray(w * h + cw * ch, w * h + 2 * cw * ch)) };
      } else tex = cam.s.out;
    }
    bpp = kind === 'linear' ? 16 : 4;
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
    if (kind === 'rgba8') { const out = new Uint8Array(w * h * 4); for (let y = 0; y < h; y++) out.set(data.subarray(y * bpr, y * bpr + w * 4), y * w * 4); return out; }
    const out = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * bpr + 4 * x, j = 3 * (y * w + x);
      out[j] = data[i]; out[j + 1] = data[i + 1]; out[j + 2] = data[i + 2];
    }
    return out;
  }

  // ---------------------------------------------------------------- queries
  /**
   * Occlusion from the camera centre to each D-frame point, in the renderer's own BVH (static + posed bodies at
   * mid-exposure). A point is visible when no surface lies between the lens centre and `tol` short of the point.
   */
  visible(name, pointsD, { tol = 2e-4, time = 0.5 } = {}) {
    const cam = this.cameras[name], T = cam.T;
    const skip = this._skipMask(name);
    const c = [T[0][3], T[1][3], T[2][3]];
    const S = this.S;
    const poses = S.blas.map((_, i) => this._poseAt(i, time));
    return pointsD.map((p) => {
      const v = sub3(p, c), dist = Math.hypot(...v);
      if (!(dist > tol)) return true;
      const d = v.map((x) => x / dist);
      const tmax = dist - tol;
      if (traverse(S.nodes, S.tris, c, d, { tmax, anyHit: true, root: S.staticRoot }).tri >= 0) return false;
      for (let i = 0; i < S.blas.length; i++) {
        const P = poses[i];
        if (!P || ((skip >>> i) & 1)) continue;
        const ol = mulRtv(P.R, sub3(c, P.p)), dl = mulRtv(P.R, d);
        if (traverse(S.nodes, S.tris, ol, dl, { tmax, anyHit: true, root: S.blas[i].root }).tri >= 0) return false;
      }
      return true;
    });
  }

  /** KB4 projection of D-frame points into a camera (float64, as lens.js). */
  project(name, pointsD) {
    const cam = this.cameras[name], T = cam.T;
    const R = [[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]], c = [T[0][3], T[1][3], T[2][3]];
    return pointsD.map((p) => project(cam.lens, mulRtv(R, sub3(p, c))));
  }

  /**
   * GPU timestamp ms. frames: per frame, per pass (tiles summed), with its submit count and largest submit;
   * maxSubmit: the largest single submit seen; over: submits above WARN_MS; rates: ms per work unit by kind.
   */
  timings() {
    const frames = this.timer.frameRecords().slice(-64);
    return { gpu: this.timer.gpu, frames, last: frames[frames.length - 1] || {}, maxSubmit: { ...this.timer.maxSubmit }, over: this.timer.over.slice(-32), rates: { ...this.timer.rates }, submits: this.timer.submits, stats: this.stats };
  }

  async idle() { await this.device.queue.onSubmittedWorkDone(); for (let i = 0; i < 50 && this.timer.pending; i++) await new Promise((r) => setTimeout(r, 5)); }

  destroy() { for (const b of Object.values(this.bufs)) b?.destroy?.(); }
}
