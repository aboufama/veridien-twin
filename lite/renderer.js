// LiteRenderer: the live WebGPU twin of the aimed dome at a small fraction of the path tracer's cost
// (docs/WEB_TWIN.md §18.1, ADR-142). Same §8 API as gpu/renderer.js, so app.js and window.twinJudge work unchanged:
//
//   const r = await LiteRenderer.create({ canvas?, scene, physics, device?, log? });   // scene: {scene, meshes, base} | URL
//   r.cameras                                  // ['CAM1', 'CAM2', 'CAM3', 'SCOPE']
//   r.setPoses(poses0, poses1?);               // mid-exposure poses are rendered (no motion blur)
//   await r.renderCamera(name, {exposure?});   // linear radiance (W sr^-1 m^-2) -> GPUTexture rgba16float (see notes)
//   r.sensor(name, {seed, frame, noise, exposure_s, gain}) -> GPUTexture rgba8unorm (display / MJPG source)
//   r.present(name, context, {fit: 'contain', rect?, clear?, background?})   // sensor texture -> WebGPU canvas
//   await r.readback(name, 'rgb8' | 'rgba8' | 'linear' | 'yuv420')
//   r.visible(name, pointsD) -> bool[]          // CPU BVH (built in a worker during create)
//   r.timings()                                // GPU timestamp ms per pass, per view
//
// Methods (what is computed, what is approximated):
//   * Geometry: WebGPU raster, 4x MSAA, of the cavity-facing static triangles (split to <= 3 mm edges) and the posed
//     bodies (sliced to <= 5 mm along their axes), KB4 per vertex for CAM1-3, a pinhole + image circle for SCOPE.
//     scripts/build_lite_twin.mjs precomputes it all (assets/lite_*.bin, 20 MB): no BVH or subdivision at load.
//   * Direct light: the scope outlet as 8 area-weighted points with L0 p(theta), the §8 BSDF per point and pixel;
//     soft shadows from one capsule per posed body against the outlet's disk (band / disk coverage of the light
//     cone); no static self-shadowing (the cavity is convex from the light, except bores and pockets).
//   * Indirect light: radiosity on 12,976 nodes (~7 mm) with a precomputed gather table of 256 cosine rays per
//     node; per pose change: node direct light, the capsule occlusion of every gather ray (a blocked ray brings
//     the body's albedo x the radiance), warm-started Jacobi steps. Static pixels interpolate the node irradiance
//     per vertex (4 nodes) and re-emit it with the BSDF's directional albedo. Posed bodies read a 128^2 cube of the
//     node radiance at the cavity centre, parallax-corrected by its distance channel (GGX lobe -> mip).
//   * Sensor (display): relative illumination (KB4 form), veiling glare (1/16 level), AE, AWB, tone + S-curve,
//     optional noise, 4:2:0 planes for the camera MJPG stage. No Bayer mosaic / demosaic / NR / optics blur.
//   * Deviation from §8: renderCamera returns an rgba16float texture (MSAA resolve); readback('linear') still
//     returns Float32Array w*h*4. Motion blur: the mid-exposure pose only.
// Measured (lite judge, Mitsuba 4096 spp references; GPU ms on the pod's RTX 4090): see docs/DECISIONS.md ADR-142
// and the numbers block at the end of this comment once the judge has run.

import { lensParams, project as kb4Project } from '../gpu/lens.js';
import { loadScene, materialsFromPhysics, emitterFromPhysics, ALBEDO_BINS } from '../gpu/scene_gpu.js';
import { sensorParams, AutoExposure, updateAWB } from '../gpu/sensor.js';
import { traverse } from '../gpu/bvh.js';
import { RADIOSITY, GATHER, ENV, ENV_MIP, CAMERA, LINEAR, SENSOR, PRESENT, MAX_EMIT, MAX_CAPS } from './shaders.js';
import { scaledLens, scopeCamera, emitterSamples, profileTable, bodyCapsules, posedCapsules, lerpPose, liteViews, bodyGroup, applyT } from './math.js';

export const LITE_DEFAULTS = Object.freeze({
  msaa: 4, envSize: 128, emitterPoints: 8, itersPerFrame: 3, itersInitial: 64, near: 0.001, far: 1.5,
  scopeSize: [1280, 720], indirect: true, direct: true,
});
export const WARN_MS = 16;

const GPU = typeof GPUBufferUsage !== 'undefined' ? GPUBufferUsage : {};
const TU = typeof GPUTextureUsage !== 'undefined' ? GPUTextureUsage : {};

class GpuTimer {
  constructor(dev, log) {
    this.dev = dev; this.log = log; this.on = dev.features.has('timestamp-query');
    this.cap = 32; this.records = []; this.maxSubmit = { ms: 0, label: null }; this.over = []; this.submits = 0; this.frame = 0;
    if (this.on) {
      this.qs = dev.createQuerySet({ type: 'timestamp', count: 2 * this.cap });
      this.res = dev.createBuffer({ size: 16 * this.cap, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.pool = [];
    }
    this.labels = [];
  }
  tw(label) {
    if (!this.on || this.labels.length >= this.cap) { this.labels.push(label); return undefined; }
    const i = this.labels.length;
    this.labels.push(label);
    return { querySet: this.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
  }
  /** Encode the resolve for this submit; returns a callback to run after queue.submit. */
  flush(enc, submitLabel) {
    const labels = this.labels; this.labels = [];
    const frame = this.frame;
    this.submits++;
    if (!this.on || !labels.length) return () => {};
    const n = Math.min(labels.length, this.cap);
    enc.resolveQuerySet(this.qs, 0, 2 * n, this.res, 0);
    const rd = this.pool.pop() || this.dev.createBuffer({ size: 16 * this.cap, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(this.res, 0, rd, 0, 16 * n);
    return () => {
      rd.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(rd.getMappedRange().slice(0, 16 * n));
        let tot = 0;
        const passes = {};
        for (let i = 0; i < n; i++) { const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6; if (ms >= 0 && ms < 1e4) { passes[labels[i]] = (passes[labels[i]] || 0) + ms; tot += ms; } }
        rd.unmap(); this.pool.push(rd);
        this.records.push({ frame, submit: submitLabel, passes, ms: tot, t: performance.now() });
        if (this.records.length > 512) this.records.splice(0, this.records.length - 512);
        if (tot > this.maxSubmit.ms) this.maxSubmit = { ms: tot, label: submitLabel };
        if (tot > WARN_MS) { this.over.push({ submit: submitLabel, ms: tot, frame }); if (this.over.length > 64) this.over.shift(); this.log(`submit ${submitLabel} took ${tot.toFixed(1)} ms of GPU time (> ${WARN_MS} ms)`); }
      }).catch(() => {});
    };
  }
}

export class LiteRenderer {
  static async create(opts) {
    const r = new LiteRenderer();
    await r._init(opts);
    return r;
  }

  get isStub() { return false; }
  get isLite() { return true; }

  async _init(opts) {
    const t0 = performance.now();
    this.opts = { ...LITE_DEFAULTS, ...opts };
    this.log = opts.log || ((...a) => console.log('[twin-lite]', ...a));
    let physics = opts.physics;
    if (typeof physics === 'string') physics = await (await fetch(physics)).json();
    this.physics = physics || {};
    let src = opts.scene;
    if (typeof src === 'string') src = await loadScene(src);
    this.scene = src.scene; this.meshes = src.meshes || {};
    const base = opts.assets || src.base || 'assets/';
    // lite assets (scripts/build_lite_twin.mjs)
    const meta = await (await fetch(base + 'lite.json', { cache: 'no-store' })).json();
    const bufs = {};
    await Promise.all(Object.entries(meta.files).map(async ([k, f]) => { bufs[k] = await (await fetch(base + f.file, { cache: 'no-store' })).arrayBuffer(); }));
    this.meta = meta;
    this.V = liteViews(meta, bufs);
    this._startOcclusion();             // the visibility BVHs build in a worker while the GPU side is set up
    this.stats = { loadMs: performance.now() - t0, staticTriangles: this.V.static.index.length / 3, dynamicTriangles: this.V.dynamic.index.length / 3, nodes: meta.files.nodes.count, K: meta.files.nodes.K };

    // device
    if (opts.device) this.device = opts.device;
    else {
      if (!navigator.gpu) throw new Error('WebGPU is not available');
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('no WebGPU adapter');
      this.adapterInfo = adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : {};
      const features = ['timestamp-query'].filter((f) => adapter.features.has(f));
      // the compute passes bind 10 storage buffers; the default limit is 8, and the M4 rejected every pass without this
      // (black frames, ~27k validation errors in the 2026-10-31 M4 bench)
      const L = adapter.limits;
      this.device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: {
        maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
    }
    const dev = this.device;
    dev.lost?.then((i) => { this.lostInfo = i.message || String(i.reason); this.log('device lost', i.reason, i.message); });
    dev.addEventListener?.('uncapturederror', (e) => { this.lastError = e.error?.message; this.log('GPU error', e.error?.message); });
    this.timer = new GpuTimer(dev, this.log);

    this._materials();
    this._emitter();
    this.caps0 = bodyCapsules(this.scene, this.meshes, this.matByName);
    this._buffers();
    await this._boardTexture(base);
    this._pipelines();
    this._camerasInit();
    this.sensorP = sensorParams(this.physics, this.opts.sensor || {});
    for (const c of Object.values(this.cams)) c.ae = new AutoExposure(this.sensorP, { exposureUnits: this.sensorP.exposureUnits0 });
    await this.occReady;
    this.poses0 = { bodies: {} }; this.poses1 = this.poses0; this.posesVersion = 0; this.solvedVersion = -1; this.solves = 0;
    this.frame = 0;
    this.stats.initMs = performance.now() - t0;
    this.log(`lite ready in ${this.stats.initMs.toFixed(0)} ms: ${this.stats.staticTriangles} static + ${this.stats.dynamicTriangles} dynamic triangles, ${this.stats.nodes} nodes x ${this.stats.K} rays, timestamp-query ${this.timer.on}`);
  }

  // ---------------------------------------------------------------- materials, emitter
  _materials() {
    const mt = this.meta.meshTable;
    const mats = materialsFromPhysics(this.physics, [...new Set(mt.map((m) => m.material))]);
    this.matByName = Object.fromEntries(mats.list.map((m) => [m.name, m]));
    const MAT_F = 20;
    const buf = new Float32Array(mt.length * MAT_F), alb = new Float32Array(mt.length * ALBEDO_BINS * 8);
    mt.forEach((m, i) => {
      const M = this.matByName[m.material];
      const o = i * MAT_F;
      const emitter = !!M.emitter || m.material === 'emitter';
      const kind = emitter ? 1 : M.texture ? 2 : 0;
      const c = M.metallic ? M.f0 : M.rho;
      buf.set([c[0], c[1], c[2], M.alpha], o);
      buf.set([M.ior, M.metallic, M.fbar ?? 0, kind], o + 4);
      if (M.coated) buf.set([M.tau[0], M.tau[1], M.tau[2], M.alphaC], o + 8);
      else if (M.texture) buf.set([0, 0, 0, M.texture.roughInk * M.texture.roughInk], o + 8);
      const hh = M.table?.hh ?? [0, 0, 0];
      buf.set([hh[0], hh[1], hh[2], 0], o + 12);
      if (M.texture) buf.set([...M.texture.ink, 0], o + 16);
      if (M.table && !emitter) for (let b = 0; b < ALBEDO_BINS; b++) {
        alb.set([M.table.spec[3 * b], M.table.spec[3 * b + 1], M.table.spec[3 * b + 2], 0], ((i * ALBEDO_BINS + b) * 2) * 4);
        alb.set([M.table.diff[3 * b], M.table.diff[3 * b + 1], M.table.diff[3 * b + 2], 0], ((i * ALBEDO_BINS + b) * 2 + 1) * 4);
      }
    });
    this.matData = buf; this.albData = alb;
  }

  _emitter() {
    const em = emitterFromPhysics(this.physics);
    const md = this.meshes[this.scene.scope?.emitter_mesh];
    if (!md) throw new Error('scene has no scope light outlet mesh (scope.emitter_mesh)');
    const es = emitterSamples(md, Math.min(MAX_EMIT, this.opts.emitterPoints));
    const a = (Number.isFinite(em.axisDeg) ? em.axisDeg : 30) * Math.PI / 180;
    this.emit = { ...es, L0: em.L0, axisK: [0, Math.sin(a), Math.cos(a)], prof: profileTable(em.theta, em.p) };
  }

  // ---------------------------------------------------------------- buffers
  _buf(data, usage, label) {
    let bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    if (bytes.byteLength % 4) { const p = new Uint8Array((bytes.byteLength + 3) & ~3); p.set(bytes); bytes = p; }
    const b = this.device.createBuffer({ size: Math.max(16, bytes.byteLength), usage: usage | GPU.COPY_DST, label });
    this.device.queue.writeBuffer(b, 0, bytes);
    return b;
  }

  _buffers() {
    const dev = this.device, V = this.V, S = this.meta.files.static, D = this.meta.files.dynamic, N = this.meta.files.nodes;
    this.vb = {
      sPos: this._buf(V.static.pos, GPU.VERTEX, 'static.pos'), sNrm: this._buf(V.static.nrm, GPU.VERTEX, 'static.nrm'),
      sNodes: this._buf(V.static.nodes, GPU.VERTEX, 'static.nodes'), sW: this._buf(V.static.w, GPU.VERTEX, 'static.w'),
      sIdx: this._buf(V.static.index, GPU.INDEX, 'static.index'),
      dPos: this._buf(V.dynamic.pos, GPU.VERTEX, 'dyn.pos'), dNrm: this._buf(V.dynamic.nrm, GPU.VERTEX, 'dyn.nrm'),
      dUv: this._buf(V.dynamic.uv, GPU.VERTEX, 'dyn.uv'), dIdx: this._buf(V.dynamic.index, GPU.INDEX, 'dyn.index'),
      envIdx: this._buf(V.nodes.env, GPU.INDEX, 'env.index'),
    };
    // nodes: nodeA = pos, area; nodeN = normal, mesh (bits)
    const n = N.count, rec = V.nodes.nodes, recU = new Uint32Array(rec.buffer, rec.byteOffset, rec.length);
    const A = new Float32Array(4 * n), Nn = new Float32Array(4 * n), NnU = new Uint32Array(Nn.buffer);
    for (let i = 0; i < n; i++) { A.set(rec.subarray(8 * i, 8 * i + 4), 4 * i); Nn.set(rec.subarray(8 * i + 4, 8 * i + 7), 4 * i); NnU[4 * i + 3] = recU[8 * i + 7]; }
    const zeros = (k, label) => dev.createBuffer({ size: Math.max(16, k), usage: GPU.STORAGE | GPU.COPY_DST | GPU.COPY_SRC, label });
    this.nb = {
      A: this._buf(A, GPU.STORAGE, 'node.A'), N: this._buf(Nn, GPU.STORAGE, 'node.N'),
      gather: this._buf(V.nodes.gather, GPU.STORAGE, 'node.gather'),
      occ: zeros(n * N.K, 'node.occ'), edir: zeros(16 * n, 'node.edir'), B0: zeros(16 * n, 'node.B0'), B1: zeros(16 * n, 'node.B1'),
      eind: zeros(16 * n, 'node.eind'),
    };
    this.nodeCount = n; this.K = N.K;
    this.ub = {
      light: dev.createBuffer({ size: 384, usage: GPU.UNIFORM | GPU.COPY_DST }),
      caps: dev.createBuffer({ size: 592, usage: GPU.UNIFORM | GPU.COPY_DST }),
      scene: dev.createBuffer({ size: 96, usage: GPU.UNIFORM | GPU.COPY_DST }),
      rad: dev.createBuffer({ size: 16, usage: GPU.UNIFORM | GPU.COPY_DST }),
    };
    dev.queue.writeBuffer(this.ub.rad, 0, new Uint32Array([n, N.K, 0, 0]));
    this.mats = this._buf(this.matData, GPU.STORAGE, 'mats');
    this.alb = this._buf(this.albData, GPU.STORAGE, 'albedo');
    // posed bodies
    this.bodyKeys = D.bodies;
    this.bodyBuf = dev.createBuffer({ size: 64 * D.bodies.length, usage: GPU.STORAGE | GPU.COPY_DST });
    const bg = new Uint32Array(4 * D.bodies.length);
    D.bodies.forEach((k, i) => { bg[4 * i] = bodyGroup(this.scene, k); });
    this.bodyGroupBuf = this._buf(bg, GPU.STORAGE, 'body.group');
    const sc = new Float32Array(24);
    sc.set([...S.bmin, 0, ...S.bmax.map((x, k) => x - S.bmin[k]), 0, ...D.bmin, 0, ...D.bmax.map((x, k) => x - D.bmin[k]), 0, ...this.meta.probe, 0, this.opts.indirect ? 1 : 0, this.opts.direct ? 1 : 0, 0, 0]);
    dev.queue.writeBuffer(this.ub.scene, 0, sc);
    // reflection cube (node radiance, distance in alpha)
    const E = this.opts.envSize, mips = Math.floor(Math.log2(E)) + 1;
    this.env = {
      size: E, mips,
      tex: dev.createTexture({ size: [E, E, 6], format: 'rgba16float', mipLevelCount: mips, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING | TU.STORAGE_BINDING }),
      depth: dev.createTexture({ size: [E, E], format: 'depth32float', usage: TU.RENDER_ATTACHMENT }),
      faceU: [0, 1, 2, 3, 4, 5].map(() => dev.createBuffer({ size: 48, usage: GPU.UNIFORM | GPU.COPY_DST })),
    };
    for (let f = 0; f < 6; f++) dev.queue.writeBuffer(this.env.faceU[f], 0, new Float32Array([...this.meta.probe, 0, 0, 0, 0, 0, 0.002, 2.0, 0, 0]));
    for (let f = 0; f < 6; f++) dev.queue.writeBuffer(this.env.faceU[f], 16, new Uint32Array([f, 0, 0, 0]));
    this.sampler = dev.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  }

  async _boardTexture(base) {
    const dev = this.device;
    const b = (this.scene.boards || [])[0];
    let tex = null;
    if (b?.texture && typeof createImageBitmap !== 'undefined') {
      try {
        const blob = await (await fetch(base + b.texture)).blob();
        const bmp = await createImageBitmap(blob);
        tex = dev.createTexture({ size: [bmp.width, bmp.height], format: 'rgba8unorm', usage: TU.TEXTURE_BINDING | TU.COPY_DST | TU.RENDER_ATTACHMENT });
        dev.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, [bmp.width, bmp.height]);
      } catch (e) { this.log('board texture not loaded:', e?.message || e); }
    }
    if (!tex) {
      tex = dev.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: TU.TEXTURE_BINDING | TU.COPY_DST });
      dev.queue.writeTexture({ texture: tex }, new Uint8Array([255, 255, 255, 255]), { bytesPerRow: 4 }, [1, 1]);
    }
    this.boardTex = tex;
  }

  // ---------------------------------------------------------------- pipelines
  _bgl(entries) {
    return this.device.createBindGroupLayout({ entries: entries.map(([binding, vis, kind]) => {
      const e = { binding, visibility: vis };
      if (kind === 'u') e.buffer = { type: 'uniform' };
      else if (kind === 'r') e.buffer = { type: 'read-only-storage' };
      else if (kind === 'w') e.buffer = { type: 'storage' };
      else if (kind === 'tcube') e.texture = { viewDimension: 'cube' };
      else if (kind === 't2') e.texture = {};
      else if (kind === 't2u') e.texture = { sampleType: 'unfilterable-float' };
      else if (kind === 't2a') e.texture = { viewDimension: '2d-array', sampleType: 'unfilterable-float' };
      else if (kind === 's') e.sampler = {};
      else if (kind.startsWith('st:')) { const [, fmt, dim] = kind.split(':'); e.storageTexture = { access: 'write-only', format: fmt, viewDimension: dim || '2d' }; }
      return e;
    }) });
  }

  _pipelines() {
    const dev = this.device, C = GPUShaderStage.COMPUTE, VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const mod = (code, label) => dev.createShaderModule({ code, label });
    // radiosity
    const g0 = this._bgl([[0, C, 'u'], [1, C, 'u'], [2, C, 'r'], [3, C, 'r'], [4, C, 'r'], [5, C, 'r'], [6, C, 'u']]);
    const rad1 = this._bgl([[0, C, 'w']]);
    const gat1 = this._bgl([[0, C, 'r'], [1, C, 'w']]);
    const gat2 = this._bgl([[0, C, 'r'], [1, C, 'w'], [2, C, 'w'], [3, C, 'r']]);
    const mR = mod(RADIOSITY, 'lite.radiosity'), mG = mod(GATHER, 'lite.gather');
    const cp = (m, entry, layouts) => dev.createComputePipeline({ layout: dev.createPipelineLayout({ bindGroupLayouts: layouts }), compute: { module: m, entryPoint: entry } });
    this.p = {
      nodeDirect: cp(mR, 'nodeDirect', [g0, rad1]),
      gatherOcc: cp(mG, 'gatherOcc', [g0, gat1]),
      jacobi: cp(mG, 'jacobi', [g0, gat1, gat2]),
    };
    const nb = this.nb, U = this.ub;
    const ent = (pairs) => pairs.map(([binding, r]) => ({ binding, resource: { buffer: r } }));
    this.g = {
      g0: dev.createBindGroup({ layout: g0, entries: ent([[0, U.light], [1, U.caps], [2, this.mats], [3, this.alb], [4, nb.A], [5, nb.N], [6, U.rad]]) }),
      rad1: dev.createBindGroup({ layout: rad1, entries: ent([[0, nb.edir]]) }),
      gat1: dev.createBindGroup({ layout: gat1, entries: ent([[0, nb.gather], [1, nb.occ]]) }),
      jac: [dev.createBindGroup({ layout: gat2, entries: ent([[0, nb.B0], [1, nb.B1], [2, nb.eind], [3, nb.edir]]) }),
        dev.createBindGroup({ layout: gat2, entries: ent([[0, nb.B1], [1, nb.B0], [2, nb.eind], [3, nb.edir]]) })],
    };
    this.bCur = 0;   // which of B0/B1 holds the latest radiance
    // reflection cube
    const envL = this._bgl([[0, VF, 'u'], [1, VF, 'r'], [2, VF, 'r']]);
    const mE = mod(ENV, 'lite.env');
    this.p.env = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [envL] }),
      vertex: { module: mE, entryPoint: 'vsEnv' }, fragment: { module: mE, entryPoint: 'fsEnv', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.g.env = [0, 1].map((b) => this.env.faceU.map((u) => dev.createBindGroup({ layout: envL, entries: ent([[0, u], [1, nb.A], [2, b ? nb.B1 : nb.B0]]) })));
    const mipL = this._bgl([[0, C, 't2a'], [1, C, 'st:rgba16float:2d-array']]);
    this.p.envDown = cp(mod(ENV_MIP, 'lite.envmip'), 'envDown', [mipL]);
    this.g.envMip = [];
    for (let l = 1; l < this.env.mips; l++) {
      this.g.envMip.push(dev.createBindGroup({ layout: mipL, entries: [
        { binding: 0, resource: this.env.tex.createView({ dimension: '2d-array', baseMipLevel: l - 1, mipLevelCount: 1 }) },
        { binding: 1, resource: this.env.tex.createView({ dimension: '2d-array', baseMipLevel: l, mipLevelCount: 1 }) }] }));
    }
    this.env.faceViews = [0, 1, 2, 3, 4, 5].map((f) => this.env.tex.createView({ dimension: '2d', baseArrayLayer: f, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 }));
    // camera
    const camG0 = this._bgl([[0, VF, 'u'], [1, VF, 'u'], [2, VF, 'r'], [3, VF, 'r'], [4, VF, 'u'], [5, VF, 'r'], [6, VF, 'r'], [7, VF, 'r'], [8, VF, 'tcube'], [9, VF, 's'], [10, VF, 't2']]);
    const camG1 = this._bgl([[0, VF, 'u']]);
    this.camG1 = camG1;
    const mC = mod(CAMERA, 'lite.camera');
    const layout = dev.createPipelineLayout({ bindGroupLayouts: [camG0, camG1] });
    const common = {
      layout, fragment: { module: mC, entryPoint: 'fsMain', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' }, multisample: { count: this.opts.msaa },
    };
    this.p.camStatic = dev.createRenderPipeline({ ...common, vertex: { module: mC, entryPoint: 'vsStatic', buffers: [
      { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'uint16x4' }] },
      { arrayStride: 4, attributes: [{ shaderLocation: 1, offset: 0, format: 'snorm8x4' }] },
      { arrayStride: 8, attributes: [{ shaderLocation: 2, offset: 0, format: 'uint16x4' }] },
      { arrayStride: 4, attributes: [{ shaderLocation: 3, offset: 0, format: 'unorm8x4' }] }] } });
    this.p.camDyn = dev.createRenderPipeline({ ...common, vertex: { module: mC, entryPoint: 'vsDyn', buffers: [
      { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'uint16x4' }] },
      { arrayStride: 8, attributes: [{ shaderLocation: 1, offset: 0, format: 'sint16x4' }] },
      { arrayStride: 4, attributes: [{ shaderLocation: 2, offset: 0, format: 'unorm16x2' }] }] } });
    this.g.cam0 = dev.createBindGroup({ layout: camG0, entries: [
      { binding: 0, resource: { buffer: U.light } }, { binding: 1, resource: { buffer: U.caps } }, { binding: 2, resource: { buffer: this.mats } },
      { binding: 3, resource: { buffer: this.alb } }, { binding: 4, resource: { buffer: U.scene } }, { binding: 5, resource: { buffer: this.bodyBuf } },
      { binding: 6, resource: { buffer: this.bodyGroupBuf } }, { binding: 7, resource: { buffer: nb.eind } },
      { binding: 8, resource: this.env.tex.createView({ dimension: 'cube' }) }, { binding: 9, resource: this.sampler },
      { binding: 10, resource: this.boardTex.createView() }] });
    // linear readback, sensor, present
    this.linL = this._bgl([[0, C, 't2u'], [1, C, 'w']]);
    this.p.toF32 = cp(mod(LINEAR, 'lite.linear'), 'toF32', [this.linL]);
    this.sensL = this._bgl([[0, C, 'u'], [1, C, 't2u'], [2, C, 't2u'], [3, C, 'st:rgba16float'], [4, C, 'st:rgba8unorm'], [5, C, 'w'], [6, C, 'w']]);
    const mS = mod(SENSOR, 'lite.sensor');
    for (const e of ['glareDown', 'glareH', 'glareV', 'sensorMain']) this.p[e] = cp(mS, e, [this.sensL]);
    this.presentModule = mod(PRESENT, 'lite.present');
    this.presentL = this._bgl([[0, GPUShaderStage.FRAGMENT, 't2'], [1, GPUShaderStage.FRAGMENT, 's'], [2, GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, 'u']]);
    this.presentPipes = {};
  }

  // ---------------------------------------------------------------- cameras
  _camerasInit() {
    this.cams = {};
    this.scene.cameras.forEach((c, i) => {
      const L = lensParams(c.lens);
      const th = Number.isFinite(c.theta_fold_rad) ? Math.min(c.theta_fold_rad, 1.45) : 1.45;
      this.cams[c.name] = { name: c.name, index: i, kind: 'kb4', meta: c, lens0: c.lens, size: [...c.size], thetaMax: th, T_D_C: c.T_D_C };
    });
    this.cams.SCOPE = { name: 'SCOPE', index: this.scene.cameras.length, kind: 'pinhole', size: [...this.opts.scopeSize] };
    this.cameras = Object.keys(this.cams);
    for (const c of Object.values(this.cams)) this._camTargets(c);
  }

  /** Render size of a view (CAMk: its lens is rescaled; SCOPE: its pinhole is sized for it). */
  setViewSize(name, size) {
    const c = this.cams[name];
    if (!c) throw new Error(`no camera ${name}`);
    if (c.size[0] === size[0] && c.size[1] === size[1]) return;
    c.size = [Math.max(16, Math.round(size[0])), Math.max(16, Math.round(size[1]))];
    for (const t of ['color', 'depth', 'lin', 'out', 'gA', 'gB']) c.t?.[t]?.destroy?.();
    for (const b of ['yuv', 'lin32']) c.t?.[b]?.destroy?.();
    c.t = null;
    this._camTargets(c);
  }

  _camTargets(c) {
    const dev = this.device, [w, h] = c.size, ms = this.opts.msaa;
    const F = 16, gw = Math.ceil(w / F), gh = Math.ceil(h / F);
    const t = {
      color: dev.createTexture({ size: [w, h], format: 'rgba16float', sampleCount: ms, usage: TU.RENDER_ATTACHMENT }),
      depth: dev.createTexture({ size: [w, h], format: 'depth32float', sampleCount: ms, usage: TU.RENDER_ATTACHMENT }),
      lin: dev.createTexture({ size: [w, h], format: 'rgba16float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING | TU.COPY_SRC }),
      out: dev.createTexture({ size: [w, h], format: 'rgba8unorm', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING | TU.COPY_SRC, label: `${c.name}.rgb8` }),
      gA: dev.createTexture({ size: [gw, gh], format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING }),
      gB: dev.createTexture({ size: [gw, gh], format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING }),
      yuv: dev.createBuffer({ size: (w * h + 2 * Math.ceil(w / 2) * Math.ceil(h / 2)) * 4, usage: GPU.STORAGE | GPU.COPY_SRC }),
      stats: dev.createBuffer({ size: 16, usage: GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST }),
      su: dev.createBuffer({ size: 256, usage: GPU.UNIFORM | GPU.COPY_DST }),
      view: dev.createBuffer({ size: 160, usage: GPU.UNIFORM | GPU.COPY_DST }),
      gw, gh, F, statsRead: [],
    };
    t.viewG = dev.createBindGroup({ layout: this.camG1, entries: [{ binding: 0, resource: { buffer: t.view } }] });
    const SG = (a, b, c2) => dev.createBindGroup({ layout: this.sensL, entries: [
      { binding: 0, resource: { buffer: t.su } }, { binding: 1, resource: t.lin.createView() }, { binding: 2, resource: a.createView() },
      { binding: 3, resource: b.createView() }, { binding: 4, resource: t.out.createView() }, { binding: 5, resource: { buffer: t.yuv } },
      { binding: 6, resource: { buffer: t.stats } }] });
    // glareDown: lin -> gA (gIn unused: gB); H: gA -> gB; V: gB -> gA; main reads gA
    t.sg = { down: SG(t.gB, t.gA), h: SG(t.gA, t.gB), v: SG(t.gB, t.gA), main: SG(t.gA, t.gB) };
    c.t = t;
    if (c.kind === 'kb4') c.lens = scaledLens(c.lens0, c.size);
  }

  /** Camera record for overlays and the page: {name, kind, size, lens|pinhole, T_D_C (rows)}. */
  cameraInfo(name) {
    const c = this.cams[name];
    if (c.kind === 'kb4') return { name, kind: 'kb4', size: [...c.size], lens: { K: [[c.lens.fx, c.lens.alpha * c.lens.fx, c.lens.cx], [0, c.lens.fy, c.lens.cy], [0, 0, 1]], D: [...c.lens.k] }, T_D_C: c.T_D_C, thetaMax: c.thetaMax };
    const s = this._scopeCam();
    return s ? { name, kind: 'pinhole', size: [...c.size], K: [[s.fx, 0, s.cx], [0, s.fy, s.cy], [0, 0, 1]], circle: { cx: s.cx, cy: s.cy, r: s.circleR }, T_D_C: s.T_D_C, hfov_deg: s.hfovDeg } : null;
  }

  _scopeCam() {
    const T = this.mid?.scope;
    if (!T) return null;
    return scopeCamera(this.scene, this.physics, T, this.cams.SCOPE.size);
  }

  // ---------------------------------------------------------------- poses
  setPoses(poses0, poses1 = null) {
    this.poses0 = poses0 || { bodies: {} };
    this.poses1 = poses1 || this.poses0;
    const B0 = this.poses0.bodies || {}, B1 = this.poses1.bodies || {};
    const mid = {};
    for (const k of new Set([...Object.keys(B0), ...Object.keys(B1)])) mid[k] = B0[k] && B1[k] ? lerpPose(B0[k], B1[k], 0.5) : (B0[k] || null);
    const key = JSON.stringify(Object.entries(mid).map(([k, T]) => [k, T ? Array.from(T, (x) => Math.round(x * 1e9)) : null]));
    if (key === this.midKey) return;
    this.midKey = key;
    this.mid = mid;
    this.posesVersion++;
  }

  _uploadPoses() {
    const dev = this.device, mid = this.mid || {};
    // bodies
    const bm = new Float32Array(16 * this.bodyKeys.length);
    this.bodyKeys.forEach((k, i) => { const T = mid[k]; if (T) bm.set(Array.from(T), 16 * i); });   // hidden: zero matrix (collapsed)
    dev.queue.writeBuffer(this.bodyBuf, 0, bm);
    // light (posed outlet)
    const T = mid.scope;
    const L = new Float32Array(96);
    if (T) {
      const R = (v) => [T[0] * v[0] + T[4] * v[1] + T[8] * v[2], T[1] * v[0] + T[5] * v[1] + T[9] * v[2], T[2] * v[0] + T[6] * v[1] + T[10] * v[2]];
      const e = this.emit;
      e.points.forEach((pt, m) => { const p = applyT(T, pt.p); L.set([p[0], p[1], p[2], pt.a], 4 * m); });
      L.set([...R(e.normal), e.points.length], 64);
      L.set([...R(e.axisK), e.prof.step], 68);
      L.set([...e.L0, e.prof.count], 72);
      L.set(e.prof.values, 76);
      L.set([...applyT(T, e.centroid), e.radius], 92);
    }
    dev.queue.writeBuffer(this.ub.light, 0, L);
    // capsules
    const caps = posedCapsules(this.scene, this.caps0, mid).slice(0, MAX_CAPS);
    this.capsPosed = caps;
    const cb = new ArrayBuffer(592), cu = new Uint32Array(cb), cf = new Float32Array(cb);
    cu[0] = caps.length;
    caps.forEach((c, i) => {
      cf.set([...c.a, c.r], 4 + 4 * i);
      cf.set([...c.b, c.rhoEff], 4 + 64 + 4 * i);
      // the scope and port C's cannula sit behind the outlet: they never shadow its direct light
      cu[4 + 128 + i] = c.group | (c.group === 3 ? 256 : 0);
    });
    dev.queue.writeBuffer(this.ub.caps, 0, cb);
  }

  /** Bring the indirect light to the current poses: node direct light, gather occlusion, Jacobi steps, the cube. */
  _ensureRadiosity(iters = null) {
    if (this.solvedVersion === this.posesVersion && iters == null) return;
    this._uploadPoses();
    if (!this.mid?.scope) { this.solvedVersion = this.posesVersion; return; }
    const n = iters ?? (this.solves === 0 ? this.opts.itersInitial : this.opts.itersPerFrame);
    const dev = this.device, enc = dev.createCommandEncoder();
    const wg = (x) => Math.ceil(x / 64);
    let pass = enc.beginComputePass({ timestampWrites: this.timer.tw('nodeDirect') });
    pass.setBindGroup(0, this.g.g0);
    pass.setPipeline(this.p.nodeDirect); pass.setBindGroup(1, this.g.rad1); pass.dispatchWorkgroups(wg(this.nodeCount));
    pass.end();
    pass = enc.beginComputePass({ timestampWrites: this.timer.tw('gatherOcc') });
    pass.setBindGroup(0, this.g.g0);
    pass.setPipeline(this.p.gatherOcc); pass.setBindGroup(1, this.g.gat1);
    const words = this.nodeCount * this.K / 4;
    if (wg(words) > 65535) throw new Error('gather table too large for one dispatch');
    pass.dispatchWorkgroups(wg(words));
    pass.end();
    this._submit(enc, 'radiosity.direct');
    // Jacobi steps in submits of <= 16 (a few ms each on the M4)
    for (let done = 0; done < n;) {
      const e2 = dev.createCommandEncoder();
      const k = Math.min(16, n - done);
      const p2 = e2.beginComputePass({ timestampWrites: this.timer.tw(`jacobi x${k}`) });
      p2.setPipeline(this.p.jacobi); p2.setBindGroup(0, this.g.g0); p2.setBindGroup(1, this.g.gat1);
      for (let i = 0; i < k; i++) { p2.setBindGroup(2, this.g.jac[this.bCur]); p2.dispatchWorkgroups(wg(this.nodeCount)); this.bCur ^= 1; }
      p2.end();
      this._submit(e2, 'radiosity.jacobi');
      done += k;
    }
    this._renderEnv();
    this.solvedVersion = this.posesVersion;
    this.solves++;
  }

  _renderEnv() {
    const dev = this.device, enc = dev.createCommandEncoder(), E = this.env;
    const nIdx = this.V.nodes.env.length;
    for (let f = 0; f < 6; f++) {
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: E.faceViews[f], loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0.2 } }],
        depthStencilAttachment: { view: E.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
        timestampWrites: f === 0 ? this.timer.tw('env') : undefined,
      });
      rp.setPipeline(this.p.env); rp.setBindGroup(0, this.g.env[this.bCur][f]);
      rp.setIndexBuffer(this.vb.envIdx, 'uint16'); rp.drawIndexed(nIdx);
      rp.end();
    }
    const cp = enc.beginComputePass();
    cp.setPipeline(this.p.envDown);
    for (let l = 1; l < E.mips; l++) { const s = Math.max(1, E.size >> l); cp.setBindGroup(0, this.g.envMip[l - 1]); cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), 6); }
    cp.end();
    this._submit(enc, 'env');
  }

  /** Judge mode: run the radiosity to convergence for the current poses (warm-started Jacobi). */
  async converge(iters = 64) {
    this._ensureRadiosity();
    this._ensureRadiosity(iters);
    await this.device.queue.onSubmittedWorkDone();
  }

  _submit(enc, label) {
    const after = this.timer.flush(enc, label);
    this.device.queue.submit([enc.finish()]);
    after();
  }

  // ---------------------------------------------------------------- camera pass
  _viewData(c) {
    const a = new Float32Array(40);
    const [w, h] = c.size;
    if (c.kind === 'kb4') {
      const R = c.T_D_C;
      // T_C_D column-major: R^T, -R^T t
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) a[4 * j + i] = R[j][i];
      for (let i = 0; i < 3; i++) a[12 + i] = -(R[0][i] * R[0][3] + R[1][i] * R[1][3] + R[2][i] * R[2][3]);
      a[15] = 1;
      const L = c.lens;
      a.set([L.fx, L.fy, L.cx, L.cy], 16); a.set(L.k, 20); a.set([w, h, this.opts.near, this.opts.far], 24);
      a.set([0, c.thetaMax, L.alpha, 0], 28); a.set([R[0][3], R[1][3], R[2][3], 0], 32); a.set([0, 0, 0, 0], 36);
      return a;
    }
    const s = this._scopeCam();
    if (!s) return null;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) a[4 * j + i] = s.R[j][i];
    for (let i = 0; i < 3; i++) a[12 + i] = -(s.R[0][i] * s.centre[0] + s.R[1][i] * s.centre[1] + s.R[2][i] * s.centre[2]);
    a[15] = 1;
    a.set([s.fx, s.fy, s.cx, s.cy], 16); a.set([0, 0, 0, 0], 20); a.set([w, h, this.opts.near, this.opts.far], 24);
    a.set([1, 3.2, 0, 0], 28); a.set([...s.centre, 0], 32); a.set([s.cx, s.cy, s.circleR, 0], 36);
    return a;
  }

  async renderCamera(name, opts = {}) {
    const c = this.cams[name];
    if (!c) throw new Error(`no camera ${name}`);
    this._ensureRadiosity();
    const vd = this._viewData(c);
    const dev = this.device, t = c.t;
    const enc = dev.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: t.color.createView(), resolveTarget: t.lin.createView(), loadOp: 'clear', storeOp: 'discard', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      depthStencilAttachment: { view: t.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
      timestampWrites: this.timer.tw(`camera ${name}`),
    });
    if (vd) {
      dev.queue.writeBuffer(t.view, 0, vd);
      rp.setBindGroup(0, this.g.cam0); rp.setBindGroup(1, t.viewG);
      rp.setPipeline(this.p.camStatic);
      rp.setVertexBuffer(0, this.vb.sPos); rp.setVertexBuffer(1, this.vb.sNrm); rp.setVertexBuffer(2, this.vb.sNodes); rp.setVertexBuffer(3, this.vb.sW);
      rp.setIndexBuffer(this.vb.sIdx, 'uint16');
      for (const g of this.meta.files.static.groups) {
        if (g.owner && g.owner === name) continue;            // a camera never images its own lens front
        for (const ch of g.chunks) rp.drawIndexed(ch.indexCount, 1, ch.firstIndex, ch.vertexBase);
      }
      rp.setPipeline(this.p.camDyn);
      rp.setVertexBuffer(0, this.vb.dPos); rp.setVertexBuffer(1, this.vb.dNrm); rp.setVertexBuffer(2, this.vb.dUv);
      rp.setIndexBuffer(this.vb.dIdx, 'uint16');
      for (const g of this.meta.files.dynamic.groups) {
        if (!this.mid?.[g.key]) continue;                      // hidden body (e.g. a board not in view)
        if (name === 'SCOPE' && g.key === 'scope') continue;   // the scope does not image itself
        for (const ch of g.chunks) rp.drawIndexed(ch.indexCount, 1, ch.firstIndex, ch.vertexBase);
      }
    }
    rp.end();
    this._submit(enc, `camera ${name}`);
    this.frame++; this.timer.frame = this.frame;
    c.rendered = true;
    if (opts.await !== false) await dev.queue.onSubmittedWorkDone();
    return t.lin;
  }

  /** A whole frame for several views, no waiting (the live loop): {cameras, sensor, sensorOpts}. */
  renderFrame(opts = {}) {
    const names = opts.cameras || this.cameras;
    const out = {};
    for (const n of names) { this.renderCamera(n, { await: false }); out[n] = { linear: this.cams[n].t.lin }; }
    if (opts.sensor) for (const n of names) out[n].rgb8 = this.sensor(n, opts.sensorOpts || {});
    out.done = this.device.queue.onSubmittedWorkDone();
    return out;
  }

  // ---------------------------------------------------------------- sensor
  exposureOf(name) {
    const c = this.cams[name];
    if (c.manual) return { ...c.manual, auto: false };
    return { time_s: c.ae.exposure, gain: c.ae.gain, gainUnits: c.ae.gainUnits, units: c.ae.units, auto: true };
  }

  setExposure(name, e) {
    const c = this.cams[name];
    c.manual = e == null || e === 'auto' ? null : typeof e === 'number' ? { time_s: e, gain: 1 } : { time_s: e.time_s ?? e.exposure_s, gain: e.gain ?? 1 };
  }

  _sensorUniform(c, o) {
    const p = { ...this.sensorP, ...(o.params || {}) };
    const ex = this.exposureOf(c.name);
    const [w, h] = c.size;
    const buf = new ArrayBuffer(256), f = new Float32Array(buf), u = new Uint32Array(buf);
    const kb4 = c.kind === 'kb4';
    const noise = o.noise === true || (o.noise !== false && this.opts.noise === true);
    const glare = kb4 && (p.glareFraction || 0) > 0;
    const yuv = String(p.chroma).startsWith('4:2:0');
    u[0] = w; u[1] = h; u[2] = (o.frame ?? this.frame) >>> 0; u[3] = (kb4 ? 1 : 0) | (noise ? 2 : 0) | (glare ? 4 : 0) | (yuv ? 8 : 0);
    if (kb4) { const L = c.lens; f.set([L.fx, L.fy, L.cx, L.cy], 4); f.set(L.k, 8); f.set([L.alpha, p.pupilExponent, 0, 0], 12); }
    f.set([Math.PI * p.transmission / (4 * p.fNumber * p.fNumber), o.exposure_s ?? ex.time_s, o.gain ?? ex.gain, glare ? p.glareFraction : 0], 16);
    for (let r = 0; r < 3; r++) f.set(p.responseMatrix[r], 20 + 4 * r);
    f.set([p.readNoise, p.fullWell, p.darkCurrent, p.conversionGain], 32);
    f.set([Math.pow(2, p.adcBits) - 1, p.blackLevel, 0, 0], 36); u[38] = (o.seed ?? 0) >>> 0;
    const wb = p.wb.map((g, k) => g * (p.awb && c.awb && !(o.params && o.params.wb) ? c.awb[k] : 1));
    f.set([...wb, 0], 40);
    for (let r = 0; r < 3; r++) f.set(p.ccm[r], 44 + 4 * r);
    f.set([p.tone === 'srgb' ? (p.sCurveC ? 3 : 0) : p.tone === 'linear' ? 2 : 1, p.gamma, p.sCurveC || 1, p.blackOffset || 0], 56);
    f.set([c.t.F, (p.glareSigmaFrac * Math.hypot(w, h)) / c.t.F, c.t.gw, c.t.gh], 60);
    return buf;
  }

  /** Radiance -> 8-bit camera RGB (light chain; docs/WEB_TWIN.md §18.1). opts: {seed, frame, noise, exposure_s, gain, params}. */
  sensor(name, opts = {}) {
    const c = this.cams[name], t = c.t, dev = this.device;
    dev.queue.writeBuffer(t.su, 0, this._sensorUniform(c, opts));
    dev.queue.writeBuffer(t.stats, 0, new Uint32Array(4));
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass({ timestampWrites: this.timer.tw(`sensor ${name}`) });
    const [w, h] = c.size;
    if (c.kind === 'kb4' && (this.sensorP.glareFraction || 0) > 0) {
      pass.setPipeline(this.p.glareDown); pass.setBindGroup(0, t.sg.down); pass.dispatchWorkgroups(Math.ceil(t.gw / 8), Math.ceil(t.gh / 8));
      pass.setPipeline(this.p.glareH); pass.setBindGroup(0, t.sg.h); pass.dispatchWorkgroups(Math.ceil(t.gw / 8), Math.ceil(t.gh / 8));
      pass.setPipeline(this.p.glareV); pass.setBindGroup(0, t.sg.v); pass.dispatchWorkgroups(Math.ceil(t.gw / 8), Math.ceil(t.gh / 8));
    }
    pass.setPipeline(this.p.sensorMain); pass.setBindGroup(0, t.sg.main); pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
    pass.end();
    const rd = t.statsRead.pop() || dev.createBuffer({ size: 16, usage: GPU.COPY_DST | GPU.MAP_READ });
    enc.copyBufferToBuffer(t.stats, 0, rd, 0, 16);
    this._submit(enc, `sensor ${name}`);
    rd.mapAsync(GPUMapMode.READ).then(() => {
      const st = new Uint32Array(rd.getMappedRange().slice(0));
      rd.unmap(); t.statsRead.push(rd);
      c.meanLuma = st[0] / (w * h * 255);
      c.meanRGB = [st[1], st[2], st[3]].map((x) => x / (w * h * 1023));
      if (!c.manual && opts.exposure_s == null) c.ae.update(c.meanLuma);
      if (this.sensorP.awb) c.awb = updateAWB(c.awb, c.meanRGB, this.sensorP.awbDamping);
    }).catch(() => {});
    return t.out;
  }

  // ---------------------------------------------------------------- present
  /**
   * Draw a view's sensor texture onto a WebGPU canvas context, letterboxed ('contain', default), cropped ('cover')
   * or stretched ('fill'). opts: {fit, rect: [x, y, w, h] (canvas px; default the whole canvas), clear (true: clear
   * the canvas first), background: [r, g, b] (0..1)}. The context is configured for this device on first use
   * (format navigator.gpu.getPreferredCanvasFormat(), alphaMode opaque) unless the caller already did.
   */
  present(name, context, opts = {}) {
    const c = this.cams[name], dev = this.device;
    if (!c) throw new Error(`no camera ${name}`);
    if (!context.__liteConfigured) {
      context.configure({ device: dev, format: opts.format || navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' });
      context.__liteConfigured = true;
    }
    const tex = context.getCurrentTexture();
    const fmt = tex.format;
    if (!this.presentPipes[fmt]) {
      this.presentPipes[fmt] = dev.createRenderPipeline({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.presentL] }),
        vertex: { module: this.presentModule, entryPoint: 'vsP' }, fragment: { module: this.presentModule, entryPoint: 'fsP', targets: [{ format: fmt }] },
        primitive: { topology: 'triangle-list' },
      });
    }
    const [W, H] = [tex.width, tex.height];
    const R = opts.rect || [0, 0, W, H];
    const [iw, ih] = c.size;
    let [x, y, w, h] = R;
    const fit = opts.fit || 'contain';
    if (fit !== 'fill') {
      const s = fit === 'cover' ? Math.max(R[2] / iw, R[3] / ih) : Math.min(R[2] / iw, R[3] / ih);
      w = iw * s; h = ih * s; x = R[0] + (R[2] - w) / 2; y = R[1] + (R[3] - h) / 2;
    }
    c.presentU ??= dev.createBuffer({ size: 16, usage: GPU.UNIFORM | GPU.COPY_DST });
    const bg = dev.createBindGroup({ layout: this.presentL, entries: [{ binding: 0, resource: c.t.out.createView() }, { binding: 1, resource: this.sampler }, { binding: 2, resource: { buffer: c.presentU } }] });
    const b = opts.background || [0.03, 0.03, 0.035];
    const enc = dev.createCommandEncoder();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: opts.clear === false ? 'load' : 'clear', storeOp: 'store', clearValue: { r: b[0], g: b[1], b: b[2], a: 1 } }] });
    // clip to the destination rect (cover crops), draw the image into its fitted rect
    const cx0 = Math.max(0, Math.floor(Math.max(x, R[0]))), cy0 = Math.max(0, Math.floor(Math.max(y, R[1])));
    const cx1 = Math.min(W, Math.ceil(Math.min(x + w, R[0] + R[2]))), cy1 = Math.min(H, Math.ceil(Math.min(y + h, R[1] + R[3])));
    if (cx1 > cx0 && cy1 > cy0) {
      rp.setViewport(x, y, w, h, 0, 1);
      rp.setScissorRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
      rp.setPipeline(this.presentPipes[fmt]); rp.setBindGroup(0, bg); rp.draw(3);
    }
    rp.end();
    dev.queue.submit([enc.finish()]);
    return { rect: [x, y, w, h] };
  }

  // ---------------------------------------------------------------- readback
  async readback(name, kind = 'rgb8') {
    const c = this.cams[name], dev = this.device, [w, h] = c.size, t = c.t;
    if (kind === 'yuv420') {
      const size = t.yuv.size;
      const st = dev.createBuffer({ size, usage: GPU.COPY_DST | GPU.MAP_READ });
      const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(t.yuv, 0, st, 0, size); dev.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const u = new Uint32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
      const cw = Math.ceil(w / 2), ch = Math.ceil(h / 2);
      return { width: w, height: h, y: Uint8Array.from(u.subarray(0, w * h)), cb: Uint8Array.from(u.subarray(w * h, w * h + cw * ch)), cr: Uint8Array.from(u.subarray(w * h + cw * ch, w * h + 2 * cw * ch)) };
    }
    if (kind === 'linear') {
      t.lin32 ??= dev.createBuffer({ size: w * h * 16, usage: GPU.STORAGE | GPU.COPY_SRC });
      const bg = dev.createBindGroup({ layout: this.linL, entries: [{ binding: 0, resource: t.lin.createView() }, { binding: 1, resource: { buffer: t.lin32 } }] });
      const st = dev.createBuffer({ size: w * h * 16, usage: GPU.COPY_DST | GPU.MAP_READ });
      const enc = dev.createCommandEncoder();
      const p = enc.beginComputePass(); p.setPipeline(this.p.toF32); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); p.end();
      enc.copyBufferToBuffer(t.lin32, 0, st, 0, w * h * 16);
      dev.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
      return out;
    }
    const bpr = Math.ceil((w * 4) / 256) * 256;
    const st = dev.createBuffer({ size: bpr * h, usage: GPU.COPY_DST | GPU.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: t.out }, { buffer: st, bytesPerRow: bpr }, [w, h]);
    dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
    if (kind === 'rgba8') { const out = new Uint8Array(w * h * 4); for (let y = 0; y < h; y++) out.set(data.subarray(y * bpr, y * bpr + w * 4), y * w * 4); return out; }
    const out = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0, i = y * bpr, j = 3 * y * w; x < w; x++, i += 4, j += 3) { out[j] = data[i]; out[j + 1] = data[i + 1]; out[j + 2] = data[i + 2]; }
    return out;
  }

  // ---------------------------------------------------------------- occlusion queries (CPU BVH, built in a worker)
  _startOcclusion() {
    const sc = this.scene, ownerIdx = Object.fromEntries(sc.cameras.map((c, i) => [c.name, i + 1]));
    let nv = 0, nt = 0;
    const st = sc.meshes.filter((m) => !m.dynamic && !m.overview_only && this.meshes[m.name]);
    for (const m of st) { nv += this.meshes[m.name].P.length / 3; nt += this.meshes[m.name].I.length / 3; }
    const P = new Float32Array(3 * (nv + 4)), I = new Uint32Array(3 * (nt + 2)), owner = new Uint8Array(nt + 2);
    let vo = 0, to = 0;
    for (const m of st) {
      const d = this.meshes[m.name];
      P.set(d.P, 3 * vo);
      for (let k = 0; k < d.I.length; k++) I[3 * to + k] = d.I[k] + vo;
      owner.fill(ownerIdx[m.owner_camera] || 0, to, to + d.I.length / 3);
      vo += d.P.length / 3; to += d.I.length / 3;
    }
    const b = sc.bench, z = b?.z ?? 0, ex = b?.extent ?? [[-0.5, -0.5], [0.5, 0.5]];
    P.set([ex[0][0], ex[0][1], z, ex[1][0], ex[0][1], z, ex[1][0], ex[1][1], z, ex[0][0], ex[1][1], z], 3 * vo);
    I.set([vo, vo + 1, vo + 2, vo, vo + 2, vo + 3], 3 * to);
    const bodies = {};
    const bodyOf = {};
    for (const ins of sc.instruments || []) for (const p of ins.parts || []) bodyOf[p.mesh] = `${ins.id}/${p.body}`;
    for (const m of sc.scope?.mesh_parts || []) bodyOf[m] = 'scope';
    if (sc.scope?.emitter_mesh) bodyOf[sc.scope.emitter_mesh] = 'scope';
    for (const bd of sc.boards || []) bodyOf[bd.mesh] = `board/${bd.name}`;
    for (const m of sc.meshes) if (m.dynamic && !bodyOf[m.name] && m.body) bodyOf[m.name] = m.body;
    for (const m of sc.meshes) {
      const k = bodyOf[m.name];
      if (!m.dynamic || !k || !this.meshes[m.name]) continue;
      (bodies[k] ??= []).push(this.meshes[m.name]);
    }
    const blist = Object.entries(bodies).map(([key, list]) => {
      let a = 0, c = 0; for (const d of list) { a += d.P.length; c += d.I.length; }
      const BP = new Float32Array(a), BI = new Uint32Array(c);
      let v = 0, t = 0;
      for (const d of list) { BP.set(d.P, v); for (let k = 0; k < d.I.length; k++) BI[t + k] = d.I[k] + v / 3; v += d.P.length; t += d.I.length; }
      return { key, P: BP, I: BI };
    });
    this.occ = null;
    this.occReady = new Promise((resolve) => {
      const finish = (res) => { this.occ = res; this.stats.occlusionBuildMs = res.ms; resolve(true); };
      try {
        const wk = new Worker(new URL('./occ_worker.js', import.meta.url), { type: 'module' });
        wk.onmessage = (e) => { finish(e.data); wk.terminate(); };
        wk.onerror = (e) => { this.log('occlusion worker failed, building here:', e.message); import('./occ_build.js').then((m) => finish(m.build({ static: { P, I, owner }, bodies: blist }).out)); };
        wk.postMessage({ static: { P, I, owner }, bodies: blist });
      } catch (e) {
        import('./occ_build.js').then((m) => finish(m.build({ static: { P, I, owner }, bodies: blist }).out));
      }
    });
  }

  /** Occlusion from the camera centre to each D point (static scene + posed bodies at mid-exposure). A point is
   * visible when nothing lies between the centre and `tol` short of it. Synchronous (create() waits for the BVHs). */
  visible(name, pointsD, { tol = 2e-4 } = {}) {
    if (!this.occ) return this.occReady.then(() => this.visible(name, pointsD, { tol }));
    const c = this.cams[name];
    let o, skipOwner = 0, skipBody = null;
    if (c.kind === 'kb4') { const T = c.T_D_C; o = [T[0][3], T[1][3], T[2][3]]; skipOwner = c.index + 1; }
    else { const s = this._scopeCam(); if (!s) return pointsD.map(() => true); o = s.centre; skipBody = 'scope'; }
    const S = this.occ.static;
    const mid = this.mid || {};
    return pointsD.map((p) => {
      const v = [p[0] - o[0], p[1] - o[1], p[2] - o[2]], dist = Math.hypot(...v);
      if (!(dist > tol)) return true;
      const d = v.map((x) => x / dist), tmax = dist - tol;
      if (traverse(S.nodes, S.T, o, d, { tmax, anyHit: true, filter: skipOwner ? (k) => S.owner[k] !== skipOwner : null }).tri >= 0) return false;
      for (const b of this.occ.bodies) {
        const M = mid[b.key];
        if (!M || b.key === skipBody) continue;
        const ob = [o[0] - M[12], o[1] - M[13], o[2] - M[14]];
        const oL = [M[0] * ob[0] + M[1] * ob[1] + M[2] * ob[2], M[4] * ob[0] + M[5] * ob[1] + M[6] * ob[2], M[8] * ob[0] + M[9] * ob[1] + M[10] * ob[2]];
        const dL = [M[0] * d[0] + M[1] * d[1] + M[2] * d[2], M[4] * d[0] + M[5] * d[1] + M[6] * d[2], M[8] * d[0] + M[9] * d[1] + M[10] * d[2]];
        if (traverse(b.nodes, b.T, oL, dL, { tmax, anyHit: true }).tri >= 0) return false;
      }
      return true;
    });
  }

  /** Pixel of each D point in a view: KB4 (float64, gpu/lens.js) or the scope pinhole; null when behind. */
  project(name, pointsD) {
    const c = this.cams[name];
    if (c.kind === 'kb4') {
      const T = c.T_D_C, o = [T[0][3], T[1][3], T[2][3]];
      return pointsD.map((p) => { const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]]; const X = [0, 1, 2].map((j) => T[0][j] * d[0] + T[1][j] * d[1] + T[2][j] * d[2]); return Math.atan2(Math.hypot(X[0], X[1]), X[2]) < c.thetaMax ? kb4Project(c.lens, X) : null; });
    }
    const s = this._scopeCam();
    return pointsD.map((p) => { if (!s) return null; const d = [p[0] - s.centre[0], p[1] - s.centre[1], p[2] - s.centre[2]]; const X = [0, 1, 2].map((j) => s.R[0][j] * d[0] + s.R[1][j] * d[1] + s.R[2][j] * d[2]); return X[2] > 0 ? [s.fx * X[0] / X[2] + s.cx, s.fy * X[1] / X[2] + s.cy] : null; });
  }

  // ---------------------------------------------------------------- timings
  timings() {
    const rec = this.timer.records.slice(-256);
    const per = {};
    for (const r of rec) for (const [k, ms] of Object.entries(r.passes)) (per[k] ??= []).push(ms);
    const passes = {};
    for (const [k, v] of Object.entries(per)) { const s = [...v].sort((a, b) => a - b); passes[k] = { median: s[s.length >> 1], p95: s[Math.min(s.length - 1, Math.floor(0.95 * s.length))], last: v[v.length - 1], n: v.length }; }
    return { gpu: this.timer.on, source: this.timer.on ? 'timestamp-query' : 'none', passes, records: rec.slice(-64), maxSubmit: { ...this.timer.maxSubmit }, over: this.timer.over.slice(-32), submits: this.timer.submits, stats: this.stats };
  }

  async idle() { await this.device.queue.onSubmittedWorkDone(); await new Promise((r) => setTimeout(r, 10)); }

  destroy() {
    for (const b of [...Object.values(this.vb), ...Object.values(this.nb), ...Object.values(this.ub)]) b?.destroy?.();
    for (const c of Object.values(this.cams)) for (const v of Object.values(c.t || {})) v?.destroy?.();
  }
}
