// godview_pt2 = godview_pt plus (2026-11-01): (1) a back-side vertex bake for the cut material, so the hollow PLA skin seen
// through the cut is lit as the two-sided path tracer lights it (raster dome_cut |rel| 1.7-3.1 -> 0.09-0.19); (2) the cut by
// clip distances, not discard (tile GPUs keep hidden-surface removal; the shadow passes are depth-only); (3) static shadow
// casters cached and rebuilt in 4 slices over 4 frames when the cut turns > 1 deg or the scope moves; (4) bake chunks and
// trace bands start small (4096 vertices, 16 rows) and grow at most x2 per measurement (no still-onset spike). ?gv_msaa=1|4.
// GodView (candidate godview_pt): "raster while moving, path-traced when still" god's-eye view of the whole rig
// (docs/WEB_TWIN.md §19, configs/twin/godview.yaml). Raw WebGPU on the twin's own scene packing and WGSL.
//
//   const g = await GodView.create({ canvas, scene, physics, godview, device? });   // scene: loadScene bundle or URL
//   g.setPoses(poses); g.setCamera({ eye, target, up, vfov_deg }); g.setCutaway(on);
//   g.render();                              // raster while anything changed in the last 150 ms, else one more
//                                            // path-traced sample (tiled submits), crossfaded over the raster
//   await g.captureLinear(w, h, { spp, mode: 'traced' | 'raster' })   // linear RGB Float32Array, top row first
//   g.timings(); g.destroy();
//
// Light (godview.yaml lights): constant environment, a directional key, the physics.yaml scope emitter times gain.
// The traced image is unbiased path tracing (BSDF sampling + NEE of the key and of the scope outlet, power-heuristic
// MIS for the outlet, Russian roulette after 3 bounces), so it converges to the Mitsuba reference of the same file.
// Machine safety (§12): every trace submit is a band of rows sized from the measured cost to <= ~8 ms, one band in
// flight at a time, and every WGSL loop is capped (gpu/wgsl/trace.wgsl.js MAX_VISITS, MAX_BOUNCE_CAP).

import { loadScene, prepareScene, ALBEDO_BINS } from '../gpu/scene_gpu.js';
import { MAX_INST } from '../gpu/wgsl/common.wgsl.js';
import { ptSource, rasterSource, displaySource, SMOOTH_WGSL } from './shaders.js';
import { loadPrepared } from './cache.js';

const SCENE_U_BYTES = 96 + MAX_INST * 192;
const GU_BYTES = 16 * 13;
const RU_BYTES = 64 * 3 + 32;
const MSAA_DEFAULT = 4;
const URLQ = (() => { try { return new URLSearchParams(globalThis.location?.search || ''); } catch { return new URLSearchParams(''); } })();   // bench knobs: gv_msaa
const SHADOW_SIZE = 2048, SCOPE_SHADOW_SIZE = 1024;
const STILL_MS = 150;
const SHADOW_SLICES = 4, SHADOW_CUT_DEG = 1.0;

const unwrap = (x) => (x && typeof x === 'object' && !Array.isArray(x) && 'value' in x ? x.value : x);
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const mulRv = (R, v) => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
const DEG = Math.PI / 180;

const quatFromMat = (m) => {
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
const rotOf = (T) => [[T[0], T[4], T[8]], [T[1], T[5], T[9]], [T[2], T[6], T[10]]];

/** eye of a godview.yaml viewpoint: centre + d (cos el cos az, cos el sin az, sin el) */
export function viewpointEye(godview, name) {
  const v = godview.viewpoints[name];
  const az = v.az_deg * DEG, el = v.el_deg * DEG, c = godview.centre;
  return [c[0] + v.d * Math.cos(el) * Math.cos(az), c[1] + v.d * Math.cos(el) * Math.sin(az), c[2] + v.d * Math.sin(el)];
}

// column-major 4x4 helpers (WGSL mat4x4f layout)
function lookAtProj(eye, target, up, vfovDeg, aspect, near, far) {
  const z = norm3(sub3(target, eye));                 // forward
  const x = norm3(cross3(z, up));                      // right
  const y = cross3(x, z);                              // up (screen)
  const f = 1 / Math.tan((vfovDeg * DEG) / 2);
  // view: camera looks down -Z (GL style); depth to [0, 1]
  const V = [x[0], y[0], -z[0], 0, x[1], y[1], -z[1], 0, x[2], y[2], -z[2], 0, -dot3(x, eye), -dot3(y, eye), dot3(z, eye), 1];
  const P = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
  return mul4(P, V);
}
function orthoLight(dirToLight, centre, halfW, depth) {
  const z = norm3(dirToLight.map((v) => -v));         // the light looks along -dirToLight
  const up = Math.abs(z[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  const x = norm3(cross3(z, up)), y = cross3(x, z);
  const eye = centre.map((c, k) => c - z[k] * depth * 0.5);
  const V = [x[0], y[0], -z[0], 0, x[1], y[1], -z[1], 0, x[2], y[2], -z[2], 0, -dot3(x, eye), -dot3(y, eye), dot3(z, eye), 1];
  const P = [1 / halfW, 0, 0, 0, 0, 1 / halfW, 0, 0, 0, 0, -1 / depth, 0, 0, 0, 0, 1];
  return mul4(P, V);
}
function mul4(A, B) {
  const C = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += A[k * 4 + r] * B[c * 4 + k]; C[c * 4 + r] = s; }
  return C;
}

export class GodView {
  static async create(opts) { const g = new GodView(); await g._init(opts); return g; }

  async _init({ canvas = null, scene, physics, godview, device = null, log = null, options = {} }) {
    this.log = log || ((...a) => console.log('[godview_pt2]', ...a));
    this.canvas = canvas;
    this.godview = godview;
    this.opts = { maxBounces: 16, tileMs: 6, maxTilesPerFrame: 4, denoiseUntil: 128, fadeFrom: 8, fadeTo: 96, bakeSpacing: 0.008, bakeRays: 64, ...options };
    const t0 = performance.now();
    // scene: a loadScene bundle {scene, meshes, base}, its URL, or a parsed scene.json (meshes then load from
    // options.base, default the twin's assets next to this module)
    let src = scene;
    if (typeof src === 'string') src = await loadScene(src);
    else if (src && Array.isArray(src.meshes)) src = await loadScene(options.base || new URL('../assets/', import.meta.url).href);
    this.scene = src.scene;
    // the god's-eye bench is bigger than the camera sims' (godview.yaml bench.extent); the static mesh is split to
    // <= bakeSpacing so the raster's baked light has a vertex every 8 mm. A prepared-scene cache skips ~4 s of CPU.
    const extent = godview.bench?.extent || src.scene.bench?.extent;
    const sc = { ...src.scene, bench: { ...(src.scene.bench || {}), extent } };
    let prepared = options.prepared || null;
    if (!prepared && options.cache !== false) prepared = await loadPrepared(new URL('./cache/', import.meta.url).href, src.scene, physics, extent, this.opts.bakeSpacing);
    this.S = prepared || prepareScene(sc, src.meshes, physics, { nodeSpacing: this.opts.bakeSpacing, leafSize: 4, log: () => {} });
    this.cached = !!prepared;
    this.prepareMs = performance.now() - t0;
    if (!device) {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('no WebGPU adapter');
      this.adapterInfo = adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description } : {};
      const features = ['timestamp-query', 'clip-distances'].filter((f) => adapter.features.has(f));
      const lim = adapter.limits;
      device = await adapter.requestDevice({
        requiredFeatures: features,
        requiredLimits: {
          maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize,
          maxStorageBuffersPerShaderStage: Math.min(10, lim.maxStorageBuffersPerShaderStage),
          maxComputeWorkgroupStorageSize: Math.min(32768, lim.maxComputeWorkgroupStorageSize),
        },
      });
    }
    this.device = device;
    device.lost?.then((i) => { this.lostInfo = i.message; this.log('device lost', i.message); });
    device.addEventListener?.('uncapturederror', (e) => { this.lastError = e.error?.message; this.log('GPU error', e.error?.message); });
    this.hasTS = device.features.has('timestamp-query');
    this.rscale = Math.min(1, Math.max(0.25, Number(options.rasterScale ?? URLQ.get('gv_scale') ?? 1)));   // live raster resolution scale (captures: 1)
    this.msaa = Number(options.msaa ?? URLQ.get('gv_msaa') ?? MSAA_DEFAULT) === 1 ? 1 : 4;
    this.clip = device.features.has('clip-distances') && !options.noClip;   // godview_pt2: cut by clip distance, no discard

    if (this.hasTS) {
      this.tsSet = device.createQuerySet({ type: 'timestamp', count: 4 });
      this.tsResolve = device.createBuffer({ size: 512, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });   // resolve offsets: multiples of 256
      this.tsRead = [0, 1].map(() => device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
      this.tsBusy = [false, false];
    }
    this.gpuRaster = []; this.gpuBand = []; this.gpuSample = 0;
    this._lights(physics);
    this._buffers();
    await this._pipelines();
    this.cam = null; this.cut = godview.cutaway?.default !== 'off' && godview.cutaway?.default !== false;
    this.size = null; this.sample = 0; this.tileRow = 0; this.changedAt = 0; this.poseVer = 0;
    this.rowsPerTile = 16; this.msPerRowPx = null; this.inFlight = 0;
    this.gpuMs = []; this.rasterMs = []; this.sampleMs = []; this.frames = 0;
    this.instances = this.S.blas.map(() => null);
    this.setPoses({ bodies: {} });
    const c = godview.centre, v = godview.viewpoints?.V1_surgeon ? 'V1_surgeon' : Object.keys(godview.viewpoints || {})[0];
    this.setCamera({ eye: v ? viewpointEye(godview, v) : [c[0] - 0.6, c[1] - 0.3, c[2] + 0.5], target: c, up: [0, 0, 1], vfov_deg: godview.vfov_deg || 35 });
    this.loadMs = performance.now() - t0;
    this.log(`ready: ${this.S.triCount ?? '?'} triangles, prepare ${this.prepareMs.toFixed(0)} ms, load ${this.loadMs.toFixed(0)} ms`);
  }

  _lights(physics) {
    const L = this.godview.lights || {};
    const env = L.environment?.radiance || [1, 1, 1];
    const key = L.key || {};
    const az = (key.from_az_deg ?? 150) * DEG, el = (key.from_el_deg ?? 65) * DEG;
    this.light = {
      env: env.map(Number), envOn: L.environment ? 1 : 0,
      keyDir: [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)], keyE: (key.irradiance || [2, 2, 2]).map(Number), keyOn: L.key ? 1 : 0,
      gain: Number(L.scope?.gain ?? 1), scopeOn: L.scope ? 1 : 0,
    };
    const ex = this.godview.display?.exposure;
    this.exposure = ex == null ? 0.35 : Number(ex);
    // materials removed by the cutaway (bit per material index)
    let mask = 0;
    for (const name of this.godview.cutaway?.materials || ['pla_matte_white']) {
      const i = this.S.materials.findIndex((m) => m.name === name);
      if (i >= 0 && i < 32) mask |= 1 << i;
    }
    this.cutMask = mask >>> 0;
    this.cutOffset = Number(this.godview.cutaway?.offset ?? 0.02);
  }

  _buf(data, usage, label) {
    const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    const b = this.device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST, label });
    this.device.queue.writeBuffer(b, 0, data.buffer ? data.buffer : data, data.byteOffset || 0, data.byteLength);
    return b;
  }

  _buffers() {
    // params layout identical to gpu/renderer.js _buildBuffers (mesh table | materials | albedo | profile | emitter)
    const S = this.S, ST = GPUBufferUsage.STORAGE;
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
    em.tris.forEach((t, k) => { P.set([...t.v0, t.cdf, ...t.e1, 0, ...t.e2, 0, ...t.n, t.area], (emitOff + 4 * k) * 4); });
    // godview_pt2: back-side bake entries for every static vertex of a cut material (the hollow skin's inside)
    const nV = S.staticVerts, TU = new Uint32Array(S.tris.buffer, S.tris.byteOffset, S.tris.length);
    const backSlot = new Uint32Array(Math.max(1, nV)).fill(0xffffffff), backList = [];
    for (let k = 0; k < S.staticTris; k++) {
      const mesh = TU[16 * k + 12], mat = S.meshTable[mesh]?.mat;
      if (mat == null || mat >= 32 || !((this.cutMask >>> mat) & 1)) continue;
      for (let j = 0; j < 3; j++) { const vi = TU[16 * k + 4 * j + 3]; if (vi < nV && backSlot[vi] === 0xffffffff) { backSlot[vi] = backList.length; backList.push(vi); } }
    }
    this.nBack = backList.length;
    this.bakeTotal = nV + this.nBack;
    this.consts = { nMesh, nMat, meshOff, matOff, albOff, albBins: ALBEDO_BINS, profOff, profN, emitOff, emitN, nStaticVerts: S.staticVerts, gatherK: 1, nodeSamples: 1, nbrK: 1 };
    this.triCount = S.tris.length / 16;
    this.bufs = {
      nodes: this._buf(S.nodes, ST, 'nodes'), tris: this._buf(S.tris, ST, 'tris'), verts: this._buf(S.verts, ST, 'verts'), params: this._buf(P, ST, 'params'),
      dbg: this.device.createBuffer({ size: 64, usage: ST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
      scn: this.device.createBuffer({ size: SCENE_U_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      gu: this.device.createBuffer({ size: GU_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      ru: this.device.createBuffer({ size: RU_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      irr: this.device.createBuffer({ size: Math.max(16, this.bakeTotal * 16), usage: ST | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label: 'irr (raster)' }),
      irrAcc: this.device.createBuffer({ size: Math.max(16, this.bakeTotal * 16), usage: ST | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label: 'irr (bake sums)' }),
      irrTmp: this.device.createBuffer({ size: Math.max(16, this.bakeTotal * 16), usage: ST, label: 'irr (smoothing)' }),
      backList: this._buf(new Uint32Array(backList.length ? backList : [0]), ST, 'backList'),
      backSlot: this._buf(backSlot, ST, 'backSlot'),
      adjStart: this._buf(S.adjacency.start, ST, 'adjStart'),
      adjList: this._buf(S.adjacency.list.length ? S.adjacency.list : new Uint32Array(4), ST, 'adjList'),
      smoothU: this._buf(new Uint32Array([S.staticVerts, this.nBack, 0, 0]), GPUBufferUsage.UNIFORM, 'smoothU'),
    };
    // one small uniform per draw (instance, triangle offset, pass kind)
    this.draws = [{ inst: 0xffffffff, off: 0, count: S.staticTris }];
    S.blas.forEach((b, i) => this.draws.push({ inst: i, off: b.triOff, count: b.triCount }));
    this.drawBufs = this.draws.map((d) => [0, 1].map((kind) => this._buf(new Uint32Array([d.inst, d.off, kind, 0]), GPUBufferUsage.UNIFORM, 'draw')));
    // godview_pt2: the static casters of both shadow maps are cached and rebuilt in SLICES draws over as many frames
    // (double-buffered) when the cut plane turns or the scope moves; only the instances are drawn every frame
    const NS = SHADOW_SLICES, q = Math.ceil(S.staticTris / NS);
    this.slices = [];
    for (let i = 0; i < NS; i++) { const off = i * q, count = Math.max(0, Math.min(q, S.staticTris - off)); if (count) this.slices.push({ off, count }); }
    this.sliceBufs = this.slices.map((sl) => [0, 1].map((kind) => this._buf(new Uint32Array([0xffffffff, sl.off, kind, 0]), GPUBufferUsage.UNIFORM, 'slice')));
    this.sceneU = new ArrayBuffer(SCENE_U_BYTES);
    this.scopeIdx = S.blas.findIndex((b) => b.body === 'scope');
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

  async _pipelines() {
    const dev = this.device;
    const ptMod = await this._module(ptSource(this.consts), 'godview_pt trace');
    this.ptPipe = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: ptMod, entryPoint: 'ptMain', constants: { CUTAWAY: 1 } } });
    this.ptG0 = dev.createBindGroup({ layout: this.ptPipe.getBindGroupLayout(0), entries: ['nodes', 'tris', 'verts', 'params', 'dbg'].map((k, i) => ({ binding: i, resource: { buffer: this.bufs[k] } })) });
    this.bakePipe = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: ptMod, entryPoint: 'bakeMain', constants: { CUTAWAY: 1 } } });
    this.bakeG0 = dev.createBindGroup({ layout: this.bakePipe.getBindGroupLayout(0), entries: ['nodes', 'tris', 'verts', 'params', 'dbg'].map((k, i) => ({ binding: i, resource: { buffer: this.bufs[k] } })) });
    this.bakeG1 = dev.createBindGroup({ layout: this.bakePipe.getBindGroupLayout(1), entries: [
      { binding: 0, resource: { buffer: this.bufs.scn } }, { binding: 1, resource: { buffer: this.bufs.gu } },
      { binding: 3, resource: { buffer: dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE }) } },          // gbuf: referenced, never written by the bake
      { binding: 4, resource: { buffer: this.bufs.irrAcc } }, { binding: 5, resource: { buffer: this.bufs.backList } }] });
    const smMod = await this._module(SMOOTH_WGSL, 'godview_pt smooth');
    this.smoothPipe = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: smMod, entryPoint: 'smoothMain' } });
    const smG = (src, dst) => dev.createBindGroup({ layout: this.smoothPipe.getBindGroupLayout(0), entries: [this.bufs.verts, this.bufs.adjStart, this.bufs.adjList, src, dst, this.bufs.smoothU, this.bufs.backList, this.bufs.backSlot].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    // acc -> tmp -> irr -> tmp -> irr ... (6 passes, ends in irr, the raster's buffer: the raster never sees a partial bake)
    this.smoothGroups = [smG(this.bufs.irrAcc, this.bufs.irrTmp), smG(this.bufs.irrTmp, this.bufs.irr), smG(this.bufs.irr, this.bufs.irrTmp)];
    this.bakeSample = 0; this.bakeVert = 0; this.bakeChunk = 4096; this.bakeMsPerRay = null;   // godview_pt2: start small, grow from measurements (no still-onset spike)
    const rMod = await this._module(rasterSource(this.consts, this.clip), 'godview_pt2 raster');
    const bgl0 = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      { binding: 8, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 9, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ] });
    const bgl1 = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    ] });
    const rLayout = dev.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] });
    this.rasterPipe = await dev.createRenderPipelineAsync({
      layout: rLayout,
      vertex: { module: rMod, entryPoint: 'vsMain' },
      fragment: { module: rMod, entryPoint: 'fsMain', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: this.msaa },
    });
    this.shadowPipe = await dev.createRenderPipelineAsync({
      layout: rLayout,
      vertex: { module: rMod, entryPoint: 'vsShadow' },
      ...(this.clip ? {} : { fragment: { module: rMod, entryPoint: 'fsShadow', targets: [] } }),   // clip distances: depth only
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 1.5 },
    });
    const SMU = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
    this.shadowTex = dev.createTexture({ size: [SHADOW_SIZE, SHADOW_SIZE], format: 'depth32float', usage: SMU });
    this.scopeShadowTex = dev.createTexture({ size: [SCOPE_SHADOW_SIZE, SCOPE_SHADOW_SIZE], format: 'depth32float', usage: SMU });
    // static-only shadow maps per kind, [front, back]
    this.smStatic = [SHADOW_SIZE, SCOPE_SHADOW_SIZE].map((n) => [0, 1].map(() => dev.createTexture({ size: [n, n], format: 'depth32float', usage: SMU })));
    this.smFront = [0, 0]; this.smKey = [null, null]; this.smBuild = [null, null];
    this.dummyDepth = dev.createTexture({ size: [1, 1], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const smp = dev.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    const g0 = (shadowView) => dev.createBindGroup({ layout: bgl0, entries: [
      { binding: 0, resource: { buffer: this.bufs.tris } }, { binding: 1, resource: { buffer: this.bufs.verts } }, { binding: 2, resource: { buffer: this.bufs.params } },
      { binding: 3, resource: { buffer: this.bufs.scn } }, { binding: 4, resource: { buffer: this.bufs.gu } }, { binding: 5, resource: { buffer: this.bufs.ru } },
      { binding: 6, resource: shadowView }, { binding: 7, resource: smp }, { binding: 8, resource: { buffer: this.bufs.irr } }, { binding: 9, resource: { buffer: this.bufs.backSlot } }] });
    this.rG0 = g0(this.shadowTex.createView());
    this.rG0shadow = g0(this.dummyDepth.createView());        // shadow passes do not sample a shadow map
    this.rG1 = this.drawBufs.map(([b0]) => dev.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: b0 } }, { binding: 1, resource: this.scopeShadowTex.createView() }] }));
    this.rG1shadow = this.drawBufs.map(([b0, b1]) => [b0, b1].map((b) => dev.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: b } }, { binding: 1, resource: this.dummyDepth.createView() }] })));
    this.rG1slice = this.sliceBufs.map(([b0, b1]) => [b0, b1].map((b) => dev.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: b } }, { binding: 1, resource: this.dummyDepth.createView() }] })));
    const dMod = await this._module(displaySource(), 'godview_pt display');
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.dispPipe = await dev.createRenderPipelineAsync({ layout: 'auto', vertex: { module: dMod, entryPoint: 'vs' }, fragment: { module: dMod, entryPoint: 'fs', targets: [{ format: this.canvasFormat }] }, primitive: { topology: 'triangle-list' } });
    this.outSizeBuf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if (this.canvas) { this.ctx = this.canvas.getContext('webgpu'); this.ctx.configure({ device: dev, format: this.canvasFormat, alphaMode: 'opaque' }); }
  }

  // ---------------------------------------------------------------- state
  _changed() { this.changedAt = performance.now(); this.sample = 0; this.tileRow = 0; this.rasterDirty = true; this.bakeStale = true; }

  setPoses(poses) {
    const b = poses?.bodies || {};
    this.instances = this.S.blas.map((bl) => {
      if (bl.kind === 'fixed') return { q: [0, 0, 0, 1], p: [0, 0, 0] };
      const T = b[bl.body];
      if (!T) return null;
      return { q: quatFromMat(T), p: [T[12], T[13], T[14]], R: rotOf(T) };
    });
    this.poseVer++;
    this._writeScene();
    this._changed();
  }

  setCamera({ eye, target, up = [0, 0, 1], vfov_deg = 35 }) {
    this.cam = { eye: [...eye], target: [...target], up: [...up], vfov: vfov_deg };
    this._writeScene();
    this._changed();
  }

  setCutaway(on) { this.cut = !!on; this._writeScene(); this._changed(); }

  _writeScene() {
    if (!this.cam) return;
    const S = this.S, f = new Float32Array(this.sceneU), u = new Uint32Array(this.sceneU);
    f.fill(0);
    const sc = this.scopeIdx;
    const scopeOn = this.light.scopeOn && sc >= 0 && this.instances?.[sc];
    const g = this.light.gain;
    f.set([S.emitter.L0[0] * g, S.emitter.L0[1] * g, S.emitter.L0[2] * g, S.emitter.area || 1e-12], 0);
    f.set([...S.emitter.axis, 0], 4);
    u[7] = S.emitter.shadowSkipsScope && sc >= 0 ? (1 << sc) >>> 0 : 0;
    // cutaway plane: dot(p - centre, n) > offset, n = unit(eye - centre)  ->  dot(n, p) > dot(n, centre) + offset
    const c = this.godview.centre;
    const n = norm3(sub3(this.cam.eye, c));
    f.set(this.cut ? [...n, dot3(n, c) + this.cutOffset] : [0, 0, 0, 1e30], 12);
    u[16] = S.blas.length; u[17] = sc < 0 ? 0 : sc;
    f.set([...S.emitter.centroid, S.emitter.radius], 20);
    S.blas.forEach((bl, i) => {
      const o = 24 + 48 * i;
      const I = this.instances?.[i];
      u[o + 11] = bl.root;
      if (!I) { u[o + 15] = 2; return; }
      f.set(I.q, o); f.set(I.q, o + 4); f.set(I.p, o + 8); u[o + 11] = bl.root; f.set(I.p, o + 12); u[o + 15] = 0;
      const R = I.R || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let k = 0; k < 8; k++) {
        const v = [k & 1 ? bl.bmax[0] : bl.bmin[0], k & 2 ? bl.bmax[1] : bl.bmin[1], k & 4 ? bl.bmax[2] : bl.bmin[2]];
        const w = mulRv(R, v);
        for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], w[a] + I.p[a]); hi[a] = Math.max(hi[a], w[a] + I.p[a]); }
      }
      f.set([lo[0] - 1e-6, lo[1] - 1e-6, lo[2] - 1e-6, 0, hi[0] + 1e-6, hi[1] + 1e-6, hi[2] + 1e-6, 0], o + 16);
      const ca = mulRv(R, bl.capsule.a).map((x, k) => x + I.p[k]), cb = mulRv(R, bl.capsule.b).map((x, k) => x + I.p[k]);
      f.set([...ca, bl.capsule.r * 1.0001 + 1e-6, ...cb, 0], o + 32);
      f.set([...ca, bl.capsule.r * 1.0001 + 1e-6, ...cb, 0], o + 40);
    });
    this.device.queue.writeBuffer(this.bufs.scn, 0, this.sceneU);
    this.scopeOn = scopeOn ? 1 : 0;
  }

  _targets(w, h) {
    const rs = this.canvas ? this.rscale : 1;
    if (this.size && this.size[0] === w && this.size[1] === h && this._rs === rs) return;
    this._rs = rs;
    const rw = Math.max(1, Math.round(w * rs)), rh = Math.max(1, Math.round(h * rs));
    const dev = this.device;
    for (const t of ['accum', 'gbuf']) this.bufs[t]?.destroy?.();
    for (const t of ['rasterMS', 'rasterTex', 'depthTex']) this[t]?.destroy?.();
    this.bufs.accum = dev.createBuffer({ size: w * h * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.bufs.gbuf = dev.createBuffer({ size: w * h * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.rasterMS = dev.createTexture({ size: [rw, rh], format: 'rgba16float', sampleCount: this.msaa, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.rasterTex = dev.createTexture({ size: [rw, rh], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.depthTex = dev.createTexture({ size: [rw, rh], format: 'depth32float', sampleCount: this.msaa, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.ptG1 = dev.createBindGroup({ layout: this.ptPipe.getBindGroupLayout(1), entries: [
      { binding: 0, resource: { buffer: this.bufs.scn } }, { binding: 1, resource: { buffer: this.bufs.gu } },
      { binding: 2, resource: { buffer: this.bufs.accum } }, { binding: 3, resource: { buffer: this.bufs.gbuf } }] });
    this.dispG = dev.createBindGroup({ layout: this.dispPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.bufs.gu } }, { binding: 1, resource: { buffer: this.bufs.accum } }, { binding: 2, resource: { buffer: this.bufs.gbuf } },
      { binding: 3, resource: this.rasterTex.createView() }, { binding: 4, resource: { buffer: this.outSizeBuf } },
      { binding: 5, resource: this.linSmp || (this.linSmp = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' })) }] });
    this.size = [w, h];
    this.sample = 0; this.tileRow = 0; this.rasterDirty = true;
  }

  _camAxes(w, h) {
    const z = norm3(sub3(this.cam.target, this.cam.eye));
    const x = norm3(cross3(z, this.cam.up)), y = cross3(z, x);
    const f = (h / 2) / Math.tan((this.cam.vfov * DEG) / 2);
    return { x, y, z, f, cx: w / 2 - 0.5, cy: h / 2 - 0.5 };
  }

  _writeGU({ sampleIndex = 0, tile = [0, 0, 0, 0], spp = 1, seed = 0, xfade = 0, denoise = 0 } = {}) {
    const [w, h] = this.size, a = this._camAxes(w, h);
    const F = new Float32Array(GU_BYTES / 4), U = new Uint32Array(F.buffer);
    F.set([...a.x, 0, ...a.y, 0, ...a.z, 0, ...this.cam.eye, 0, a.f, a.f, a.cx, a.cy], 0);
    U.set([w, h, sampleIndex, this.opts.maxBounces], 20);
    U.set(tile, 24);
    F.set([...this.light.keyDir, this.light.keyOn], 28);
    F.set([...this.light.keyE, 0], 32);
    F.set([...this.light.env, this.light.envOn], 36);
    U.set([this.cutMask, seed >>> 0, spp, this.scopeOn], 40);
    F.set([this.exposure, xfade, denoise, 0], 44);
    this.device.queue.writeBuffer(this.bufs.gu, 0, F);
  }

  _writeRU() {
    const [w, h] = this.size;
    const c = this.godview.centre;
    const VP = lookAtProj(this.cam.eye, this.cam.target, this.cam.up, this.cam.vfov, w / h, 0.02, 6.0);
    const LP = orthoLight(this.light.keyDir, [c[0], c[1], c[2]], 0.45, 3.0);
    // scope outlet: perspective from the outlet centroid along its axis (shadow map of the inside light)
    let SP = new Array(16).fill(0);
    const I = this.instances?.[this.scopeIdx];
    if (I) {
      const R = I.R, pc = mulRv(R, this.S.emitter.centroid).map((x, k) => x + I.p[k]), ax = norm3(mulRv(R, this.S.emitter.axis));
      const eye = pc.map((x, k) => x + ax[k] * 1e-4);
      const up = Math.abs(ax[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
      SP = lookAtProj(eye, eye.map((x, k) => x + ax[k]), up, 120, 1, 0.0015, 1.0);     // the profile is 0 past 55 deg
    }
    const F = new Float32Array(RU_BYTES / 4);
    F.set(VP, 0); F.set(LP, 16); F.set(SP, 32);
    F.set([1.5 / SHADOW_SIZE, 0, 0, 0], 48);
    F.set([...(this.S.emitter.normal || this.S.emitter.axis), 0], 52);
    this.device.queue.writeBuffer(this.bufs.ru, 0, F);
  }

  // ---------------------------------------------------------------- passes
  _smStateKey(kind) {
    const f = new Float32Array(this.sceneU);
    const k = { cut: [f[12], f[13], f[14], f[15]] };
    if (kind === 1) { const I = this.instances?.[this.scopeIdx]; k.scope = I ? [...I.p, ...I.q] : null; }
    return k;
  }
  _smFar(a, b, kind) {
    const same = a.cut.every((v, i) => v === b.cut[i]);
    const c = a.cut[0] * b.cut[0] + a.cut[1] * b.cut[1] + a.cut[2] * b.cut[2];
    if (!same && c < Math.cos(SHADOW_CUT_DEG * DEG) || Math.abs(a.cut[3] - b.cut[3]) > 1e-3) return true;
    if (kind === 1) { if (!a.scope !== !b.scope) return true; if (a.scope) for (let i = 0; i < 7; i++) if (Math.abs(a.scope[i] - b.scope[i]) > 1e-5) return true; }
    return false;
  }
  _encodeStaticSlices(enc, tex, kind, s0, s1, ts = () => ({})) {
    for (let s = s0; s < s1; s++) {
      const sp = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: tex.createView(), depthClearValue: 1, depthLoadOp: s === 0 ? 'clear' : 'load', depthStoreOp: 'store' }, ...ts() });
      sp.setPipeline(this.shadowPipe);
      sp.setBindGroup(0, this.rG0shadow);
      sp.setBindGroup(1, this.rG1slice[s][kind]); sp.draw(3 * this.slices[s].count);
      sp.end();
    }
  }

  _encodeRaster(enc, force = false) {
    this._writeRU();
    // shadow maps: key (ortho) and scope (perspective from the outlet)
    const tsK = this.hasTS && !this.tsBusy[0];
    let first = true;
    for (const [tex, kind] of [[this.shadowTex, 0], [this.scopeShadowTex, 1]]) {
      if (kind === 1 && !this.scopeOn) continue;
      const ts = () => { if (!(tsK && first)) return {}; first = false; return { timestampWrites: { querySet: this.tsSet, beginningOfPassWriteIndex: 0 } }; };
      // 1) static casters: rebuild the back map one slice per frame when the cut plane or the scope moved (all at once
      //    on the first frame and for captures), then swap it in
      const key = this._smStateKey(kind);
      const front = this.smStatic[kind][this.smFront[kind]];
      if (force || !this.smKey[kind]) {
        this.smBuild[kind] = null;
        this._encodeStaticSlices(enc, this.smStatic[kind][this.smFront[kind]], kind, 0, this.slices.length, ts);
        this.smKey[kind] = key;
      } else {
        if (!this.smBuild[kind] && this._smFar(this.smKey[kind], key, kind)) this.smBuild[kind] = { next: 0, key };
        const B = this.smBuild[kind];
        if (B) {
          const back = this.smStatic[kind][1 - this.smFront[kind]];
          this._encodeStaticSlices(enc, back, kind, B.next, B.next + 1, ts);
          B.next++;
          if (B.next >= this.slices.length) { this.smFront[kind] = 1 - this.smFront[kind]; this.smKey[kind] = B.key; this.smBuild[kind] = null; }
        }
      }
      // 2) the working map = the static map + the instances drawn on top
      const st = this.smStatic[kind][this.smFront[kind]];
      enc.copyTextureToTexture({ texture: st }, { texture: tex }, [tex.width, tex.height]);
      const sp = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: tex.createView(), depthLoadOp: 'load', depthStoreOp: 'store' }, ...ts() });
      sp.setPipeline(this.shadowPipe);
      sp.setBindGroup(0, this.rG0shadow);
      this.draws.forEach((d, i) => {
        if (d.inst === 0xffffffff || !this.instances[d.inst]) return;
        if (kind === 1 && d.inst === this.scopeIdx) return;          // the scope does not shadow its own outlet
        sp.setBindGroup(1, this.rG1shadow[i][kind]); sp.draw(3 * d.count);
      });
      sp.end();
    }
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: (this.msaa > 1 ? this.rasterMS : this.rasterTex).createView(), ...(this.msaa > 1 ? { resolveTarget: this.rasterTex.createView(), storeOp: 'discard' } : { storeOp: 'store' }), loadOp: 'clear', clearValue: [...this.light.env.map((v) => v * this.light.envOn), 1] }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      ...(tsK ? { timestampWrites: { querySet: this.tsSet, endOfPassWriteIndex: 1 } } : {}),
    });
    rp.setPipeline(this.rasterPipe);
    rp.setBindGroup(0, this.rG0);
    this.draws.forEach((d, i) => {
      if (d.inst !== 0xffffffff && !this.instances[d.inst]) return;
      rp.setBindGroup(1, this.rG1[i]); rp.draw(3 * d.count);
    });
    rp.end();
    if (tsK) this._tsCollect(enc, 0, (ms) => { this.gpuRaster.push(ms); if (this.gpuRaster.length > 300) this.gpuRaster.shift(); });
  }

  _encodeDisplay(enc, xfade, denoise) {
    if (!this.ctx) return;
    this.device.queue.writeBuffer(this.outSizeBuf, 0, new Float32Array([this.canvas.width, this.canvas.height, 0, 0]));
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    rp.setPipeline(this.dispPipe); rp.setBindGroup(0, this.dispG); rp.draw(3); rp.end();
  }

  /** timestamps [2k, 2k+1] of kind k (0 raster frame, 1 traced band) -> GPU ms, read back without waiting */
  _tsCollect(enc, k, onMs) {
    if (!this.hasTS || this.tsBusy[k]) return;
    enc.resolveQuerySet(this.tsSet, 2 * k, 2, this.tsResolve, 256 * k);
    enc.copyBufferToBuffer(this.tsResolve, 256 * k, this.tsRead[k], 0, 16);
    this.tsBusy[k] = true;
    this._tsPending = this._tsPending || [];
    this._tsPending.push(() => this.tsRead[k].mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigUint64Array(this.tsRead[k].getMappedRange().slice(0, 16));
      this.tsRead[k].unmap(); this.tsBusy[k] = false;
      const ms = Number(t[1] - t[0]) / 1e6;
      if (ms >= 0 && ms < 1e4) onMs(ms);
    }).catch(() => { this.tsBusy[k] = false; }));
  }
  _tsFlush() { const p = this._tsPending || []; this._tsPending = []; for (const f of p) f(); }

  /** one chunk of the vertex bake (the raster's light: everything but the key's and the outlet's direct light) */
  _submitBake(rays = 1) {
    if (this.bakeStale) { this.bakeSample = 0; this.bakeVert = 0; this.bakeStale = false; }
    const nV = this.bakeTotal;
    const count = Math.max(64, Math.min(nV - this.bakeVert, this.bakeChunk));
    if (!this.size) this._targets(64, 64);
    this._writeGU({ sampleIndex: this.bakeSample, tile: [this.bakeVert, nV, count, 1], spp: rays, seed: 0x51ed });
    const enc = this.device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.bakePipe); p.setBindGroup(0, this.bakeG0); p.setBindGroup(1, this.bakeG1);
    p.dispatchWorkgroups(Math.ceil(count / 64));
    p.end();
    this.device.queue.submit([enc.finish()]);
    const t0 = performance.now(), work = count * rays;
    this.inFlight++;
    this.device.queue.onSubmittedWorkDone().then(() => {
      this.inFlight--;
      const ms = performance.now() - t0;
      const per = ms / work;
      this.bakeMsPerRay = this.bakeMsPerRay == null ? per : 0.7 * this.bakeMsPerRay + 0.3 * per;
      // godview_pt2: no 32k floor (one 32k chunk is ~20 ms on the M4) and at most x2 growth per measurement
      this.bakeChunk = Math.max(1024, Math.min(2 * this.bakeChunk, 1 << 20, Math.floor(this.opts.tileMs / (this.bakeMsPerRay * rays) / 64) * 64));
      this.subMax = Math.max(this.subMax || 0, ms);
      if (ms > 64) this.log(`bake overrun: ${ms.toFixed(1)} ms (${count} vertices)`);
    });
    this.bakeVert += count;
    if (this.bakeVert >= nV) {
      this.bakeVert = 0; this.bakeSample += rays;
      if (this.bakeSample >= this.opts.bakeRays) this._smooth(this.opts.smoothPasses ?? 6);
    }
  }

  _smooth(passes) {
    const enc = this.device.createCommandEncoder();
    const seq = [0, 1];
    for (let k = 2; k < passes; k += 2) seq.push(2, 1);
    for (const gi of seq) {
      const p = enc.beginComputePass();
      p.setPipeline(this.smoothPipe); p.setBindGroup(0, this.smoothGroups[gi]); p.dispatchWorkgroups(Math.ceil(this.bakeTotal / 64)); p.end();
    }
    this.device.queue.submit([enc.finish()]);
    this.rasterDirty = true;
    this.bakeVersion = (this.bakeVersion || 0) + 1;
  }

  bakeDone() { return !this.bakeStale && this.bakeSample >= this.opts.bakeRays; }

  /** one band of rows of one path-traced sample (one submit) */
  _submitBand(spp = 1, seed = 0) {
    const [w, h] = this.size;
    const rows = Math.max(8, Math.min(h - this.tileRow, this.rowsPerTile));
    this._writeGU({ sampleIndex: this.sample, tile: [0, this.tileRow, w, rows], spp, seed });
    const enc = this.device.createCommandEncoder();
    const tsB = this.hasTS && !this.tsBusy[1];
    const p = enc.beginComputePass(tsB ? { timestampWrites: { querySet: this.tsSet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } } : {});
    p.setPipeline(this.ptPipe); p.setBindGroup(0, this.ptG0); p.setBindGroup(1, this.ptG1);
    p.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(rows / 8));
    p.end();
    const frac = rows / h;
    if (tsB) this._tsCollect(enc, 1, (ms) => { this.gpuBand.push(ms / frac); if (this.gpuBand.length > 300) this.gpuBand.shift(); });
    this.device.queue.submit([enc.finish()]);
    this._tsFlush();
    const t0 = performance.now(), px = w * rows * spp;
    this.inFlight++;
    const ver = this.size;
    this.device.queue.onSubmittedWorkDone().then(() => {
      this.inFlight--;
      const ms = performance.now() - t0;
      // adapt the band so one submit stays near opts.tileMs of GPU time (wall time from submit: an upper bound)
      const perPx = ms / px;
      this.msPerRowPx = this.msPerRowPx == null ? perPx : 0.7 * this.msPerRowPx + 0.3 * perPx;
      this.rowsPerTile = Math.max(8, Math.min(2 * this.rowsPerTile, 1024, Math.floor(this.opts.tileMs / (this.msPerRowPx * w * spp) / 8) * 8));
      this.subMax = Math.max(this.subMax || 0, ms);
      if (ms > 64) this.log(`band overrun: ${ms.toFixed(1)} ms (${rows} rows)`);
      this.sampleMs.push(ms / (rows / ver[1]));
      if (this.sampleMs.length > 200) this.sampleMs.shift();
    });
    this.tileRow += rows;
    if (this.tileRow >= h) { this.tileRow = 0; this.sample += spp; }
  }

  /** Draw the display image to the canvas; never waits on the GPU. */
  render() {
    if (!this.cam) return;
    const w = this.canvas ? this.canvas.width : 1280, h = this.canvas ? this.canvas.height : 800;
    this._targets(w, h);
    const now = performance.now();
    const still = now - this.changedAt > STILL_MS;
    const enc = this.device.createCommandEncoder();
    if (this.rasterDirty) { this._writeGU({}); this._encodeRaster(enc); this.rasterDirty = false; }
    this.device.queue.submit([enc.finish()]);
    this._tsFlush();
    if (still && this.inFlight === 0 && !this.bakeDone()) {
      this._submitBake(1);
    } else if (still && this.inFlight === 0 && this.sample < (this.opts.maxSamples ?? 4096)) {
      for (let k = 0; k < this.opts.maxTilesPerFrame && this.inFlight < 2; k++) {
        if (this.sample === 0 && this.tileRow === 0 && k > 0) break;
        this._submitBand(1, this.sample * 2654435761);
        if (this.tileRow === 0) break;                 // a sample completed this frame
      }
    }
    // crossfade: the traced image takes over once one full sample exists, over the first few samples
    const done = this.sample;
    const u = Math.min(1, Math.max(0, (done - this.opts.fadeFrom) / (this.opts.fadeTo - this.opts.fadeFrom)));
    const xfade = done <= 0 ? 0 : u * u * (3 - 2 * u);
    const denoise = done > 0 && done < this.opts.denoiseUntil ? 2.0 * Math.sqrt(1 - done / this.opts.denoiseUntil) : 0;
    this._writeGU({ sampleIndex: this.sample, xfade, denoise });
    const enc2 = this.device.createCommandEncoder();
    this._encodeDisplay(enc2, xfade, denoise);
    this.device.queue.submit([enc2.finish()]);
    this.frames++;
  }

  /** Scene-referred linear RGB (w*h*3, top row first) of the current view. opts: {spp, mode: 'traced' | 'raster', seed} */
  async captureLinear(width, height, opts = {}) {
    const mode = opts.mode || 'traced';
    const keepCanvas = this.canvas; this.canvas = null;
    try {
      this._targets(width, height);
      const dev = this.device;
      if (mode === 'raster') {
        if (!opts.noBake) {          // noBake: keep the bake made under the previous view (the stale-bake probe)
          this.bakeStale = true;
          do { this._submitBake(1); await dev.queue.onSubmittedWorkDone(); } while (!this.bakeDone());
        }
        this._targets(width, height);
        const enc = dev.createCommandEncoder();
        this._writeGU({});
        this._encodeRaster(enc, true);
        dev.queue.submit([enc.finish()]);
        this._tsFlush();
        const bpr = Math.ceil(width * 8 / 256) * 256;
        const rb = dev.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const e2 = dev.createCommandEncoder();
        e2.copyTextureToBuffer({ texture: this.rasterTex }, { buffer: rb, bytesPerRow: bpr }, [width, height]);
        dev.queue.submit([e2.finish()]);
        await rb.mapAsync(GPUMapMode.READ);
        const src = new Uint16Array(rb.getMappedRange().slice(0));
        rb.unmap(); rb.destroy();
        const out = new Float32Array(width * height * 3);
        for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
          const i = (y * width + x) * 3, j = y * (bpr / 2) + x * 4;
          out[i] = half(src[j]); out[i + 1] = half(src[j + 1]); out[i + 2] = half(src[j + 2]);
        }
        return out;
      }
      const spp = opts.spp ?? 256;
      this.sample = 0; this.tileRow = 0;
      const t0 = performance.now();
      let last = performance.now();
      while (this.sample < spp) {
        this._submitBand(1, (opts.seed ?? 0) * 7919 + this.sample * 2654435761);
        await dev.queue.onSubmittedWorkDone();
        if (performance.now() - last > 1500) { await new Promise((r) => setTimeout(r, 50)); last = performance.now(); }   // §12 pacing
      }
      this.captureMs = performance.now() - t0;
      const rb = dev.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = dev.createCommandEncoder();
      enc.copyBufferToBuffer(this.bufs.accum, 0, rb, 0, width * height * 16);
      dev.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const A = new Float32Array(rb.getMappedRange().slice(0));
      rb.unmap(); rb.destroy();
      const out = new Float32Array(width * height * 3);
      for (let i = 0; i < width * height; i++) { const n = Math.max(1, A[4 * i + 3]); out[3 * i] = A[4 * i] / n; out[3 * i + 1] = A[4 * i + 1] / n; out[3 * i + 2] = A[4 * i + 2] / n; }
      return out;
    } finally {
      this.canvas = keepCanvas;
    }
  }

  async debugCounters() {
    const dev = this.device;
    const rb = dev.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(this.bufs.dbg, 0, rb, 0, 64); dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const a = Array.from(new Uint32Array(rb.getMappedRange().slice(0))); rb.unmap(); rb.destroy();
    return a;
  }

  timings() {
    const med = (a) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
    // gpu_ms_*: GPU timestamp ms of the last / median raster frame (shadow maps + main pass; what draws while things
    // move); trace_ms_per_sample_*: GPU ms of one full-frame path-traced sample (band timestamps scaled to the frame);
    // wall_ms_per_sample_median: submit-to-done wall time (the fallback without timestamp-query)
    return { gpu_ms_last: this.gpuRaster.at(-1) ?? null, gpu_ms_median: med(this.gpuRaster), trace_ms_per_sample_median: med(this.gpuBand),
      trace_ms_per_sample_last: this.gpuBand.at(-1) ?? null, wall_ms_per_sample_median: med(this.sampleMs), frames: this.frames, samples: this.sample,
      rowsPerTile: this.rowsPerTile, timestamp_query: this.hasTS, cached_scene: this.cached, clip_distances: !!this.clip, back_entries: this.nBack, submit_wall_ms_max: this.subMax ?? null, bake_chunk: this.bakeChunk };
  }

  destroy() {
    for (const b of Object.values(this.bufs)) b?.destroy?.();
    for (const t of ['rasterMS', 'rasterTex', 'depthTex', 'shadowTex', 'scopeShadowTex', 'dummyDepth']) this[t]?.destroy?.();
    for (const p of this.smStatic || []) for (const t of p) t.destroy();
    this.ctx?.unconfigure?.();
  }
}

function half(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 5.960464477539063e-8;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}
