// STUB renderer: the docs/WEB_TWIN.md §8 API on a plain WebGPU rasteriser, so the page, the model socket and the
// judges can be built before gpu/renderer.js exists. It is NOT the twin's light transport: flat-ish shading, a point
// light at the scope tip, KB4 applied per vertex (straight edges between projected vertices), no indirect light, no
// sensor model. Every frame it renders is labelled STUB on the page. app.js switches to gpu/renderer.js automatically.

import { buildBVH, packTriangles, traverse } from '../gpu/bvh.js';

const COLORS = {           // linear albedo by material (stub only)
  pla_matte_white: [0.80, 0.80, 0.77], bench_white: [0.86, 0.86, 0.84], insulation_black: [0.03, 0.03, 0.03],
  steel_satin: [0.55, 0.56, 0.58], steel_jaw: [0.62, 0.63, 0.65], scope_steel: [0.5, 0.5, 0.52], scope_glass: [0.05, 0.06, 0.08],
  emitter: [1.0, 1.0, 0.95], charuco_print: [0.8, 0.8, 0.8], lens_glass: [0.05, 0.05, 0.06],
};

const WGSL = /* wgsl */`
struct View {
  T_C_D : mat4x4f,       // D -> camera (OpenCV)
  K     : vec4f,         // fx fy cx cy (kb4) | f_y-normalised pinhole: (1/tan(fovy/2)/aspect, 1/tan(fovy/2), 0, 0)
  D     : vec4f,         // kb4 k1..k4
  size  : vec4f,         // W H near far
  light : vec4f,         // xyz light position in D, w: 1 = scope light (feeds), 2 = viewing aid (overview)
  ldir  : vec4f,         // xyz light axis, w: exposure gain
  cut   : vec4f,         // cutaway plane n.xyz, offset (discard static fragments with dot(n, p) > offset)
  mode  : vec4f,         // x: 0 kb4, 1 pinhole; y: theta_max; z: 1 = cutaway on
};
struct Obj { M : mat4x4f, color : vec4f, flags : vec4f };   // flags.x: 1 static, 2 emitter, 3 board
@group(0) @binding(0) var<uniform> V : View;
@group(0) @binding(1) var<storage, read> objs : array<Obj>;

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) pD : vec3f,
  @location(1) nD : vec3f,
  @location(2) pL : vec3f,
  @location(3) @interpolate(flat) obj : u32,
};

@vertex fn vs(@location(0) p : vec3f, @location(1) n : vec3f, @builtin(instance_index) oi : u32) -> VOut {
  let o = objs[oi];
  let pD = (o.M * vec4f(p, 1.0)).xyz;
  let nD = (o.M * vec4f(n, 0.0)).xyz;
  let pc = (V.T_C_D * vec4f(pD, 1.0)).xyz;
  var out : VOut;
  let near = V.size.z; let far = V.size.w;
  if (V.mode.x < 0.5) {
    let rho = length(pc.xy);
    let th = atan2(rho, pc.z);
    let t2 = th * th;
    let thd = th * (1.0 + t2 * (V.D.x + t2 * (V.D.y + t2 * (V.D.z + t2 * V.D.w))));
    var uv = V.K.zw;
    if (rho > 1e-12) { uv = vec2f(V.K.x * thd * pc.x / rho, V.K.y * thd * pc.y / rho) + V.K.zw; }
    let ndc = vec2f((uv.x + 0.5) / V.size.x * 2.0 - 1.0, 1.0 - (uv.y + 0.5) / V.size.y * 2.0);
    let dist = length(pc);
    var z = clamp((dist - near) / (far - near), 0.0, 1.0);
    if (th > V.mode.y) { z = 2.0; }                      // past the lens's field: clipped
    out.pos = vec4f(ndc, z, 1.0);
  } else {
    // OpenCV camera -> GL clip: x right, y up, depth z forward
    let f = V.K.xy;
    out.pos = vec4f(pc.x * f.x, -pc.y * f.y, (pc.z - near) * far / (far - near), pc.z);
  }
  out.pD = pD; out.nD = nD; out.pL = p; out.obj = oi;
  return out;
}

@fragment fn fs(v : VOut) -> @location(0) vec4f {
  let o = objs[v.obj];
  if (V.mode.z > 0.5 && o.flags.x == 1.0 && dot(V.cut.xyz, v.pD) > V.cut.w) { discard; }
  var n = normalize(v.nD);
  var c = o.color.rgb;
  if (o.flags.x == 3.0) {                              // board: procedural 12 mm checker in the board frame
    let q = floor(v.pL.xy / 0.012);
    let k = (i32(q.x) + i32(q.y)) & 1;
    c = select(vec3f(0.85), vec3f(0.04), k == 1);
  }
  if (o.flags.x == 2.0) { return vec4f(c * 8.0, 1.0); }
  let toL = V.light.xyz - v.pD;
  let d = length(toL);
  let l = toL / max(d, 1e-6);
  let lam = abs(dot(n, l));
  var e = 0.0;
  if (V.light.w < 1.5) {
    let cosA = max(dot(-l, normalize(V.ldir.xyz)), 0.0);
    let prof = exp(-pow(acos(cosA) / 0.733, 2.0));      // exp(-(theta / 42 deg)^2), physics.yaml first pass
    e = 0.004 * lam * prof / max(d * d, 1e-5) + 0.02;
    e = e * V.ldir.w;
  } else {
    e = 0.25 + 0.75 * lam;
  }
  return vec4f(c * e, 1.0);
}

struct FOut { @builtin(position) pos : vec4f };
@vertex fn vsq(@builtin(vertex_index) i : u32) -> FOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o : FOut; o.pos = vec4f(p[i], 0.0, 1.0); return o;
}
@group(0) @binding(0) var lin : texture_2d<f32>;
@fragment fn fsq(f : FOut) -> @location(0) vec4f {
  let c = textureLoad(lin, vec2i(f.pos.xy), 0).rgb;
  let g = pow(clamp(c, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
  return vec4f(g, 1.0);
}
`;

const VIEW_BYTES = 4 * (16 + 4 * 7);
const OBJ_BYTES = 4 * (16 + 8);

function bodyOfMesh(scene) {
  const map = {};
  for (const ins of scene.instruments || []) for (const p of ins.parts || []) map[p.mesh] = `${ins.id}/${p.body}`;
  for (const m of scene.scope?.mesh_parts || []) map[m] = 'scope';
  if (scene.scope?.emitter_mesh) map[scene.scope.emitter_mesh] = 'scope';
  for (const b of scene.boards || []) if (b.mesh) map[b.mesh] = `board/${b.name}`;
  for (const m of scene.meshes) {
    if (m.body && !map[m.name]) map[m.name] = m.body;
    if (m.dynamic && !map[m.name] && m.group === 'scope') map[m.name] = 'scope';
  }
  return map;
}

export class StubRenderer {
  static async create({ canvas = null, scene, physics = null, device = null } = {}) {
    const r = new StubRenderer();
    await r._init(canvas, scene, physics, device);
    return r;
  }

  get isStub() { return true; }

  async _init(canvas, bundle, physics, device) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    this.scene = bundle.scene ?? bundle;
    this.meshData = bundle.meshes ?? {};
    this.physics = physics;
    if (!device) {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('no WebGPU adapter');
      this.adapterInfo = adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture}`.trim() : '';
      device = await adapter.requestDevice();
    }
    this.device = device;
    const dev = device;
    this.module = dev.createShaderModule({ code: WGSL });
    this.bodyMap = bodyOfMesh(this.scene);
    // geometry
    this.draws = [];
    const mats = [];
    const addDraw = (name, md, meta, flags, body) => {
      const vb = dev.createBuffer({ size: Math.max(16, md.P.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(vb, 0, new Float32Array(md.P));
      const nb = dev.createBuffer({ size: Math.max(16, md.N.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(nb, 0, new Float32Array(md.N));
      const ib = dev.createBuffer({ size: Math.max(16, md.I.byteLength), usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(ib, 0, new Uint32Array(md.I));
      this.draws.push({ name, vb, nb, ib, count: md.I.length, obj: this.draws.length, body, flags, lens: meta?.group === 'lens_front' || !!meta?.overview_only });
      mats.push({ color: COLORS[meta?.material] ?? [0.6, 0.6, 0.6], flags });
    };
    for (const m of this.scene.meshes) {
      const md = this.meshData[m.name];
      if (!md) continue;
      const body = m.dynamic ? this.bodyMap[m.name] ?? null : null;
      const flags = !m.dynamic ? 1 : m.material === 'emitter' ? 2 : body?.startsWith('board/') ? 3 : 0;
      addDraw(m.name, md, m, flags, body);
    }
    // the bench: a quad at bench.z over its extent
    const bz = this.scene.bench?.z ?? 0, ex = this.scene.bench?.extent ?? [[-0.4, -0.3], [0.3, 0.36]];
    addDraw('bench', {
      P: new Float32Array([ex[0][0], ex[0][1], bz, ex[1][0], ex[0][1], bz, ex[1][0], ex[1][1], bz, ex[0][0], ex[1][1], bz]),
      N: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]), I: new Uint32Array([0, 1, 2, 0, 2, 3]),
    }, { material: 'bench_white' }, 1, null);
    this.objBuf = dev.createBuffer({ size: OBJ_BYTES * this.draws.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.objData = new Float32Array(this.draws.length * (OBJ_BYTES / 4));
    this.mats = mats;
    // pipelines
    const vbl = [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
      { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] }];
    this.bgl = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }] });
    const layout = dev.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    this._mkPipe = (format, sampleCount) => dev.createRenderPipeline({
      layout, vertex: { module: this.module, entryPoint: 'vs', buffers: vbl },
      fragment: { module: this.module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: sampleCount },
    });
    this.pipeLinear = this._mkPipe('rgba32float', 1);
    this.qbgl = dev.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }] });
    this.pipeSensor = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.qbgl] }),
      vertex: { module: this.module, entryPoint: 'vsq' }, fragment: { module: this.module, entryPoint: 'fsq', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });
    // cameras
    this.cams = {};
    for (const c of this.scene.cameras) {
      const [W, H] = c.size;
      const lin = dev.createTexture({ size: [W, H], format: 'rgba32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
      const sen = dev.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
      const depth = dev.createTexture({ size: [W, H], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
      const ub = dev.createBuffer({ size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const bg = dev.createBindGroup({ layout: this.bgl, entries: [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: this.objBuf } }] });
      const sbg = dev.createBindGroup({ layout: this.qbgl, entries: [{ binding: 0, resource: lin.createView() }] });
      this.cams[c.name] = { rec: c, W, H, lin, sen, depth, ub, bg, sbg, T_C_D: invRows(c.T_D_C), stage: [] };
    }
    // overview
    this.canvas = canvas;
    if (canvas) {
      this.ctx = canvas.getContext('webgpu');
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.ctx.configure({ device: dev, format: this.format, alphaMode: 'opaque' });
      this.pipeOverview = this._mkPipe(this.format, 4);
    }
    // CPU occlusion: one BVH of the static meshes, one per dynamic mesh in its body frame
    this._buildOcclusion();
    this.poses0 = null; this.poses1 = null;
    this.times = [];
    this.lightPos = [0, 0, 0.2]; this.lightDir = [0, 0, -1];
  }

  _buildOcclusion() {
    const st = this.scene.meshes.filter((m) => !m.dynamic && this.meshData[m.name] && m.group !== 'lens_front');
    let nv = 0, ni = 0;
    for (const m of st) { nv += this.meshData[m.name].P.length; ni += this.meshData[m.name].I.length; }
    const P = new Float32Array(nv), I = new Uint32Array(ni);
    let vo = 0, io = 0;
    for (const m of st) {
      const d = this.meshData[m.name];
      P.set(d.P, vo);
      for (let k = 0; k < d.I.length; k++) I[io + k] = d.I[k] + vo / 3;
      vo += d.P.length; io += d.I.length;
    }
    this.occStatic = ni ? (() => { const b = buildBVH(P, I); return { nodes: b.nodes, T: packTriangles(P, I, b.order) }; })() : null;
    this.occDyn = [];
    for (const m of this.scene.meshes) {
      if (!m.dynamic || !this.meshData[m.name] || m.material === 'emitter') continue;
      const d = this.meshData[m.name];
      const b = buildBVH(d.P, d.I);
      this.occDyn.push({ body: this.bodyMap[m.name], nodes: b.nodes, T: packTriangles(d.P, d.I, b.order) });
    }
  }

  setPoses(poses0, poses1 = null) {
    this.poses0 = poses0; this.poses1 = poses1 ?? poses0;
    const B = this.poses0.bodies || {};
    const od = this.objData;
    for (const d of this.draws) {
      const o = d.obj * (OBJ_BYTES / 4);
      let T = d.body ? B[d.body] : IDENT;
      let hidden = false;
      if (d.body && !T) { T = IDENT; hidden = true; }
      for (let k = 0; k < 16; k++) od[o + k] = T[k];
      if (hidden) for (let k = 0; k < 12; k++) od[o + k] = 0;   // collapse a hidden body to a point
      const c = this.mats[d.obj].color;
      od[o + 16] = c[0]; od[o + 17] = c[1]; od[o + 18] = c[2]; od[o + 19] = 1;
      od[o + 20] = this.mats[d.obj].flags;
    }
    this.device.queue.writeBuffer(this.objBuf, 0, od);
    const S = B.scope;
    if (S) {
      // light outlet: the scope tip, emitting along the direction of view (30 deg about +x toward +y)
      const a = 30 * Math.PI / 180;
      this.lightPos = [S[12], S[13], S[14]];
      const v = [0, Math.sin(a), Math.cos(a)];
      this.lightDir = [S[0] * v[0] + S[4] * v[1] + S[8] * v[2], S[1] * v[0] + S[5] * v[1] + S[9] * v[2], S[2] * v[0] + S[6] * v[1] + S[10] * v[2]];
      this.lightPos = [this.lightPos[0] + this.lightDir[0] * 0.0005, this.lightPos[1] + this.lightDir[1] * 0.0005, this.lightPos[2] + this.lightDir[2] * 0.0005];
    }
  }

  _viewData(T_C_D, K, D, size, near, far, lightMode, gain, cut, mode, thMax) {
    const a = new Float32Array(VIEW_BYTES / 4);
    a.set(T_C_D, 0);
    a.set(K, 16); a.set(D, 20); a.set([size[0], size[1], near, far], 24);
    a.set([...this.lightPos, lightMode], 28); a.set([...this.lightDir, gain], 32);
    a.set(cut ?? [0, 0, 0, 0], 36); a.set([mode, thMax, cut ? 1 : 0, 0], 40);
    return a;
  }

  async renderCamera(name, { exposure = 0.01 } = {}) {
    const c = this.cams[name];
    if (!c) throw new Error(`no camera ${name}`);
    const t0 = performance.now();
    const L = c.rec.lens;
    const gain = exposure / 0.01;
    this.device.queue.writeBuffer(c.ub, 0, this._viewData(c.T_C_D, [L.K[0][0], L.K[1][1], L.K[0][2], L.K[1][2]], L.D, [c.W, c.H], 0.001, 0.6, 1, gain, null, 0, 1.5));
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: c.lin.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      depthStencilAttachment: { view: c.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
    });
    pass.setPipeline(this.pipeLinear);
    pass.setBindGroup(0, c.bg);
    this._drawAll(pass, true);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    this._time(`camera ${name}`, performance.now() - t0);
    return c.lin;
  }

  sensor(name) {
    const c = this.cams[name];
    const t0 = performance.now();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: c.sen.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.pipeSensor);
    pass.setBindGroup(0, c.sbg);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this._time(`sensor ${name}`, performance.now() - t0);
    return c.sen;
  }

  _drawAll(pass, camera = false) {
    for (const d of this.draws) {
      if (camera && d.lens) continue;             // a camera never images the lens fronts (its own sits on its pupil)
      pass.setVertexBuffer(0, d.vb); pass.setVertexBuffer(1, d.nb);
      pass.setIndexBuffer(d.ib, 'uint32');
      pass.drawIndexed(d.count, 1, 0, 0, d.obj);
    }
  }

  /** Same signature as gpu/renderer.js: view {eye, target, up?, fovy_deg, size: [w, h]}, opts {cut: {n, d} | null
   * (discard static surfaces with n.p > d), aid: false | {...} (a neutral headlight; feeds never use it)}. */
  renderOverview(view, opts = {}) {
    return this._pinhole(this.canvas, this.ctx, view, opts, 'overview');
  }

  /** A pinhole view into any canvas (the scope panel): attaches a context on first use. */
  renderPinhole(canvas, view, opts = {}) {
    if (!canvas._stubCtx) {
      canvas._stubCtx = canvas.getContext('webgpu');
      canvas._stubCtx.configure({ device: this.device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' });
    }
    return this._pinhole(canvas, canvas._stubCtx, view, opts, 'scope_view');
  }

  _pinhole(canvas, ctx, view, opts, label) {
    if (!ctx) return null;
    const t0 = performance.now();
    const W = canvas.width, H = canvas.height;
    const key = `${label}_${W}x${H}`;
    this._pin ??= {};
    let tg = this._pin[label];
    if (!tg || tg.key !== key) {
      tg?.color.destroy(); tg?.depth.destroy();
      tg = this._pin[label] = {
        key, color: this.device.createTexture({ size: [W, H], format: this.format ?? navigator.gpu.getPreferredCanvasFormat(), sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT }),
        depth: this.device.createTexture({ size: [W, H], format: 'depth24plus', sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT }),
        ub: this.device.createBuffer({ size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      };
      tg.bg = this.device.createBindGroup({ layout: this.bgl, entries: [{ binding: 0, resource: { buffer: tg.ub } }, { binding: 1, resource: { buffer: this.objBuf } }] });
    }
    if (!this.pipeOverview) this.pipeOverview = this._mkPipe(this.format ?? navigator.gpu.getPreferredCanvasFormat(), 4);
    const T_D_V = view.T_D_V ?? lookAtRows(view.eye, view.target, view.up ?? [0, 0, 1]);
    const f = 1 / Math.tan((view.fovy_deg * Math.PI / 180) / 2);
    const T_C_D = invRows(T_D_V);
    const cut = opts.cut ? [...opts.cut.n, opts.cut.d] : null;
    const save = [this.lightPos, this.lightDir];
    const aid = opts.aid !== false && opts.aid != null;
    if (aid) this.lightPos = [T_D_V[0][3], T_D_V[1][3], T_D_V[2][3]];
    this.device.queue.writeBuffer(tg.ub, 0, this._viewData(T_C_D, [f / (W / H), f, 0, 0], [0, 0, 0, 0], [W, H], 0.002, 3.0, aid ? 2 : 1, 1, cut, 1, 3.2));
    [this.lightPos, this.lightDir] = save;
    const enc = this.device.createCommandEncoder();
    const bg = aid ? { r: 0.953, g: 0.953, b: 0.937, a: 1 } : { r: 0, g: 0, b: 0, a: 1 };
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: tg.color.createView(), resolveTarget: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'discard', clearValue: bg }],
      depthStencilAttachment: { view: tg.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
    });
    pass.setPipeline(this.pipeOverview);
    pass.setBindGroup(0, tg.bg);
    this._drawAll(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this._time(label, performance.now() - t0);
    return null;
  }

  /** The exposure the stub uses (fixed; the real renderer runs the UVC auto-exposure loop). */
  exposureOf() { return { time_s: 0.010, gain: 1, auto: false }; }

  async readback(name, kind = 'rgb8') {
    const c = this.cams[name];
    const t0 = performance.now();
    const lin = kind === 'linear';
    const bpp = lin ? 16 : 4;
    const bpr = Math.ceil((c.W * bpp) / 256) * 256;
    const size = bpr * c.H;
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: lin ? c.lin : c.sen }, { buffer: buf, bytesPerRow: bpr }, [c.W, c.H]);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = buf.getMappedRange();
    let out;
    if (lin) {
      out = new Float32Array(c.W * c.H * 4);
      for (let y = 0; y < c.H; y++) out.set(new Float32Array(src, y * bpr, c.W * 4), y * c.W * 4);
    } else if (kind === 'rgba8') {
      out = new Uint8Array(c.W * c.H * 4);
      for (let y = 0; y < c.H; y++) out.set(new Uint8Array(src, y * bpr, c.W * 4), y * c.W * 4);
    } else {
      const s = new Uint8Array(src);
      out = new Uint8Array(c.W * c.H * 3);
      for (let y = 0; y < c.H; y++) {
        let i = y * bpr, o = y * c.W * 3;
        for (let x = 0; x < c.W; x++, i += 4, o += 3) { out[o] = s[i]; out[o + 1] = s[i + 1]; out[o + 2] = s[i + 2]; }
      }
    }
    buf.unmap(); buf.destroy();
    this._time(`readback ${name}`, performance.now() - t0);
    return out;
  }

  /** Occlusion: is the straight segment from the camera centre to each D point free of any surface? */
  visible(name, pointsD) {
    const c = this.cams[name];
    const T = c.rec.T_D_C;
    const o = [T[0][3], T[1][3], T[2][3]];
    const B = this.poses0?.bodies || {};
    return pointsD.map((p) => {
      const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
      const L = Math.hypot(...d);
      const u = d.map((x) => x / L);
      const tmax = L - 0.0015;                          // the point's own surface (a jaw tip) does not occlude it
      if (this.occStatic && traverse(this.occStatic.nodes, this.occStatic.T, o, u, { tmin: 0.002, tmax, anyHit: true }).t < Infinity) return false;
      for (const dyn of this.occDyn) {
        const M = B[dyn.body];
        if (!M) continue;
        // ray into the body frame: R^T (o - t), R^T u
        const ob = [o[0] - M[12], o[1] - M[13], o[2] - M[14]];
        const oL = [M[0] * ob[0] + M[1] * ob[1] + M[2] * ob[2], M[4] * ob[0] + M[5] * ob[1] + M[6] * ob[2], M[8] * ob[0] + M[9] * ob[1] + M[10] * ob[2]];
        const uL = [M[0] * u[0] + M[1] * u[1] + M[2] * u[2], M[4] * u[0] + M[5] * u[1] + M[6] * u[2], M[8] * u[0] + M[9] * u[1] + M[10] * u[2]];
        if (traverse(dyn.nodes, dyn.T, oL, uL, { tmin: 0.002, tmax, anyHit: true }).t < Infinity) return false;
      }
      return true;
    });
  }

  _time(pass, ms) {
    this.times.push({ pass, ms, t: performance.now() });
    if (this.times.length > 600) this.times.splice(0, this.times.length - 600);
  }

  /** Per-pass ms (CPU wall clock around submit -> done; the stub has no timestamp queries), last N frames. */
  timings() {
    const by = {};
    for (const r of this.times.slice(-300)) (by[r.pass] ??= []).push(r.ms);
    const out = {};
    for (const [k, v] of Object.entries(by)) { const s = [...v].sort((a, b) => a - b); out[k] = { last: v[v.length - 1], median: s[s.length >> 1], n: v.length }; }
    return { source: 'stub (CPU wall clock)', passes: out };
  }
}

function lookAtRows(eye, target, up) {
  const z = norm3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  let x = cross3(z, up);
  if (Math.hypot(...x) < 1e-9) x = [0, -1, 0];
  x = norm3(x);
  const y = cross3(z, x);
  return [[x[0], y[0], z[0], eye[0]], [x[1], y[1], z[1], eye[1]], [x[2], y[2], z[2], eye[2]], [0, 0, 0, 1]];
}
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

const IDENT = new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** Column-major inverse (T_C_D) of a row-major rigid T_D_C. */
function invRows(M) {
  const R = [[M[0][0], M[0][1], M[0][2]], [M[1][0], M[1][1], M[1][2]], [M[2][0], M[2][1], M[2][2]]];
  const t = [M[0][3], M[1][3], M[2][3]];
  const out = new Float32Array(16);
  // R^T and -R^T t, column-major
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[j * 4 + i] = R[j][i];
  for (let i = 0; i < 3; i++) out[12 + i] = -(R[0][i] * t[0] + R[1][i] * t[1] + R[2][i] * t[2]);
  out[15] = 1;
  return out;
}
