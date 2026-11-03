// The tip net's input on the GPU (ADR-150): preprocess.js's exact integer chain as WebGPU compute shaders, so a
// camera frame goes VideoFrame / ImageBitmap -> GPU texture (copyExternalImageToTexture) -> the net input buffer the
// ONNX session reads (IO binding), with no pixel round trip through the CPU:
//   resize  2x INTER_AREA ((a + b + c + d + 2) >> 2), then INTER_LINEAR as OpenCV 5 runs it (vertical pass rounded,
//           horizontal pass rounded, 11-bit taps from preprocess.taps), packed B | G << 8 | R << 16 per pixel;
//   hist    3 x 256 histograms of the [::4, ::4] subsample (only on the sets that refresh the white balance; the
//           gains themselves are preprocess.wbGainsFromHist on the CPU, float32 as numpy);
//   pack    the per-slot LUT (identity when off), the float32 NCHW net input, and the OpenCV 5 BGR2GRAY of the
//           LUT'd image ((B 3735 + G 19235 + R 9798 + 16384) >> 15), 4 px per u32.
// One readback per set: the net output and the grays share one staging buffer (stagingLayout). Pure helpers (the
// layout, the tables) are tested in node; the shaders are judged by the self-test (selftest.js) and the browser judge.

import { taps, IN_W, IN_H } from './preprocess.js';

export const FW = 1280, FH = 720, N = 3, OW = IN_W / 4, OH = IN_H / 4;
export const PLANE = IN_W * IN_H, OPLANE = OW * OH;
export const IN_LEN = N * 3 * PLANE, OUT_LEN = N * 6 * OPLANE;
export const N_HIST = (IN_H / 4) * (IN_W / 4);

/** Byte layout of the one staging buffer read back per set, for n cameras: the net output of slots 0..n-1 (float32),
 *  then their grays (u8, PLANE each). Offsets 8-aligned and sizes 4-aligned (WebGPU's copy and map rules). */
export function stagingLayout(n = N) {
  if (!(n >= 1 && n <= N)) throw new Error(`stagingLayout: ${n} cameras (1..${N})`);
  const yBytes = n * 6 * OPLANE * 4, grayOff = yBytes, grayBytes = n * PLANE;
  return { n, yOff: 0, yBytes, grayOff, grayBytes, total: grayOff + grayBytes, max: N * 6 * OPLANE * 4 + N * PLANE };
}

/** INTER_LINEAR taps as one Int32Array of (i0, i1, w0, w1) per destination index (the shader's table). */
export function tapTable(src, dst) {
  const t = taps(src, dst), a = new Int32Array(4 * dst);
  for (let d = 0; d < dst; d++) { a[4 * d] = t.i0[d]; a[4 * d + 1] = t.i1[d]; a[4 * d + 2] = t.w0[d]; a[4 * d + 3] = t.w1[d]; }
  return a;
}

/** The per-slot LUT table (Uint32Array 3 slots x 768: B, G, R) from preprocess.wbLut results; null -> identity. */
export function lutTable(luts, out = new Uint32Array(N * 768)) {
  for (let i = 0; i < N; i++) {
    const l = luts[i] ?? null;
    for (let k = 0; k < 768; k++) out[i * 768 + k] = l ? l[k] : k & 255;
  }
  return out;
}

/** Unpack the shader's packed small image (u32 B | G << 8 | R << 16) of one slot into interleaved BGR bytes. */
export function unpackSmall(words, slot = 0) {
  const out = new Uint8Array(3 * PLANE), o = slot * PLANE;
  for (let p = 0; p < PLANE; p++) { const v = words[o + p]; out[3 * p] = v & 255; out[3 * p + 1] = (v >>> 8) & 255; out[3 * p + 2] = (v >>> 16) & 255; }
  return out;
}

const W2 = FW / 2, H2 = FH / 2;
export const WGSL_RESIZE = /* wgsl */`
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> tx: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> ty: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read_write> small: array<u32>;
fn px(l: i32, x: i32, y: i32) -> vec3<i32> { return vec3<i32>(round(textureLoad(src, vec2<i32>(x, y), l, 0).bgr * 255.0)); }
fn area(l: i32, x: i32, y: i32) -> vec3<i32> {
  return (px(l, 2 * x, 2 * y) + px(l, 2 * x + 1, 2 * y) + px(l, 2 * x, 2 * y + 1) + px(l, 2 * x + 1, 2 * y + 1) + vec3<i32>(2)) >> vec3<u32>(2u);
}
fn vert(l: i32, x: i32, t: vec4<i32>) -> vec3<i32> {
  return (area(l, x, t.x) * t.z + area(l, x, t.y) * t.w + vec3<i32>(1024)) >> vec3<u32>(11u);
}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= ${IN_W}u || g.y >= ${IN_H}u) { return; }
  let l = i32(g.z); let a = tx[g.x]; let b = ty[g.y];
  let v = (vert(l, a.x, b) * a.z + vert(l, a.y, b) * a.w + vec3<i32>(1024)) >> vec3<u32>(11u);
  small[(g.z * ${IN_H}u + g.y) * ${IN_W}u + g.x] = u32(v.x) | (u32(v.y) << 8u) | (u32(v.z) << 16u);
}`;
export const WGSL_HIST = /* wgsl */`
@group(0) @binding(0) var<storage, read> small: array<u32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;
@compute @workgroup_size(16, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= ${IN_W / 4}u || g.y >= ${IN_H / 4}u) { return; }
  let v = small[(g.z * ${IN_H}u + 4u * g.y) * ${IN_W}u + 4u * g.x];
  let h = g.z * 768u;
  atomicAdd(&hist[h + (v & 255u)], 1u); atomicAdd(&hist[h + 256u + ((v >> 8u) & 255u)], 1u); atomicAdd(&hist[h + 512u + ((v >> 16u) & 255u)], 1u);
}`;
export const WGSL_PACK = /* wgsl */`
@group(0) @binding(0) var<storage, read> small: array<u32>;
@group(0) @binding(1) var<storage, read> lut: array<u32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> gray: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let q = g.x; if (q >= ${(N * PLANE) / 4}u) { return; }
  let cam = (4u * q) / ${PLANE}u; let lb = cam * 768u; let xb = cam * ${3 * PLANE}u;
  var gw = 0u;
  for (var k = 0u; k < 4u; k++) {
    let i = 4u * q + k; let p = i - cam * ${PLANE}u; let v = small[i];
    let b = lut[lb + (v & 255u)]; let gg = lut[lb + 256u + ((v >> 8u) & 255u)]; let r = lut[lb + 512u + ((v >> 16u) & 255u)];
    x[xb + p] = f32(b); x[xb + ${PLANE}u + p] = f32(gg); x[xb + ${2 * PLANE}u + p] = f32(r);
    gw |= (((b * 3735u + gg * 19235u + r * 9798u + 16384u) >> 15u) & 255u) << (8u * k);
  }
  gray[q] = gw;
}`;

/** The GPU preprocess on one device (onnxruntime-web's own, so the net reads `x` in place). */
export class GpuPre {
  constructor(device) {
    const d = (this.device = device), U = GPUBufferUsage;
    this.tex = d.createTexture({ size: [FW, FH, N], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    const buf = (a) => { const b = d.createBuffer({ size: a.byteLength, usage: U.STORAGE | U.COPY_DST }); d.queue.writeBuffer(b, 0, a); return b; };
    this.tx = buf(tapTable(W2, IN_W)); this.ty = buf(tapTable(H2, IN_H));
    this.small = d.createBuffer({ size: N * PLANE * 4, usage: U.STORAGE | U.COPY_SRC });
    this.hist = d.createBuffer({ size: N * 768 * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.histRead = d.createBuffer({ size: N * 768 * 4, usage: U.MAP_READ | U.COPY_DST });
    this.lut = d.createBuffer({ size: N * 768 * 4, usage: U.STORAGE | U.COPY_DST });
    this.x = d.createBuffer({ size: IN_LEN * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.gray = d.createBuffer({ size: N * PLANE, usage: U.STORAGE | U.COPY_SRC });
    const pipe = (code) => d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint: 'main' } });
    this.pR = pipe(WGSL_RESIZE); this.pH = pipe(WGSL_HIST); this.pP = pipe(WGSL_PACK);
    const bg = (p, rs) => d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: rs.map((r, i) => ({ binding: i,
      resource: r === this.tex ? r.createView({ dimension: '2d-array' }) : { buffer: r } })) });
    this.bR = bg(this.pR, [this.tex, this.tx, this.ty, this.small]);
    this.bH = bg(this.pH, [this.small, this.hist]);
    this.bP = bg(this.pP, [this.small, this.lut, this.x, this.gray]);
    this.lutHost = new Uint32Array(N * 768);
  }

  /** Frames (VideoFrame / ImageBitmap / OffscreenCanvas, FW x FH) into texture layers 0..n-1. */
  upload(sources) {
    sources.forEach((s, l) => this.device.queue.copyExternalImageToTexture({ source: s }, { texture: this.tex, origin: [0, 0, l] }, [FW, FH]));
  }

  /** Resize n layers (and their histograms when asked) in one submission; -> histograms (Uint32Array n x 768) or null. */
  async resize(n, withHist = false) {
    const d = this.device, e = d.createCommandEncoder();
    let c = e.beginComputePass();
    c.setPipeline(this.pR); c.setBindGroup(0, this.bR); c.dispatchWorkgroups(IN_W / 16, IN_H / 16, n); c.end();
    if (!withHist) { d.queue.submit([e.finish()]); return null; }
    e.clearBuffer(this.hist);
    c = e.beginComputePass();
    c.setPipeline(this.pH); c.setBindGroup(0, this.bH); c.dispatchWorkgroups(IN_W / 4 / 16, IN_H / 4 / 8, n); c.end();
    e.copyBufferToBuffer(this.hist, 0, this.histRead, 0, N * 768 * 4);
    d.queue.submit([e.finish()]);
    await this.histRead.mapAsync(GPUMapMode.READ);
    const h = new Uint32Array(this.histRead.getMappedRange().slice(0, n * 768 * 4));
    this.histRead.unmap();
    return h;
  }

  /** LUT + pack + gray for all slots; slots n.. of the net input are zeroed (as toTensor pads a missing camera). */
  pack(n, luts) {
    const d = this.device;
    d.queue.writeBuffer(this.lut, 0, lutTable(luts, this.lutHost));
    const e = d.createCommandEncoder(), c = e.beginComputePass();
    c.setPipeline(this.pP); c.setBindGroup(0, this.bP); c.dispatchWorkgroups(Math.ceil(N * PLANE / 4 / 64)); c.end();
    if (n < N) e.clearBuffer(this.x, n * 3 * PLANE * 4, (N - n) * 3 * PLANE * 4);
    d.queue.submit([e.finish()]);
  }

  /** Debug / self-test readback: {resize: [BGR u8 per slot], x: Float32Array, grays: [u8 per slot]} of slots 0..n-1. */
  async read(n = N) {
    const d = this.device, U = GPUBufferUsage, sb = N * PLANE * 4, xb = IN_LEN * 4, gb = N * PLANE;
    const rd = d.createBuffer({ size: sb + xb + gb, usage: U.MAP_READ | U.COPY_DST }), e = d.createCommandEncoder();
    e.copyBufferToBuffer(this.small, 0, rd, 0, sb); e.copyBufferToBuffer(this.x, 0, rd, sb, xb); e.copyBufferToBuffer(this.gray, 0, rd, sb + xb, gb);
    d.queue.submit([e.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const sm = new Uint32Array(rd.getMappedRange(0, sb).slice(0)), x = new Float32Array(rd.getMappedRange(sb, xb).slice(0));
    const g = new Uint8Array(rd.getMappedRange(sb + xb, gb).slice(0));
    rd.unmap(); rd.destroy();
    const slots = Array.from({ length: n }, (_, i) => i);
    return { resize: slots.map((i) => unpackSmall(sm, i)), x, grays: slots.map((i) => g.slice(i * PLANE, (i + 1) * PLANE)) };
  }

  destroy() { for (const b of [this.tex, this.tx, this.ty, this.small, this.hist, this.histRead, this.lut, this.x, this.gray]) b.destroy?.(); }
}
