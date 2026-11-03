// The tip net on WebGPU, end to end on the GPU (ADR-150): onnxruntime-web's WebGPU EP with IO binding (the input is
// GpuPre's buffer, the output a buffer of ours: Tensor.fromGpuBuffer both ways, preferredOutputLocation gpu-buffer)
// fed by gpu_pre.js from the frames themselves. No graph capture (enableGraphCapture): on this Mac (Chrome 152,
// onnxruntime-web 1.30) its replays left the bound output holding the previous run's result on every recorded set,
// so the readback saw stale tips; `capture: true` turns it on for experiments, and the self-test's second, permuted
// pass catches that failure. Per set: copyExternalImageToTexture of each frame,
// resize (+ histograms on white-balance sets), LUT + pack + gray, the forward, and one readback (net output + grays).
// onnxruntime-web 1.30 allows one WebGPU session per JS context: this module makes exactly one.
//
// API (createGpuTipNet): {ep: 'webgpu', run(x) -> y (a CPU input, for frames the GPU path does not take),
//   runFrames(sources, wb) -> {y, grays}, checkFrames(sources, images) -> {ok, slots}, selftest(ref) -> report,
//   release()}. The decision to use it is selftest.decideBackend's, made in net.js.

import { GpuPre, stagingLayout, FW, FH, N, IN_LEN, OUT_LEN, PLANE, OPLANE, N_HIST } from './gpu_pre.js';
import { toSmall, WhiteBalance, wbGainsFromHist } from './preprocess.js';
import { selftestFrames, digests, compareSelftest } from './selftest.js';

const DIMS_IN = [N, 3, 288, 512], DIMS_OUT = [N, 6, 72, 128];
const now = () => performance.now();

export async function createGpuTipNet({ ort, bytes, capture = false }) {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
    preferredOutputLocation: 'gpu-buffer', enableGraphCapture: !!capture });
  const device = await ort.env.webgpu.device;
  const U = GPUBufferUsage;
  const pre = new GpuPre(device);
  const yBuf = device.createBuffer({ size: OUT_LEN * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
  const xT = ort.Tensor.fromGpuBuffer(pre.x, { dataType: 'float32', dims: DIMS_IN });
  const yT = ort.Tensor.fromGpuBuffer(yBuf, { dataType: 'float32', dims: DIMS_OUT });
  const stage = device.createBuffer({ size: stagingLayout(N).max, usage: U.MAP_READ | U.COPY_DST });
  const yHost = new Float32Array(OUT_LEN);
  let lost = null;
  device.lost?.then((i) => { if (lost) return; lost = i?.message || 'device lost'; console.warn('[tip net] WebGPU device lost:', lost); });
  const timing = {};

  /** forward + the one readback: net output of slots 0..n-1 into yHost (the rest zero), grays of slots 0..n-1. */
  async function forward(n, withGray) {
    if (lost) throw new Error(`WebGPU device lost (${lost})`);
    let t = now();
    await session.run({ x: xT }, { y: yT });
    timing.run_ms = now() - t; t = now();
    const L = stagingLayout(n), e = device.createCommandEncoder();
    e.copyBufferToBuffer(yBuf, 0, stage, L.yOff, L.yBytes);
    if (withGray) e.copyBufferToBuffer(pre.gray, 0, stage, L.grayOff, L.grayBytes);
    device.queue.submit([e.finish()]);
    await stage.mapAsync(GPUMapMode.READ, 0, withGray ? L.total : L.yBytes);
    yHost.set(new Float32Array(stage.getMappedRange(L.yOff, L.yBytes)));
    if (n < N) yHost.fill(0, n * 6 * OPLANE);
    let grays = null;
    if (withGray) {
      const g = new Uint8Array(stage.getMappedRange(L.grayOff, L.grayBytes));
      grays = Array.from({ length: n }, (_, i) => g.slice(i * PLANE, (i + 1) * PLANE));
    }
    stage.unmap();
    timing.readback_ms = now() - t;
    return { y: yHost, grays };
  }

  /** A CPU-built net input (toTensor) -> y: the JS-preprocess path on the same session. */
  async function run(x) {
    device.queue.writeBuffer(pre.x, 0, x.buffer, x.byteOffset, IN_LEN * 4);
    return (await forward(N, false)).y;
  }

  /**
   * Frames (n <= 3, FW x FH, rig order of the cameras present) -> {y, grays}. wb: {need: bool (some camera refreshes
   * its gains this set), onHist(hists: Uint32Array n x 768, nPix), luts() -> [3 x 256 LUT | null per slot]}.
   */
  async function runFrames(sources, wb) {
    const n = sources.length;
    let t = now();
    pre.upload(sources);
    const h = await pre.resize(n, !!wb.need);
    if (h) wb.onHist(h, N_HIST);
    pre.pack(n, wb.luts());
    timing.pre_ms = now() - t;
    return forward(n, true);
  }

  /** The frames' resize on the GPU vs preprocess.toSmall of the CPU's pixels of the same frames (images: ImageData-
   *  like RGBA): whether copyExternalImageToTexture sees the pixels the canvas path does. -> {ok, slots: [equal?]}. */
  async function checkFrames(sources, images) {
    pre.upload(sources);
    await pre.resize(sources.length, false);
    const r = await pre.read(sources.length);
    const slots = images.map((im, i) => { const a = toSmall(im).data, b = r.resize[i]; return a.length === b.length && a.every((v, k) => v === b[k]); });
    return { ok: slots.every(Boolean), slots };
  }

  /** The start-up self-test (selftest.js): -> {ok, fails, checked, ms, y_abs_sum}. */
  async function selftest(ref) {
    const t0 = now();
    const frames = selftestFrames({ shifts: ref?.frames?.shifts, greySlot: ref?.frames?.grey_slot });
    const bms = await Promise.all(frames.map((f) => createImageBitmap(new ImageData(f.data, f.width, f.height),
      { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })));
    const pass = async (order) => {
      const wb = new WhiteBalance(), names = order.map((k) => `S${k}`);
      wb.tick();
      const o = await runFrames(order.map((k) => bms[k]), { need: true,
        onHist: (hh, np) => names.forEach((nm, i) => wb.set(nm, wbGainsFromHist(hh.subarray(i * 768, (i + 1) * 768), np))),
        luts: () => names.map((nm) => wb.lut.get(nm)) });
      return { y: o.y.slice(), grays: o.grays, gains: names.map((nm) => wb.gains.get(nm)) };
    };
    const a = await pass([0, 1, 2]), r = await pre.read(N);
    const got = await digests({ resize: r.resize, grays: a.grays, x: r.x, y: a.y });
    got.wb_gains = a.gains;
    const cmp = compareSelftest(got, ref);
    let abs = 0;
    for (let i = 0; i < a.y.length; i++) abs += Math.abs(a.y[i]);
    // a second pass with the slots rotated: every slot's output must move with its frame (a stale or reordered
    // output buffer fails here; one forward on the same input would not show it)
    const order = [1, 2, 0], b = await pass(order);
    bms.forEach((bm) => bm.close());
    const gb = await digests({ grays: b.grays, y: b.y });
    const rot = { ...ref, gray_sha256: order.map((k) => ref.gray_sha256[k]), y_slot_sha256: order.map((k) => ref.y_slot_sha256[k]), y_sha256: null };
    order.forEach((k, i) => { if (gb.y_slot_sha256[i] !== rot.y_slot_sha256[i]) cmp.fails.push(`rotated pass: y slot ${i}`); });
    order.forEach((k, i) => { if (gb.gray_sha256[i] !== rot.gray_sha256[i]) cmp.fails.push(`rotated pass: gray slot ${i}`); });
    if (cmp.fails.length) cmp.ok = false;
    cmp.checked.push('rotated pass');
    return { ...cmp, ms: Math.round(now() - t0), y_abs_sum: abs, y_abs_sum_ref: ref?.y_abs_sum ?? null };
  }

  /** Free the session, our buffers and the device (onnxruntime-web's, used by nothing else in this context: the
   *  WASM fallback is a separate bundle). Idempotent; the net is unusable afterwards. */
  let released = false;
  function release() {
    if (released) return;
    released = true; lost ??= 'released';
    try { session.release?.(); } catch { /* */ }
    try { pre.destroy(); yBuf.destroy(); stage.destroy(); } catch { /* */ }
    try { device.destroy?.(); } catch { /* */ }
  }

  return { ep: 'webgpu', ort, device, session, pre, run, runFrames, checkFrames, selftest, release, timing, size: [FW, FH],
    get lost() { return lost; } };
}
