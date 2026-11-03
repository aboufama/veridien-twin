// The tip net (TipNet w7, assets/multicam/tipnet/w7/best.onnx) under onnxruntime-web (MIT; ADR in docs/DECISIONS.md),
// pinned to an exact version on jsDelivr. The ONNX batch is fixed at 3 ([3, 3, 288, 512] float32 BGR 0..255 in;
// y [3, 6, 72, 128] logits + offsets out).
//
// Backend (ADR-150, replacing ADR-147's WASM default): WebGPU end to end (gpu_net.js: frames copied to the GPU, the
// exact preprocess in WGSL, the forward with IO binding, one readback), but only when its start-up
// self-test (selftest.js) reproduces the live Python chain bit for bit on this machine; otherwise the plain SIMD WASM
// bundle (not the asyncify one inside ort.webgpu.min.mjs: 1-4x slower), the backend scripts/judge_browser_tracker.mjs
// judges. ?ep=wasm forces WASM; ?ep=webgpu keeps WebGPU even when the self-test fails (debugging). GitHub Pages is not
// cross-origin isolated, so WASM runs on one thread there.
//
// Node tests and the judge inject `ort` (the onnxruntime-web module, or onnxruntime-node) and the model bytes: that
// path is WASM (or the injected ep) with no self-test, as before.

import { decideBackend } from './selftest.js';

export const ORT_VERSION = '1.30.0';
export const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
export const NET_BATCH = 3;
export const NET_INPUT = 'x', NET_OUTPUT = 'y';

/** Try the WebGPU net and its self-test. -> {gpu: {available, error?, selftest?}, net: the GPU net or null}. */
export async function probeWebGpu(bytes, { nav = globalThis.navigator, fetch: fetchFn = globalThis.fetch, ref = null,
  importer = importOrt, create = null, capture = false } = {}) {
  let net = null;
  try {
    if (!nav?.gpu) throw new Error('no navigator.gpu');
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter');
    const ort = await importer('webgpu');
    const make = create ?? (await import('./gpu_net.js')).createGpuTipNet;
    net = await make({ ort, bytes, capture });
    const want = ref ?? await (await fetchFn(new URL('./selftest.json', import.meta.url).href, { cache: 'no-cache' })).json();
    const selftest = await net.selftest(want);
    return { gpu: { available: true, selftest }, net };
  } catch (err) {
    try { net?.release(); } catch { /* */ }
    return { gpu: { available: false, error: String(err?.message ?? err).slice(0, 300) }, net: null };
  }
}

/** Import the matching onnxruntime-web bundle from the pinned CDN dist (browser). */
export async function importOrt(ep) {
  const url = ORT_DIST + (ep === 'webgpu' ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs');
  const mod = await import(/* @vite-ignore */ url);
  const ort = mod.default ?? mod;
  if (ort.env?.wasm) ort.env.wasm.wasmPaths = ORT_DIST;
  return ort;
}

/**
 * Load the tip net. opts: {model: URL string | ArrayBuffer | Uint8Array, ort?: injected module, ep?: null/'auto'
 * (default: WebGPU iff its self-test passes) | 'wasm' | 'webgpu', numThreads?, fetch?, selftestRef?, graphCapture?
 * (experiments only; see gpu_net.js)}. -> {ep,
 * run(x: Float32Array [3*3*288*512]) -> Promise<Float32Array y>, release(), backend: {ep, reason}, selftest?,
 * gpu? (the WebGPU net: runFrames / checkFrames, for the frames path)}.
 */
export async function createTipNet(opts = {}) {
  const fetchFn = opts.fetch ?? globalThis.fetch;
  let bytes = opts.model;
  if (typeof bytes === 'string' || bytes instanceof URL) bytes = new Uint8Array(await (await fetchFn(String(bytes))).arrayBuffer());
  else if (bytes instanceof ArrayBuffer) bytes = new Uint8Array(bytes);
  const tryEp = async (ep, ort0 = null) => {
    const ort = ort0 ?? opts.ort ?? await importOrt(ep);
    if (ort.env?.wasm) {
      const coi = globalThis.crossOriginIsolated === true;
      ort.env.wasm.numThreads = opts.numThreads ?? (coi ? Math.min(4, globalThis.navigator?.hardwareConcurrency || 1) : 1);
    }
    const session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' });
    const shape = [NET_BATCH, 3, 288, 512];
    const run = async (x) => {
      const res = await session.run({ [NET_INPUT]: new ort.Tensor('float32', x, shape) });
      const y = res[NET_OUTPUT];
      const d = y.data instanceof Float32Array ? y.data : await y.getData();
      return d;
    };
    await run(new Float32Array(NET_BATCH * 3 * 288 * 512));          // warm-up
    return { ep, run, release: () => session.release?.(), ort };
  };
  if (opts.ort) {                                                       // node: the injected runtime, no self-test
    const net = await tryEp(opts.ep && opts.ep !== 'auto' ? opts.ep : 'wasm');
    net.backend = { ep: net.ep, reason: 'injected runtime' };
    return net;
  }
  let probe = { gpu: { available: false, error: 'not tried' }, net: null };
  if (String(opts.ep ?? '').toLowerCase() !== 'wasm') probe = await probeWebGpu(bytes, { fetch: fetchFn, ref: opts.selftestRef, capture: !!opts.graphCapture });
  const backend = decideBackend(opts.ep, probe.gpu);
  if (backend.ep === 'webgpu') {
    const g = probe.net;
    console.info('[tip net] WebGPU:', backend.reason);
    return { ep: 'webgpu', run: g.run, release: g.release, gpu: g, backend, selftest: probe.gpu.selftest };
  }
  probe.net?.release();
  if (String(opts.ep ?? '').toLowerCase() !== 'wasm') console.warn('[tip net] WASM:', backend.reason);
  const net = await tryEp('wasm');
  return Object.assign(net, { backend, selftest: probe.gpu.selftest ?? null });
}
