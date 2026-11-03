// The WebGPU path's start-up self-test, pure parts (ADR-150). The tracker uses the WebGPU tip net (gpu_net.js: frames
// copied to the GPU, preprocess in WGSL, the ONNX forward through onnxruntime-web's WebGPU EP with IO binding) only
// when, on this machine, it reproduces the live Python chain bit for bit on a fixed input:
//   three integer-formula 1280x720 frames (selftestFrames; slot 2 grey, so one white balance is the identity) ->
//   GPU resize, white-balance gains, gray and the packed net input must equal the exact JS preprocess (preprocess.js,
//   itself md5-equal to TipNet.preprocess), and the net output must equal cv2.dnn's (the live server's forward),
//   all as SHA-256 digests in selftest.json (scripts/make_tracker_selftest.py). Anything else -> WASM, with the reason.
// No DOM, no GPU here: the generator, the JS reference, the digest comparison and the backend decision.

import { toSmall, WhiteBalance, gray, toTensor } from './preprocess.js';

export const SELFTEST_SCHEMA = 'veridien.tracker.selftest/1';
const pmod = (a, m) => ((a % m) + m) % m;

/** make_tracker_selftest.frame(k): RGBA ImageData-like frames (w x h), one per shift; slot `greySlot` copies G to B, R. */
export function selftestFrames({ shifts = [0, 97, 211], greySlot = 2, w = 1280, h = 720 } = {}) {
  return shifts.map((sh, k) => {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) for (let x0 = 0; x0 < w; x0++) {
      const x = x0 + sh;
      let b = pmod(x * 7 + y * 3 + pmod(x * y, 23), 256);
      let g = pmod(x * 2 + y * 5 + pmod(Math.floor(x / 9) * Math.floor(y / 7), 31), 256);
      let r = pmod(128 + pmod(x - 2 * y, 97) + pmod(Math.floor((x + y) / 5), 29), 256);
      if (pmod(Math.floor(x / 40) + Math.floor(y / 30), 5) === 0) [b, g, r] = [200, 230, 240];
      if (Math.abs(3 * x - 4 * y - 600) < 90) [b, g, r] = [b >> 2, g >> 2, r >> 2];
      if (k === greySlot) b = r = g;
      const o = (y * w + x0) * 4;
      data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = 255;
    }
    return { width: w, height: h, data };
  });
}

/** The exact JS chain on the frames: {resize: [BGR u8], smalls: [WB'd BGR u8], grays: [u8], gains, x: Float32Array}. */
export function jsReference(frames) {
  const wb = new WhiteBalance();
  wb.tick();
  const resize = frames.map((f) => toSmall(f));
  const smalls = frames.map((f, i) => wb.apply(`S${i}`, toSmall(f)));
  return { resize: resize.map((s) => s.data), smalls: smalls.map((s) => s.data), grays: smalls.map((s) => gray(s).data),
    gains: frames.map((_, i) => wb.gains.get(`S${i}`)), x: toTensor(smalls) };
}

const u8 = (a) => (a instanceof Uint8Array ? a : new Uint8Array(a.buffer, a.byteOffset, a.byteLength));

/** SHA-256 hex of a typed array's bytes (WebCrypto: browsers in a secure context, workers, node >= 19). */
export async function sha256Hex(a, subtle = globalThis.crypto?.subtle) {
  const d = new Uint8Array(await subtle.digest('SHA-256', u8(a)));
  return Array.from(d, (v) => v.toString(16).padStart(2, '0')).join('');
}

/** Digests of one run (the GPU's or the JS's): {resize, small?, gray, x?, y?} arrays per slot or one tensor. */
export async function digests({ resize, smalls, grays, x, y, ySlots = 3 }) {
  const all = (l) => (l ? Promise.all(l.map((a) => sha256Hex(a))) : null);
  const out = { resize_sha256: await all(resize), small_sha256: await all(smalls), gray_sha256: await all(grays),
    x_sha256: x ? await sha256Hex(x) : null, y_sha256: y ? await sha256Hex(y) : null };
  if (y) {
    const n = y.length / ySlots;
    out.y_slot_sha256 = await Promise.all(Array.from({ length: ySlots }, (_, i) => sha256Hex(y.subarray(i * n, (i + 1) * n))));
  }
  return out;
}

/**
 * Compare a run's digests and gains with the reference (selftest.json). Only the fields present in `got` are checked.
 * -> {ok, fails: ['y slot 1', ...], checked: [...]}.
 */
export function compareSelftest(got, ref) {
  const fails = [], checked = [];
  if (!ref || ref.schema !== SELFTEST_SCHEMA) return { ok: false, fails: [`reference schema ${ref?.schema}`], checked };
  for (const k of ['resize', 'small', 'gray']) {
    const g = got[`${k}_sha256`];
    if (!g) continue;
    checked.push(k);
    g.forEach((h, i) => { if (h !== ref[`${k}_sha256`][i]) fails.push(`${k} slot ${i}`); });
  }
  if (got.wb_gains) {
    checked.push('wb gains');
    got.wb_gains.forEach((g, i) => { if (!g || g.some((v, j) => v !== ref.wb_gains[i][j])) fails.push(`wb gains slot ${i}`); });
  }
  if (got.x_sha256) { checked.push('x'); if (got.x_sha256 !== ref.x_sha256) fails.push('x'); }
  if (got.y_sha256) {
    checked.push('y');
    if (got.y_sha256 !== ref.y_sha256) {
      const bad = (got.y_slot_sha256 ?? []).map((h, i) => (h !== ref.y_slot_sha256[i] ? i : -1)).filter((i) => i >= 0);
      fails.push(bad.length ? `y slot ${bad.join(', ')}` : 'y');
    }
  }
  return { ok: fails.length === 0 && checked.length > 0, fails, checked };
}

/**
 * Which backend the tracker runs. forced: the page's ?ep ('wasm' | 'webgpu' | null/'' / 'auto' for the default);
 * gpu: {available: bool, error?: string, selftest?: compareSelftest result}. -> {ep: 'webgpu' | 'wasm', reason}.
 * Default: WebGPU only when it is available and its self-test passed; ?ep=wasm forces WASM; ?ep=webgpu keeps WebGPU
 * even when the self-test failed (debugging; the reason says so), but not when it could not start.
 */
export function decideBackend(forced, gpu = {}) {
  const f = forced == null || forced === '' || forced === 'auto' ? null : String(forced).toLowerCase();
  if (f === 'wasm') return { ep: 'wasm', reason: 'forced by ?ep=wasm' };
  if (f && f !== 'webgpu') return { ep: 'wasm', reason: `unknown ?ep=${forced}; WASM` };
  if (!gpu.available) return { ep: 'wasm', reason: `WebGPU unavailable: ${gpu.error ?? 'no adapter'}` };
  const st = gpu.selftest;
  if (st?.ok) return { ep: 'webgpu', reason: `self-test bit-identical (${st.checked.join(', ')})` };
  const why = st ? `self-test differs: ${st.fails.join('; ')}` : 'self-test did not run';
  if (f === 'webgpu') return { ep: 'webgpu', reason: `forced by ?ep=webgpu although the ${why}` };
  return { ep: 'wasm', reason: `WebGPU ${why}` };
}
