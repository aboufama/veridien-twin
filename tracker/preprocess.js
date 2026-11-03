// The tip net's input, bit-exact with the live server's Python (tipnet.TipNet.resize / preprocess with wb=True, on
// OpenCV 5.0 + numpy 2): a camera frame (ImageData-like, RGBA or RGB, any size; 1280x720 live) ->
//   1. an exact 2x INTER_AREA box when the frame is >= 2x the net input: (s00 + s01 + s10 + s11 + 2) >> 2;
//   2. INTER_LINEAR to 512x288 as OpenCV 5 runs it: a vertical pass rounded to an integer, then a horizontal pass
//      rounded again, 11-bit weights (for the 5:4 step of 1280x720 frames these are k/8, k in {1, 3, 5, 7}: verified
//      against cv2 on 15 session frames, 0 mismatches); other ratios follow the same rule (not verified);
//   3. white balance: per camera, every 10th set, per-channel gains g = clip(max(p97) / max(p97_c, 1), 1, 4) from
//      the 97th percentile of the [::4, ::4] subsample (numpy 'linear'), float32 throughout; no LUT when every gain
//      is < 1.03; LUT[v] = trunc(clip(v * g, 0, 255));
//   4. gray (cv2.COLOR_BGR2GRAY, OpenCV 5's 15-bit coefficients) for the shaft-darkness and steady stages.
// Images here are interleaved BGR Uint8Array (OpenCV order), gray is Uint8Array. Never resize with canvas drawImage:
// its filter is browser-defined.

import { f32 } from './math.js';

export const IN_W = 512, IN_H = 288;

/** Interleaved BGR (3 channels) copy of an ImageData-like {width, height, data} (RGBA, or RGB when
 *  data.length == 3wh; `order: 'bgr'` marks data that is already BGR). */
export function toBGR(img) {
  const { width: w, height: h, data } = img;
  const n = w * h, ch = img.channels ?? (data.length === 3 * n ? 3 : 4);
  const out = new Uint8Array(3 * n);
  const bgr = img.order === 'bgr';
  for (let i = 0, j = 0; i < n; i++, j += ch) {
    out[3 * i] = data[j + (bgr ? 0 : 2)];
    out[3 * i + 1] = data[j + 1];
    out[3 * i + 2] = data[j + (bgr ? 2 : 0)];
  }
  return out;
}

/** Exact 2x INTER_AREA of an ImageData-like frame straight from RGBA/RGB into BGR: (w/2) x (h/2). */
export function area2(img) {
  const { width: w, height: h, data } = img;
  const ch = img.channels ?? (data.length === 3 * w * h ? 3 : 4);
  const bgr = img.order === 'bgr';
  const ow = Math.floor(w / 2), oh = Math.floor(h / 2);
  const out = new Uint8Array(ow * oh * 3);
  const row = w * ch;
  const cB = bgr ? 0 : 2, cR = bgr ? 2 : 0;
  for (let y = 0; y < oh; y++) {
    const r0 = 2 * y * row, r1 = r0 + row;
    let o = y * ow * 3;
    for (let x = 0; x < ow; x++, o += 3) {
      const a = r0 + 2 * x * ch, b = r1 + 2 * x * ch;
      out[o] = (data[a + cB] + data[a + ch + cB] + data[b + cB] + data[b + ch + cB] + 2) >> 2;
      out[o + 1] = (data[a + 1] + data[a + ch + 1] + data[b + 1] + data[b + ch + 1] + 2) >> 2;
      out[o + 2] = (data[a + cR] + data[a + ch + cR] + data[b + cR] + data[b + ch + cR] + 2) >> 2;
    }
  }
  return { width: ow, height: oh, data: out };
}

/** Per-axis INTER_LINEAR taps: source index i0, i1 and 11-bit weights w0, w1 for each destination index (also the
 *  tables of the WebGPU preprocess, gpu_pre.js). */
export function taps(sn, dn) {
  const s = sn / dn;
  const i0 = new Int32Array(dn), i1 = new Int32Array(dn), w0 = new Int32Array(dn), w1 = new Int32Array(dn);
  for (let d = 0; d < dn; d++) {
    let f = (d + 0.5) * s - 0.5;
    let i = Math.floor(f);
    let fr = f - i;
    if (i < 0) { i = 0; fr = 0; }
    if (i >= sn - 1) { i = sn - 1; fr = 0; }
    const wb = Math.round(fr * 2048);
    i0[d] = i; i1[d] = Math.min(i + 1, sn - 1); w0[d] = 2048 - wb; w1[d] = wb;
  }
  return { i0, i1, w0, w1 };
}

/** INTER_LINEAR resize of interleaved BGR (OpenCV 5 order: vertical pass rounded, then horizontal pass rounded). */
export function resizeLinear(src, dw, dh) {
  const { width: sw, height: sh, data } = src;
  const tx = taps(sw, dw), ty = taps(sh, dh);
  const out = new Uint8Array(dw * dh * 3);
  const r = new Int32Array(sw * 3);
  const rowS = sw * 3;
  for (let y = 0; y < dh; y++) {
    const a = ty.i0[y] * rowS, b = ty.i1[y] * rowS, wa = ty.w0[y], wb = ty.w1[y];
    for (let x = 0; x < rowS; x++) r[x] = (data[a + x] * wa + data[b + x] * wb + 1024) >> 11;
    let o = y * dw * 3;
    for (let x = 0; x < dw; x++, o += 3) {
      const p = tx.i0[x] * 3, q = tx.i1[x] * 3, xa = tx.w0[x], xb = tx.w1[x];
      out[o] = (r[p] * xa + r[q] * xb + 1024) >> 11;
      out[o + 1] = (r[p + 1] * xa + r[q + 1] * xb + 1024) >> 11;
      out[o + 2] = (r[p + 2] * xa + r[q + 2] * xb + 1024) >> 11;
    }
  }
  return { width: dw, height: dh, data: out };
}

/** TipNet.resize: frame -> the 512x288 BGR net input (before white balance). */
export function toSmall(img, inW = IN_W, inH = IN_H) {
  let im;
  if (img.width >= 2 * inW && img.height >= 2 * inH) im = area2(img);
  else im = { width: img.width, height: img.height, data: toBGR(img) };
  if (im.width === inW && im.height === inH) return im;
  return resizeLinear(im, inW, inH);
}

/** numpy percentile(v, q, method 'linear') of uint8 values, cast to float32. */
export function percentileU8(vals, q) {
  const hist = new Uint32Array(256);
  for (let i = 0; i < vals.length; i++) hist[vals[i]]++;
  return percentileHist(hist, vals.length, q);
}

/** percentileU8 from a 256-bin histogram of n uint8 values (the same arithmetic; the WebGPU path's histograms). */
export function percentileHist(hist, n, q) {
  const at = (k) => { let c = 0; for (let v = 0; v < 256; v++) { c += hist[v]; if (c > k) return v; } return 255; };
  const idx = (n - 1) * (q / 100);
  const lo = Math.floor(idx), t = idx - lo;
  const a = at(lo), b = at(Math.min(lo + 1, n - 1)), diff = b - a;
  return f32(t >= 0.5 ? b - diff * (1 - t) : a + diff * t);
}

/** Histograms (Uint32Array 3 x 256, channels B, G, R) of the [::step, ::step] subsample of a BGR image, and its
 *  pixel count: what wbGains takes the percentiles of. */
export function subsampleHist(small, step = 4) {
  const { width: w, height: h, data } = small;
  const hist = new Uint32Array(768);
  let n = 0;
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step, n++) {
    const o = (y * w + x) * 3;
    hist[data[o]]++; hist[256 + data[o + 1]]++; hist[512 + data[o + 2]]++;
  }
  return { hist, n };
}

/** White-balance gains [gB, gG, gR] (float32 values) from subsample histograms (3 x 256, B, G, R) of n pixels. */
export function wbGainsFromHist(hist, n, q = 97, maxGain = 4) {
  const p = [0, 1, 2].map((c) => percentileHist(hist.subarray(c * 256, c * 256 + 256), n, q));
  const pm = Math.max(p[0], p[1], p[2]);
  return p.map((pc) => f32(Math.min(Math.max(f32(pm / Math.max(pc, 1)), 1), maxGain)));
}

/** White-balance gains [gB, gG, gR] (float32 values) of a 512x288 BGR image, tipnet's rule. */
export function wbGains(small, step = 4, q = 97, maxGain = 4) {
  const { hist, n } = subsampleHist(small, step);
  return wbGainsFromHist(hist, n, q, maxGain);
}

/** The LUT (Uint8Array 3 x 256, per channel B, G, R) for gains g, or null when every gain is < 1.03 (float32). */
export function wbLut(g) {
  const lim = f32(1.03);
  if (g.every((x) => x < lim)) return null;
  const lut = new Uint8Array(768);
  for (let c = 0; c < 3; c++) for (let v = 0; v < 256; v++) lut[c * 256 + v] = Math.trunc(Math.min(Math.max(f32(v * g[c]), 0), 255));
  return lut;
}

/** Apply a 3 x 256 LUT to interleaved BGR in place. */
export function applyLut(img, lut) {
  if (!lut) return img;
  const d = img.data;
  for (let i = 0; i < d.length; i += 3) {
    d[i] = lut[d[i]]; d[i + 1] = lut[256 + d[i + 1]]; d[i + 2] = lut[512 + d[i + 2]];
  }
  return img;
}

/** Per-camera white balance as TipNet.preprocess runs it (wb=True): gains refreshed when n % 10 == 1 (n counts
 *  predict calls since start; never reset) or the camera has no LUT yet. Keyed by camera name (Python keys by batch
 *  slot, which is the same while all three cameras are present). */
export class WhiteBalance {
  constructor({ every = 10, step = 4, q = 97, maxGain = 4 } = {}) {
    Object.assign(this, { every, step, q, maxGain });
    this.n = 0;
    this.lut = new Map();
    this.gains = new Map();
  }
  /** Start of a predict call (one frame set). */
  tick() { this.n += 1; }
  /** Whether this set refreshes the camera's gains (n % every == 1, or no LUT yet). */
  due(name) { return this.n % this.every === 1 || !this.lut.has(name); }
  /** Set a camera's gains (and LUT): from wbGains, or from the GPU's subsample histograms (wbGainsFromHist). */
  set(name, g) { this.gains.set(name, g); this.lut.set(name, wbLut(g)); }
  /** White-balance one camera's small image in place; returns it. */
  apply(name, small) {
    if (this.due(name)) this.set(name, wbGains(small, this.step, this.q, this.maxGain));
    return applyLut(small, this.lut.get(name));
  }
}

/** cv2.cvtColor(BGR2GRAY) on uint8, OpenCV 5: Y = (B*3735 + G*19235 + R*9798 + 16384) >> 15. */
export function gray(img) {
  const { width: w, height: h, data } = img;
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < out.length; i++, j += 3) out[i] = (data[j] * 3735 + data[j + 1] * 19235 + data[j + 2] * 9798 + 16384) >> 15;
  return { width: w, height: h, data: out };
}

/** The net input tensor: Float32Array [batch, 3, h, w] of BGR values 0..255 (no normalisation: the ONNX folds it);
 *  slots beyond the given images stay zero. */
export function toTensor(smalls, batch = 3, w = IN_W, h = IN_H) {
  const plane = w * h;
  const x = new Float32Array(batch * 3 * plane);
  smalls.forEach((im, i) => {
    if (!im) return;
    const d = im.data, o = i * 3 * plane;
    for (let p = 0, j = 0; p < plane; p++, j += 3) {
      x[o + p] = d[j]; x[o + plane + p] = d[j + 1]; x[o + 2 * plane + p] = d[j + 2];
    }
  });
  return x;
}

/** Round half to even to uint8 (OpenCV saturate_cast of a float). */
function roundU8(v) {
  const r = Math.round(v);
  const e = (r - v === 0.5 && (r & 1)) ? r - 1 : r;
  return e < 0 ? 0 : e > 255 ? 255 : e;
}

/** cv2.remap INTER_LINEAR of a uint8 gray image at one float32 point (OpenCV 5: float bilinear, rounded half to
 *  even); `border`: 'replicate' clamps the neighbours, 'constant' reads 0 outside. */
export function sampleBilinear(g, x, y, border = 'replicate') {
  const xf = f32(x), yf = f32(y);
  const w = g.width, h = g.height, d = g.data;
  const x0 = Math.floor(xf), y0 = Math.floor(yf);
  const fx = xf - x0, fy = yf - y0;
  const px = (xx, yy) => {
    if (border === 'replicate') {
      xx = xx < 0 ? 0 : xx >= w ? w - 1 : xx;
      yy = yy < 0 ? 0 : yy >= h ? h - 1 : yy;
      return d[yy * w + xx];
    }
    return (xx < 0 || yy < 0 || xx >= w || yy >= h) ? 0 : d[yy * w + xx];
  };
  const v = (1 - fx) * (1 - fy) * px(x0, y0) + fx * (1 - fy) * px(x0 + 1, y0) + (1 - fx) * fy * px(x0, y0 + 1) + fx * fy * px(x0 + 1, y0 + 1);
  return roundU8(v);
}
