// Baseline JPEG encoder for the web twin's simulated MJPG frames (docs/WEB_TWIN.md §9 step 5).
//
// Why our own: canvas.toBlob / OffscreenCanvas.convertToBlob cannot be given quant tables, and the twin must
// encode with the real camera's own tables (lifted from its MJPG stream with parseJPEG / tablesFromJPEG).
//
// Format: SOI, JFIF APP0, optional COM, DQT (2 tables), SOF0 (8-bit baseline, YCbCr 4:2:0: Y 2x2, Cb 1x1,
// Cr 1x1), DHT (the four standard tables), optional DRI, SOS, entropy-coded data, EOI.
// Colour: JFIF full-range BT.601 in libjpeg's own fixed point; chroma is box-averaged over 2x2 with libjpeg's
// h2v2 bias, so the 8-bit samples entering the DCT are libjpeg's. DCT: the separable AAN float FDCT, quantised
// with libjpeg's float rounding (half up). Edges are replicated to the 16x16 MCU grid, as libjpeg does inside
// the image area.
//
// Licence: written for Veridien from the ITU-T T.81 standard; no third-party code is copied. The quantisation
// and Huffman tables are the example tables of T.81 Annex K. The IJG quality scaling (qualityTables) follows the
// documented libjpeg rule so its tables match cv2.imencode's exactly (checked by web/twin/tests/ui_jpeg.test.mjs).

// ------------------------------------------------------------------------------------------- tables

/** Annex K.1 luminance quantisation table, natural (row-major) order. */
export const STD_LUMA_Q = Uint8Array.from([
  16, 11, 10, 16, 24, 40, 51, 61,
  12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56,
  14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77,
  24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101,
  72, 92, 95, 98, 112, 100, 103, 99,
]);

/** Annex K.2 chrominance quantisation table, natural order. */
export const STD_CHROMA_Q = Uint8Array.from([
  17, 18, 24, 47, 99, 99, 99, 99,
  18, 21, 26, 66, 99, 99, 99, 99,
  24, 26, 56, 99, 99, 99, 99, 99,
  47, 66, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
]);

/** ZIGZAG[k] = natural index of the k-th coefficient in zigzag order (T.81 Figure A.6). */
export const ZIGZAG = (() => {
  const zz = new Uint8Array(64);
  let k = 0;
  for (let s = 0; s < 15; s++) {
    const lo = Math.max(0, s - 7), hi = Math.min(s, 7);
    if (s % 2 === 1) for (let r = lo; r <= hi; r++) zz[k++] = r * 8 + (s - r);
    else for (let r = hi; r >= lo; r--) zz[k++] = r * 8 + (s - r);
  }
  return zz;
})();

// Annex K.3 Huffman tables: BITS[1..16] (count of codes per length) and HUFFVAL.
const DC_LUMA_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_LUMA_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const DC_CHROMA_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_CHROMA_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUMA_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_LUMA_VALS = [
  0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
  0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
  0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
  0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
  0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
  0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
  0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
  0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
  0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
  0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa,
];
const AC_CHROMA_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHROMA_VALS = [
  0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
  0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
  0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
  0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
  0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
  0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
  0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
  0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
  0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
  0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa,
];

/** Code and size per symbol (T.81 Annex C). */
function huffCodes(bits, vals) {
  let n = 0;
  for (const b of bits) n += b;
  if (n !== vals.length) throw new Error(`Huffman table: ${n} codes for ${vals.length} values`);
  const code = new Uint16Array(256), size = new Uint8Array(256);
  let c = 0, k = 0;
  for (let len = 1; len <= 16; len++) {
    for (let i = 0; i < bits[len - 1]; i++) { code[vals[k]] = c; size[vals[k]] = len; k++; c++; }
    c <<= 1;
  }
  return { code, size };
}
const H_DC_L = huffCodes(DC_LUMA_BITS, DC_LUMA_VALS);
const H_AC_L = huffCodes(AC_LUMA_BITS, AC_LUMA_VALS);
const H_DC_C = huffCodes(DC_CHROMA_BITS, DC_CHROMA_VALS);
const H_AC_C = huffCodes(AC_CHROMA_BITS, AC_CHROMA_VALS);

/** IJG quality scaling of the Annex K tables (libjpeg jpeg_set_quality, force_baseline): natural order. */
export function qualityTables(quality = 90) {
  let q = Math.round(quality);
  q = Math.min(100, Math.max(1, q));
  const scale = q < 50 ? Math.floor(5000 / q) : 200 - 2 * q;
  const f = (base) => Uint8Array.from(base, (b) => Math.min(255, Math.max(1, Math.floor((b * scale + 50) / 100))));
  return { qY: f(STD_LUMA_Q), qC: f(STD_CHROMA_Q) };
}

function checkTable(t, what) {
  if (!t || t.length !== 64) throw new Error(`${what}: 64 quant values needed (natural order)`);
  const out = new Uint8Array(64);
  for (let i = 0; i < 64; i++) {
    const v = t[i];
    if (!Number.isInteger(v) || v < 1 || v > 255) throw new Error(`${what}[${i}] = ${v}: baseline quant values are integers 1..255`);
    out[i] = v;
  }
  return out;
}

// AAN scale factors: cos(k·π/16)·√2 for k > 0, 1 for k = 0.
const AAN = Float64Array.from([1.0, 1.387039845, 1.306562965, 1.175875602, 1.0, 0.785694958, 0.541196100, 0.275899379]);

/** Reciprocal divisors, in ZIGZAG order, folding in the AAN output scaling (libjpeg jcdctmgr float path). */
function divisors(q) {
  const d = new Float64Array(64);
  for (let k = 0; k < 64; k++) {
    const i = ZIGZAG[k];
    d[k] = 1.0 / (q[i] * AAN[i >> 3] * AAN[i & 7] * 8.0);
  }
  return d;
}

// RGB -> YCbCr in libjpeg's fixed point (jccolor.c: 16 fractional bits, rounded), so the samples entering the
// DCT are the same 8-bit values libjpeg (and a camera's JPEG engine) would encode.
const FIX = (x) => Math.round(x * 65536);
const HALF = 1 << 15, CBCR_OFF = 128 << 16;
const T_YR = new Int32Array(256), T_YG = new Int32Array(256), T_YB = new Int32Array(256);
const T_UR = new Int32Array(256), T_UG = new Int32Array(256), T_UB = new Int32Array(256);
const T_VG = new Int32Array(256), T_VB = new Int32Array(256);
for (let v = 0; v < 256; v++) {
  T_YR[v] = FIX(0.29900) * v; T_YG[v] = FIX(0.58700) * v; T_YB[v] = FIX(0.11400) * v + HALF;
  T_UR[v] = -FIX(0.16874) * v; T_UG[v] = -FIX(0.33126) * v;
  T_UB[v] = FIX(0.50000) * v + CBCR_OFF + HALF - 1;               // also the R term of Cr (both 0.5)
  T_VG[v] = -FIX(0.41869) * v; T_VB[v] = -FIX(0.08131) * v;
}

// Per-size scratch planes, reused between frames (one module instance per worker, encode is synchronous).
let scratch = null;
function planes(Wp, Hp) {
  if (!scratch || scratch.Wp !== Wp || scratch.Hp !== Hp) {
    const n = (Wp * Hp) >> 2;
    scratch = { Wp, Hp, Y: new Float32Array(Wp * Hp), U: new Float32Array(n), V: new Float32Array(n),
      Us: new Int32Array(n), Vs: new Int32Array(n) };
  }
  return scratch;
}

// ------------------------------------------------------------------------------------------- encoder

/**
 * Encode packed RGB (or RGBA with opts.channels = 4) as a baseline 4:2:0 JPEG.
 * @param {Uint8Array|Uint8ClampedArray} rgb  w*h*channels bytes, row-major, top row first
 * @param {number} width
 * @param {number} height
 * @param {{quality?: number, qY?: ArrayLike<number>, qC?: ArrayLike<number>, restartInterval?: number,
 *          comment?: string, channels?: 3|4}} [opts]
 * @returns {Uint8Array} the JPEG file
 */
export function encodeJPEG(rgb, width, height, opts = {}) {
  const w = width | 0, h = height | 0;
  if (w < 1 || h < 1 || w > 65535 || h > 65535 || w !== width || h !== height) throw new Error(`bad size ${width}x${height}`);
  const ch = opts.channels ?? 3;
  if (ch !== 3 && ch !== 4) throw new Error('channels must be 3 or 4');
  if (!rgb || rgb.length < w * h * ch) throw new Error(`need ${w * h * ch} bytes, got ${rgb?.length}`);
  const std = (opts.qY && opts.qC) ? null : qualityTables(opts.quality ?? 90);
  const qY = checkTable(opts.qY ?? std.qY, 'qY');
  const qC = checkTable(opts.qC ?? std.qC, 'qC');
  const restart = Math.max(0, opts.restartInterval | 0);
  if (restart > 65535) throw new Error('restartInterval must be <= 65535');

  // ---- colour conversion (8-bit YCbCr, as libjpeg) + 2x2 chroma average on the MCU-padded grid (edges replicated)
  const Wp = (w + 15) & ~15, Hp = (h + 15) & ~15, Cw = Wp >> 1;
  const P = planes(Wp, Hp);
  const Y = P.Y, U = P.U, V = P.V, Us = P.Us, Vs = P.Vs;
  Us.fill(0); Vs.fill(0);
  for (let y = 0; y < Hp; y++) {
    const sy = y < h ? y : h - 1;
    const srow = sy * w * ch;
    const yrow = y * Wp;
    const crow = (y >> 1) * Cw;
    for (let x = 0; x < Wp; x++) {
      const s = srow + (x < w ? x : w - 1) * ch;
      const r = rgb[s], g = rgb[s + 1], b = rgb[s + 2];
      Y[yrow + x] = ((T_YR[r] + T_YG[g] + T_YB[b]) >> 16) - 128;
      const ci = crow + (x >> 1);
      Us[ci] += (T_UR[r] + T_UG[g] + T_UB[b]) >> 16;
      Vs[ci] += (T_UB[r] + T_VG[g] + T_VB[b]) >> 16;
    }
  }
  // libjpeg h2v2_downsample: (sum of 4 + bias) >> 2, bias alternating 1, 2 along the row
  for (let i = 0, n = Us.length; i < n; i++) {
    const bias = (i & 1) + 1;            // Cw is even, so column parity = index parity
    U[i] = ((Us[i] + bias) >> 2) - 128;
    V[i] = ((Vs[i] + bias) >> 2) - 128;
  }

  return entropyEncode(Y, U, V, Wp, Hp, w, h, qY, qC, restart, opts);
}

/**
 * Encode 4:2:0 YCbCr planes as they come from a camera ISP (the renderer's readback(name, 'yuv420'): full-range
 * JFIF Y, Cb, Cr, chroma already subsampled to ceil(w/2) x ceil(h/2)), so chroma is not subsampled twice
 * (docs/WEB_TWIN.md §9 step 5). Same tables, markers and entropy coder as encodeJPEG.
 * @param {{y: Uint8Array, cb: Uint8Array, cr: Uint8Array, width: number, height: number}} yuv
 */
export function encodeJPEGYUV420(yuv, opts = {}) {
  const w = yuv.width | 0, h = yuv.height | 0;
  if (w < 1 || h < 1 || w > 65535 || h > 65535) throw new Error(`bad size ${yuv.width}x${yuv.height}`);
  const cw = (w + 1) >> 1, chh = (h + 1) >> 1;
  if (!yuv.y || yuv.y.length < w * h || !yuv.cb || yuv.cb.length < cw * chh || !yuv.cr || yuv.cr.length < cw * chh) {
    throw new Error(`yuv420 planes too small for ${w}x${h}`);
  }
  const std = (opts.qY && opts.qC) ? null : qualityTables(opts.quality ?? 90);
  const qY = checkTable(opts.qY ?? std.qY, 'qY');
  const qC = checkTable(opts.qC ?? std.qC, 'qC');
  const restart = Math.max(0, opts.restartInterval | 0);
  if (restart > 65535) throw new Error('restartInterval must be <= 65535');
  // whole MCUs (1280x720 is 80 x 45): read the planes in place, no padded copy (byte-identical, ui_jpeg.test.mjs)
  if ((w & 15) === 0 && (h & 15) === 0 && !opts.reference) return encodeYUV420Fast(yuv, w, h, qY, qC, restart, opts);
  const Wp = (w + 15) & ~15, Hp = (h + 15) & ~15, Cw = Wp >> 1, Ch = Hp >> 1;
  const P = planes(Wp, Hp);
  const Y = P.Y, U = P.U, V = P.V;
  for (let y = 0; y < Hp; y++) {
    const srow = (y < h ? y : h - 1) * w, yrow = y * Wp;
    for (let x = 0; x < Wp; x++) Y[yrow + x] = yuv.y[srow + (x < w ? x : w - 1)] - 128;
  }
  for (let y = 0; y < Ch; y++) {
    const srow = (y < chh ? y : chh - 1) * cw, crow = y * Cw;
    for (let x = 0; x < Cw; x++) {
      const si = srow + (x < cw ? x : cw - 1);
      U[crow + x] = yuv.cb[si] - 128;
      V[crow + x] = yuv.cr[si] - 128;
    }
  }
  return entropyEncode(Y, U, V, Wp, Hp, w, h, qY, qC, restart, opts);
}

/** SOI, JFIF APP0, optional COM, DQT (both tables), SOF0 4:2:0, DHT (Annex K), optional DRI, SOS. */
function jpegHeader(w, h, qY, qC, restart, comment) {
  const c = comment ? new TextEncoder().encode(String(comment)).subarray(0, 65533) : null;
  const out = new Uint8Array(1024 + (c ? c.length + 4 : 0));
  let pos = 0;
  const byte = (v) => { out[pos++] = v; };
  const u16 = (v) => { out[pos++] = (v >> 8) & 0xff; out[pos++] = v & 0xff; };
  u16(0xffd8);                                                        // SOI
  u16(0xffe0); u16(16);                                               // APP0 JFIF 1.01, aspect 1:1, no thumbnail
  byte(0x4a); byte(0x46); byte(0x49); byte(0x46); byte(0);
  byte(1); byte(1); byte(0); u16(1); u16(1); byte(0); byte(0);
  if (c) { u16(0xfffe); u16(c.length + 2); out.set(c, pos); pos += c.length; }
  u16(0xffdb); u16(2 + 2 * 65);                                       // DQT, both tables, 8-bit, zigzag order
  byte(0x00); for (let k = 0; k < 64; k++) byte(qY[ZIGZAG[k]]);
  byte(0x01); for (let k = 0; k < 64; k++) byte(qC[ZIGZAG[k]]);
  u16(0xffc0); u16(17); byte(8); u16(h); u16(w); byte(3);             // SOF0
  byte(1); byte(0x22); byte(0);
  byte(2); byte(0x11); byte(1);
  byte(3); byte(0x11); byte(1);
  const dht = [[0x00, DC_LUMA_BITS, DC_LUMA_VALS], [0x10, AC_LUMA_BITS, AC_LUMA_VALS],
    [0x01, DC_CHROMA_BITS, DC_CHROMA_VALS], [0x11, AC_CHROMA_BITS, AC_CHROMA_VALS]];
  u16(0xffc4); u16(2 + dht.reduce((a, [, , v]) => a + 17 + v.length, 0));   // DHT
  for (const [tc, b, v] of dht) { byte(tc); for (const x of b) byte(x); for (const x of v) byte(x); }
  if (restart) { u16(0xffdd); u16(4); u16(restart); }                 // DRI
  u16(0xffda); u16(12); byte(3);                                      // SOS
  byte(1); byte(0x00); byte(2); byte(0x11); byte(3); byte(0x11);
  byte(0); byte(63); byte(0);
  return out.subarray(0, pos);
}

function entropyEncode(Y, U, V, Wp, Hp, w, h, qY, qC, restart, opts) {
  const Cw = Wp >> 1;
  // ---- output buffer and bit writer
  let out = new Uint8Array(Math.max(1 << 16, (w * h) >> 1));
  let pos = 0, bitBuf = 0, bitCnt = 0;
  const ensure = (n) => {
    if (pos + n <= out.length) return;
    let cap = out.length * 2;
    while (cap < pos + n) cap *= 2;
    const o = new Uint8Array(cap);
    o.set(out.subarray(0, pos));
    out = o;
  };
  const byte = (v) => { out[pos++] = v; };
  const u16 = (v) => { out[pos++] = (v >> 8) & 0xff; out[pos++] = v & 0xff; };
  const bits = (code, size) => {          // size <= 16, bitCnt < 8 on entry: at most 23 bits held
    bitBuf = (bitBuf << size) | code;
    bitCnt += size;
    while (bitCnt >= 8) {
      const b = (bitBuf >>> (bitCnt - 8)) & 0xff;
      out[pos++] = b;
      if (b === 0xff) out[pos++] = 0;      // byte stuffing
      bitCnt -= 8;
    }
    bitBuf &= (1 << bitCnt) - 1;
  };
  const flushBits = () => { if (bitCnt > 0) bits((1 << (8 - bitCnt)) - 1, 8 - bitCnt); };

  // ---- headers
  const hdr = jpegHeader(w, h, qY, qC, restart, opts.comment);
  ensure(hdr.length);
  out.set(hdr, 0); pos = hdr.length;

  // ---- blocks
  const dY = divisors(qY), dC = divisors(qC);
  const ws = new Float64Array(64);
  const zz = new Int32Array(64);
  let predY = 0, predU = 0, predV = 0;

  const block = (plane, stride, x0, y0, div, pred, dcH, acH) => {
    // load
    for (let r = 0, o = y0 * stride + x0; r < 8; r++, o += stride) {
      const i = r * 8;
      ws[i] = plane[o]; ws[i + 1] = plane[o + 1]; ws[i + 2] = plane[o + 2]; ws[i + 3] = plane[o + 3];
      ws[i + 4] = plane[o + 4]; ws[i + 5] = plane[o + 5]; ws[i + 6] = plane[o + 6]; ws[i + 7] = plane[o + 7];
    }
    // AAN float FDCT: rows, then columns (outputs scaled by 8·AAN[u]·AAN[v], folded into the divisors)
    for (let p = 0; p < 64; p += 8) fdct8(ws, p, 1);
    for (let p = 0; p < 8; p++) fdct8(ws, p, 8);
    // quantise in zigzag order (libjpeg float rounding: half up)
    let last = 0;
    for (let k = 0; k < 64; k++) {
      const v = ((ws[ZIGZAG[k]] * div[k] + 16384.5) | 0) - 16384;
      zz[k] = v;
      if (v !== 0) last = k;
    }
    // DC
    const dc = zz[0];
    let diff = dc - pred;
    if (diff === 0) bits(dcH.code[0], dcH.size[0]);
    else {
      const n = 32 - Math.clz32(diff < 0 ? -diff : diff);
      bits(dcH.code[n], dcH.size[n]);
      if (diff < 0) diff = (diff - 1) & ((1 << n) - 1);
      bits(diff, n);
    }
    // AC
    let run = 0;
    for (let k = 1; k <= last; k++) {
      let v = zz[k];
      if (v === 0) { run++; continue; }
      while (run > 15) { bits(acH.code[0xf0], acH.size[0xf0]); run -= 16; }
      const a = v < 0 ? -v : v;
      const n = 32 - Math.clz32(a);
      const sym = (run << 4) | n;
      bits(acH.code[sym], acH.size[sym]);
      if (v < 0) v = (v - 1) & ((1 << n) - 1);
      bits(v, n);
      run = 0;
    }
    if (last < 63) bits(acH.code[0x00], acH.size[0x00]);         // EOB
    return dc;
  };

  const mcuCols = Wp >> 4, mcuRows = Hp >> 4;
  let mcu = 0, rst = 0;
  for (let my = 0; my < mcuRows; my++) {
    ensure(mcuCols * 6 * 450 + 64);       // worst case per block (~ 64 x 27 bits, doubled by stuffing) with margin
    for (let mx = 0; mx < mcuCols; mx++) {
      if (restart && mcu > 0 && mcu % restart === 0) {
        flushBits();
        byte(0xff); byte(0xd0 + (rst & 7)); rst++;
        predY = predU = predV = 0;
      }
      const x0 = mx << 4, y0 = my << 4;
      predY = block(Y, Wp, x0, y0, dY, predY, H_DC_L, H_AC_L);
      predY = block(Y, Wp, x0 + 8, y0, dY, predY, H_DC_L, H_AC_L);
      predY = block(Y, Wp, x0, y0 + 8, dY, predY, H_DC_L, H_AC_L);
      predY = block(Y, Wp, x0 + 8, y0 + 8, dY, predY, H_DC_L, H_AC_L);
      predU = block(U, Cw, x0 >> 1, y0 >> 1, dC, predU, H_DC_C, H_AC_C);
      predV = block(V, Cw, x0 >> 1, y0 >> 1, dC, predV, H_DC_C, H_AC_C);
      mcu++;
    }
  }
  flushBits();
  ensure(2);
  u16(0xffd9);                                                        // EOI
  return out.slice(0, pos);
}

// The fast 4:2:0 path: the same tables, the same AAN float DCT and rounding and the same Huffman codes as
// entropyEncode, so the bytes are identical; it reads the u8 planes in place (w and h multiples of 16, so no MCU
// crosses the edge), keeps the bit writer in locals and emits each Huffman code with its value bits in one go.
const HF = (() => {
  const mk = (b, v) => { const { code, size } = huffCodes(b, v); return { code: Int32Array.from(code), size: Int32Array.from(size) }; };
  return { dl: mk(DC_LUMA_BITS, DC_LUMA_VALS), al: mk(AC_LUMA_BITS, AC_LUMA_VALS), dc: mk(DC_CHROMA_BITS, DC_CHROMA_VALS), ac: mk(AC_CHROMA_BITS, AC_CHROMA_VALS) };
})();
const FWS = new Float64Array(64), FZZ = new Int32Array(64);
let fastOut = null;

function encodeYUV420Fast(yuv, w, h, qY, qC, restart, opts) {
  const Y = yuv.y, CB = yuv.cb, CR = yuv.cr, cw = w >> 1, mcuCols = w >> 4, mcuRows = h >> 4;
  const hdr = jpegHeader(w, h, qY, qC, restart, opts.comment);
  const worst = hdr.length + mcuCols * mcuRows * 6 * 450 + 64 + (restart ? 2 * Math.ceil(mcuCols * mcuRows / restart) : 0);
  if (!fastOut || fastOut.length < worst) fastOut = new Uint8Array(worst);
  const out = fastOut;
  out.set(hdr, 0);
  let pos = hdr.length, bb = 0, bc = 0;
  let pY = 0, pU = 0, pV = 0, mcu = 0, rst = 0;
  const dY = divisors(qY), dC = divisors(qC), ws = FWS, zz = FZZ;
  const dlC = HF.dl.code, dlS = HF.dl.size, alC = HF.al.code, alS = HF.al.size;
  const dcC0 = HF.dc.code, dcS0 = HF.dc.size, acC0 = HF.ac.code, acS0 = HF.ac.size;
  for (let my = 0; my < mcuRows; my++) {
    for (let mx = 0; mx < mcuCols; mx++) {
      if (restart && mcu > 0 && mcu % restart === 0) {
        if (bc > 0) { const pad = 8 - bc; bb = (bb << pad) | ((1 << pad) - 1); const t = bb & 255; out[pos++] = t; if (t === 255) out[pos++] = 0; bb = 0; bc = 0; }
        out[pos++] = 0xff; out[pos++] = 0xd0 + (rst & 7); rst++;
        pY = pU = pV = 0;
      }
      mcu++;
      for (let b = 0; b < 6; b++) {
        let plane, stride, o0, div, dcC, dcS, acC, acS;
        if (b < 4) { plane = Y; stride = w; o0 = ((my << 4) + ((b >> 1) << 3)) * w + (mx << 4) + ((b & 1) << 3); div = dY; dcC = dlC; dcS = dlS; acC = alC; acS = alS; }
        else { plane = b === 4 ? CB : CR; stride = cw; o0 = (my << 3) * cw + (mx << 3); div = dC; dcC = dcC0; dcS = dcS0; acC = acC0; acS = acS0; }
        for (let r = 0, o = o0; r < 8; r++, o += stride) {
          const i = r * 8;
          ws[i] = plane[o] - 128; ws[i + 1] = plane[o + 1] - 128; ws[i + 2] = plane[o + 2] - 128; ws[i + 3] = plane[o + 3] - 128;
          ws[i + 4] = plane[o + 4] - 128; ws[i + 5] = plane[o + 5] - 128; ws[i + 6] = plane[o + 6] - 128; ws[i + 7] = plane[o + 7] - 128;
        }
        for (let p = 0; p < 64; p += 8) fdct8(ws, p, 1);
        for (let p = 0; p < 8; p++) fdct8(ws, p, 8);
        let last = 0;
        for (let k = 0; k < 64; k++) { const v = ((ws[ZIGZAG[k]] * div[k] + 16384.5) | 0) - 16384; zz[k] = v; if (v !== 0) last = k; }
        // DC: the category code and the value bits in one emission (split when longer than 23 bits; bc < 8 on entry)
        const dc = zz[0], pred = b < 4 ? pY : b === 4 ? pU : pV;
        let diff = dc - pred, n = 0;
        if (diff !== 0) { n = 32 - Math.clz32(diff < 0 ? -diff : diff); if (diff < 0) diff = (diff - 1) & ((1 << n) - 1); }
        let code = (dcC[n] << n) | diff, size = dcS[n] + n;
        if (size > 23) { const hi = size - 16; bb = (bb << hi) | (code >>> 16); bc += hi; while (bc >= 8) { const t = (bb >>> (bc - 8)) & 255; out[pos++] = t; if (t === 255) out[pos++] = 0; bc -= 8; } bb &= (1 << bc) - 1; code &= 0xffff; size = 16; }
        bb = (bb << size) | code; bc += size; while (bc >= 8) { const t = (bb >>> (bc - 8)) & 255; out[pos++] = t; if (t === 255) out[pos++] = 0; bc -= 8; } bb &= (1 << bc) - 1;
        if (b < 4) pY = dc; else if (b === 4) pU = dc; else pV = dc;
        // AC: run/size symbol and value bits in one emission, ZRL for runs over 15, EOB unless the last is non-zero
        let run = 0;
        for (let k = 1; k <= last; k++) {
          let v = zz[k];
          if (v === 0) { run++; continue; }
          while (run > 15) { bb = (bb << acS[0xf0]) | acC[0xf0]; bc += acS[0xf0]; while (bc >= 8) { const t = (bb >>> (bc - 8)) & 255; out[pos++] = t; if (t === 255) out[pos++] = 0; bc -= 8; } bb &= (1 << bc) - 1; run -= 16; }
          const a = v < 0 ? -v : v, m = 32 - Math.clz32(a), sym = (run << 4) | m;
          if (v < 0) v = (v - 1) & ((1 << m) - 1);
          code = (acC[sym] << m) | v; size = acS[sym] + m;
          if (size > 23) { const hi = size - 16; bb = (bb << hi) | (code >>> 16); bc += hi; while (bc >= 8) { const t = (bb >>> (bc - 8)) & 255; out[pos++] = t; if (t === 255) out[pos++] = 0; bc -= 8; } bb &= (1 << bc) - 1; code &= 0xffff; size = 16; }
          bb = (bb << size) | code; bc += size; while (bc >= 8) { const t = (bb >>> (bc - 8)) & 255; out[pos++] = t; if (t === 255) out[pos++] = 0; bc -= 8; } bb &= (1 << bc) - 1;
          run = 0;
        }
        if (last < 63) { bb = (bb << acS[0]) | acC[0]; bc += acS[0]; while (bc >= 8) { const t = (bb >>> (bc - 8)) & 255; out[pos++] = t; if (t === 255) out[pos++] = 0; bc -= 8; } bb &= (1 << bc) - 1; }
      }
    }
  }
  if (bc > 0) { const pad = 8 - bc; bb = (bb << pad) | ((1 << pad) - 1); const t = bb & 255; out[pos++] = t; if (t === 255) out[pos++] = 0; }
  out[pos++] = 0xff; out[pos++] = 0xd9;                                // EOI
  return out.slice(0, pos);
}

/** One 8-point AAN forward DCT, in place on d[p + s·i], i = 0..7 (unnormalised: see divisors()). */
function fdct8(d, p, s) {
  const i0 = p, i1 = p + s, i2 = p + 2 * s, i3 = p + 3 * s, i4 = p + 4 * s, i5 = p + 5 * s, i6 = p + 6 * s, i7 = p + 7 * s;
  const t0 = d[i0] + d[i7], t7 = d[i0] - d[i7];
  const t1 = d[i1] + d[i6], t6 = d[i1] - d[i6];
  const t2 = d[i2] + d[i5], t5 = d[i2] - d[i5];
  const t3 = d[i3] + d[i4], t4 = d[i3] - d[i4];
  // even part
  const t10 = t0 + t3, t13 = t0 - t3, t11 = t1 + t2, t12 = t1 - t2;
  d[i0] = t10 + t11;
  d[i4] = t10 - t11;
  const z1 = (t12 + t13) * 0.707106781;
  d[i2] = t13 + z1;
  d[i6] = t13 - z1;
  // odd part
  const o10 = t4 + t5, o11 = t5 + t6, o12 = t6 + t7;
  const z5 = (o10 - o12) * 0.382683433;
  const z2 = 0.541196100 * o10 + z5;
  const z4 = 1.306562965 * o12 + z5;
  const z3 = o11 * 0.707106781;
  const z11 = t7 + z3, z13 = t7 - z3;
  d[i5] = z13 + z2;
  d[i3] = z13 - z2;
  d[i1] = z11 + z4;
  d[i7] = z11 - z4;
}

// ------------------------------------------------------------------------------------------- parser

/**
 * Read the header markers of a JPEG (a real camera's MJPG frame, or ours): size, components, quant tables
 * (natural order), whether it carries DHT (UVC MJPG frames often do not), and the restart interval.
 * Stops at SOS. 16-bit quant tables come back as Uint16Array.
 */
export function parseJPEG(bytes) {
  const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (b[0] !== 0xff || b[1] !== 0xd8) throw new Error('not a JPEG (no SOI)');
  const res = { width: 0, height: 0, precision: 0, sof: null, components: [], qTables: {}, hasDHT: false, restartInterval: 0, comments: [] };
  let i = 2;
  while (i + 4 <= b.length) {
    if (b[i] !== 0xff) throw new Error(`marker expected at byte ${i}`);
    let m = b[i + 1];
    while (m === 0xff && i + 2 < b.length) { i++; m = b[i + 1]; }     // fill bytes
    if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { i += 2; continue; }
    if (m === 0xd9) break;
    const len = (b[i + 2] << 8) | b[i + 3];
    const s = i + 4, e = i + 2 + len;
    if (e > b.length) throw new Error(`segment 0x${m.toString(16)} runs past the end`);
    if (m === 0xdb) {
      let j = s;
      while (j < e) {
        const pq = b[j] >> 4, tq = b[j] & 15;
        j++;
        const t = pq ? new Uint16Array(64) : new Uint8Array(64);
        for (let k = 0; k < 64; k++) t[ZIGZAG[k]] = pq ? (b[j + 2 * k] << 8) | b[j + 2 * k + 1] : b[j + k];
        j += pq ? 128 : 64;
        res.qTables[tq] = t;
      }
    } else if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {
      res.sof = m;
      res.precision = b[s];
      res.height = (b[s + 1] << 8) | b[s + 2];
      res.width = (b[s + 3] << 8) | b[s + 4];
      const n = b[s + 5];
      res.components = [];
      for (let c = 0; c < n; c++) {
        const o = s + 6 + 3 * c;
        res.components.push({ id: b[o], h: b[o + 1] >> 4, v: b[o + 1] & 15, tq: b[o + 2] });
      }
    } else if (m === 0xc4) {
      res.hasDHT = true;
    } else if (m === 0xdd) {
      res.restartInterval = (b[s] << 8) | b[s + 1];
    } else if (m === 0xfe) {
      res.comments.push(new TextDecoder().decode(b.subarray(s, e)));
    } else if (m === 0xda) {
      break;
    }
    i = e;
  }
  return res;
}

/** The {qY, qC} a camera's MJPG frame was encoded with, ready to pass to encodeJPEG. */
export function tablesFromJPEG(bytes) {
  const p = parseJPEG(bytes);
  const y = p.components[0], c = p.components[1] ?? p.components[0];
  const qY = p.qTables[y?.tq], qC = p.qTables[c?.tq];
  if (!qY || !qC) throw new Error('JPEG has no quant table for its components');
  if (qY instanceof Uint16Array || qC instanceof Uint16Array) throw new Error('16-bit quant tables are not baseline');
  return { qY: Uint8Array.from(qY), qC: Uint8Array.from(qC) };
}
