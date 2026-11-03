// The pure parts of the in-page ("browser") tracker source (twinpage/browser_source.js, tracker_worker.js): no DOM, no
// worker globals, so node:test runs them (web/twin/tests/browser_source.test.mjs) and the worker imports them as is.
//   finalizePayload   the in-page tracker's result made exactly the shape of the server's GET /api/tips
//   displayWB         per-camera display white balance, the server's display_wb rule (preview.py:727-749)
//   areaGray / viewSignature / matchViews   which camera is which, the server's view_signature / match_views
//   crc32 / ZipStore  a store-only ZIP (no compression: JPEGs do not shrink) for recordings downloaded by the page
//   recordFrameLine / recordTrackLine       the server recorder's frames.jsonl / tracks.jsonl lines (preview.py:1255, :1317)

export const CAM_NAMES = Object.freeze(['CAM1', 'CAM2', 'CAM3']);

// ------------------------------------------------------------------ the tips payload

/**
 * The in-page tracker's result as GET /api/tips returns it (preview.py twin_tips): rig_link.js and real_cams.js read
 * it unchanged. Keys the tracker already set win, except frame_id / t / source / replay, which the page owns.
 *   ctx: {frame_id, t (capture unix s), source ('live'|'replay'), replay (name|null), image_size, now (unix s), model}
 */
export function finalizePayload(p, ctx) {
  const out = { type: 'predictions', ...(p && typeof p === 'object' ? p : {}) };
  out.type = 'predictions';
  out.frame_id = ctx.frame_id;
  out.t = Number.isFinite(ctx.t) ? ctx.t : null;
  out.source = ctx.source ?? 'live';
  out.replay = ctx.replay ?? null;
  out.model = out.model ?? ctx.model ?? null;
  const now = Number.isFinite(ctx.now) ? ctx.now : Date.now() / 1000;
  out.stale = !out.t || now - out.t > 0.5;
  if (!Array.isArray(out.instruments)) out.instruments = [];
  out.instruments = [...out.instruments].sort((a, b) => String(a?.id).localeCompare(String(b?.id)));
  out.keypoints_px = out.keypoints_px && typeof out.keypoints_px === 'object' ? out.keypoints_px : {};
  out.image_size = { ...(ctx.image_size ?? {}), ...(out.image_size ?? {}) };
  out.tracker = 'browser';
  return out;
}

// ------------------------------------------------------------------ status

/** The camera bar's text for the browser source (twinpage/browser_source.js; calib_embed.js shows it). */
export function browserNote(src) {
  const t = (typeof src.timing === 'function' ? src.timing() : src.timing) ?? null, w = src.workerInfo;
  switch (src.status) {
    case 'idle': return src.replay ? 'replay ended' : 'cameras off';
    case 'starting': return w ? 'opening cameras…' : 'loading the tracker…';
    case 'error': case 'moved': return src.error || 'stopped';
    default: break;
  }
  if (!t) return w ? 'waiting for frames' : 'loading the tracker…';
  const parts = [`${(t.hz || 0).toFixed(0)} frame sets/s`];
  if (t.tracker === 'stub') parts.push('views only (no tracker)');
  else if (Number.isFinite(t.track_ms)) parts.push(`tracker ${t.track_ms.toFixed(0)} ms${t.ep ? ` (${t.ep})` : ''}`);
  if (src.status === 'replay') parts.push(`replay ${src.replayName ?? src.replay}`);
  if (src.namesFrom === 'order') parts.push('check which camera is which');
  return parts.join(' · ');
}

// ------------------------------------------------------------------ display white balance

/** The 97th percentile (numpy 'linear') of a 0..255 histogram of n values. */
export function histPercentile(hist, n, q = 97) {
  if (!n) return 0;
  const idx = (n - 1) * (q / 100), lo = Math.floor(idx), t = idx - lo;
  let acc = 0, a = null, b = null;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (a == null && acc > lo) a = v;
    if (acc > lo + 1) { b = v; break; }
  }
  if (b == null) b = a;
  return a + (b - a) * t;
}

/** Display gains [gR, gG, gB] from RGBA pixels: each channel's p97 brought to the brightest one's, clipped to 1..3. */
export function wbGains(rgba, { maxGain = 3 } = {}) {
  const h = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  const n = rgba.length >> 2;
  for (let i = 0; i < rgba.length; i += 4) { h[0][rgba[i]]++; h[1][rgba[i + 1]]++; h[2][rgba[i + 2]]++; }
  const p = h.map((x) => histPercentile(x, n));
  const top = Math.max(...p);
  return p.map((v) => Math.min(maxGain, Math.max(1, top / Math.max(v, 1))));
}

/** The server's low-pass: g0 + 0.3 (g - g0), at most every minS seconds; rounded to 0.001. */
export class DisplayWB {
  constructor({ minS = 0.5, k = 0.3 } = {}) { this.minS = minS; this.k = k; this.g = {}; this.at = -1e9; }
  due(t) { return t - this.at >= this.minS; }
  update(cam, g, t) {
    const g0 = this.g[cam];
    const v = g0 ? g0.map((x, i) => x + this.k * (g[i] - x)) : [...g];
    this.g[cam] = v.map((x) => Math.round(x * 1000) / 1000);
    this.at = t;
    return this.g[cam];
  }
  gains() { return Object.fromEntries(Object.entries(this.g).map(([k, v]) => [k, [...v]])); }
}

// ------------------------------------------------------------------ which camera is which (view signatures)

/** RGBA w x h -> gray (w/f) x (h/f) by an f x f box mean (INTER_AREA for an integer factor), BT.601 luma. */
export function areaGray(rgba, w, h, f = 8) {
  const W = Math.floor(w / f), H = Math.floor(h / f), out = new Uint8Array(W * H), nn = f * f;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let r = 0, g = 0, b = 0;
    for (let dy = 0; dy < f; dy++) {
      let o = ((y * f + dy) * w + x * f) * 4;
      for (let dx = 0; dx < f; dx++, o += 4) { r += rgba[o]; g += rgba[o + 1]; b += rgba[o + 2]; }
    }
    out[y * W + x] = Math.round((0.299 * r + 0.587 * g + 0.114 * b) / nn);
  }
  return out;
}

const reflect101 = (i, n) => { if (n === 1) return 0; while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i; return i; };

/** Separable Gaussian blur (sigma, ksize 2*round(4 sigma)+1 as cv2 for float images, BORDER_REFLECT_101). */
export function gaussianBlur(src, w, h, sigma = 1.5) {
  const r = Math.round(sigma * 4), k = [];
  let s = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); s += v; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0; for (let i = -r; i <= r; i++) a += k[i + r] * src[y * w + reflect101(x + i, w)];
    tmp[y * w + x] = a;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0; for (let i = -r; i <= r; i++) a += k[i + r] * tmp[reflect101(y + i, h) * w + x];
    out[y * w + x] = a;
  }
  return out;
}

const meanStd = (a) => {
  let m = 0; for (let i = 0; i < a.length; i++) m += a[i]; m /= a.length;
  let v = 0; for (let i = 0; i < a.length; i++) v += (a[i] - m) ** 2;
  return [m, Math.sqrt(v / a.length)];
};

/** preview.py view_signature on a 160x90 gray: null for a blank frame (std < minStd), else the blurred, z-scored view. */
export function viewSignature(gray, w = 160, h = 90, { sigma = 1.5, minStd = 4 } = {}) {
  if (meanStd(gray)[1] < minStd) return null;
  const g = gaussianBlur(gray, w, h, sigma);
  const [m, sd] = meanStd(g);
  for (let i = 0; i < g.length; i++) g[i] = (g[i] - m) / (sd + 1e-6);
  return g;
}

const ncc = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s / a.length; };

function permutations(n) {
  if (n <= 1) return [[...Array(n).keys()]];
  const out = [];
  for (const p of permutations(n - 1)) for (let i = 0; i <= p.length; i++) out.push([...p.slice(0, i), n - 1, ...p.slice(i)]);
  return out;
}

/**
 * preview.py match_views: tiles {id: signature} vs refs {CAMk: [signature...]} -> {id: CAMk}, the best joint assignment
 * (brute force: at most 3 x 3), kept only where NCC >= minNcc and >= minMargin above that tile's next camera.
 * Also returns the matrix: {names, M, tiles, cams}.
 */
export function matchViews(sigs, refs, { minNcc = 0.4, minMargin = 0.12 } = {}) {
  const tiles = Object.keys(sigs).filter((t) => sigs[t]).sort(), cams = Object.keys(refs).sort();
  const M = tiles.map((t) => cams.map((c) => Math.max(...refs[c].map((r) => ncc(sigs[t], r)))));
  const names = {};
  if (!tiles.length || !cams.length) return { names, M, tiles, cams };
  const n = Math.max(tiles.length, cams.length);
  let best = null, bestS = -Infinity;
  for (const p of permutations(n)) {           // p[i] = column of row i (indices beyond the matrix are "unassigned")
    let s = 0;
    for (let i = 0; i < tiles.length; i++) if (p[i] < cams.length) s += M[i][p[i]];
    if (s > bestS) { bestS = s; best = p; }
  }
  for (let i = 0; i < tiles.length; i++) {
    const j = best[i];
    if (j >= cams.length) continue;
    const others = M[i].filter((_, k) => k !== j);
    if (M[i][j] >= minNcc && (!others.length || M[i][j] - Math.max(...others) >= minMargin)) names[tiles[i]] = cams[j];
  }
  return { names, M, tiles, cams };
}

/** base64 -> Uint8Array (atob in browsers and workers, Buffer in node). */
export function b64bytes(s) {
  if (typeof atob === 'function') { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }
  return new Uint8Array(Buffer.from(s, 'base64'));
}

/** rig.json view_refs -> {CAMk: [signature...]} (refs too flat to tell anything are skipped). */
export function refSignatures(viewRefs, opts = {}) {
  const out = {};
  for (const [cam, list] of Object.entries(viewRefs || {})) {
    const s = (list || []).map((r) => viewSignature(b64bytes(r.gray_u8_b64), r.w, r.h, opts)).filter(Boolean);
    if (s.length) out[cam] = s;
  }
  return out;
}

/**
 * Slot names for the cameras found: a saved name per device id wins; the rest take the names nobody holds, in order.
 * devices: [deviceId], saved: {deviceId: CAMk} -> [CAMk per device] (null beyond the known names).
 */
export function assignNames(devices, saved = {}, names = CAM_NAMES) {
  const out = devices.map((d) => (names.includes(saved[d]) ? saved[d] : null));
  // a name held twice (two devices saved with it): the first keeps it
  out.forEach((n, i) => { if (n && out.indexOf(n) !== i) out[i] = null; });
  const free = names.filter((n) => !out.includes(n));
  return out.map((n) => n ?? free.shift() ?? null);
}

/** Swap two slots' names: [names], a, b -> the new list (a and b are CAM names). */
export function swapNames(slots, a, b) {
  return slots.map((n) => (n === a ? b : n === b ? a : n));
}

// ------------------------------------------------------------------ recording: a store-only ZIP

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();

/** CRC-32 (ZIP / PNG), optionally continuing from crc. */
export function crc32(u8, crc = 0) {
  let c = (crc ^ 0xffffffff) >>> 0;
  for (let i = 0; i < u8.length; i++) c = CRC_TABLE[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const dosTime = (d) => ({
  time: ((d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1)) & 0xffff,
  date: (((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate()) & 0xffff,
});

/**
 * ZipStore: files added as bytes (with their CRC when the caller has it), kept as Blob parts so a long recording need
 * not sit in one buffer; blob() writes the central directory. Stored (method 0), UTF-8 names, no ZIP64: under 4 GB
 * and 65535 files (the page caps recordings well below that).
 */
export class ZipStore {
  constructor({ date = new Date() } = {}) { this.parts = []; this.entries = []; this.offset = 0; this.dt = dosTime(date); }
  get bytes() { return this.offset; }
  get count() { return this.entries.length; }
  add(name, data, crc = null) {
    const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
    const nm = new TextEncoder().encode(name);
    if (this.entries.length >= 65535 || this.offset + u8.length + 30 + nm.length > 0xffffffff) throw new Error('recording too large for a ZIP without ZIP64');
    const c = crc ?? crc32(u8);
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint16(8, 0, true);
    h.setUint16(10, this.dt.time, true); h.setUint16(12, this.dt.date, true); h.setUint32(14, c, true);
    h.setUint32(18, u8.length, true); h.setUint32(22, u8.length, true); h.setUint16(26, nm.length, true); h.setUint16(28, 0, true);
    this.entries.push({ nm, crc: c, size: u8.length, offset: this.offset });
    this.parts.push(new Uint8Array(h.buffer), nm, typeof Blob !== 'undefined' ? new Blob([u8]) : u8);
    this.offset += 30 + nm.length + u8.length;
  }
  addText(name, text) { this.add(name, new TextEncoder().encode(text)); }
  /** The central directory and end record: the parts of a complete archive. */
  finishParts() {
    const cd = [];
    let size = 0;
    for (const e of this.entries) {
      const h = new DataView(new ArrayBuffer(46));
      h.setUint32(0, 0x02014b50, true); h.setUint16(4, 20, true); h.setUint16(6, 20, true); h.setUint16(8, 0x0800, true);
      h.setUint16(10, 0, true); h.setUint16(12, this.dt.time, true); h.setUint16(14, this.dt.date, true); h.setUint32(16, e.crc, true);
      h.setUint32(20, e.size, true); h.setUint32(24, e.size, true); h.setUint16(28, e.nm.length, true);
      h.setUint32(42, e.offset, true);
      cd.push(new Uint8Array(h.buffer), e.nm);
      size += 46 + e.nm.length;
    }
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, this.entries.length, true); end.setUint16(10, this.entries.length, true);
    end.setUint32(12, size, true); end.setUint32(16, this.offset, true);
    return [...this.parts, ...cd, new Uint8Array(end.buffer)];
  }
  blob() { return new Blob(this.finishParts(), { type: 'application/zip' }); }
}

/** Local time as the server writes it: 2026-11-01T15:09:44.514 (no zone). */
export function isoLocal(ms) {
  const d = new Date(ms), p = (x, n = 2) => String(x).padStart(n, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
}

/** The recording's folder name: <YYYYMMDD>_browser_<HHMMSS> (the server's is <YYYYMMDD>_<tag>_NN). */
export function sessionName(ms, tag = 'browser') {
  const s = isoLocal(ms);
  return `${s.slice(0, 10).replace(/-/g, '')}_${tag}_${s.slice(11, 19).replace(/:/g, '')}`;
}

/** frames.jsonl, one line per recorded set (preview.py:1255-1260): scripts/eval_tips.py reads it unchanged. */
export function recordFrameLine({ index, wallMs, tMs, cameras, tiles = {}, capMs = {}, phase = null }) {
  return JSON.stringify({ index, wall: isoLocal(wallMs), t_ms: tMs, phase, cameras: [...cameras].sort(), tiles, cap_ms: capMs });
}

/** tracks.jsonl (preview.py:1317-1324): instruments is the tracker's rig-frame dict when it gives one; the full tips
 * payload (frame D) rides along as `tips`. */
export function recordTrackLine({ index, wallMs, tMs, cameras, ms, payload }) {
  const rig = payload?.instruments_rig ?? payload?.rig?.instruments ?? {};
  return JSON.stringify({ wall: isoLocal(wallMs), t_ms: tMs, index, cameras: [...cameras].sort(), ms, instruments: rig, tips: payload ?? null });
}
