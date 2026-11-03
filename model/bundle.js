// Frame bundles on the model socket (docs/WEB_TWIN.md §7.2). Runs in the browser and in node.
//
//   [u32 header_len][header JSON, UTF-8, header_len bytes][jpeg_0][jpeg_1]...[jpeg_{n-1}]
//
// header_len is LITTLE-endian. The contract did not fix the byte order; little-endian is what every
// machine on this project (Apple silicon, x86) is natively, and Python reads it with struct.unpack("<I", ...).
// The header is {"type": "frames", "frame_id", "source", "cameras": [{"name", "t_capture", "bytes",
// "width", "height", ...}]}; cameras[i].bytes is the length of jpeg_i and the JPEGs follow in the same order.

const MAX_HEADER = 1 << 20;          // 1 MiB: a header is a few hundred bytes; anything larger is garbage

const enc = new TextEncoder();
const dec = new TextDecoder('utf-8', { fatal: true });

/**
 * Pack one bundle. Sets header.cameras[i].bytes = jpegs[i].length (the header object is modified).
 * @param {object} header  must have a `cameras` array with one entry per JPEG
 * @param {Uint8Array[]} jpegs
 * @returns {ArrayBuffer}
 */
export function packBundle(header, jpegs) {
  if (!header || !Array.isArray(header.cameras)) throw new TypeError('packBundle: header.cameras must be an array');
  if (!Array.isArray(jpegs)) throw new TypeError('packBundle: jpegs must be an array');
  if (header.cameras.length !== jpegs.length) {
    throw new RangeError(`packBundle: ${header.cameras.length} cameras in the header but ${jpegs.length} JPEGs`);
  }
  let body = 0;
  jpegs.forEach((j, i) => {
    if (!(j instanceof Uint8Array)) throw new TypeError(`packBundle: jpeg ${i} is not a Uint8Array`);
    header.cameras[i].bytes = j.length;
    body += j.length;
  });
  const h = enc.encode(JSON.stringify(header));
  if (h.length > MAX_HEADER) throw new RangeError(`packBundle: header is ${h.length} bytes (max ${MAX_HEADER})`);
  const out = new Uint8Array(4 + h.length + body);
  new DataView(out.buffer).setUint32(0, h.length, true);
  out.set(h, 4);
  let o = 4 + h.length;
  for (const j of jpegs) { out.set(j, o); o += j.length; }
  return out.buffer;
}

/**
 * Unpack and validate one bundle. The returned JPEGs are views into the input buffer (no copy).
 * @param {ArrayBuffer|Uint8Array} buf
 * @returns {{header: object, jpegs: Uint8Array[]}}
 */
export function unpackBundle(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u8.length < 4) throw new RangeError(`unpackBundle: ${u8.length} bytes, too short for the header length`);
  const hlen = new DataView(u8.buffer, u8.byteOffset, u8.byteLength).getUint32(0, true);
  if (hlen === 0 || hlen > MAX_HEADER) throw new RangeError(`unpackBundle: header_len ${hlen} is not sane`);
  if (4 + hlen > u8.length) throw new RangeError(`unpackBundle: header_len ${hlen} runs past the end (${u8.length} bytes)`);
  let header;
  try {
    header = JSON.parse(dec.decode(u8.subarray(4, 4 + hlen)));
  } catch (e) {
    throw new SyntaxError(`unpackBundle: header is not UTF-8 JSON (${e.message})`);
  }
  if (!header || typeof header !== 'object' || !Array.isArray(header.cameras)) {
    throw new TypeError('unpackBundle: header has no cameras array');
  }
  const jpegs = [];
  let o = 4 + hlen;
  for (const [i, c] of header.cameras.entries()) {
    const n = c?.bytes;
    if (!Number.isInteger(n) || n < 0) throw new TypeError(`unpackBundle: camera ${i} has no valid bytes field`);
    if (o + n > u8.length) throw new RangeError(`unpackBundle: camera ${i} (${n} bytes) runs past the end`);
    jpegs.push(u8.subarray(o, o + n));
    o += n;
  }
  if (o !== u8.length) throw new RangeError(`unpackBundle: ${u8.length - o} trailing bytes after the last JPEG`);
  return { header, jpegs };
}
