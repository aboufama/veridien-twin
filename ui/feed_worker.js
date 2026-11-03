// One worker per tracking camera: encode the sensor's rgb8 frame as the camera's MJPG frame (baseline JPEG, 4:2:0,
// the configured quant tables) and draw the decoded JPEG, i.e. exactly what a model receives, into the feed's
// OffscreenCanvas. Nothing here runs on the UI thread.
//
//   {type: 'init', name, canvas?: OffscreenCanvas}                 (canvas transferred)
//   {type: 'tables', qY?, qC?, quality?}
//   {type: 'resize', width, height}
//   {type: 'visible', on}                                          (the panel is on screen; off: no drawing)
//   {type: 'frame', id, rgb: ArrayBuffer (rgb8, or rgba8 with channels 4), width, height, channels?, draw}
//   {type: 'frame', id, y, cb, cr: ArrayBuffer (the ISP's 4:2:0 planes, renderer readback 'yuv420'), width, height, draw}     -> {type: 'jpeg', id, name, jpeg: ArrayBuffer, ms, drawMs}
//
// The JPEG goes back BEFORE the feed is drawn (the bundle waits for the bytes, not for the panel), and drawing is
// latest-only: a frame that arrives while the previous one is still being decoded replaces any frame waiting to be
// drawn. drawMs in a reply is the last finished draw's time.
import { encodeJPEG, encodeJPEGYUV420 } from './jpeg_encoder.js';

let name = '', canvas = null, ctx = null, tables = { quality: 90 }, visible = true;
let drawing = false, waiting = null, lastDrawMs = 0;

async function drawLatest(blob) {
  if (drawing) { waiting = blob; return; }
  drawing = true;
  try {
    for (let b = blob; b; b = waiting, waiting = null) {
      const t1 = performance.now();
      const bmp = await createImageBitmap(b);
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
      bmp.close();
      lastDrawMs = performance.now() - t1;
    }
  } finally { drawing = false; }
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      name = m.name;
      if (m.canvas) { canvas = m.canvas; ctx = canvas.getContext('2d', { alpha: false }); }
    } else if (m.type === 'tables') {
      tables = { ...(m.qY ? { qY: m.qY } : {}), ...(m.qC ? { qC: m.qC } : {}), ...(m.quality ? { quality: m.quality } : {}) };
      if (!tables.qY && !tables.quality) tables.quality = 90;
    } else if (m.type === 'resize') {
      if (canvas && (canvas.width !== m.width || canvas.height !== m.height)) { canvas.width = m.width; canvas.height = m.height; }
    } else if (m.type === 'visible') {
      visible = !!m.on;
    } else if (m.type === 'frame') {
      const t0 = performance.now();
      const jpeg = m.y
        ? encodeJPEGYUV420({ y: new Uint8Array(m.y), cb: new Uint8Array(m.cb), cr: new Uint8Array(m.cr), width: m.width, height: m.height }, tables)
        : encodeJPEG(new Uint8Array(m.rgb), m.width, m.height, { ...tables, channels: m.channels ?? 3 });
      const ms = performance.now() - t0;
      // the Blob copies the bytes, so the JPEG's own buffer can go back to the page first
      const blob = m.draw && ctx && visible ? new Blob([jpeg], { type: 'image/jpeg' }) : null;
      const buf = jpeg.buffer.byteLength === jpeg.byteLength ? jpeg.buffer : jpeg.slice().buffer;
      self.postMessage({ type: 'jpeg', id: m.id, name, jpeg: buf, ms, drawMs: lastDrawMs }, [buf]);
      if (blob) drawLatest(blob).catch((err) => self.postMessage({ type: 'drawError', name, error: String(err?.stack || err) }));
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, name, error: String(err?.stack || err) });
  }
};
