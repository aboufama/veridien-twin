// The tracking-camera feeds: one worker per camera (ui/feed_worker.js) encodes each rgb8 sensor frame as MJPG and
// draws the decoded JPEG into the feed canvas through an OffscreenCanvas, so neither the encode nor the draw ever
// runs on the UI thread. Falls back to inline encoding (and no drawing) without workers.
import { encodeJPEG, encodeJPEGYUV420 } from './jpeg_encoder.js';

export class Feeds {
  constructor(cameras, canvases = {}) {
    this.workers = new Map();
    this.pending = new Map();
    this.next = 1;
    this.tables = { quality: 90 };
    this.stats = { encoded: 0, msSum: 0, drawSum: 0, lastMs: 0 };
    this.inline = typeof Worker !== 'function';
    if (this.inline) return;
    for (const cam of cameras) {
      const w = new Worker(new URL('./feed_worker.js', import.meta.url), { type: 'module' });
      w.onmessage = (e) => this._reply(e.data);
      w.onerror = (e) => console.error(`feed worker ${cam.name}`, e.message || e);
      const cv = canvases[cam.name];
      let off = null;
      if (cv && cv.transferControlToOffscreen) {
        cv.width = 640; cv.height = 360;
        off = cv.transferControlToOffscreen();
      }
      w.postMessage({ type: 'init', name: cam.name, canvas: off }, off ? [off] : []);
      this.workers.set(cam.name, w);
    }
  }

  setTables(t) {
    this.tables = { ...t };
    for (const w of this.workers.values()) w.postMessage({ type: 'tables', ...t });
  }

  resize(name, width, height) { this.workers.get(name)?.postMessage({ type: 'resize', width, height }); }

  /** The feed panel is on screen (or not): an off-screen feed's worker still encodes but does not draw. */
  setVisible(name, on) {
    if (this.visible?.[name] === !!on) return;
    (this.visible ??= {})[name] = !!on;
    this.workers.get(name)?.postMessage({ type: 'visible', on: !!on });
  }

  _reply(m) {
    if (m.type === 'drawError') { if (!this.drawErr) console.error(`feed ${m.name} draw`, m.error); this.drawErr = m.error; return; }
    const p = this.pending.get(m.id);
    if (!p) return;
    this.pending.delete(m.id);
    if (m.type === 'error') { p.reject(new Error(m.error)); return; }
    this.stats.encoded++; this.stats.msSum += m.ms; this.stats.drawSum += m.drawMs; this.stats.lastMs = m.ms;
    p.resolve({ jpeg: new Uint8Array(m.jpeg), ms: m.ms, drawMs: m.drawMs });
  }

  /** Encode (and, when draw, display) one frame. rgb's buffer is transferred to the worker. */
  encode(name, rgb, width, height, { draw = true, channels = 3 } = {}) {
    if (this.inline || !this.workers.has(name)) {
      const t0 = performance.now();
      const jpeg = encodeJPEG(rgb, width, height, { ...this.tables, channels });
      return Promise.resolve({ jpeg, ms: performance.now() - t0, drawMs: 0 });
    }
    const id = this.next++;
    const buf = rgb.byteOffset === 0 && rgb.byteLength === rgb.buffer.byteLength ? rgb.buffer : rgb.slice().buffer;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.workers.get(name).postMessage({ type: 'frame', id, rgb: buf, width, height, channels, draw }, [buf]);
    });
  }

  /** Encode the ISP's 4:2:0 planes ({y, cb, cr, width, height}, renderer readback 'yuv420'); planes are transferred. */
  encodeYUV(name, yuv, { draw = true } = {}) {
    if (this.inline || !this.workers.has(name)) {
      const t0 = performance.now();
      const jpeg = encodeJPEGYUV420(yuv, this.tables);
      return Promise.resolve({ jpeg, ms: performance.now() - t0, drawMs: 0 });
    }
    const id = this.next++;
    const own = (a) => (a.byteOffset === 0 && a.byteLength === a.buffer.byteLength ? a.buffer : a.slice().buffer);
    const y = own(yuv.y), cb = own(yuv.cb), cr = own(yuv.cr);
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.workers.get(name).postMessage({ type: 'frame', id, y, cb, cr, width: yuv.width, height: yuv.height, draw }, [y, cb, cr]);
    });
  }

  meanEncodeMs() { return this.stats.encoded ? this.stats.msSum / this.stats.encoded : 0; }
}
