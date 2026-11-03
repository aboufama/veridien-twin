// A pool of JPEG workers, one per camera, so encoding the three tracking cameras' frames never runs on the
// UI thread. Without Worker (node, or a page that refuses workers) it encodes inline with the same encoder.
//
//   const pool = new JpegPool({ size: 3 });
//   pool.setTables({ qY, qC });                 // the real camera's tables (tablesFromJPEG on a real frame)
//   const jpeg = await pool.encode('CAM1', rgb8, 1280, 720);
//
// encode() TRANSFERS rgb's buffer to the worker when rgb is a standalone view (the caller's array is
// detached afterwards); a view into a larger or shared buffer is copied first.
import { encodeJPEG } from './jpeg_encoder.js';

/** Synchronous encode with the pool's table semantics: tables = {qY?, qC?, quality?, channels?}. */
export function encodeInline(rgb, width, height, tables = {}) {
  return encodeJPEG(rgb, width, height, tables);
}

export class JpegPool {
  constructor({ size = 3, workerUrl = new URL('./jpeg_worker.js', import.meta.url), inline } = {}) {
    this.tables = {};
    this.workers = [];
    this.byName = new Map();
    this.pending = new Map();
    this.nextId = 1;
    this.rr = 0;
    this.done = 0;
    this.msSum = 0;
    this.errors = 0;
    const noWorker = typeof Worker !== 'function' || (typeof process !== 'undefined' && !!process.versions?.node);
    this.inline = inline ?? noWorker;
    if (!this.inline) {
      try {
        for (let i = 0; i < Math.max(1, size); i++) {
          const w = new Worker(workerUrl, { type: 'module' });
          w.onmessage = (e) => this._reply(e.data);
          w.onerror = (e) => this._fail(w, e);
          this.workers.push(w);
        }
      } catch (err) {
        console.warn('JpegPool: workers unavailable, encoding inline', err);
        for (const w of this.workers) w.terminate();
        this.workers = [];
        this.inline = true;
      }
    }
  }

  /** Tables used for every later encode: {qY, qC} (natural order) and/or {quality}. */
  setTables({ qY, qC, quality } = {}) {
    this.tables = {};
    if (qY) this.tables.qY = Array.from(qY);
    if (qC) this.tables.qC = Array.from(qC);
    if (quality != null) this.tables.quality = quality;
  }

  encode(name, rgb, width, height, tables = {}) {
    const t = { ...this.tables, ...tables };
    if (this.inline) {
      const t0 = performance.now();
      try {
        const out = encodeInline(rgb, width, height, t);
        this.done++; this.msSum += performance.now() - t0;
        return Promise.resolve(out);
      } catch (err) {
        this.errors++;
        return Promise.reject(err);
      }
    }
    let k = this.byName.get(name);
    if (k === undefined) { k = this.rr++ % this.workers.length; this.byName.set(name, k); }
    const w = this.workers[k];
    const standalone = rgb.byteOffset === 0 && rgb.byteLength === rgb.buffer.byteLength &&
      !(typeof SharedArrayBuffer !== 'undefined' && rgb.buffer instanceof SharedArrayBuffer);
    const buf = standalone ? rgb.buffer : rgb.slice().buffer;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, w });
      w.postMessage({ id, name, rgb: buf, width, height, qY: t.qY, qC: t.qC, quality: t.quality, channels: t.channels }, [buf]);
    });
  }

  _reply({ id, jpeg, ms, error }) {
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id);
    if (error) { this.errors++; p.reject(new Error(error)); return; }
    this.done++; this.msSum += ms;
    p.resolve(new Uint8Array(jpeg));
  }

  _fail(w, e) {
    this.errors++;
    for (const [id, p] of this.pending) {
      if (p.w === w) { this.pending.delete(id); p.reject(new Error(`JPEG worker failed: ${e?.message ?? e}`)); }
    }
  }

  stats() {
    return { inFlight: this.pending.size, done: this.done, meanMs: this.done ? this.msSum / this.done : 0,
      errors: this.errors, workers: this.workers.length, inline: this.inline };
  }

  close() {
    for (const w of this.workers) w.terminate();
    for (const [, p] of this.pending) p.reject(new Error('JpegPool closed'));
    this.pending.clear();
    this.workers = [];
  }
}
