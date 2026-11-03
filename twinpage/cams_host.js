// The page's side of the camera-column worker (twinpage/cams_worker.js, ADR-137): starts the worker, hands it the tile
// canvases (transferControlToOffscreen) and forwards tile and bundle requests. The renderer's seconds of set-up CPU
// and every tile's encoding then run off the main thread, so the god's-eye view never stalls on the column.
// Use CamsHost.supported() first; the page falls back to the renderer on its own thread (?camsWorker=0 forces that).

export class CamsHost {
  static supported() {
    return typeof Worker === 'function' && typeof OffscreenCanvas === 'function'
      && typeof HTMLCanvasElement !== 'undefined' && typeof HTMLCanvasElement.prototype.transferControlToOffscreen === 'function';
  }

  /** -> CamsHost with a renderer, or null when none of `order` started in the worker (the worker is then closed). */
  static async create({ order, base, physics, onLost, onGpuError, timeoutMs = 120000 }) {
    const h = new CamsHost();
    h.worker = new Worker(new URL('./cams_worker.js', import.meta.url), { type: 'module', name: 'twin camera column' });
    h.worker.onmessage = (ev) => h._onMessage(ev.data);
    h.worker.onerror = (e) => { h.error = String(e?.message || 'worker error'); for (const p of h.pending.values()) p.reject(new Error(h.error)); h.pending.clear(); };
    h.onLost = onLost; h.onGpuError = onGpuError;
    let info;
    try {
      info = await Promise.race([h.call('init', { order, base, physics }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('camera worker did not start in time')), timeoutMs))]);
    } catch (e) { h.worker.terminate(); throw e; }
    h.failed = info.failed || [];
    if (!info.dir) { h.worker.terminate(); return Object.assign(new CamsHost(), { dir: null, failed: h.failed }); }
    Object.assign(h, { dir: info.dir, cameras: info.cameras, kind: info.kind, loadMs: info.loadMs });
    return h;
  }

  constructor() { this.pending = new Map(); this.nextId = 1; this.keys = new WeakMap(); this.nKeys = 0; this.lastTimings = null; this.isWorkerHost = true; }

  _onMessage(m) {
    if (m?.type === 'lost') { this.lost = m.message; this.onLost?.(m.message); return; }
    if (m?.type === 'gpuerror') { this.onGpuError?.(m.message); return; }
    const p = this.pending.get(m?.id);
    if (!p) return;
    this.pending.delete(m.id);
    if (m.ok) p.resolve(m.out); else p.reject(new Error(m.error));
  }

  call(type, payload = {}, transfer = []) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try { this.worker.postMessage({ ...payload, id, type }, transfer); } catch (e) { this.pending.delete(id); reject(e); }
    });
  }

  /** The worker's key for a page canvas, transferring it on first use (the canvas is the worker's from then on). */
  canvasKey(canvas) {
    let k = this.keys.get(canvas);
    if (k == null) {
      k = `c${this.nKeys++}`;
      const off = canvas.transferControlToOffscreen();
      this.keys.set(canvas, k);
      this.worker.postMessage({ id: 0, type: 'canvas', key: k, canvas: off }, [off]);
    }
    return k;
  }

  /** Render, sense and present one camera into `canvases` ([{canvas, w, h}]); resolves when its GPU work is done. */
  tile({ name, size, poses = null, seed, frame, canvases = [] }) {
    const targets = canvases.map(({ canvas, w, h }) => ({ key: this.canvasKey(canvas), w, h }));
    return this.call('tile', { name, size, poses, seed, frame, targets });
  }

  /** The model's bundle: CAM1-CAM3 at native size; -> {cams: [{name, vis, rb}]}. */
  bundle({ poses, cams, pts, seed, frame }) { return this.call('bundle', { poses, cams, pts, seed, frame }); }

  /** The renderer's last timings (refreshed in the background on every call). */
  timings() {
    if (!this.timingsBusy) {
      this.timingsBusy = true;
      this.call('timings').then((t) => { this.lastTimings = t; }).catch(() => {}).finally(() => { this.timingsBusy = false; });
    }
    return this.lastTimings;
  }

  destroy() { this.worker?.terminate(); }
}
