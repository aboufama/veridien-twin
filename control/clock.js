// Free-running camera clocks (docs/WEB_TWIN.md §4 cameras[].fps, phase_s): every tracking camera captures on its
// own schedule t_n = phase + n / fps, as the real unsynchronised UVC cameras do, with a global-shutter exposure
// [t_n, t_n + exposure]. Truth and t_capture are at mid-exposure (§6). Pure module (node + page).

export class CameraClock {
  /** cameras: [{name, fps, phase_s}] (scene.cameras); exposure_s: exposure time (s). */
  constructor(cameras, { exposure_s = 0.010, phaseOverride = null } = {}) {
    this.cams = cameras.map((c, k) => ({
      name: c.name, fps: c.fps ?? 20, phase: phaseOverride ? phaseOverride[k] ?? 0 : (c.phase_s ?? 0),
    }));
    this.exposure = exposure_s;
  }

  capture(name, n) {
    const c = this.cams.find((x) => x.name === name);
    const t0 = c.phase + n / c.fps;
    return { name, index: n, t_start: t0, t_end: t0 + this.exposure, t_mid: t0 + this.exposure / 2 };
  }

  /** Index of the latest capture of `name` whose exposure (default: the clock's) has ended by t (or -1). */
  latestIndex(name, t, exposure = this.exposure) {
    const c = this.cams.find((x) => x.name === name);
    return Math.floor((t - exposure - c.phase) * c.fps + 1e-9);
  }

  /** Every capture (all cameras) whose exposure starts in [t0, t1), in time order (ties: camera order). */
  between(t0, t1) {
    const out = [];
    for (const c of this.cams) {
      let n = Math.ceil((t0 - c.phase) * c.fps - 1e-9);
      for (; ; n++) {
        const ts = c.phase + n / c.fps;
        if (ts >= t1 - 1e-12) break;
        if (n >= 0) out.push(this.capture(c.name, n));
      }
    }
    const order = new Map(this.cams.map((c, k) => [c.name, k]));
    out.sort((a, b) => a.t_start - b.t_start || order.get(a.name) - order.get(b.name));
    return out;
  }

  /** Group captures into bundles: the k-th bundle holds every camera's k-th capture (frame_id = k). */
  bundles(t0, t1) {
    const caps = this.between(t0, t1);
    const by = new Map();
    for (const c of caps) { if (!by.has(c.index)) by.set(c.index, []); by.get(c.index).push(c); }
    const n = this.cams.length;
    return [...by.entries()].filter(([, v]) => v.length === n).sort((a, b) => a[0] - b[0])
      .map(([index, caps2]) => ({ frame_id: index, captures: caps2, t_ref: Math.max(...caps2.map((c) => c.t_mid)) }));
  }
}
