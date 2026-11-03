// The in-browser tip tracker: the live camera server's chain (preview.TipNetTracker._track, src/veridien/multicam/
// preview.py) in pure JS, so the hosted twin page runs with no backend. Per frame set:
//   frames -> TipNet.resize + white balance (preprocess.js) -> tip net (net.js, onnxruntime-web) -> sigmoid + 3x3 NMS +
//   CellHysteresis (heat.js) -> solve_frame (rcm.js: RCM hypotheses, shaft darkness, assignment) -> hysteresis
//   after_solve -> TipFilter per port (filter.js) -> SteadyTips (steady.js) -> the rig-frame instrument dict and the
//   twin's /api/tips payload (twin_tips: frame D, metres), the same JSON the page reads from the server.
// Config and calibration come from tracker/rig.json (scripts/export_rig_json.py), generated from the server's own
// loaders. No DOM: images are ImageData-like {width, height, data (RGBA or RGB)}.

import { camModels, projectPx, project, toCam } from './lens.js';
import { toSmall, gray, toTensor, WhiteBalance, wbGainsFromHist, IN_W, IN_H } from './preprocess.js';
import { sigmoidHeat, decodeSet, CellHysteresis } from './heat.js';
import { solveFrame, SOLVE_DEFAULTS, JAW_M } from './rcm.js';
import { TipFilter } from './filter.js';
import { SteadyTips } from './steady.js';
import { npRound, pyRound, mulMv, sub, scale, norm, linspace, add } from './math.js';

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/** rig.json -> {names, cams, ports, T_D_R: {R, t}, cfg} with the live defaults filled in. */
export function loadRig(rig) {
  if (rig.schema !== 'veridien.tracker.rig/1') throw new Error(`rig.json schema ${rig.schema}: expected veridien.tracker.rig/1`);
  const cams = camModels(rig);
  const ports = rig.ports.map((p) => ({ name: p.name, pivot: p.pivot_rig_m.slice(), depth_m: (p.depth_m ?? [0.02, 0.4]).slice() }));
  const T = rig.T_D_R;
  const solve = { ...SOLVE_DEFAULTS, ...Object.fromEntries(Object.entries(rig.solve ?? {}).map(([k, v]) => [k, v ?? (k === 'min_dark' ? -Infinity : v)])) };
  return { rig, names: cams.map((c) => c.name), cams, ports, T_D_R: { R: T.slice(0, 3).map((r) => r.slice(0, 3)), t: T.slice(0, 3).map((r) => r[3]) },
    solve, hyst: rig.hysteresis ?? {}, filter: rig.tip_filter ?? {}, steady: rig.steady ?? {}, net: rig.net ?? {}, model: rig.net?.model_name ?? 'tipnet@unknown' };
}

const r6 = (v) => v.map((x) => npRound(x, 6));

/** The rig-frame instrument dict of one port (preview.TipNetTracker._track) . */
function instrument(port, tip, pt, filt, mode, names, cams, present) {
  const v = sub(tip, port.pivot);
  const d = norm(v);
  const u = scale(v, 1 / d);
  const hinge = add(port.pivot, scale(u, d - JAW_M));
  const conf = pt == null ? 0.05 : Math.min(Math.max(1.0 - pt.rms_px / 4.0, 0.05), 1.0);
  const e = {
    views: pt == null ? [] : pt.views.map(([k]) => k).sort((a, b) => a - b).map((k) => names[k]),
    tip_rig_mm: tip.map((x) => npRound(x * 1e3, 2)),
    tip_raw_rig_mm: pt == null ? null : pt.tip.map((x) => npRound(x * 1e3, 2)),
    held: !!filt.held,
    direction_rig: r6(u),
    hinge_rig_mm: hinge.map((x) => npRound(x * 1e3, 2)),
    axis_point_rig_mm: port.pivot.map((x) => npRound(x * 1e3, 2)),
    pivot_dist_mm: 0.0,
    rms_px: pt == null ? null : pyRound(pt.rms_px, 2),
    conf: pyRound(conf, 3),
    plane_rms_mm: 0.0,
    px: {},
    source: pt != null ? 'net' : 'held',
    mode: mode ?? 'measured',
    estimated: mode === 'estimated',
  };
  for (const n of present) {
    const cam = cams[names.indexOf(n)];
    const end = linspace(0.0, d, 24)[23];
    const Xc = toCam(cam, add(port.pivot, scale(u, end)));
    if (Xc[2] > 0.01) e.px[n] = project(cam.lens, Xc).map((x) => npRound(x, 1));
  }
  return e;
}

/** preview.twin_tips: rig-frame instrument dicts -> the twin's predictions payload (frame D, metres). */
export function twinTips(rigm, instruments, { frameId = null, t = null, model, imageSize = {}, nowS = Date.now() / 1e3, extra = {} } = {}) {
  const { R, t: tD } = rigm.T_D_R;
  const out = { type: 'predictions', frame_id: frameId, t, source: 'live', replay: null, model: model ?? rigm.model,
    stale: !t || nowS - t > 0.5, instruments: [] };
  for (const port of Object.keys(instruments).sort()) {
    const d = instruments[port];
    if (!d.tip_rig_mm) continue;
    const tip = add(mulMv(R, d.tip_rig_mm.map((x) => x * 1e-3)), tD);
    const u = mulMv(R, d.direction_rig);
    const ins = { id: port, tip_D: r6(tip), shaft_dir_D: r6(scale(u, 1 / norm(u))), conf: d.conf ?? 0.5, residual_px: d.rms_px };
    if (d.hinge_rig_mm) ins.hinge_D = r6(add(mulMv(R, d.hinge_rig_mm.map((x) => x * 1e-3)), tD));
    out.instruments.push(ins);
  }
  const kp = {};
  for (const [port, d] of Object.entries(instruments)) for (const [cam, uv] of Object.entries(d.px ?? {})) (kp[cam] ??= {})[port] = { tip: uv };
  out.keypoints_px = kp;
  out.image_size = imageSize;
  return Object.assign(out, extra);
}

/** preview.twin_calibration from rig.json: {type: "calibration", cameras: [{name, size, lens, T_D_C}], ports, placement}. */
export function twinCalibration(rig) {
  const T = rig.T_D_R;
  const R = T.slice(0, 3).map((r) => r.slice(0, 3)), tD = T.slice(0, 3).map((r) => r[3]);
  return {
    type: 'calibration', frame: 'D (dome CAD frame, metres, z up)', source: 'real',
    placement: { method: 'rig calibration fitted to the aimed CAD camera poses (coverage.align_to_design), shipped in tracker/rig.json',
      residuals: rig.align_to_design ?? [] },
    cameras: rig.cameras.map((c) => ({ name: c.name, size: c.size, lens: { model: c.lens.model, K: c.lens.K, D: c.lens.D, size: c.size }, T_D_C: c.T_D_C })),
    ports: rig.ports.map((p) => {
      const e = { id: p.name, pivot_cad_D: p.pivot_cad_D };
      if (p.pivot_D) e.pivot_D = p.pivot_D;
      else e.pivot_D = r6(add(mulMv(R, p.pivot_rig_m), tD));
      if (p.pivot_fit) e.pivot_fit = Object.fromEntries(Object.entries(p.pivot_fit).filter(([k]) => ['axes', 'inliers', 'residual_median_mm', 'residual_p95_mm'].includes(k)));
      return e;
    }).sort((a, b) => (a.id < b.id ? -1 : 1)),
  };
}

/**
 * The tracker. opts: {rig: rig.json object, net: {run(x) -> Promise<Float32Array>} (createTipNet), hyst?: true}.
 * track(images {CAMk: ImageData-like}, tSeconds, {frameId}) -> Promise<{payload, instruments, res, cands, filtered,
 * steady, timing}>. tSeconds: the set's capture time (unix s live; any increasing clock works for the filters).
 */
export class Tracker {
  constructor({ rig, net = null, hyst = true } = {}) {
    this.rigm = loadRig(rig);
    this.net = net;
    this.wb = new WhiteBalance();
    this.hyst = hyst ? new CellHysteresis(this.rigm.hyst) : null;
    this.filters = Object.fromEntries(this.rigm.ports.map((p) => [p.name, new TipFilter(this.rigm.filter)]));
    this.steady = new SteadyTips(this.rigm.ports, this.rigm.cams, this.rigm.steady);
    this.prev = {};
    this.timing = {};
    this.n = 0;
    this.model = this.rigm.model;
  }

  get names() { return this.rigm.names; }

  /** Forget the temporal state (a new source). The white-balance counter is kept, as on the server. */
  reset() {
    Object.values(this.filters).forEach((f) => f.reset());
    this.prev = {};
    this.steady.reset();
    this.hyst?.reset();
  }

  /** Frames -> WB'd 512x288 BGR smalls and their grays, in rig order of the cameras present. */
  preprocess(images) {
    const present = this.names.filter((n) => images[n]);
    this.wb.tick();
    const smalls = present.map((n) => this.wb.apply(n, toSmall(images[n])));
    return { present, smalls, grays: smalls.map(gray) };
  }

  async track(images, tS, { frameId = null, nowS = null } = {}) {
    const t0 = now();
    const { present, smalls, grays } = this.preprocess(images);
    const t1 = now();
    const y = present.length ? await this.net.run(toTensor(smalls)) : null;
    const t2 = now();
    const sizes = Object.fromEntries(present.map((n) => [n, [images[n].width, images[n].height]]));
    const out = this._finish(y, present, grays, sizes, tS, { frameId, nowS, t0, t1, t2 });
    return { ...out, smalls };
  }

  /** Whether trackFrames can take these frames: a WebGPU net, and every frame at the net's frame size (1280x720). */
  framesOk(frames) {
    const g = this.net?.gpu;
    if (!g || g.lost) return false;
    const fs = Object.values(frames).filter(Boolean);
    return fs.length > 0 && fs.length <= 3 && fs.every((f) => (f.displayWidth ?? f.width) === g.size[0] && (f.displayHeight ?? f.height) === g.size[1]);
  }

  /**
   * The WebGPU path (ADR-150): frames {CAMk: VideoFrame | ImageBitmap} straight to the GPU (gpu_net.runFrames: copy,
   * the exact resize, white balance and gray in WGSL, the forward, one readback), then the same decode, solve, filter
   * and steady chain as track(). The white-balance state is this.wb's, as in track(). Does not close the frames.
   */
  async trackFrames(frames, tS, { frameId = null, nowS = null } = {}) {
    const t0 = now();
    const present = this.names.filter((n) => frames[n]);
    this.wb.tick();
    const due = present.map((n) => this.wb.due(n));
    const r = await this.net.gpu.runFrames(present.map((n) => frames[n]), {
      need: due.some(Boolean),
      onHist: (h, nPix) => present.forEach((n, i) => { if (due[i]) this.wb.set(n, wbGainsFromHist(h.subarray(i * 768, (i + 1) * 768), nPix)); }),
      luts: () => present.map((n) => this.wb.lut.get(n) ?? null),
    });
    const t2 = now();
    const grays = r.grays.map((d) => ({ width: IN_W, height: IN_H, data: d }));
    const sizes = Object.fromEntries(present.map((n) => [n, [frames[n].displayWidth ?? frames[n].width, frames[n].displayHeight ?? frames[n].height]]));
    const gt = this.net.gpu.timing;
    return this._finish(r.y, present, grays, sizes, tS, { frameId, nowS, t0, t1: t0 + (gt.pre_ms ?? 0), t2, gpu: true });
  }

  /** The chain after the forward, shared by track() and trackFrames(). */
  _finish(y, present, grays, sizes, tS, { frameId, nowS, t0, t1, t2, gpu = false }) {
    let cands = {};
    if (present.length) {
      sigmoidHeat(y, present.length);
      const cs = Object.fromEntries(present.map((n) => [n, this.rigm.cams[this.names.indexOf(n)].size]));
      cands = decodeSet(y, present, cs, { hyst: this.hyst, k: this.rigm.net.k ?? 4, thresh: this.rigm.net.thresh ?? 0.2 });
    }
    const t3 = now();
    const out = this.post(cands, Object.fromEntries(present.map((n, i) => [n, grays[i]])), tS, present);
    const t4 = now();
    this.n += 1;
    const imageSize = Object.fromEntries(present.map((n) => {
      const full = this.rigm.cams[this.names.indexOf(n)].size;
      const [w, h] = sizes[n];
      return [n, (w === full[0] && h === full[1]) ? [w, h] : full.slice()];
    }));
    this.timing = { pre_ms: t1 - t0, net_ms: t2 - t1, decode_ms: t3 - t2, solve_post_ms: t4 - t3, total_ms: t4 - t0, path: gpu ? 'webgpu-frames' : 'js-pre' };
    const payload = twinTips(this.rigm, out.instruments, { frameId: frameId ?? this.n, t: tS, model: this.model, imageSize,
      nowS: nowS ?? tS, extra: { tracker: 'browser' } });
    return { ...out, cands, payload, timing: this.timing, grays };
  }

  /** Everything after the net: candidates {CAMk: {tip, hinge}} (full-res px) + grays {CAMk: 512x288 gray} -> output. */
  post(cands, grays, tS, present = Object.keys(cands)) {
    const { cams, ports, names } = this.rigm;
    const byIdx = {}, gIdx = {};
    for (const n of names) if (cands[n]) byIdx[names.indexOf(n)] = cands[n];
    for (const n of names) if (grays[n]) gIdx[names.indexOf(n)] = grays[n];
    const res = solveFrame(byIdx, cams, ports, Object.keys(gIdx).length ? gIdx : null, this.prev, this.rigm.solve);
    if (this.hyst) this.hyst.afterSolve(Object.fromEntries(Object.entries(res).map(([p, pt]) => [p, pt && { ...pt }])), names, cams, projectPx, JAW_M);
    const filtered = {};
    for (const p of ports) filtered[p.name] = this.filters[p.name].update(res[p.name] ? res[p.name].tip : null, tS);
    const raw = { ...filtered };
    const mode = {};
    let steadyOut = null;
    if (present.length) {
      const k0 = names.indexOf(present[0]);
      const g0 = gIdx[k0];
      const sc = g0 ? g0.width / cams[k0].size[0] : 1.0;
      steadyOut = this.steady.step(filtered, tS, gIdx, sc);
      for (const [pn, o] of Object.entries(steadyOut)) {
        filtered[pn] = o === null ? null : o[0];
        mode[pn] = o === null ? null : o[1];
      }
    }
    const instruments = {};
    for (const p of ports) {
      const pt = res[p.name];
      const tip = filtered[p.name];
      if (tip == null) { delete this.prev[p.name]; continue; }
      if ((mode[p.name] ?? 'measured') !== 'estimated') this.prev[p.name] = tip;
      instruments[p.name] = instrument(p, tip, pt, this.filters[p.name], mode[p.name], names, cams, present);
    }
    return { res, filtered: raw, steady: steadyOut, instruments };
  }
}

/** The server's display white balance (/api/wb): {CAMk: [gR, gG, gB]}, p97 per channel of a ~160x90 view,
 *  g = clip(max / p, 1, 3), low-passed (g0 + 0.3 (g - g0)) and recomputed at most every `everyS` seconds. */
export class DisplayWB {
  constructor({ everyS = 0.5, alpha = 0.3 } = {}) { this.everyS = everyS; this.alpha = alpha; this.g = {}; this.at = {}; }

  update(name, img, tS = Date.now() / 1e3) {
    if (this.at[name] != null && tS - this.at[name] < this.everyS) return this.g[name];
    this.at[name] = tS;
    const { width: w, height: h, data } = img;
    const ch = img.channels ?? (data.length === 3 * w * h ? 3 : 4);
    const sx = Math.max(1, Math.floor(w / 160)), sy = Math.max(1, Math.floor(h / 90));
    const hist = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
    let n = 0;
    for (let y = 0; y < h; y += sy) for (let x = 0; x < w; x += sx, n++) {
      const o = (y * w + x) * ch;
      hist[0][data[o]]++; hist[1][data[o + 1]]++; hist[2][data[o + 2]]++;
    }
    const p97 = hist.map((hh) => { const k = 0.97 * (n - 1); let c = 0; for (let v = 0; v < 256; v++) { c += hh[v]; if (c > k) return v; } return 255; });
    const m = Math.max(...p97);
    const g = p97.map((p) => Math.min(Math.max(m / Math.max(p, 1), 1), 3));       // RGB order
    const g0 = this.g[name];
    this.g[name] = g0 ? g0.map((x, i) => x + this.alpha * (g[i] - x)) : g;
    return this.g[name];
  }

  get gains() { return { ...this.g }; }
}
