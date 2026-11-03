// Judge page of the godview_pt candidate (docs/WEB_TWIN.md §19): window.godJudge.
//   ready; capture(viewpoint, state, {spp, mode}) -> base64 Float32Array (w*h*3, linear RGB, top row first) at
//   godview.yaml image size, eye from the viewpoint formula, cutaway on, all lights.
// URL: judge.html?gain=<scope gain>&exposure=<display exposure>&spp=<default spp>&bounces=<max bounces>
import { loadScene } from '../gpu/scene_gpu.js';
import * as K from '../control/kinematics.js';
import { Limits } from '../control/limits.js';
import { GodView, viewpointEye } from './godview.js';

const Q = new URLSearchParams(location.search);
const num = (k, d) => (Q.get(k) != null && Q.get(k) !== '' ? Number(Q.get(k)) : d);
const msg = (t) => { document.getElementById('msg').textContent = t; };
const app = {};

async function getJSON(urls) {
  for (const u of urls) { try { const r = await fetch(u, { cache: 'no-store' }); if (r.ok) return await r.json(); } catch { /* next */ } }
  throw new Error(`none of ${urls.join(', ')} loaded`);
}

async function boot() {
  const t0 = performance.now();
  const [bundle, physics, godview] = await Promise.all([
    loadScene('../assets/'), getJSON(['/veridien-twin/configs/twin/physics.json', './physics.json']), getJSON(['/veridien-twin/configs/twin/godview.json', './godview.json']),
  ]);
  if (Q.get('gain')) godview.lights.scope.gain = num('gain', 1);
  if (Q.get('exposure')) godview.display.exposure = num('exposure', 1);
  app.scene = bundle.scene; app.godview = godview; app.physics = physics;
  const spacing = num('bakeSpacing', 0.008);
  app.g = await GodView.create({ canvas: document.getElementById('view'), scene: bundle, physics, godview, options: { maxBounces: num('bounces', 16), tileMs: num('tileMs', 6), bakeSpacing: spacing, cache: !Q.get('nocache') } });
  app.loadMs = performance.now() - t0;
  app.cached = app.g.cached;
  app.limits = new Limits(bundle.scene, bundle.meshes);
  const st = app.limits.clampState({ ...K.defaultState(app.scene) }).state;
  app.g.setPoses(K.posesFromState(app.scene, st));
  app.g.render();
  await app.g.device.queue.onSubmittedWorkDone();
  app.firstFrameMs = performance.now() - t0;
  msg(`godview_pt3 ready: load ${app.loadMs.toFixed(0)} ms, first frame ${app.firstFrameMs.toFixed(0)} ms`);
}

let readyResolve, readyReject;
const ready = new Promise((a, b) => { readyResolve = a; readyReject = b; });
boot().then(readyResolve, (e) => { msg(`error: ${e.message}`); console.error(e); readyReject(e); });

function b64(f32) {
  const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}

function setView(name, state) {
  const st = app.limits.clampState({ ...K.defaultState(app.scene), ...(state || {}) }).state;
  app.g.setPoses(K.posesFromState(app.scene, st));
  app.g.setCamera({ eye: viewpointEye(app.godview, name), target: app.godview.centre, up: [0, 0, 1], vfov_deg: app.godview.vfov_deg });
  const cq = Q.get('cut');                       // ?cut=fixed judges the fixed section (refs /workspace/godrefs_fixed)
  app.g.setCutaway(cq == null || cq === '' ? true : cq === '0' ? false : cq);
}

const lum = (A, i) => 0.2126 * A[3 * i] + 0.7152 * A[3 * i + 1] + 0.0722 * A[3 * i + 2];

window.godJudge = {
  ready,
  async capture(name, state, opts = {}) {
    await ready;
    setView(name, state);
    const [w, h] = opts.size || app.godview.image;
    const out = await app.g.captureLinear(w, h, { spp: opts.spp ?? num('spp', 256), mode: opts.mode || Q.get('mode') || 'traced', seed: opts.seed ?? 1 });
    return b64(out);
  },
  /** progressive convergence: luminance |rel| of the n-spp image vs the final one (pixels with Y > 2 % of p99) */
  async convergence(name, state, checkpoints = [1, 2, 4, 8, 16, 32, 64, 128], final = 1024, size = null) {
    await ready;
    setView(name, state);
    const [w, h] = size || app.godview.image;
    const g = app.g;
    const snaps = {};
    const tStart = performance.now();
    const times = {};
    for (const n of checkpoints) {
      const t0 = performance.now();
      snaps[n] = await g.captureLinear(w, h, { spp: n, seed: 3 });
      times[n] = performance.now() - t0;
    }
    const t0 = performance.now();
    const F = await g.captureLinear(w, h, { spp: final, seed: 11 });
    times[final] = performance.now() - t0;
    const N = w * h, ys = new Float64Array(N);
    for (let i = 0; i < N; i++) ys[i] = lum(F, i);
    const p99 = Float64Array.from(ys).sort()[Math.floor(0.99 * N)];
    const res = {};
    for (const n of checkpoints) {
      const rel = [];
      for (let i = 0; i < N; i++) if (ys[i] > 0.02 * p99) rel.push(Math.abs(lum(snaps[n], i) - ys[i]) / ys[i]);
      rel.sort((a, b) => a - b);
      res[n] = { median_rel: rel[Math.floor(rel.length / 2)], p95_rel: rel[Math.floor(rel.length * 0.95)], ms: times[n] };
    }
    return { res, final, finalMs: times[final], totalMs: performance.now() - tStart };
  },
  /** GPU cost: raster frames at the canvas size (moving camera) and path-traced samples (still) */
  async bench(n = 60) {
    await ready;
    const g = app.g, dev = g.device, c = app.godview.centre;
    const eye0 = viewpointEye(app.godview, 'V1_surgeon');
    const ras = [];
    for (let k = 0; k < n; k++) {
      const a = 0.01 * k;
      g.setCamera({ eye: [c[0] + (eye0[0] - c[0]) * Math.cos(a) - (eye0[1] - c[1]) * Math.sin(a), c[1] + (eye0[0] - c[0]) * Math.sin(a) + (eye0[1] - c[1]) * Math.cos(a), eye0[2]], target: c, up: [0, 0, 1], vfov_deg: 35 });
      await dev.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      g.render();
      await dev.queue.onSubmittedWorkDone();
      ras.push(performance.now() - t0);
    }
    // still: wait for the still threshold, then time path-traced samples
    await new Promise((r) => setTimeout(r, 200));
    const W = g.size[0], H = g.size[1];
    g.sample = 0; g.tileRow = 0;
    const smp = [];
    for (let k = 0; k < 16; k++) {
      await dev.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      const s0 = g.sample;
      while (g.sample === s0) { g._submitBand(1, k * 977); await dev.queue.onSubmittedWorkDone(); }
      smp.push(performance.now() - t0);
    }
    const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
    await new Promise((r) => setTimeout(r, 100));
    return { timings: g.timings(), size: [W, H], raster_ms_median: med(ras.slice(5)), raster_ms_p95: [...ras.slice(5)].sort((x, y) => x - y)[Math.floor(0.95 * (ras.length - 5))], sample_ms_median: med(smp), rowsPerTile: g.rowsPerTile, loadMs: app.loadMs, firstFrameMs: app.firstFrameMs };
  },
  /** the live display path: setView, then n path-traced samples through render() (0 = the raster frame) */
  async live(name, state, n = 0) {
    await ready;
    setView(name, state);
    const g = app.g, dev = g.device;
    g.render(); await dev.queue.onSubmittedWorkDone();
    if (n > 0) {
      await new Promise((r) => setTimeout(r, 170));
      let guard = 0;
      while (g.sample < n && guard++ < 100000) { g.render(); await dev.queue.onSubmittedWorkDone(); }
      g.render(); await dev.queue.onSubmittedWorkDone();
    }
    return g.timings();
  },
  /** stale-bake probe: bake under `name`, orbit the eye about the vertical through the centre by each angle, and
   *  compare the raster drawn with the old bake against the raster with a fresh bake there (luminance |rel|) */
  async staleProbe(name, state, orbits = [30, 90], size = [640, 400]) {
    await ready;
    setView(name, state);
    const g = app.g, [w, h] = size, c = app.godview.centre, e0 = viewpointEye(app.godview, name);
    await g.captureLinear(w, h, { mode: 'raster' });
    const out = {};
    for (const deg of orbits) {
      const a = deg * Math.PI / 180, dx = e0[0] - c[0], dy = e0[1] - c[1];
      const eye = [c[0] + dx * Math.cos(a) - dy * Math.sin(a), c[1] + dx * Math.sin(a) + dy * Math.cos(a), e0[2]];
      g.setCamera({ eye, target: c, up: [0, 0, 1], vfov_deg: app.godview.vfov_deg });
      g.bakeStale = false;                 // keep the old bake (setCamera marks it stale)
      const stale = await g.captureLinear(w, h, { mode: 'raster', noBake: true });
      const fresh = await g.captureLinear(w, h, { mode: 'raster' });
      const N = w * h, ys = new Float64Array(N);
      for (let i = 0; i < N; i++) ys[i] = lum(fresh, i);
      const p99 = Float64Array.from(ys).sort()[Math.floor(0.99 * N)];
      const rel = [], sg = [];
      for (let i = 0; i < N; i++) if (ys[i] > 0.02 * p99) { const r = (lum(stale, i) - ys[i]) / ys[i]; rel.push(Math.abs(r)); sg.push(r); }
      rel.sort((x, y) => x - y); sg.sort((x, y) => x - y);
      out[deg] = { median_rel: rel[rel.length >> 1], p95_rel: rel[Math.floor(rel.length * 0.95)], p05_signed: sg[Math.floor(sg.length * 0.05)], p95_signed: sg[Math.floor(sg.length * 0.95)], n: rel.length };
      // back to the start view's bake for the next angle
      setView(name, state); await g.captureLinear(w, h, { mode: 'raster' });
    }
    return out;
  },
  async debug() { return { dbg: await app.g.debugCounters(), err: app.g.lastError || null, lost: app.g.lostInfo || null }; },
  info() { return { cached: app.cached, loadMs: app.loadMs, firstFrameMs: app.firstFrameMs, prepareMs: app.g?.prepareMs, adapter: app.g?.adapterInfo, image: app.godview?.image, gain: app.godview?.lights?.scope?.gain, exposure: app.g?.exposure }; },
  timings() { return app.g ? app.g.timings() : {}; },
};
// TwinPage (src/veridien/webtwin/headless.py) waits for window.twinJudge.ready
window.twinJudge = { ready, info: () => window.godJudge.info() };
