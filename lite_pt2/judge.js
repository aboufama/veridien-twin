// Judge page of the lite_pt2 candidate: window.twinJudge as judges.py (capture_twin / judge_mitsuba / TwinPage) expects.
// URL: lite_pt2/judge.html?factor=3&acc=16&sigma=1.0&edgeMax=0.006&spp=1&refine=0&ls=1&live=1 (the defaults)
//   live=1: the judge's render is ONE live frame (seed from the judge, no accumulation)
//   live=0 / settle=N: N accumulated frames (what a still view converges to) before the read-back
import { loadScene } from '../gpu/scene_gpu.js';
import * as K from '../control/kinematics.js';
import { Limits } from '../control/limits.js';
import { LitePTRenderer } from './renderer.js';

const Q = new URLSearchParams(location.search);
const num = (k, d) => (Q.get(k) != null ? Number(Q.get(k)) : d);
const msg = (t) => { document.getElementById('msg').textContent = t; };
const app = {};

async function boot() {
  const t0 = performance.now();
  const bundle = await loadScene('../assets/');
  let physics = {};
  try { const r = await fetch('/veridien-twin/configs/twin/physics.json'); if (r.ok) physics = await r.json(); } catch { /* static server */ }
  app.scene = bundle.scene; app.bundle = bundle; app.physics = physics;
  app.limits = new Limits(bundle.scene, bundle.meshes);
  app.r = await LitePTRenderer.create({
    scene: bundle, physics, preset: Q.get('preset') || 'live', factor: num('factor', 3), accumulate: num('acc', 16),
    sigma: num('sigma', 1.0), edgeMax: num('edgeMax', 0.006), spp: num('spp', 1), refine: num('refine', 0), lightSamples: num('ls', 1),
    ...(Q.get('radEvery') ? { radiosityEvery: num('radEvery', 2) } : {}),
    // any TwinRenderer option as tr_<name>=<number> (e.g. tr_nodeSpacing=0.005, tr_gatherK=32)
    ...Object.fromEntries([...Q.entries()].filter(([k]) => k.startsWith('tr_')).map(([k, v]) => [k.slice(3), Number(v)])),
  });
  app.loadMs = performance.now() - t0;
  // first frame of the default state, presented (load-to-first-frame)
  const st = app.limits.clampState({ ...K.defaultState(app.scene) }).state;
  app.r.setPoses(K.posesFromState(app.scene, st));
  await app.r.renderCamera('SCOPE');
  app.r.sensor('SCOPE');
  app.r.present('SCOPE', document.getElementById('view'));
  await app.r.device.queue.onSubmittedWorkDone();
  app.firstFrameMs = performance.now() - t0;
  // lite_pt2: the radiosity gather table builds after create(); the judge's renders and its speed bench start once
  // it is in (converge() waits for it too)
  await app.r.gatherReady;
  app.gatherReadyMs = performance.now() - t0;
  msg(`lite_pt2 ready: load ${app.loadMs.toFixed(0)} ms, first frame ${app.firstFrameMs.toFixed(0)} ms`);
}

let readyResolve, readyReject;
const ready = new Promise((a, b) => { readyResolve = a; readyReject = b; });
boot().then(readyResolve, (e) => { msg(`error: ${e.message}`); console.error(e); readyReject(e); });

const camSize = (n) => app.r.cameraSize(n);

window.twinJudge = {
  ready,
  setState(state) { return app.limits.clampState({ ...K.defaultState(app.scene), ...state }).state; },
  async poses(state) { return K.posesFromState(app.scene, this.setState(state)); },
  /** render(state, {cameras, linear, rgb8, exposure, state1, seed, settle}) -> {CAM1: {linear, rgb8, width, height, truth}} */
  async render(state, opts = {}) {
    await ready;
    const r = app.r;
    const names = opts.cameras ?? ['CAM1', 'CAM2', 'CAM3'];
    const st0 = this.setState(state);
    const st1 = opts.state1 ? app.limits.clampState({ ...st0, ...opts.state1 }).state : st0;
    r.setPoses(K.posesFromState(app.scene, st0), K.posesFromState(app.scene, st1));
    if (opts.converge !== 0) await r.converge(opts.converge ?? 200);
    const live = num('live', 1) === 1 && !opts.settle;
    const settle = opts.settle ?? num('settle', 16);
    const truth = K.truthFromState(app.scene, K.lerpState(st0, st1, 0.5));
    const out = {};
    for (const name of names) {
      const ex = opts.exposure != null ? { exposure: opts.exposure } : {};
      if (live) await r.renderCamera(name, { seed: opts.seed ?? 0, ...ex });
      else for (let k = 0; k < settle; k++) { await r.renderCamera(name, { ...ex }); if (k % 4 === 3) await r.device.queue.onSubmittedWorkDone(); }
      r.sensor(name, { seed: opts.seed ?? 0, frame: opts.frame ?? 0, ...(opts.exposure != null ? { exposure_s: opts.exposure } : {}) });
      const [w, h] = camSize(name);
      const rec = { width: w, height: h, truth: { t: truth.t, scope: truth.scope } };
      if (opts.linear) rec.linear = await r.readback(name, 'linear');
      if (opts.rgb8 !== false) rec.rgb8 = await r.readback(name, 'rgb8');
      out[name] = rec;
    }
    if (names.includes('SCOPE') || opts.present) r.present(names.includes('SCOPE') ? 'SCOPE' : names[0], document.getElementById('view'));
    return out;
  },
  /** GPU ms per frame of one camera: n live frames (moving: a new pose each frame, so no accumulation skip). */
  async bench(name = 'SCOPE', n = 30, { moving = true, state = null } = {}) {
    await ready;
    const r = app.r;
    const st = this.setState(state || {});
    const frames = [];
    r.inner.timer.frames.clear();
    const t0 = performance.now();
    for (let k = 0; k < n; k++) {
      const s = JSON.parse(JSON.stringify(st));
      if (moving) s.instruments.forEach((q, i) => { q.yaw += 0.002 * Math.sin(k * 0.3 + i); q.jaw = Math.max(0, Math.min(1, q.jaw + 0.05 * Math.sin(k * 0.2))); });
      if (moving === 'scope' && s.scope) { s.scope.yaw += 0.003 * Math.sin(k * 0.25); s.scope.pitch += 0.002 * Math.cos(k * 0.2); }
      r.setPoses(K.posesFromState(app.scene, s));
      r.inner.frame++;
      await r.renderCamera(name, moving ? {} : {});
      r.sensor(name);
      if (name === 'SCOPE') r.present(name, document.getElementById('view'));
      await r.device.queue.onSubmittedWorkDone();
      if (k % 20 === 19) await new Promise((res) => setTimeout(res, 50));
    }
    const wall = (performance.now() - t0) / n;
    await new Promise((res) => setTimeout(res, 200));
    const recs = r.inner.timer.frameRecords().slice(-n);
    for (const f of recs) {
      const passes = Object.fromEntries(Object.entries(f).filter(([k]) => !['frame', 'submits', 'maxSubmit'].includes(k)));
      frames.push({ total: Object.values(passes).reduce((a, b) => a + b, 0), passes, submits: f.submits, maxSubmit: f.maxSubmit });
    }
    return { name, n, wallMs: wall, frames, loadMs: app.loadMs, firstFrameMs: app.firstFrameMs, fallback: await r.fallbackCount(), gpu: r.inner.timer.gpu, rates: { ...r.inner.timer.rates } };
  },
  /** SCOPE (no Mitsuba reference yet): the lite live frame against the inner path tracer at full resolution and judge
   * settings (spp 4, refine 6, 4 light samples), the validated transport, on the same poses. Luminance |rel| stats. */
  async scopeCheck(state = {}, seed = 1) {
    await ready;
    const r = app.r;
    const st = this.setState(state);
    r.setPoses(K.posesFromState(app.scene, st));
    await r.converge(200);
    await r.renderCamera('SCOPE', { seed });
    const A = await r.readback('SCOPE', 'linear');
    await r.inner.renderCamera('SCOPE', { seed, spp: 4, refine: 6, lightSamples: 4, refineLightSamples: 1 });
    await r.device.queue.onSubmittedWorkDone();
    const B = await r.readback('SCOPE', 'linear');
    const [w, h] = r.cameraSize('SCOPE');
    const Y = (X, i) => 0.2126 * X[4 * i] + 0.7152 * X[4 * i + 1] + 0.0722 * X[4 * i + 2];
    const ys = []; for (let i = 0; i < w * h; i++) ys.push(Y(B, i));
    const sorted = Float64Array.from(ys).sort(); const p99 = sorted[Math.floor(0.99 * sorted.length)];
    const rel = []; let sa = 0, sb = 0;
    const r2 = (0.95 * w / 2) ** 2;
    for (let i = 0; i < w * h; i++) { const yb = ys[i], px = i % w - (w / 2 - 0.5), py = Math.floor(i / w) - (h / 2 - 0.5); if (yb > 0.02 * p99 && px * px + py * py < r2) { const ya = Y(A, i); rel.push(Math.abs(ya - yb) / yb); sa += ya; sb += yb; } }
    rel.sort((a, b) => a - b);
    // leave the history consistent: the next lite frame re-renders
    r.state.SCOPE.accKey = null;
    return { n: rel.length, median_rel: rel[Math.floor(0.5 * rel.length)], p95_rel: rel[Math.floor(0.95 * rel.length)], level: sa / sb };
  },
  timings() { return app.r ? app.r.timings() : {}; },
  info() { return { loadMs: app.loadMs, firstFrameMs: app.firstFrameMs, cameras: app.r.cameras, stats: app.r.stats }; },
};
