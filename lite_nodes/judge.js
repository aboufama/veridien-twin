// Judge page for lite_nodes: window.twinJudge as judges.py capture expects (render -> {CAM: {linear, width, height}}),
// plus bench() for the speed proxy. States are rendered as given (the driver passes the clamped judge states).
import { loadScene } from '../gpu/scene_gpu.js';
import * as K from '../control/kinematics.js';
import { LiteNodesRenderer } from './renderer.js';

const Q = new URLSearchParams(location.search);
const $log = document.getElementById('log');
const say = (s) => { $log.textContent += '\n' + s; console.log('[judge]', s); };
let readyResolve, readyReject;
const ready = new Promise((a, b) => { readyResolve = a; readyReject = b; });
const app = {};

async function boot() {
  const t0 = performance.now();
  const bundle = await loadScene('../assets/');
  let physics = null;
  try { const r = await fetch('/veridien-twin/configs/twin/physics.json'); if (r.ok) physics = await r.json(); } catch {}
  app.scene = bundle.scene; app.physics = physics || {};
  app.tFetch = performance.now() - t0;
  const tracerOpts = {};
  if (Q.get('nodeSpacing')) tracerOpts.nodeSpacing = Number(Q.get('nodeSpacing'));
  if (Q.get('gatherK')) tracerOpts.gatherK = Number(Q.get('gatherK'));
  if (Q.get('skipInterior')) tracerOpts.skipInterior = Q.get('skipInterior') === '1';
  app.r = await LiteNodesRenderer.create({ scene: bundle, physics: app.physics, tracerOpts, debug: Number(Q.get('debug') || 0), fast: Q.get('fast') === '1' });
  app.tLoad = performance.now() - t0;
  app.state = K.defaultState(app.scene);
  app.r.setPoses(K.posesFromState(app.scene, app.state));
  await app.r.renderCamera('SCOPE');
  app.r.sensor('SCOPE');
  app.r.present('SCOPE', document.getElementById('view').getContext('webgpu'));
  await app.r.idle();
  app.tFirst = performance.now() - t0;
  say(`ready: fetch ${app.tFetch.toFixed(0)} ms, renderer ${app.tLoad.toFixed(0)} ms, first frame ${app.tFirst.toFixed(0)} ms`);
}

let lastKey = null;
const twinJudge = {
  ready,
  async render(state, opts = {}) {
    await ready;
    const r = app.r;
    const names = opts.cameras ?? app.scene.cameras.map((c) => c.name);
    const st0 = { ...K.defaultState(app.scene), ...state };
    const st1 = opts.state1 ? { ...st0, ...opts.state1 } : null;
    const p0 = K.posesFromState(app.scene, st0), p1 = st1 ? K.posesFromState(app.scene, st1) : null;
    r.setPoses(p0, p1);
    const key = JSON.stringify([st0, st1]);
    if (opts.converge !== 0 && key !== lastKey) { await r.converge(opts.converge ?? 200); lastKey = key; }
    const out = {};
    for (const name of names) {
      await r.renderCamera(name);
      const [w, h] = r.cams[name].size;
      const rec = { width: w, height: h };
      if (opts.linear !== false) rec.linear = await r.readback(name, 'linear');
      if (opts.rgb8) { r.sensor(name, {}); await r.idle(); r.sensor(name, {}); rec.rgb8 = await r.readback(name, 'rgb8'); }
      out[name] = rec;
    }
    return out;
  },
  /** Speed proxy: n frames per camera with the tools moving (incremental radiosity + dynamic shadow maps each frame). */
  async bench({ cameras = ['SCOPE', 'CAM1'], frames = 60, sensor = true } = {}) {
    await ready;
    const r = app.r, res = {};
    const base = K.defaultState(app.scene);
    for (const name of cameras) {
      r.hist.length = 0;
      const wall = [];
      for (let i = 0; i < frames; i++) {
        const st = structuredClone(base);
        for (const ins of st.instruments) { ins.insertion = (ins.insertion ?? 0.15) + 0.002 * Math.sin(i * 0.3); ins.jaw = 0.3 + 0.2 * Math.sin(i * 0.2); }
        r.setPoses(K.posesFromState(app.scene, st));
        const t = performance.now();
        await r.renderCamera(name, { wait: false });
        if (sensor) r.sensor(name, {});
        await r.device.queue.onSubmittedWorkDone();
        wall.push(performance.now() - t);
      }
      await r.idle();
      await new Promise((ok) => setTimeout(ok, 200));
      const g = r.hist.filter((x) => x.camera === name).map((x) => x.total_ms).sort((a, b) => a - b);
      const ws = [...wall].sort((a, b) => a - b);
      res[name] = { gpu_main_finish_ms_median: g[g.length >> 1], gpu_p95: g[Math.floor(g.length * 0.95)], wall_ms_median: ws[ws.length >> 1], wall_p95: ws[Math.floor(ws.length * 0.95)], n: g.length };
    }
    // shadow-map cost alone (dynamic casters, every frame) and static (on scope motion)
    const enc0 = () => r.device.createCommandEncoder();
    for (const which of [2, 1]) {
      const ts = [];
      for (let i = 0; i < 20; i++) {
        const e = enc0(); const t = performance.now();
        r._shadowPasses(e, which); r.device.queue.submit([e.finish()]);
        await r.device.queue.onSubmittedWorkDone(); ts.push(performance.now() - t);
      }
      ts.sort((a, b) => a - b);
      res[which === 2 ? 'shadow_dynamic_wall_ms' : 'shadow_static_wall_ms'] = ts[ts.length >> 1];
      const g = [];
      for (let i = 0; i < 10; i++) g.push(await r.shadowGpuMs(which));
      g.sort((a, b) => a - b);
      res[which === 2 ? 'shadow_dynamic_gpu_ms' : 'shadow_static_gpu_ms'] = g[g.length >> 1];
    }
    // radiosity incremental update cost alone (tools moved)
    {
      const ts = [];
      for (let i = 0; i < 20; i++) {
        const st = structuredClone(base);
        for (const ins of st.instruments) ins.insertion = (ins.insertion ?? 0.15) + 0.002 * Math.sin(i * 0.7);
        r.setPoses(K.posesFromState(app.scene, st));
        const t = performance.now(); r.tr._ensureRadiosity(); await r.device.queue.onSubmittedWorkDone(); ts.push(performance.now() - t);
      }
      ts.sort((a, b) => a - b);
      res.radiosity_incremental_wall_ms = ts[ts.length >> 1];
      await r.idle();
      res.tracer_timings = r.tr.timings?.();
    }
    res.load = { fetch_ms: app.tFetch, renderer_ms: app.tLoad, first_frame_ms: app.tFirst, stats: r.stats, tracerStats: r.tr.stats };
    return res;
  },
  timings: () => app.r.timings(),
  info: () => ({ renderer: 'lite_nodes', cameras: app.r.cameras, load_ms: app.tLoad, first_frame_ms: app.tFirst }),
  poses: (state) => K.posesFromState(app.scene, { ...K.defaultState(app.scene), ...state }),
};
window.twinJudge = twinJudge;
boot().then(() => readyResolve(true)).catch((e) => { say('could not start: ' + (e?.stack || e)); readyReject(e); });
