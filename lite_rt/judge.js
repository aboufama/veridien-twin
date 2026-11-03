// lite_rt judge page: window.twinJudge with the contract API (docs/WEB_TWIN.md §8) on the lite_rt renderer, for
// judges.py (capture via headless.TwinPage(path='lite_rt/judge.html')). Query: ?accum=0 (no temporal accumulation:
// every render is a single moving-scene frame), ?ls=1|2 (light samples), ?cull=0|1, ?present=1 (draw to a canvas).
import { LiteRTRenderer } from './renderer.js';
import * as K from '../control/kinematics.js';

const tLoad0 = performance.now();
const log = (s) => { const el = document.getElementById('log'); if (el) el.textContent += '\n' + s; console.log('[lite_rt judge]', s); };
const params = new URLSearchParams(location.search);
const sceneDir = params.get('scene') || '../assets/';
let resolveReady, rejectReady;
const ready = new Promise((a, b) => { resolveReady = a; rejectReady = b; });
const J = { ready, kind: 'lite_rt', renderer: null, scene: null };
window.twinJudge = J;

(async () => {
  try {
    const scene = await (await fetch(sceneDir + 'scene.json')).json();
    let physics = {};
    for (const u of [params.get('physics'), '/veridien-twin/configs/twin/physics.json'].filter(Boolean)) {
      try { const r = await fetch(u); if (r.ok) { physics = await r.json(); break; } } catch (e) { /* next */ }
    }
    const o = { scene: sceneDir, physics, log };
    if (params.has('accum')) o.accumulate = params.get('accum') !== '0';
    if (params.has('ls')) o.lightSamples = Number(params.get('ls'));
    if (params.has('cull')) o.cullShadows = params.get('cull') !== '0';
    if (params.has('cap')) o.accCap = Number(params.get('cap'));
    const r = await LiteRTRenderer.create(o);
    J.renderer = r; J.scene = scene;
    J.options = { accumulate: r.opts.accumulate, lightSamples: r.opts.lightSamples, accCap: r.opts.accCap, cullShadows: o.cullShadows ?? r.tr.opts.cullShadows };
    // load to first frame: the default state, SCOPE, presented
    r.setPoses(K.posesFromState(scene, K.defaultState(scene)));
    await r.renderCamera('SCOPE', {});
    const cv = document.getElementById('view');
    if (cv) r.present('SCOPE', cv.getContext('webgpu'));
    await r.idle();
    J.loadMs = performance.now() - tLoad0;
    J.loadFromNavMs = performance.now();
    log(`ready; first SCOPE frame ${J.loadMs.toFixed(0)} ms after module start (${J.loadFromNavMs.toFixed(0)} ms after navigation)`);
    resolveReady(true);
  } catch (e) { log('ERROR ' + (e.stack || e)); rejectReady(e); }
})();

J.poses = (state) => K.posesFromState(J.scene, state);
J.defaultState = () => K.defaultState(J.scene);
J.timings = () => J.renderer.timings();
J.info = () => ({ adapter: J.renderer.adapterInfo, stats: J.renderer.stats, opts: J.options, loadMs: J.loadMs, loadFromNavMs: J.loadFromNavMs });

// render(state, {cameras, linear, rgb8, state1, seed, repeat, accumulate})
J.render = async (state, o = {}) => {
  const r = J.renderer;
  const p0 = K.posesFromState(J.scene, state);
  const p1 = o.state1 ? K.posesFromState(J.scene, o.state1) : null;
  r.setPoses(p0, p1);
  const out = { _meta: {} };
  for (const cam of o.cameras || Object.keys(r.cameras)) {
    const reps = o.repeat ?? 1;
    let acc = null;
    for (let k = 0; k < reps; k++) {
      await r.renderCamera(cam, { seed: (o.seed ?? 0) + 7919 * k, accumulate: o.accumulate });
      if (o.linear === false) continue;
      const lin = await r.readback(cam, 'linear');
      if (!acc) acc = new Float64Array(lin.length);
      for (let i = 0; i < lin.length; i++) acc[i] += lin[i];
    }
    const [w, h] = r.cameras[cam].size;
    if (acc) {
      const mean = new Float32Array(acc.length);
      for (let i = 0; i < acc.length; i++) mean[i] = acc[i] / reps;
      out[cam] = { linear: mean, width: w, height: h };
    } else out[cam] = { width: w, height: h };
    if (o.rgb8) out[cam].rgb8 = await r.readback(cam, 'rgb8');
  }
  await r.idle();
  out._meta.timings = r.timings();
  return out;
};

// speed: render `name` repeatedly for `seconds` with the instruments moving (every frame a reset, no accumulation)
// or still (accumulating); returns per-frame GPU ms by pass label from the timestamp queries.
J.speed = async (name = 'SCOPE', seconds = 5, o = {}) => {
  const r = J.renderer, scene = J.scene;
  const base = K.defaultState(scene);
  const stateAt = (t) => ({ ...base, t, instruments: base.instruments.map((q, i) => ({ ...q, yaw: q.yaw + 0.08 * Math.sin(2 * t + i), pitch: q.pitch + 0.05 * Math.cos(1.3 * t + i), insertion: q.insertion + 0.01 * Math.sin(0.7 * t + i), jaw: 0.4 + 0.3 * Math.sin(3 * t + i) })) });
  const t0 = performance.now();
  const wall = [];
  let k = 0;
  const f0 = r.frameNo + 1;
  while (performance.now() - t0 < seconds * 1000) {
    const ts = performance.now();
    const st = o.moving === false ? base : stateAt(k / 60);
    r.setPoses(K.posesFromState(scene, st));
    const names = Array.isArray(name) ? name : [name];
    for (const n of names) await r.renderCamera(n, {});
    if (o.present && document.getElementById('view')) r.present(names[0], document.getElementById('view').getContext('webgpu'));
    await r.idle();
    wall.push(performance.now() - ts);
    k++;
  }
  await r.idle();
  await new Promise((a) => setTimeout(a, 200));
  const frames = r.tr.timer.frameRecords().filter((f) => f.frame >= f0);
  return { frames, wall, n: k, adapter: r.adapterInfo, name };
};
