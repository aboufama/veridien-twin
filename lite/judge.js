// Development / judge page of the lite renderer (docs/WEB_TWIN.md §18.1): window.twinJudge with the subset the
// judges use (src/veridien/webtwin/judges.py capture_twin, judge_mitsuba, judge_perf), backed by LiteRenderer.
//   lite/judge.html            ?size=320x180 renders every view at that size (small-first runs)
//                              ?view=SCOPE shows that view on the page canvas after each render (headless: unused)
import { loadScene } from '../gpu/scene_gpu.js';
import * as K from '../control/kinematics.js';
import { Limits } from '../control/limits.js';
import { LiteRenderer } from './renderer.js';

const Q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
let readyResolve, readyReject;
const ready = new Promise((a, b) => { readyResolve = a; readyReject = b; });
const page = { r: null, scene: null, limits: null, state: null, info: {} };

async function boot() {
  const t0 = performance.now();
  const bundle = await loadScene('../assets/');
  const physics = await (await fetch('/veridien-twin/configs/twin/physics.json', { cache: 'no-store' })).json();
  const t1 = performance.now();
  const r = await LiteRenderer.create({ scene: bundle, physics, assets: '../assets/', log: (...a) => console.log('[lite]', ...a) });
  const size = Q.get('size');
  if (size) { const [w, h] = size.split('x').map(Number); for (const n of r.cameras) r.setViewSize(n, [w, h]); }
  page.r = r; page.scene = bundle.scene; page.meshes = bundle.meshes;
  page.limits = new Limits(bundle.scene, bundle.meshes);
  page.state = page.limits.clampState(K.defaultState(bundle.scene)).state;
  r.setPoses(K.posesFromState(page.scene, page.state));
  await r.renderCamera(r.cameras.includes('SCOPE') ? 'SCOPE' : 'CAM1');
  const t2 = performance.now();
  page.info = { fetchMs: t1 - t0, createMs: r.stats.initMs, firstFrameMs: t2 - t0, renderer: 'lite', stats: r.stats, adapter: r.adapterInfo };
  $('status').textContent = `lite renderer: ready (${(t2 - t0).toFixed(0)} ms to first frame; ${r.stats.staticTriangles + r.stats.dynamicTriangles} triangles, ${r.stats.nodes} nodes)`;
}

function camTruth(truth, name, vis) {
  return truth.instruments.map((ins) => ({ ...ins, px: truth.px[name]?.[ins.id] ?? null, in_image: truth.in_image[name]?.[ins.id] ?? null, visible: vis[ins.id] ?? null }));
}

const twinJudge = {
  ready,
  /** render(state, {cameras, linear, rgb8, exposure, state1, converge, sensor}) -> {CAM1: {linear, rgb8, width, height, truth}} */
  async render(state, opts = {}) {
    await ready;
    const r = page.r, scene = page.scene;
    const names = opts.cameras ?? scene.cameras.map((c) => c.name);
    const st0 = page.limits.clampState({ ...K.defaultState(scene), ...state }).state;
    const st1 = opts.state1 ? page.limits.clampState({ ...st0, ...opts.state1 }).state : st0;
    r.setPoses(K.posesFromState(scene, st0), K.posesFromState(scene, st1));
    if (opts.converge !== 0) await r.converge(opts.converge ?? 64);
    const truth = K.truthFromState(scene, K.lerpState(st0, st1, 0.5));
    const kps = K.keypointList(truth);
    const out = {};
    for (const name of names) {
      const vis = r.visible(name, kps.map((x) => x.p));
      const vm = {};
      kps.forEach((k, i) => { (vm[k.id] ??= {})[k.kp] = vis[i]; });
      await r.renderCamera(name, opts.exposure != null ? { exposure: opts.exposure } : {});
      const rec = { truth: { t: truth.t, instruments: camTruth(truth, name, vm) }, width: r.cams[name].size[0], height: r.cams[name].size[1] };
      if (opts.linear) rec.linear = await r.readback(name, 'linear');
      if (opts.rgb8 !== false || opts.sensor) {
        r.sensor(name, { seed: opts.seed ?? 0, frame: opts.frame ?? 0, ...(opts.exposure != null ? { exposure_s: opts.exposure } : {}) });
        if (opts.rgb8 !== false) rec.rgb8 = await r.readback(name, 'rgb8');
      }
      out[name] = rec;
    }
    const view = Q.get('view');
    if (view && out[view]) r.present(view, $('view').getContext('webgpu'), { fit: 'contain' });
    return out;
  },
  /** GPU ms per view at a size: n frames of one view (camera pass + sensor), poses fixed (radiosity solved once),
   * then n frames with the tools moving (radiosity update every frame). Returns the timestamp-query medians. */
  async bench({ view = 'CAM1', size = null, frames = 60, moving = true } = {}) {
    await ready;
    const r = page.r, scene = page.scene;
    const prev = [...r.cams[view].size];
    if (size) r.setViewSize(view, size);
    const base = page.limits.clampState(K.defaultState(scene)).state;
    r.setPoses(K.posesFromState(scene, base));
    await r.converge(64);
    const t0 = r.timer.records.length;
    const wall = [];
    for (let i = 0; i < frames; i++) {
      const s = structuredClone(base);
      if (moving) for (const q of s.instruments) { q.yaw += 0.05 * Math.sin(i * 0.21 + q.yaw); q.jaw = 0.3 + 0.3 * Math.sin(i * 0.37); }
      const w0 = performance.now();
      r.setPoses(K.posesFromState(scene, page.limits.clampState(s).state));
      await r.renderCamera(view, { await: false });
      r.sensor(view, { frame: i });
      await r.device.queue.onSubmittedWorkDone();
      wall.push(performance.now() - w0);
      if (i % 20 === 19) await new Promise((res) => setTimeout(res, 50));       // §12: idle gaps
    }
    await new Promise((res) => setTimeout(res, 200));
    const recs = r.timer.records.slice(t0);
    const sum = {};
    const perFrame = new Map();
    for (const x of recs) { const f = perFrame.get(x.frame) || 0; perFrame.set(x.frame, f + x.ms); for (const [k, ms] of Object.entries(x.passes)) (sum[k.replace(/ (CAM\d|SCOPE)$/, '')] ??= []).push(ms); }
    const med = (a) => { const s = [...a].sort((p, q) => p - q); return s.length ? s[s.length >> 1] : null; };
    const p95 = (a) => { const s = [...a].sort((p, q) => p - q); return s.length ? s[Math.min(s.length - 1, Math.floor(0.95 * s.length))] : null; };
    const out = { view, size: [...r.cams[view].size], frames, moving, passes: Object.fromEntries(Object.entries(sum).map(([k, v]) => [k, { median: med(v), p95: p95(v), n: v.length }])),
      frame_gpu_ms: { median: med([...perFrame.values()]), p95: p95([...perFrame.values()]) }, wall_ms: { median: med(wall), p95: p95(wall) }, maxSubmit: r.timer.maxSubmit, over: r.timer.over.length };
    if (size) r.setViewSize(view, prev);
    return out;
  },
  timings: () => page.r?.timings(),
  state: () => structuredClone(page.state),
  setState(s) { page.state = page.limits.clampState({ ...page.state, ...s }).state; return page.state; },
  defaultState: () => page.limits.clampState(K.defaultState(page.scene)).state,
  poses: (state) => K.posesFromState(page.scene, page.limits.clampState({ ...K.defaultState(page.scene), ...state }).state),
  info: () => ({ ...page.info, cameras: page.r?.cameras, lastError: page.r?.lastError ?? null }),
  get renderer() { return page.r; },
};
window.twinJudge = twinJudge;

boot().then(() => readyResolve(true)).catch((err) => {
  console.error(err);
  $('status').textContent = `could not start: ${err?.message || err}`;
  readyReject(err);
});
