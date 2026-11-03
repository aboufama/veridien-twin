// Veridien dome twin, the live page (docs/WEB_TWIN.md §5-§10, the one-view page of §18.2). Owns the device-independent
// parts: scene + physics loading, the one kinematics (control/), mouse + keyboard in the main view's image directions,
// free-running camera clocks, the camera pipeline (render -> sensor -> readback -> MJPG in workers -> model socket),
// model-vs-truth markers, scripted autoplay, recording through the server, and the judge hooks (window.twinJudge,
// ?judge=1).
//
// The page. One main view fills the window (SCOPE by default, CAM1 when the renderer has no SCOPE); "4 views" (or V)
// shows SCOPE, CAM1, CAM2 and CAM3 in a 2x2 grid, a click makes a tile the main view, Esc closes. "?" lists the keys.
// One quiet status line appears only while a problem lasts. Nothing else is on screen.
//
// Display. Camera images go to the screen on the GPU (ui/present.js: the sensor texture is blitted, letterboxed, to a
// WebGPU canvas on the renderer's device; no CPU readback). Only visible views render, one renderCamera + sensor each.
//   * raster renderers (lite, stub), nothing consuming camera frames: on requestAnimationFrame, one frame in flight,
//     the period at least twice the last frame's time (the GPU idles as long as it works), 10 fps when still;
//   * the path tracer, or while a model, the ONNX slot or a recording needs every tracking camera: through the GPU
//     calendar (control/gpu_sched.js) as a viewing-aid job, so the §12 gaps and the bundle schedule hold;
//   * tracking-camera views show the bundle's own sensor texture whenever bundles run (20 fps, exactly what the model
//     receives), and judge pages (?judge=1) render nothing extra: they show what the judge rendered.
//
// The GPU calendar (control/gpu_sched.js, docs/WEB_TWIN.md §12, §17): each camera of a bundle is issued at its own
// exposure end, the next job is queued before the current one ends, a bundle is rendered whole or skipped whole
// (counted, with the reason), and an idle gap of >= 50 ms comes at least every 1.8 s. The 3 tracking cameras render
// offscreen at 20 fps only while a model, the ONNX slot or a recording is attached (§18.1). The live render seed is
// fixed per camera, so the tracer's static first-hit and occlusion caches hold (§15); sensor noise is drawn fresh.

import { loadScene } from './gpu/scene_gpu.js';
import * as K from './control/kinematics.js';
import { Limits } from './control/limits.js';
import { Controls } from './control/input.js';
import { CameraClock } from './control/clock.js';
import { makeScript, SCRIPTS } from './control/scripts.js';
import { viewBasis, toolBasis } from './control/view_basis.js';
import { OrbitView } from './ui/view.js';
import { Feeds } from './ui/feeds.js';
import { StubRenderer } from './ui/stub_renderer.js';
import { Presenter } from './ui/present.js';
import { drawMarkers, clearOverlay } from './ui/markers.js';
import { aimScope, toolsTarget } from './ui/scope_aim.js';
import { GRID, KEYS, viewLabel, initialViewState, viewReducer, visibleViews, fitCap, backingSize, displayDue, statusLine } from './ui/views.js';
import { TwinSocket } from './model/socket.js';
import { encodeJPEG, encodeJPEGYUV420, qualityTables } from './ui/jpeg_encoder.js';
import { GpuScheduler } from './control/gpu_sched.js';

const Q = new URLSearchParams(location.search);
const JUDGE = Q.get('judge') === '1';
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 1) => (x == null || !Number.isFinite(x) ? '–' : x.toFixed(d));
const DEFAULT_EXPOSURE_S = 0.010;     // assumed until the renderer's UVC auto-exposure reports one (exposureOf)
const STILL_FPS = Number(Q.get('stillfps')) || 10;     // display rate of a still scene (noise and AE stay alive)

const app = {
  scene: null, meshes: null, sceneLabel: '', physics: null, renderer: null, stub: true, kind: null, limits: null,
  state: null, controls: null, clock: null, feeds: null, socket: null,
  mode: 'manual', script: null, scriptT0: 0,
  history: [], t0: performance.now(), t: 0, lastMotion: performance.now(), stateKey: '',
  sched: null, sendChain: Promise.resolve(), inFlight: 0,
  frames: {},            // latest bundle frame per tracking camera: {cam, truth, visible, frame_id}
  lastTruth: null, lastScore: null, scores: [],
  stats: { bundles: 0, overviewMs: [], bundleMs: [], issueMs: [], stageMs: [], periodMs: [], encodeMs: [], bundleTimes: [], ovTimes: [],
    lastSent: null, traced: 0, traceReused: 0 },
  trace: {},             // per camera: the poses of its last live trace (a still scene re-senses that radiance)
  onnx: null, onnxBusy: false, frameId: 0, refs: {}, sceneRef: {},
  aidsOff: false, ovCount: 0,
  // the one-view page
  views: null, slots: {}, presenter: null, rcams: new Set(), scopeMode: null, camTex: {}, marks: {}, viewSize: {},
  disp: { at: -Infinity, ms: 0, busy: false, direct: false, dirty: true, frames: 0, frame: 0, times: [], msHist: [], raf: [], P: null, posesV: 0, traced: {}, err: null, firstAt: null },
  autoName: 'task_set', autoSeed: 0, note: null, flash: null, socketEverOpen: false, modelAttached: false,
};
// the profiling tools (web/twin/tests/ui_perf_*) read these two
Object.defineProperty(app, 'lastBundle', { get: () => app.sched?.lastAdmitted ?? -1, configurable: true });
Object.defineProperty(app, 'camBusy', { get: () => !!app.sched?.busy, configurable: true });
window.twinApp = app;

// ------------------------------------------------------------------ boot

async function sha256(text) {
  const h = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
}

async function fetchOk(url, kind = 'text') {
  try {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) return null;
    return kind === 'json' ? await r.json() : await r.text();
  } catch { return null; }
}

async function loadAll() {
  const want = Q.get('scene');
  const bases = want ? [want] : ['assets/', 'tests/stub/'];
  for (const base of bases) {
    const text = await fetchOk(base + 'scene.json');
    if (!text) continue;
    const bundle = await loadScene(base);
    app.scene = bundle.scene; app.meshes = bundle.meshes; app.bundle = bundle;
    app.sceneLabel = bundle.scene.provenance?.stub ? 'stub scene' : 'CAD export';
    app.sceneRef.scene = base + 'scene.json';
    app.sceneRef.scene_sha = await sha256(text);
    app.sceneRef.scene_commit = bundle.scene.provenance?.git_commit ?? null;
    break;
  }
  if (!app.scene) throw new Error('no scene: neither assets/scene.json nor tests/stub/scene.json could be loaded');
  // physics: parsed by the server (YAML -> JSON); a static server cannot parse YAML, so the renderer gets {}.
  app.physics = await fetchOk('/veridien-twin/configs/twin/physics.json', 'json');
  const yaml = await fetchOk('/veridien-twin/configs/twin/physics.yaml');
  app.sceneRef.physics_sha = yaml ? await sha256(yaml) : null;
  app.sceneRef.physics = app.physics ? 'configs/twin/physics.yaml' : null;
}

/** ?renderer=lite|trace|stub. Live pages default to lite, judge pages to the path tracer; each falls back in turn
 * (lite -> trace -> stub), and the status line says so. */
const RENDERER_ORDER = { lite: ['lite', 'trace', 'stub'], trace: ['trace', 'stub'], stub: ['stub'] };

async function makeRenderer() {
  const want = RENDERER_ORDER[Q.get('renderer')] ? Q.get('renderer') : (JUDGE ? 'trace' : 'lite');
  const failed = [];
  for (const kind of RENDERER_ORDER[want]) {
    const t0 = performance.now();
    try {
      setBoot(kind === 'stub' ? 'Starting the stub renderer' : kind === 'lite' ? 'Preparing the renderer' : 'Preparing the path tracer');
      const r = await createRenderer(kind);
      if (kind === 'lite') {
        try { await probeFrame(r); } catch (e) { try { r.destroy?.(); } catch { /* best effort */ } throw e; }
      }
      const s = ((performance.now() - t0) / 1000).toFixed(1);
      app.rendererInfo = kind === 'lite' ? `lite renderer${app.liteDir && app.liteDir !== 'lite' ? ` (${app.liteDir})` : ''}, ${s} s to prepare`
        : kind === 'trace' ? `WebGPU path tracer, ${app.preset} preset, ${s} s to prepare` : 'stub rasteriser';
      if (failed.length) app.fallbackFrom = failed.join(' and ');
      return { r, kind };
    } catch (err) {
      console.warn(`renderer ${kind} unavailable:`, err);
      failed.push(kind === 'lite' ? 'the lite renderer' : kind === 'trace' ? 'the path tracer' : 'the stub');
      app.rendererError = `${kind}: ${String(err?.message || err)}`;
    }
  }
  throw new Error(`no renderer could start (${app.rendererError})`);
}

async function createRenderer(kind) {
  const log = (...a) => console.log(`[twin-${kind}]`, ...a);
  if (kind === 'stub') return StubRenderer.create({ scene: app.bundle, physics: app.physics });
  if (kind === 'lite') {
    app.preset = Q.get('preset') || 'live';
    // ?lite=<dir>: a candidate lite renderer in web/twin/<dir>/renderer.js (lite_* only), for A/B through this page
    const dir = /^lite[a-z0-9_]*$/.test(Q.get('lite') || '') ? Q.get('lite') : 'lite';
    const mod = await import(`./${dir}/renderer.js`);
    const Lite = mod.LiteRenderer ?? mod.default ?? Object.values(mod).find((v) => typeof v === 'function' && typeof v.create === 'function');
    if (!Lite?.create) throw new Error(`${dir}/renderer.js has no LiteRenderer.create`);
    app.liteDir = dir;
    return Lite.create({ scene: app.bundle, physics: app.physics || {}, preset: app.preset, log });
  }
  const mod = await import('./gpu/renderer.js');
  app.preset = Q.get('preset') || (JUDGE ? 'judge' : 'live');
  // the presets count radiosityEvery in setPoses calls, and the renderer's bench calls setPoses once per frame;
  // this page calls it once per camera (each camera has its own exposure), three per bundle, so live and
  // balanced take 3x the count: the indirect light refreshes at 10 Hz, the bench's operating point (§15)
  const rad = Q.get('radEvery') != null ? Number(Q.get('radEvery')) : (mod.PRESETS?.[app.preset]?.radiosityEvery ?? 0) * 3;
  return mod.TwinRenderer.create({ scene: app.bundle, physics: app.physics || {}, preset: app.preset,
    ...(rad > 0 && app.preset !== 'judge' ? { radiosityEvery: rad } : {}), log });
}

/** One small first frame under a validation error scope: a renderer whose shaders or pipelines are broken draws
 * black and floods the console, so it counts as not loaded and the page falls back (the status line says so). */
async function probeFrame(r) {
  const dev = r.device;
  let jsErr = null;
  dev.pushErrorScope('validation');
  try {
    r.setPoses(K.posesFromState(app.scene, K.defaultState(app.scene)));
    const name = rendererCameras(r).includes('SCOPE') ? 'SCOPE' : app.scene.cameras[0].name;
    if (r.renderFrame) { const f = r.renderFrame({ cameras: [name], sensor: true, sensorOpts: { seed: 1, frame: 0 } }); await f.done; }
    else { await r.renderCamera(name, {}); r.sensor(name, { seed: 1, frame: 0 }); await dev.queue.onSubmittedWorkDone(); }
  } catch (e) { jsErr = e; }
  const gpuErr = await dev.popErrorScope();
  if (jsErr) throw jsErr;
  if (gpuErr) throw new Error(`GPU validation error on its first frame: ${String(gpuErr.message).split('\n')[0]}`);
}

/** The camera names a renderer can render: r.cameras as a list (lite), an object keyed by name (tracer), or r.cams (stub). */
function rendererCameras(r) {
  if (Array.isArray(r.cameras)) return r.cameras.map((c) => (typeof c === 'string' ? c : c.name));
  if (r.cameras && typeof r.cameras === 'object') return Object.keys(r.cameras);
  if (r.cams && typeof r.cams === 'object') return Object.keys(r.cams);
  return app.scene.cameras.map((c) => c.name);
}

function calibration() {
  const s = app.scene;
  return {
    cameras: s.cameras.map((c) => ({ name: c.name, size: c.size, lens: c.lens, T_D_C: c.T_D_C })),
    ports: s.ports, instruments: s.instruments.map((i) => ({ id: i.id, port: i.port, kind: i.kind, spec: i.spec })),
    source: 'sim',
  };
}

/** Domain randomisation for dataset sessions (scripts/gen_twin_dataset.py sets window.__TWIN_DR__ before the page
 * loads): physics overrides by dotted path ({value} leaves keep their metadata) and per-camera T_D_C (the jittered,
 * true pose of each camera in this session). Applied before the renderer and the kinematics see the scene, so the
 * images and the truth use the same cameras. Recorded in scene_ref.json through the hello. */
function applyDR(dr) {
  if (!dr) return;
  const set = (root, path, v) => {
    const ks = path.split('.');
    let o = root;
    for (const k of ks.slice(0, -1)) { if (o[k] == null || typeof o[k] !== 'object') o[k] = {}; o = o[k]; }
    const leaf = ks.at(-1);
    if (o[leaf] && typeof o[leaf] === 'object' && !Array.isArray(o[leaf]) && 'value' in o[leaf]) o[leaf] = { ...o[leaf], value: v, status: 'randomised' };
    else o[leaf] = v;
  };
  if (dr.physics && app.physics) for (const [path, v] of Object.entries(dr.physics)) set(app.physics, path, v);
  for (const c of app.scene.cameras) { const T = dr.cameras?.[c.name]?.T_D_C; if (T) c.T_D_C = T; }
  app.sceneRef.domain_randomisation = dr;
}

function setBoot(text, err = false) {
  const b = $('boot');
  if (!b) return;
  b.textContent = text;
  b.classList.toggle('err', err);
  b.hidden = false;
}

async function boot() {
  await loadAll();
  applyDR(globalThis.__TWIN_DR__ ?? null);
  const { r, kind } = await makeRenderer();
  app.renderer = r; app.kind = kind; app.stub = kind === 'stub' || !!r.isStub; app.device = r.device;
  // the bundles' readback for the camera JPEG: the ISP's 4:2:0 planes (the tracer; a renderer that lists 'yuv420' in
  // r.readbackKinds), packed rgb8 otherwise, rgba8 from the stub
  app.rbKind = app.stub ? 'rgba8' : (kind === 'trace' || r.isLite || r.readbackKinds?.includes?.('yuv420')) ? 'yuv420' : 'rgb8';
  app.limits = new Limits(app.scene, app.meshes);
  app.state = app.limits.clampState(K.defaultState(app.scene)).state;
  // live pages open on the scope view: aim it at the instruments (ui/scope_aim.js; ?aim=0 keeps the scene's pose)
  if (!JUDGE && Q.get('aim') !== '0') {
    const clamp = (st) => app.limits.clampState(st).state;
    app.state = aimScope(app.scene, app.state, toolsTarget(app.scene, app.state), { clamp }).state;
  }
  app.pivots = Object.fromEntries(app.scene.instruments.map((i) => [i.id, K.port(app.scene, i.port).pivot]));
  app.clock = new CameraClock(app.scene.cameras, { exposure_s: DEFAULT_EXPOSURE_S });
  app.camIndex = Object.fromEntries(app.scene.cameras.map((c, i) => [c.name, i]));
  app.camFps = Object.fromEntries(app.scene.cameras.map((c) => [c.name, c.fps ?? 20]));
  app.rcams = new Set(rendererCameras(r));
  // SCOPE: the renderer's own camera (lite), else a pinhole from the kinematics (the tracer's overview pass, the stub's)
  app.scopeMode = app.rcams.has('SCOPE') ? 'renderer' : (r.renderOverview || r.renderPinhole) ? 'pinhole' : null;
  const available = GRID.filter((n) => (n === 'SCOPE' ? !!app.scopeMode : app.rcams.has(n)));
  // the main view: ?view=, else SCOPE when the renderer has it, else CAM1 (judge pages: CAM1, what the judges render)
  const asked = Q.get('view');
  const first = asked || (JUDGE || !app.rcams.has('SCOPE') ? 'CAM1' : 'SCOPE');
  app.views = initialViewState({ available, requested: first, grid: Q.get('grid') === '1', help: Q.get('help') === '1' });
  try { app.presenter = new Presenter(app.device); } catch (e) { console.error('presenter', e); app.disp.err = `display: ${e.message}`; }
  app.device?.lost?.then?.((i) => { app.deviceLost = i?.message || String(i?.reason || 'lost'); });
  // a GPU error while running shows in the status line for a few seconds after the last one
  app.device?.addEventListener?.('uncapturederror', (e) => { app.gpuErr = { text: String(e.error?.message || e.error || 'error').split('\n')[0], at: performance.now() }; });
  const home = new OrbitView(K.workspaceCentre(app.scene));
  const fallbackBasis = home.screenBasis();
  app.controls = new Controls({
    surface: $('main-view'), scene: app.scene,
    // drags and arrows act in the main view's image directions: the selected tool's hinge moves along image right / up
    // (control/view_basis.js toolBasis: solved from the view's image Jacobian at the selected tool, in the plane its
    // port allows), the overview's home view basis when a view has none
    screenBasis: () => toolBasis(app.scene, app.state, app.views.main, app.controls?.selected, { renderer: app.renderer, physics: app.physics }) ?? fallbackBasis,
    orbit: () => {},
  });
  app.controls.addEventListener('select', () => paintChrome());
  app.live = !JUDGE || Q.get('live') === '1';
  app.noDisplay = JUDGE && (Q.get('feeds') === '0' || Q.get('ov') === '0');
  app.aidsOff = Q.get('ov') === '0';
  app.visibleCams = new Set();
  buildPage();
  app.feeds = new Feeds(app.scene.cameras, {});       // the camera-JPEG workers (encode only; display is on the GPU)
  const qt = jpegTables();
  app.feeds.setTables(qt);
  app.jpegTables = qt;
  // the socket (only meaningful when served by `veridien webtwin serve`; offline otherwise, with retries)
  app.socket = new TwinSocket({ hello: () => ({ calibration: calibration(), scene_ref: { ...app.sceneRef, renderer: app.stub ? 'stub' : app.kind === 'lite' ? 'webgpu-lite' : 'webgpu', renderer_preset: app.stub ? null : app.preset, jpeg: app.jpegTables?.source } }) });
  app.socket.addEventListener('state', () => { if (app.socket.state === 'open') app.socketEverOpen = true; paintChrome(); });
  app.socket.addEventListener('status', (e) => { app.serverStatus = e.detail; paintChrome(); });
  app.socket.addEventListener('score', (e) => onScore(e.detail));
  wireAutoplay();
  app.sched = new GpuScheduler({
    now: () => performance.now(),
    setTimer: (fn, ms) => { const h = setTimeout(fn, ms); return () => clearTimeout(h); },
    source: liveSource, aid: aidJob, log: (m) => console.warn('[twin-gpu-calendar]', m),
    // asynchronous renderers (the stub, a renderer without renderFrame) run one job at a time: their awaits would
    // otherwise interleave two jobs' poses
    depth: app.stub || !app.renderer.renderFrame ? 1 : Number(Q.get('depth')) || 2,
    ...schedOverrides(),
  });
  $('boot').hidden = true;
  app.bootMs = performance.now();                      // navigation start -> page ready (the first frame follows)
  setInterval(controlTick, 8);
  setInterval(pollGpuFlags, 5000);
  setInterval(paintChrome, 250);
  requestAnimationFrame(tick);
  paintChrome();
}

/** ?sched=lateMs:25,maxQueuedMs:80 overrides calendar options (numbers; the A/B judges use it). */
function schedOverrides() {
  const out = {};
  for (const kv of (Q.get('sched') || '').split(',')) {
    const [k, v] = kv.split(':');
    if (k && v != null && Number.isFinite(Number(v))) out[k.trim()] = Number(v);
  }
  return out;
}

/** The live socket's JPEG stage (docs/WEB_TWIN.md §9 step 5): the camera's own MJPG, IJG-equivalent quality 91
 * (physics.yaml jpeg.quality, measured), with a measured luma table when physics.yaml gives one (natural 8x8 or flat
 * 64, row-major) and the IJG chroma table at that quality otherwise. Recordings get OpenCV's q95 stage on the server. */
function jpegTables() {
  const j = app.physics?.jpeg;
  const val = (x) => (x && typeof x === 'object' && !Array.isArray(x) && 'value' in x ? x.value : x);
  const q = Number(val(j?.quality)) || 91;
  const std = qualityTables(q);
  const flat = (t) => (t == null ? null : Uint8Array.from(Array.isArray(t[0]) ? t.flat() : t));
  const qt = j?.quant_tables;
  const luma = flat(val(qt?.luma) ?? val(qt)?.luma), chroma = flat(val(qt?.chroma) ?? val(qt)?.chroma);
  const qY = luma?.length === 64 ? luma : std.qY, qC = chroma?.length === 64 ? chroma : std.qC;
  const src = `IJG q${q}${j?.quality ? ` (${j.quality.status ?? 'physics.yaml'})` : ' (default)'}` + (luma?.length === 64 ? ', luma table from physics.yaml' : '');
  return { qY, qC, quality: q, source: src };
}

// ------------------------------------------------------------------ the page: one main view, the grid, the keys

function buildPage() {
  const main = $('main-view');
  app.slots.main = { slot: 'main', el: main, canvas: main.querySelector('canvas.img'), ov: main.querySelector('canvas.ov'), shown: null, rect: null, name: null };
  const grid = $('grid');
  GRID.forEach((name, i) => {
    const t = document.createElement('button');
    t.className = 'tile';
    t.type = 'button';
    t.dataset.view = name;
    t.innerHTML = `<span class="label">${viewLabel(name)}</span><span class="frame"><canvas class="img"></canvas><canvas class="ov"></canvas><span class="note" hidden></span></span>`;
    t.addEventListener('click', () => dispatch({ type: 'pick', name }));
    grid.appendChild(t);
    app.slots[`tile${i}`] = { slot: `tile${i}`, el: t.querySelector('.frame'), tile: t, canvas: t.querySelector('canvas.img'), ov: t.querySelector('canvas.ov'), shown: null, rect: null, name };
  });
  $('keys').innerHTML = KEYS.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  // the header controls hand the keyboard back to the view (Space is the grasp, not a button press)
  $('grid-btn').addEventListener('click', (e) => { e.currentTarget.blur(); dispatch({ type: 'toggleGrid' }); });
  $('help-btn').addEventListener('click', (e) => { e.currentTarget.blur(); dispatch({ type: 'toggleHelp' }); });
  document.addEventListener('pointerdown', (e) => { if (app.views.help && !e.target.closest('#help, #help-btn')) dispatch({ type: 'closeHelp' }); });
  main.addEventListener('pointerdown', (e) => { if (e.button === 0) main.classList.add('dragging'); });
  for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) main.addEventListener(ev, () => main.classList.remove('dragging'));
  window.addEventListener('keydown', onPageKey);
  // canvases: backing size = CSS size x devicePixelRatio (<= 2), measured when they change
  const ro = new ResizeObserver((entries) => {
    for (const e of entries) {
      const s = Object.values(app.slots).find((x) => x.el === e.target);
      if (!s) continue;
      const cw = e.contentRect.width, ch = e.contentRect.height;
      if (!(cw > 0 && ch > 0)) continue;                         // hidden (the grid or the main view is off)
      const box = e.devicePixelContentBoxSize?.[0];
      const [w, h] = box ? [Math.round(Math.min(box.inlineSize, 2 * cw)), Math.round(Math.min(box.blockSize, 2 * ch))] : backingSize(cw, ch, window.devicePixelRatio);
      s.css = [cw, ch];
      if (w > 0 && h > 0 && (s.canvas.width !== w || s.canvas.height !== h)) { s.canvas.width = w; s.canvas.height = h; s.shown = null; s.sized = true; app.disp.dirty = true; }
      else s.sized = true;
    }
  });
  for (const s of Object.values(app.slots)) {
    try { ro.observe(s.el, { box: 'device-pixel-content-box' }); } catch { ro.observe(s.el); }
  }
  // the in-page ONNX model: the file input in the help, or a file dropped on the page
  $('onnx').onchange = (e) => { const f = e.target.files?.[0]; if (f) loadOnnx(f).catch((err) => flash(`ONNX model: ${err.message}`)); };
  window.addEventListener('dragover', (e) => { if ([...(e.dataTransfer?.items ?? [])].some((i) => i.kind === 'file')) { e.preventDefault(); document.body.classList.add('drop'); } });
  window.addEventListener('dragleave', (e) => { if (!e.relatedTarget) document.body.classList.remove('drop'); });
  window.addEventListener('drop', (e) => {
    e.preventDefault(); document.body.classList.remove('drop');
    const f = [...(e.dataTransfer?.files ?? [])].find((x) => /\.onnx$/i.test(x.name));
    if (f) loadOnnx(f).catch((err) => flash(`ONNX model: ${err.message}`));
  });
  applyViews();
}

function dispatch(action) {
  const prev = app.views;
  app.views = viewReducer(app.views, action);
  if (app.views !== prev) applyViews(prev);
  return app.views;
}

/** The DOM follows the view state: main view or grid, the help, the controls' pressed states, the labels. */
function applyViews(prev = null) {
  const v = app.views;
  const grid = v.mode === 'grid';
  $('main-view').hidden = grid;
  $('grid').hidden = !grid;
  $('help').hidden = !v.help;
  $('grid-btn').setAttribute('aria-pressed', String(grid));
  $('help-btn').setAttribute('aria-expanded', String(v.help));
  const main = app.slots.main;
  if (main.name !== v.main) { main.name = v.main; main.shown = null; main.rect = null; clearOverlay(main.ov); }
  for (const s of Object.values(app.slots)) {
    if (!s.tile) continue;
    const ok = v.available.includes(s.name);
    s.tile.classList.toggle('current', s.name === v.main);
    s.tile.classList.toggle('off', !ok);
    s.tile.disabled = !ok;
    const n = s.el.querySelector('.note');
    n.hidden = ok;
    n.textContent = ok ? '' : (s.name === 'SCOPE' ? 'No scope view in this renderer' : 'Not rendered by this renderer');
  }
  if (!v.markers) for (const s of Object.values(app.slots)) clearOverlay(s.ov);
  if (!prev || prev.mode !== v.mode || prev.main !== v.main) {
    for (const s of Object.values(app.slots)) s.shown = null;
    app.disp.dirty = true;
    if (!grid) $('main-view').focus({ preventScroll: true });
  }
  app.visibleCams = new Set(visibleViews(v).map((x) => x.name).filter((n) => n !== 'SCOPE'));
  paintChrome();
}

function onPageKey(e) {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName ?? '') || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === '?' || (e.code === 'Slash' && e.shiftKey)) { e.preventDefault(); dispatch({ type: 'toggleHelp' }); return; }
  if (e.repeat) return;
  if (e.code === 'Escape') dispatch({ type: 'escape' });
  else if (e.code === 'KeyV') dispatch({ type: 'toggleGrid' });
  else if (e.code === 'KeyM') { dispatch({ type: 'toggleMarkers', modelAttached: modelAttached() }); if (!modelAttached()) note('Markers need a model attached', 2500); }
  else if (e.code === 'KeyP') { if (app.mode === 'autoplay') stopAutoplay('stopped'); else if (app.mode === 'manual') startAutoplay(app.autoName, app.autoSeed); }
  else if (e.code === 'KeyR' && e.shiftKey) recordScripted();
}

const TOOL_NAMES = { L: 'fenestrated grasper', R: 'Maryland', scope: 'scope' };

/** The header: the main view's name, a quiet caption (the tool being moved, autoplay, the model), the status line. */
function paintChrome() {
  if (!app.views) return;
  const v = app.views;
  $('view-name').textContent = v.mode === 'grid' ? 'Four views' : viewLabel(v.main ?? '');
  const parts = [];
  const sel = app.controls?.selected;
  if (app.mode === 'autoplay' && app.script) parts.push(`Autoplay, ${app.autoName.replace(/_/g, ' ')}, seed ${app.autoSeed}; any input takes over`);
  else if (app.note?.sticky && performance.now() < app.note.until) parts.push(app.note.text);
  else if (sel) parts.push(`Moving the ${TOOL_NAMES[sel] ?? sel}`);
  const m = app.serverStatus?.model;
  if (app.onnx) parts.push(`model ${app.onnx.name ?? 'onnx'} in the page`);
  else if (m?.connected) parts.push(m.oracle ? 'oracle model (plumbing test)' : `model ${m.name || 'attached'}`);
  if (v.markers) parts.push('markers on');
  if (app.note && !app.note.sticky && performance.now() < app.note.until) parts.push(app.note.text);
  const sub = parts.join(' · ');
  if ($('view-sub').textContent !== sub) $('view-sub').textContent = sub;
  // model attached / detached: markers follow
  const att = modelAttached();
  if (att !== app.modelAttached) { app.modelAttached = att; if (!att) dispatch({ type: 'modelDetached' }); }
  const r = app.renderer;
  const fl = app.flash && performance.now() < app.flash.until ? app.flash.text : null;
  const line = statusLine({
    deviceLost: app.deviceLost ?? r?.lostInfo ?? null, tripped: r?.timer?.tripped || r?.tripped || null,
    socketLost: app.socketEverOpen && app.socket?.state !== 'open', renderer: app.kind, fallbackFrom: app.fallbackFrom,
    error: fl ?? app.disp.err ?? (app.gpuErr && performance.now() - app.gpuErr.at < 3000 ? `The renderer reported a GPU error: ${app.gpuErr.text}` : null),
  });
  const st = $('status');
  if (st.textContent !== line) st.textContent = line;
  st.hidden = !line;
  const info = `Renderer: ${app.rendererInfo ?? '–'}${app.fallbackFrom && app.rendererError ? ` (${app.rendererError})` : ''}. Scene: ${app.sceneLabel}${app.sceneRef.scene_commit ? ` @ ${String(app.sceneRef.scene_commit).slice(0, 7)}` : ''}.`;
  if ($('help-renderer').textContent !== info) $('help-renderer').textContent = info;
}

function note(text, ms = 6000, sticky = false) { app.note = { text, until: performance.now() + ms, sticky }; paintChrome(); }
function flash(text, ms = 8000) { app.flash = { text, until: performance.now() + ms }; paintChrome(); }
function modelAttached() { return !!(app.serverStatus?.model?.connected || app.onnx); }

// ------------------------------------------------------------------ the loop

function stateAtLive(t) {
  const H = app.history;
  if (!H.length) return app.state;
  if (t <= H[0].t) return H[0].state;
  if (t >= H[H.length - 1].t) return H[H.length - 1].state;
  let lo = 0, hi = H.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (H[m].t <= t) lo = m; else hi = m; }
  const a = H[lo], b = H[hi];
  return K.lerpState(a.state, b.state, (t - a.t) / (b.t - a.t));
}

let lastControl = performance.now(), lastTruthAt = 0;

/** Input, autoplay and limits, integrated up to now and appended to the history that every camera samples its
 * exposure from. Called on the 125 Hz tick and again right before a camera is issued, so its exposure end is in. */
function sampleInput() {
  const now = performance.now();
  const dt = Math.min(0.1, Math.max(0, (now - lastControl) / 1000));
  lastControl = now;
  if (app.mode === 'judge') return;          // the judge drives time
  app.t = (now - app.t0) / 1000;
  if (app.mode === 'autoplay') {
    if (app.controls.activity > app.scriptWall) stopAutoplay('you took over');
    else {
      const ts = app.t - app.scriptT0;
      if (ts > app.script.duration) stopAutoplay('finished');
      else { const c = app.limits.clampState(app.script.stateAt(ts)); app.state = c.state; app.limited = c.limited; }
    }
  }
  if (app.mode === 'manual') { const c = app.limits.clampState(app.controls.step(app.state, dt)); app.state = c.state; app.limited = c.limited; }
  else app.controls.step(app.state, dt);
  app.state = { ...app.state, t: app.t };
  const key = JSON.stringify({ ...app.state, t: 0 });
  if (key !== app.stateKey) { app.stateKey = key; app.lastMotion = now; }
  app.history.push({ t: app.t, state: app.state });
  while (app.history.length > 2 && app.history[0].t < app.t - 2.0) app.history.shift();
}

/** On a 125 Hz timer, NOT on requestAnimationFrame: rAF is paced by the compositor and slows down whenever the GPU
 * queue is long; input must not. The GPU calendar is pumped here, at every job's completion and at the next release. */
function controlTick() {
  sampleInput();
  app.sched?.pump();
}

/** Display: present what bundles and judge renders produced, and the direct (rAF) display frame of a raster renderer. */
function tick(now) {
  requestAnimationFrame(tick);
  if (!app.renderer || !app.views) return;
  push(app.disp.raf, now, 240);
  presentLanded();
  if (!presentOnly() && displayFast() && !bundlesActive()) maybeDirectFrame(now);
  if (now - lastTruthAt > 100) { lastTruthAt = now; app.lastTruthNow = K.truthFromState(app.scene, app.state); }
}

/** Every camera bundle needs all cameras when something consumes the frames: a model on the socket, the in-page
 * ONNX model, or a recording (and the judge's scripted runs). Otherwise no bundles run: the display renders. */
function needAllCameras() {
  return !!(app.serverStatus?.model?.connected || app.onnx || app.serverStatus?.recording || app.mode === 'judge');
}
const bundlesActive = () => !!(app.live && app.mode !== 'judge' && needAllCameras());
/** Judge pages and scripted runs render nothing for display: they show what the judge or the script rendered. */
const presentOnly = () => app.mode === 'judge' || !app.live;
/** Raster renderers display on rAF; the path tracer goes through the GPU calendar (the §12 gaps). */
const displayFast = () => app.kind !== 'trace';
const isTracking = (name) => name !== 'SCOPE' && app.camIndex?.[name] != null;
const tripped = () => !!(app.renderer?.timer?.tripped || app.renderer?.tripped || app.deviceLost);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
/** record_stop waits for the server's disk thread to drain: 60 s plus 150 ms per bundle sent. */
const recordStopTimeoutMs = (bundles) => 60000 + 150 * Math.max(0, bundles | 0);
const LIVE_SEED = (idx) => ((idx + 1) * 104729) >>> 0;      // fixed per camera: the static hit and occlusion caches hold
const REUSE = Q.get('reuse') !== '0';                       // ?reuse=0: trace every frame even in a still scene
const STILL_MS = 2500;                                      // ... after the scene has been still this long
const STRIDE = Math.max(1, Number(Q.get('stride')) || 1);   // ?stride=4: every 4th bundle only (bring-up at 5 /s)

/** Wait for the GPU to finish everything submitted so far. */
async function gpuDone(r) { await (r.device ?? app.device)?.queue.onSubmittedWorkDone(); }

/** The exposure (s) the sensor will use next for this camera: the renderer's UVC AE, at most one frame. */
function exposureOf(name) {
  let ex = null;
  try { ex = app.renderer?.exposureOf ? app.renderer.exposureOf(name) : null; } catch { ex = null; }
  return Math.min(1 / (app.camFps[name] ?? 20), ex?.time_s ?? app.clock.exposure);
}
const pageMs = (tSim) => app.t0 + tSim * 1000;

/** The tracking cameras the live bundles render: all of them while something consumes the frames, else none. */
function wantedCams() {
  return needAllCameras() ? app.scene.cameras : [];
}

/** A still scene re-senses a camera's last radiance instead of tracing it again (fresh sensor noise; AE runs on). */
function reusable(name) { return REUSE && !app.stub && !!app.trace[name] && performance.now() - app.lastMotion >= STILL_MS; }

function posesEqual(a, b) {
  const A = a?.bodies ?? {}, B = b?.bodies ?? {};
  const ka = Object.keys(A);
  if (!a || !b || ka.length !== Object.keys(B).length) return false;
  for (const k of ka) {
    const x = A[k], y = B[k];
    if (x === y) continue;
    if (!x || !y || x.length !== y.length) return false;
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  }
  return true;
}

/** Which of the next n setPoses calls run a radiosity update in the renderer (its radiosityEvery counter). */
function radPlan(n) {
  const r = app.renderer, every = r?.opts?.radiosityEvery || 1;
  let pv = r?.posesVersion ?? 0, sv = r?.solvedVersion ?? 0;
  const out = [];
  for (let i = 0; i < n; i++) { pv++; const run = sv < 0 || pv - sv >= every; if (run) sv = pv; out.push(run); }
  return out;
}

/** Cost keys of the calendar's model: a trace, a trace that carries the radiosity update, or a re-sense. */
function jobKeys(names) {
  const plan = radPlan(names.length);
  let m = 0;
  return names.map((n) => `cam:${n}${reusable(n) ? ':reuse' : plan[m++] ? ':rad' : ''}`);
}

/** Live bundles for the calendar: bundle n holds every wanted camera's n-th capture; camera k's job is released at
 * its own exposure end (page clock). */
const liveSource = {
  ready() {
    if (!app.live || app.mode === 'judge') return 'off';
    if (app.renderer?.timer?.tripped || app.renderer?.tripped) return 'tripped';
    if (document.hidden) return 'hidden';
    if (!wantedCams().length) return 'nocams';
    if (app.inFlight >= 2) return 'inflight';
    if (app.disp.busy && app.disp.direct) return 'display';     // a direct display frame is on the GPU
    return true;
  },
  newest(now) {
    const t = (now - app.t0) / 1000;
    let n = -1;
    for (const c of wantedCams()) n = Math.max(n, app.clock.latestIndex(c.name, t, exposureOf(c.name)));
    return n < 0 ? n : Math.floor(n / STRIDE) * STRIDE;
  },
  describe(id) {
    if (id < 0 || id % STRIDE) return null;
    const cams = wantedCams();
    if (!cams.length) return null;
    const keys = jobKeys(cams.map((c) => c.name));
    const jobs = cams.map((c, i) => ({ key: keys[i], cam: c.name, release: pageMs(app.clock.capture(c.name, id).t_start + exposureOf(c.name)) }));
    return { jobs: jobs.sort((a, b) => a.release - b.release) };
  },
  admit(id) {
    const cams = wantedCams();
    if (!cams.length) return null;
    const caps = cams.map((c) => app.clock.capture(c.name, id));
    return makeBundle(id, caps, stateAtLive, { live: true, draw: true, partial: cams.length < app.scene.cameras.length }).jobs;
  },
};

/**
 * One bundle: a calendar job per camera (each at its own capture instant: motion blur between exposure start and
 * end, truth at mid-exposure, visibility from the renderer's BVH), each camera's JPEG started the moment its
 * readback lands (CAM1 encodes while CAM2 and CAM3 are still on the GPU), then the truth message and the §7.2
 * bundle, sent in frame order through app.sendChain. draw: its sensor textures go to the views. Returns {jobs, done}.
 */
function makeBundle(frameId, caps, stateAt, { draw = true, mustDeliver = false, seed = null, judgeOut = null, rgbOut = null, linearOut = null, partial = false, live = false, hashOut = null } = {}) {
  app.inFlight++;
  const r = app.renderer;
  // the socket's JPEG is encoded from the ISP's 4:2:0 planes when the renderer has them (§9 step 5), else from rgb8
  const kind = rgbOut ? 'rgb8' : app.rbKind;
  const n = caps.length, per = new Array(n), encs = new Array(n);
  const onnxImgs = app.onnx && !app.onnxBusy ? {} : null;     // latest only: one in-page prediction at a time
  if (onnxImgs) app.onnxBusy = true;
  let onnxStarted = false;
  const B = { id: frameId, tAdmit: performance.now(), tIssue: null, tLanded: null, reused: 0,
    due: live ? Math.max(...caps.map((c) => pageMs(c.t_start + exposureOf(c.name)))) : null };
  let left = n, settle, fail;
  const landed = new Promise((res, rej) => { settle = res; fail = rej; });
  landed.catch(() => {});
  const keys = live ? jobKeys(caps.map((c) => c.name)) : caps.map((c) => `cam:${c.name}:script`);
  B.jobs = caps.map((cap, k) => ({
    key: keys[k], cam: cap.name, release: live ? pageMs(cap.t_start + exposureOf(cap.name)) : -Infinity,
    issue: () => {
      B.tIssue ??= performance.now();
      return issueCamera(frameId, cap, k, stateAt, { seed, kind, linear: !!linearOut, live, draw }).then((rec) => {
        per[k] = rec;
        if (rec.reused) B.reused++;
        if (rgbOut) rgbOut[cap.name] = rec.rb.slice();
        if (linearOut) linearOut[cap.name] = rec.lin;
        if (hashOut) (hashOut[frameId] ??= {})[cap.name] = hashPixels(rec.rb);
        // the in-page ONNX slot sees the same images as a socket model (copied before the encoder takes the planes)
        const ch = rec.rb?.y ? 'yuv420' : kind === 'rgba8' ? 4 : 3;
        if (onnxImgs) onnxImgs[cap.name] = { rgb: rgbOf(rec.rb, ch), width: camSize(cap.name)[0], height: camSize(cap.name)[1] };
        encs[k] = ch === 'yuv420'
          ? app.feeds.encodeYUV(cap.name, rec.rb, { draw: false })
          : app.feeds.encode(cap.name, rec.rb, ...camSize(cap.name), { draw: false, channels: ch });
        if (--left === 0) { B.tLanded = performance.now(); settle(); }
      }, (e) => { fail(e); throw e; });
    },
  })).sort((a, b) => a.release - b.release);
  const done = app.sendChain.then(() => landed).then(async () => {
    onnxStarted = !!onnxImgs;
    const onnxP = onnxImgs ? runOnnx(frameId, onnxImgs) : null;
    const encsDone = await Promise.all(encs);
    const jpegs = encsDone.map((e) => e.jpeg);
    app.stats.encodeMs.push(Math.max(...encsDone.map((e) => e.ms)));
    // truth message (stays on the server) + the §7.2 bundle
    const tRef = Math.max(...per.map((p) => p.cap.t_mid));
    const ref = K.truthFromState(app.scene, stateAt(tRef));
    const truthMsg = {
      frame_id: frameId, t_ref: tRef,
      ref: { instruments: ref.instruments },
      cameras: Object.fromEntries(per.map((p) => [p.cap.name, {
        t_capture: p.cap.t_mid, index: p.cap.index, exposure_s: p.cap.exposure,
        instruments: camTruth(p.truth, p.cap.name, p.vis),
      }])),
      state: stateAt(tRef),
    };
    const header = {
      type: 'frames', frame_id: frameId, source: 'sim',
      cameras: per.map((p) => ({ name: p.cap.name, t_capture: p.cap.t_mid, bytes: 0, width: camSize(p.cap.name)[0], height: camSize(p.cap.name)[1], index: p.cap.index })),
    };
    let sent = false;
    if (app.socket?.state === 'open' && !partial) sent = await app.socket.sendFrame(truthMsg, header, jpegs, { mustDeliver });
    if (onnxP) onnxP.then((pred) => sendOnnx(frameId, pred, sent));
    if (judgeOut) per.forEach((p, i) => { judgeOut[p.cap.name] = { jpeg: jpegs[i], truth: truthMsg.cameras[p.cap.name], t_mid: p.cap.t_mid }; });
    for (const p of per) {
      app.frames[p.cap.name] = { cam: K.ensureLenses(app.scene).find((c) => c.name === p.cap.name), truth: p.truth, visible: p.vis, frame_id: frameId, t: p.cap.t_mid, stub: app.stub };
    }
    app.lastTruth = { frame_id: frameId, ref };
    const S = app.stats, tSent = performance.now(), t0 = B.tIssue ?? B.tAdmit;
    S.bundles++;
    // bundle_ms: live, from the bundle's due instant (its last exposure end) to sent; scripted, from its first issue
    push(S.bundleMs, live ? tSent - B.due : tSent - t0);
    push(S.issueMs, tSent - t0);
    push(S.stageMs, (B.tLanded ?? tSent) - t0);
    if (S.lastSent != null) push(S.periodMs, tSent - S.lastSent);
    S.lastSent = tSent;
    S.bundleTimes.push(tSent);
    while (S.bundleTimes.length > 600) S.bundleTimes.shift();
    return { sent, header, truthMsg };
  }).finally(() => { app.inFlight--; if (onnxImgs && !onnxStarted) app.onnxBusy = false; });
  app.sendChain = done.catch((e) => console.error('bundle', e));
  B.done = done;
  return B;
}

const push = (a, v, max = 1024) => { a.push(v); if (a.length > max) a.splice(0, a.length - max); };

/** A camera's newest sensor texture (a bundle's or a judge render's), for the display to present. */
function noteTex(name, tex, frameId, truth = null) {
  if (!tex) return;
  const prev = app.camTex[name];
  app.camTex[name] = { tex, v: (prev?.v ?? 0) + 1, frameId, truth };
}

/**
 * Submit one camera's GPU work now (synchronously: poses, visibility, trace, sensor, readback copy, in queue order)
 * and return a promise for its record once the readback lands, i.e. once the GPU is done with it.
 */
function issueCamera(frameId, cap, k, stateAt, { seed = null, kind, linear = false, live = false, draw = true }) {
  const r = app.renderer, idx = app.camIndex[cap.name];
  const exposure = exposureOf(cap.name);                       // the AE's exposure: end-of-exposure state
  const t_end = cap.t_start + exposure, t_mid = cap.t_start + exposure / 2;
  if (live) sampleInput();                                     // the input history reaches t_end first
  const s0 = stateAt(cap.t_start), s1 = stateAt(t_end), sm = stateAt(t_mid);
  const truth = K.truthFromState(app.scene, sm);
  const kps = K.keypointList(truth), pts = kps.map((x) => x.p);
  const P0 = K.posesFromState(app.scene, s0), P1 = K.posesFromState(app.scene, s1);
  // live: a fixed render seed per camera (the static caches hold) and a fresh sensor seed per frame; scripted runs
  // and recordings keep their per-frame render seed
  const sensorSeed = seed == null ? (frameId * 7919 + idx * 104729) >>> 0 : (seed * 1000003 + frameId * 7919 + k * 104729) >>> 0;
  const renderSeed = seed == null ? LIVE_SEED(idx) : sensorSeed;
  const sensorOpts = { seed: sensorSeed, frame: frameId };
  const rec = (rb, lin, vis, reused) => ({ cap: { ...cap, t_end, t_mid, exposure }, truth, vis: visMap(kps, vis), rb, lin, state: sm, reused });
  ensureViewSize(cap.name, camSize(cap.name));                 // the camera's own size (a grid tile may have shrunk it)
  app.disp.P = null;                                           // the display sets its own poses again
  app.disp.traced[cap.name] = null;
  if (!r.renderFrame) {                                        // asynchronous renderers: one job at a time
    return (async () => {
      r.setPoses(P0, P1);
      const vis = await r.visible(cap.name, pts);
      await r.renderCamera(cap.name, { seed: renderSeed });
      const tex = r.sensor(cap.name, sensorOpts);
      if (draw) noteTex(cap.name, tex, frameId, truth);
      const rb = await r.readback(cap.name, kind);
      return rec(rb, linear ? await r.readback(cap.name, 'linear') : null, vis, false);
    })();
  }
  const last = app.trace[cap.name];
  const reuse = live && reusable(cap.name) && posesEqual(last.P0, P0) && posesEqual(last.P1, P1);
  let vis, tex;
  if (reuse) {
    r.renderFrame({ cameras: [] }).done?.catch?.(() => {});   // no GPU work: a frame of its own in the GPU timings
    tex = r.sensor(cap.name, sensorOpts);                      // the radiance of its last trace: the same poses
    vis = r.visible(cap.name, pts);
    app.stats.traceReused++;
  } else {
    r.setPoses(P0, P1);
    app.lastPosesAt = performance.now();
    vis = r.visible(cap.name, pts);
    const f = r.renderFrame({ cameras: [cap.name], sensor: true, seed: renderSeed, sensorOpts });
    f.done?.catch?.(() => {});
    tex = f[cap.name]?.rgb8;
    app.trace[cap.name] = live ? { P0, P1 } : null;
    app.stats.traced++;
  }
  if (draw) noteTex(cap.name, tex, frameId, truth);
  const rbP = r.readback(cap.name, kind);                      // its copy is queued behind this camera's sensor pass
  const linP = linear ? r.readback(cap.name, 'linear') : null;
  return Promise.all([rbP, linP, vis]).then(([rb, lin, v]) => rec(rb, lin, v, reuse));
}

/** FNV-1a of a readback (the 4:2:0 planes or a packed array), for the pipelining judge. */
function hashPixels(rb) {
  let h = 0x811c9dc5;
  for (const a of rb.y ? [rb.y, rb.cb, rb.cr] : [rb instanceof Uint8Array ? rb : new Uint8Array(rb.buffer, rb.byteOffset, rb.byteLength)]) {
    for (let i = 0; i < a.length; i++) h = Math.imul(h ^ a[i], 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

// ------------------------------------------------------------------ display (ui/present.js)

/** The visible view slots, each with its render size (the main view <= 1280x720, a tile <= 640x360). */
function visibleSlots() {
  if (!app.views) return [];
  return visibleViews(app.views).filter((v) => v.available).map((v) => {
    const s = app.slots[v.slot];
    const cam = app.scene.cameras.find((c) => c.name === v.name);
    const aspect = cam ? cam.size[0] / cam.size[1] : 16 / 9;
    return { ...v, s, size: fitCap(aspect, v.cap) };
  });
}

/** The views the display itself renders now: none on judge pages; no tracking camera while bundles run (their own
 * frames are shown). */
function displayTargets() {
  if (presentOnly() || !app.presenter) return [];
  const b = bundlesActive();
  return visibleSlots().filter((v) => !(b && isTracking(v.name)) && v.s.sized);
}

/** Present the newest bundle / judge frame of each visible tracking-camera view that does not render itself. */
function presentLanded() {
  if (!app.presenter || app.noDisplay || app.aidsOff) return;
  if (!(presentOnly() || bundlesActive())) return;
  for (const v of visibleSlots()) {
    if (!isTracking(v.name)) continue;
    const t = app.camTex[v.name];
    if (!t || v.s.shown === `b${t.v}`) continue;
    try {
      v.s.rect = app.presenter.present(t.tex, v.s.canvas);
      v.s.shown = `b${t.v}`;
      app.disp.firstAt ??= performance.now();
      // the markers of exactly this frame: its truth from the issue (visibility once the bundle has landed)
      const f = app.frames[v.name];
      const frame = f && f.frame_id === t.frameId ? f
        : t.truth ? { cam: K.ensureLenses(app.scene).find((c) => c.name === v.name), truth: t.truth, visible: null, frame_id: t.frameId } : null;
      afterPresent(v, frame, t.frameId);
    } catch (e) { if (!app.disp.err) console.error('present', e); app.disp.err = `display: ${e.message}`; }
  }
}

/** Markers over a presented view (M, only with a model attached, tracking cameras only). */
function afterPresent(v, frame, frameId) {
  if (!app.views.markers || !isTracking(v.name)) return;
  const s = v.s, css = s.css ?? [s.canvas.clientWidth, s.canvas.clientHeight];
  if (!s.rect || !frame) { clearOverlay(s.ov); return; }
  const kx = css[0] / s.canvas.width, ky = css[1] / s.canvas.height;
  const rect = { x: s.rect.x * kx, y: s.rect.y * ky, w: s.rect.w * kx, h: s.rect.h * ky };
  app.marks[v.name] = drawMarkers(s.ov, { ...frame, prediction: predictionFor(frameId ?? frame.frame_id) }, rect);
}

function maybeDirectFrame(now) {
  const D = app.disp;
  if (D.busy || app.sched?.busy || app.noDisplay || document.hidden || tripped()) return;
  const targets = displayTargets();
  if (!targets.length) return;
  const moving = performance.now() - app.lastMotion < 300;
  if (!displayDue({ now, lastAt: D.at, lastMs: D.ms, dirty: D.dirty || targets.some((t) => t.s.shown == null), moving, stillFps: STILL_FPS })) return;
  runDisplay(targets, true);
}

/** The calendar's viewing-aid job: the display of the path tracer, or the scope view while bundles run. */
function aidJob(now) {
  if (presentOnly() || app.aidsOff || app.noDisplay || document.hidden || !app.renderer || tripped()) return null;
  if (displayFast() && !bundlesActive()) return null;          // the rAF path has it
  const D = app.disp;
  if (D.busy) return null;
  const targets = displayTargets();
  if (!targets.length) return null;
  const moving = performance.now() - app.lastMotion < 300;
  if (!D.dirty && !moving && targets.every((t) => t.s.shown != null) && now - D.at < 1000 / STILL_FPS) return null;
  const fast = displayFast();
  return { key: `aid:view:${targets.map((t) => t.name).join('+')}:${targets[0].size.join('x')}`, since: now - D.at,
    minMs: fast ? 15 : 50, maxMs: fast ? 40 : 110, run: () => runDisplay(targets, false), onService: (ms) => push(app.stats.overviewMs, ms) };
}

/** One display frame: render + sense + present every target, then wait for the GPU. */
function runDisplay(targets, direct) {
  const D = app.disp;
  D.busy = true; D.direct = direct; D.dirty = false; D.at = performance.now();
  const t0 = D.at;
  return renderViews(targets).then(() => {
    D.ms = performance.now() - t0;
    push(D.msHist, D.ms, 240);
    D.frames++;
    D.times.push(performance.now());
    while (D.times.length > 240) D.times.shift();
    app.ovCount = D.frames;
    if (D.err && D.err.startsWith('display')) D.err = null;
  }, (e) => {
    if (!D.err) console.error('display', e);
    D.err = `display: ${e?.message || e}`;
    D.ms = Math.max(D.ms, 100);
  }).finally(() => { D.busy = false; D.direct = false; });
}

async function renderViews(targets) {
  const r = app.renderer, D = app.disp;
  const P = K.posesFromState(app.scene, app.state);
  if (!posesEqual(D.P, P)) { r.setPoses(P); D.P = P; D.posesV++; app.lastPosesAt = performance.now(); }
  for (const t of targets) {
    let tex;
    if (t.name === 'SCOPE' && app.scopeMode === 'pinhole') tex = await scopePinhole(t);
    else tex = await renderCam(t.name, t.size);
    if (tex) { t.s.rect = app.presenter.present(tex, t.s.canvas); t.s.shown = `d${D.frames}`; D.firstAt ??= performance.now(); }
    else if (tex === null) t.s.shown = `d${D.frames}`;      // drawn by the renderer itself (the stub's pinhole)
    afterPresent(t, null, null);
  }
  await gpuDone(r);
}

/** Render + sense one view for display. A still scene re-senses the last radiance of that view. */
async function renderCam(name, size) {
  const r = app.renderer, D = app.disp, idx = app.camIndex[name] ?? 7;
  const frame = ++D.frame;
  const sensorOpts = { seed: (frame * 7919 + idx * 104729) >>> 0, frame };
  const key = `${D.posesV}|${size.join('x')}`;
  const still = performance.now() - app.lastMotion >= 600;
  if (REUSE && still && D.traced[name] === key) return r.sensor(name, sensorOpts);
  ensureViewSize(name, size);
  let tex;
  if (r.renderFrame) {
    const f = r.renderFrame({ cameras: [name], sensor: true, seed: LIVE_SEED(idx), sensorOpts });
    f.done?.catch?.(() => {});
    tex = f[name]?.rgb8 ?? f[name]?.sensor ?? null;
  } else {
    const lt = r.renderCamera(name, { seed: LIVE_SEED(idx), await: false, wait: false });   // no GPU stall per view
    if (lt?.then) await lt;
    tex = r.sensor(name, sensorOpts);
  }
  if (!tex) throw new Error(`the renderer gave no sensor texture for ${name}`);
  D.traced[name] = key;
  if (isTracking(name)) app.trace[name] = null;               // the live bundles' re-sense key is no longer valid
  return tex;
}

/** A renderer with per-view render sizes (lite: r.setViewSize) renders `name` at `size`: the main view <= 1280x720,
 * a grid tile 640x360, and the tracking cameras at their native size whenever a bundle or a judge renders them (the
 * model's JPEGs, the truth px and the judges are all at the calibration's size). A resize reallocates the view's
 * targets, so its last texture is dropped (never presented after it is destroyed). */
function ensureViewSize(name, size) {
  const r = app.renderer;
  if (!r?.setViewSize || !size) return;
  const cur = app.viewSize[name];
  if (cur && cur[0] === size[0] && cur[1] === size[1]) return;
  r.setViewSize(name, size);
  app.viewSize[name] = [...size];
  delete app.camTex[name];
  app.disp.traced[name] = null;
  if (app.trace) app.trace[name] = null;
}

/** The scope view of a renderer without SCOPE: a pinhole from the kinematics (§18.1 numbers; no sensor model). */
function scopePinhole(t) {
  const r = app.renderer;
  const sc = K.truthFromState(app.scene, app.state).scope;
  const b = viewBasis(app.scene, app.state, 'SCOPE', { physics: app.physics });
  if (!sc || !b) return Promise.resolve(undefined);
  const hv = app.physics?.scope?.camera?.hfov_deg;
  const hfov = Number((hv && typeof hv === 'object' ? hv.value : hv) ?? 75) * Math.PI / 180;
  if (r.isStub) {
    const c = t.s.canvas, W = c.width, H = c.height;
    const fovy = 2 * Math.atan(Math.tan(hfov / 2) * H / W) * 180 / Math.PI;
    r.renderPinhole(c, { eye: sc.lens_D, target: K.add(sc.lens_D, b.forward), up: b.up, fovy_deg: fovy, size: [W, H] }, { aid: false, cut: null });
    t.s.rect = { x: 0, y: 0, w: W, h: H };
    return Promise.resolve(null);
  }
  const [W, H] = t.size;
  const fovy = 2 * Math.atan(Math.tan(hfov / 2) * H / W) * 180 / Math.PI;
  const view = { eye: sc.lens_D, target: K.add(sc.lens_D, b.forward), up: b.up, fovy_deg: fovy, size: [W, H] };
  const opts = { aid: false, cut: null, spp: 1, background: null };
  if (r.renderFrame) { const f = r.renderFrame({ cameras: [], overview: view, overviewOpts: opts }); f.done?.catch?.(() => {}); return Promise.resolve(f.overview); }
  return Promise.resolve(r.renderOverview(view, { ...opts, present: false }));
}

function displayIdle() {
  return new Promise((res) => { const w = () => (app.disp.busy ? setTimeout(w, 4) : res()); w(); });
}

/** GPU loop caps and the submit budget (docs/WEB_TWIN.md §12): logged to the console; a trip shows in the status line. */
async function pollGpuFlags() {
  const r = app.renderer;
  if (!r || r.isStub || document.hidden || !app.sched) return;
  try {
    const f = r.debugFlags ? await app.sched.side('flags', () => r.debugFlags()) : {};
    const bad = Object.entries({ traversal: f.traversalCap, leaf: f.leafCap, 'corrupt tree': f.corruptTree }).filter(([, v]) => v > 0);
    const tm = r.timings?.() ?? {};
    const msgs = bad.map(([k, v]) => `${k} cap hit ${v}x`);
    if ((tm.over?.length ?? 0) > (app.overSeen ?? 0)) { msgs.push(`${tm.over.length} submits over 16 ms (largest ${fmt(tm.maxSubmit?.ms, 0)} ms)`); app.overSeen = tm.over.length; }
    if (app.sched.stats.stalls > (app.stallsSeen ?? 0)) { msgs.push(`${app.sched.stats.stalls} GPU jobs ran 1 s past their prediction`); app.stallsSeen = app.sched.stats.stalls; }
    app.gpuWarn = msgs;
    if (msgs.length) console.warn('[twin-gpu]', msgs.join(' · '));
  } catch (e) { console.warn('debugFlags', e); }
}

/** A camera's calibration size (scene.json); a view the scene does not list (SCOPE): the renderer's size for it. */
function camSize(name) {
  const c = app.scene.cameras.find((x) => x.name === name);
  if (c) return c.size;
  try { return app.renderer?.cameraInfo?.(name)?.size ?? app.viewSize[name] ?? [1280, 720]; } catch { return [1280, 720]; }
}

/** §6 truth of one camera: per instrument its 3-D points, px, in_image, unoccluded and visible.
 *  unoccluded: the renderer's ray from the lens centre reaches the point (§6 ray cast; other tool and shell occlude);
 *  visible:    unoccluded AND its px lies inside the image (what a label "seen in this frame" needs).
 *  px, unoccluded and visible are null past the camera's KB4 fold (§13), where a direction projects back in. */
function camTruth(truth, name, vis) {
  return truth.instruments.map((ins) => {
    const px = truth.px[name][ins.id], inside = truth.in_image[name][ins.id];
    const un = Object.fromEntries(Object.entries(vis[ins.id] ?? {}).map(([k, b]) => [k, px[k] == null ? null : b]));
    const v = Object.fromEntries(Object.entries(un).map(([k, b]) => [k, b == null ? null : !!(b && inside[k])]));
    return { ...ins, px, in_image: inside, unoccluded: un, visible: v };
  });
}

function visMap(kps, vis) {
  const out = {};
  kps.forEach((k, i) => { (out[k.id] ??= {})[k.kp] = !!vis[i]; });
  return out;
}

// ------------------------------------------------------------------ model scores and the ONNX slot

function onScore(s) {
  app.lastScore = s;
  app.scores.push({ t: performance.now(), s });
  while (app.scores.length > 200) app.scores.shift();
}

/** The prediction to draw over a frame: the score for exactly that frame_id, else the newest one if it is at most
 * two frames older (a prediction is never drawn over images it was not made from by more than 0.1 s). */
function predictionFor(frameId) {
  for (let i = app.scores.length - 1; i >= 0 && i >= app.scores.length - 60; i--) {
    const s = app.scores[i].s;
    if (s.frame_id === frameId) return s.prediction;
  }
  const last = app.scores.at(-1);
  if (last && frameId - last.s.frame_id <= 2 && frameId - last.s.frame_id >= 0) return last.s.prediction;
  return null;
}

/** Packed RGB for the in-page model from a readback (4:2:0 planes, rgb8 or rgba8), copied before the encoder takes it. */
function rgbOf(a, channels) {
  if (channels === 'yuv420') return yuvToRgb(a);
  if (channels === 3) return a.slice();
  const out = new Uint8Array((a.length / 4) * 3);
  for (let i = 0, j = 0; i < a.length; i += 4, j += 3) { out[j] = a[i]; out[j + 1] = a[i + 1]; out[j + 2] = a[i + 2]; }
  return out;
}

/** Run the in-page ONNX model on one bundle's images ({CAM1: {rgb, width, height}, ...}); app.onnxBusy is set by the
 * bundle that took the slot and cleared here. */
async function runOnnx(frameId, images) {
  try {
    return await app.onnx.predict(images, calibration(), { frame_id: frameId });
  } catch (e) { console.error('onnx', e); return null; } finally { app.onnxBusy = false; }
}

/** JFIF full-range 4:2:0 YCbCr -> packed RGB (nearest chroma), for the in-page model. */
function yuvToRgb({ y, cb, cr, width: w, height: h }) {
  const out = new Uint8Array(w * h * 3), cw = (w + 1) >> 1;
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const Y = y[j * w + i], c = (j >> 1) * cw + (i >> 1), U = cb[c] - 128, V = cr[c] - 128, o = 3 * (j * w + i);
    out[o] = Math.max(0, Math.min(255, Y + 1.402 * V));
    out[o + 1] = Math.max(0, Math.min(255, Y - 0.344136 * U - 0.714136 * V));
    out[o + 2] = Math.max(0, Math.min(255, Y + 1.772 * U));
  }
  return out;
}

/** In-page predictions are scored by the server exactly as a socket model's, after their truth has reached it. */
function sendOnnx(frameId, pred, bundleSent) {
  if (!pred) return;
  if (app.socket?.state === 'open' && bundleSent) app.socket.ws.send(JSON.stringify({ ...pred, type: 'predictions', frame_id: frameId, origin: 'page-onnx' }));
  else onScore({ type: 'score', frame_id: frameId, model: pred.model, instruments: [], prediction: pred, note: 'not scored: no server' });
}

async function loadOnnx(file) {
  const { OnnxModel } = await import('./model/onnx_adapter.js');
  const bytes = new Uint8Array(await file.arrayBuffer());
  app.onnx = await OnnxModel.load(bytes, { name: file.name.replace(/\.onnx$/, '') });
  note(`ONNX model ${file.name} loaded (${app.onnx.ep ?? 'ep?'})`);
}

// ------------------------------------------------------------------ autoplay and recording

function wireAutoplay() {
  const ap = Q.get('autoplay');
  app.autoName = SCRIPTS.includes(ap) ? ap : SCRIPTS[0];
  app.autoSeed = Number(Q.get('seed')) || 0;
  if (app.autoName.startsWith('charuco') && !(app.scene.boards || []).length) app.autoName = 'task_set';
  $('help-script').textContent = app.autoName.replace(/_/g, ' ');
  $('help-seed').textContent = String(app.autoSeed);
  if (ap && ap !== '0' && !JUDGE) setTimeout(() => { if (app.mode === 'manual') startAutoplay(app.autoName, app.autoSeed); }, 300);
}

function startAutoplay(name, seed) {
  app.script = makeScript(name, app.scene, seed, { limits: app.limits });
  app.scriptT0 = app.t; app.scriptWall = performance.now(); app.mode = 'autoplay';
  paintChrome();
}

function stopAutoplay(why) {
  if (app.mode !== 'autoplay') return;
  app.mode = 'manual';
  app.controls.adopt(app.state);
  note(`Autoplay ${why}`, 3000);
}

/** Shift R: the autoplay script recorded through the server (§10), on the simulated clock. */
async function recordScripted() {
  if (app.mode === 'judge') return;
  if (app.socket?.state !== 'open') { flash('Recording needs the twin server: run veridien webtwin serve and open its address', 6000); return; }
  const tag = (Q.get('tag') || 'twin').trim();
  if (!/^[a-z0-9-]+$/.test(tag)) { flash('Record tag: lower-case letters, digits and dashes', 6000); return; }
  if (app.mode === 'autoplay') stopAutoplay('stopped');
  try {
    const res = await twinJudge.runScript(app.autoName, app.autoSeed, { record: tag });
    note(`Recorded ${res.path}: ${res.bundles} bundles, ${fmt(res.duration, 1)} s`, 15000);
  } catch (e) { flash(`Recording failed: ${e.message}`); }
}

/** Median of the last 60. */
const med = (a) => { const s = [...a].slice(-60).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
function rate(times, now = performance.now(), win = 2000) {
  const t = times.filter((x) => x >= now - win);
  return t.length > 1 ? (t.length - 1) / ((t[t.length - 1] - t[0]) / 1000) : 0;
}

// ------------------------------------------------------------------ judge hooks (§8)

let readyResolve, readyReject;
const ready = new Promise((res, rej) => { readyResolve = res; readyReject = rej; });

function b64(u8) {
  let s = '';
  const step = 0x8000;
  for (let i = 0; i < u8.length; i += step) s += String.fromCharCode.apply(null, u8.subarray(i, i + step));
  return btoa(s);
}

const twinJudge = {
  ready,
  /** render(state, {cameras, linear, rgb8, exposure, state1}) -> {CAM1: {linear, rgb8, jpeg, truth}} */
  async render(state, opts = {}) {
    await ready;
    const names = opts.cameras ?? app.scene.cameras.map((c) => c.name);
    const r = app.renderer;
    const st0 = app.limits.clampState({ ...K.defaultState(app.scene), ...state }).state;
    const st1 = opts.state1 ? app.limits.clampState({ ...st0, ...opts.state1 }).state : st0;
    const out = {};
    const prevMode = app.mode;
    app.mode = 'judge';
    try {
      // alone on the GPU: the calendar lets admitted bundles finish, drains, runs this, then resumes
      await app.sched.exclusive(async () => {
        await displayIdle();                         // a direct display frame finishes first
        app.trace = {};                              // the cameras' radiance changes: no live re-sense of it
        app.disp.P = null; app.disp.traced = {};
        const truth = K.truthFromState(app.scene, K.lerpState(st0, st1, 0.5));
        const kps = K.keypointList(truth);
        r.setPoses(K.posesFromState(app.scene, st0), K.posesFromState(app.scene, st1));
        // judge mode: converge the radiosity for this pose (warm-started Jacobi; paced by the renderer)
        if (r.converge && opts.converge !== 0) await r.converge(opts.converge ?? 200);
        for (const name of names) ensureViewSize(name, camSize(name));     // the calibration's size
        for (const name of names) {                  // same poses for every camera: no new setPoses (keeps the solve)
          const vis = await r.visible(name, kps.map((x) => x.p));
          const camOpts = { seed: opts.seed ?? 0, ...(opts.spp ? { spp: opts.spp } : {}), ...(opts.exposure != null ? { exposure: opts.exposure } : {}) };
          const sensorOpts = { seed: opts.seed ?? 0, frame: opts.frame ?? 0, ...(opts.exposure != null ? { exposure_s: opts.exposure } : {}) };
          let tex;
          if (r.renderFrame) { const f = r.renderFrame({ cameras: [name], sensor: true, ...camOpts, sensorOpts }); await f.done; tex = f[name]?.rgb8; }
          else { await r.renderCamera(name, camOpts); tex = r.sensor(name, sensorOpts); }
          noteTex(name, tex, opts.frame ?? 0);
          const rec = { truth: { t: truth.t, instruments: camTruth(truth, name, visMap(kps, vis)), scope: truth.scope } };
          if (opts.linear) rec.linear = await r.readback(name, 'linear');
          rec.rgb8 = await r.readback(name, 'rgb8');
          rec.yuv = r.isStub ? null : await r.readback(name, 'yuv420');
          out[name] = rec;
        }
      });
      for (const name of names) {
        const rec = out[name];
        const [w, h] = camSize(name);
        // the camera's own MJPG stage, as the socket sends it (from the ISP's 4:2:0 planes when available)
        rec.jpeg = rec.yuv ? encodeJPEGYUV420(rec.yuv, app.jpegTables) : encodeJPEG(rec.rgb8, w, h, app.jpegTables);
        delete rec.yuv;
        if (opts.rgb8 === false) delete rec.rgb8;
      }
    } finally { app.mode = prevMode; app.disp.dirty = true; }
    return out;
  },
  /** The same render, with the arrays base64-encoded (cheap to move through Playwright's evaluate). */
  async renderB64(state, opts = {}) {
    const out = await twinJudge.render(state, opts);
    const enc = {};
    for (const [k, v] of Object.entries(out)) {
      enc[k] = { truth: v.truth, jpeg: b64(v.jpeg) };
      if (v.rgb8) enc[k].rgb8 = b64(v.rgb8);
      if (v.linear) enc[k].linear = b64(new Uint8Array(v.linear.buffer, v.linear.byteOffset, v.linear.byteLength));
    }
    return enc;
  },
  /** Autoplay a scripted session on the sim clock: every capture of every camera, nothing dropped; with
   * {record: tag} through the server recorder (§10). Returns {script, seed, duration, bundles, path, frames}.
   * opts: {duration, cameras, draw (show the frames in their views), overview: false (no display at all), pipeline:
   * true (two GPU jobs queued, as the live loop; the default is one at a time, so a recording's bytes do not depend on
   * when the renderer's background eAvg read-back lands), hashes: true (FNV-1a of every camera's readback, per
   * frame_id), stopTimeoutMs, ...}. */
  async runScript(name, seed = 0, opts = {}) {
    await ready;
    const { camera, varyStart, reaches, graspCycles, crossings, stillHold, reaims, poses } = opts;
    const script = makeScript(name, app.scene, seed, { limits: app.limits, camera, varyStart, reaches, graspCycles, crossings, stillHold, reaims, poses });
    const duration = Math.min(script.duration, opts.duration ?? Infinity);
    const stateAt = (t) => app.limits.clampState(script.stateAt(t)).state;
    let path = null, frames = null;
    const prevMode = app.mode, prevLive = app.live;
    app.mode = 'judge'; app.live = false;
    await app.sched.exclusive(async () => {});      // the live loop's admitted bundles finish on the GPU ...
    await displayIdle();                            // ... and any display frame ...
    await app.sendChain;                            // ... and are sent, before a recording can start
    if (opts.record) {
      try {
        for (let i = 0; i < 100 && app.socket?.state !== 'open'; i++) await new Promise((r) => setTimeout(r, 50));
        if (app.socket?.state !== 'open') throw new Error('recording needs the server: run `veridien webtwin serve` and open its URL');
        const rep = await app.socket.request('record_start', { tag: opts.record, seed, script: name, note: opts.note ?? '' }, 'record_started', 10000);
        path = rep.path;
      } catch (e) {
        app.mode = prevMode === 'judge' ? 'manual' : prevMode; app.live = prevLive;
        throw e;
      }
    }
    const bundles = app.clock.bundles(0, duration);
    const camNames = opts.cameras;
    const hashes = opts.hashes ? {} : null;
    const depth0 = app.sched.o.depth, aidsOff0 = app.aidsOff;
    if (opts.pipeline !== true) app.sched.o.depth = 1;
    if (opts.overview === false) app.aidsOff = true;
    app.trace = {};
    app.disp.P = null; app.disp.traced = {};
    let n = 0;
    const t0 = performance.now();
    try {
      // every bundle in order through the calendar (two GPU jobs queued, the §12 gaps); the GPU work of bundle k+1
      // overlaps the encode and send of bundle k (sends stay in order); <= 2 bundles in flight
      const inflight = [];
      for (const b of bundles) {
        const caps = camNames ? b.captures.filter((c) => camNames.includes(c.name)) : b.captures;
        app.state = stateAt(b.t_ref); app.t = b.t_ref; app.lastMotion = performance.now();
        const B = makeBundle(b.frame_id, caps, stateAt, { draw: opts.draw ?? true, mustDeliver: !!opts.record, seed, hashOut: hashes });
        await app.sched.admitBatch(b.frame_id, B.jobs);
        inflight.push(B.done);
        if (inflight.length >= 2) await inflight.shift();
        else await sleep(0);
        n++;
        if (opts.onProgress && n % 20 === 0) opts.onProgress(n, bundles.length);
        if (n % 20 === 0) { note(`${opts.record ? 'Recording' : 'Scripted run'}: ${name.replace(/_/g, ' ')}, seed ${seed}, ${n} of ${bundles.length} bundles`, 60000, true); await new Promise((r) => setTimeout(r, 0)); }
      }
      await Promise.all(inflight);
    } finally {
      app.sched.o.depth = depth0; app.aidsOff = aidsOff0;
      app.mode = prevMode === 'judge' ? 'manual' : prevMode; app.live = prevLive;
      app.t0 = performance.now() - app.t * 1000;
      app.history = [];
      app.trace = {};
      app.disp.P = null; app.disp.traced = {}; app.disp.dirty = true;
      if (app.note?.sticky) app.note = null;
      app.sched.resetLive();
    }
    const wall = (performance.now() - t0) / 1000;
    if (opts.record) {
      // the server's disk thread drains everything queued before it answers: allow for all of it (60 s plus
      // 150 ms per bundle sent, as if none had been written yet; ~3 x 50 ms of decode + re-encode per bundle)
      const rep = await app.socket.request('record_stop', {}, 'record_stopped', opts.stopTimeoutMs ?? recordStopTimeoutMs(n));
      frames = rep.frames;
    }
    return { script: name, seed, duration, bundles: n, path, frames, wall_s: wall, labels: script.labels, ...(hashes ? { hashes } : {}) };
  },
  /** The true calibration of this page's cameras (jittered by the domain randomisation when there is one). */
  calibration: () => calibration(),
  /** renderer: its timings(); page: the app's clocks. bundle_ms (live) runs from a bundle's due instant (its last
   * camera's exposure end) to sent; issue_to_sent_ms from its first camera's issue; stage_ms from that issue to its
   * last readback; period_ms between sends; skipped and skipped_by from the GPU calendar (hidden: not counted).
   * display: the one-view page's frames (overview_* keep their old names for the perf tools: the display's). */
  timings() {
    const tm = app.renderer?.timings?.() ?? {};
    const S = app.stats, sc = app.sched, D = app.disp, mc = app.slots.main?.canvas;
    return { renderer: tm, page: {
      bundle_ms: S.bundleMs.slice(-64), issue_to_sent_ms: S.issueMs.slice(-64), stage_ms: S.stageMs.slice(-64), period_ms: S.periodMs.slice(-64),
      overview_ms: D.msHist.slice(-64), encode_ms: S.encodeMs.slice(-64), skipped: sc?.stats.skipped ?? 0, skipped_by: { ...(sc?.stats.skippedBy ?? {}) },
      bundles: S.bundles, bundles_per_s: rate(S.bundleTimes), traced: S.traced, trace_reused: S.traceReused, overview_frames: D.frames, overview_size: mc ? [mc.width, mc.height] : null,
      display: { mode: app.views?.mode ?? null, main: app.views?.main ?? null, views: visibleSlots().map((v) => ({ name: v.name, render: v.size, canvas: [v.s.canvas.width, v.s.canvas.height] })),
        frames: D.frames, fps: rate(D.times), frame_ms: D.msHist.slice(-64), frame_ms_p50: med(D.msHist), path: displayFast() && !bundlesActive() ? 'raf' : 'calendar',
        presents: app.presenter?.count ?? 0, renderer: app.kind, error: D.err, raf_fps: rate(D.raf),
        first_frame_ms: D.firstAt, boot_ms: app.bootMs ?? null },
      bundle_ms_from: 'live: the last camera exposure end (due); scripted: the first camera issue',
      calendar: sc ? { ...sc.stats, skippedBy: undefined, depth: sc.o.depth, costs: sc.costs.snapshot(), queued_ms: sc.queuedMs() } : null,
      render_seed: 'live: fixed per camera; scripted: per frame', reuse: REUSE, radiosity_every: app.renderer?.opts?.radiosityEvery ?? null,
    } };
  },
  // inspection helpers (app checks; not part of §8)
  state: () => structuredClone(app.state),
  setState(s) { app.state = app.limits.clampState({ ...app.state, ...s }).state; return app.state; },
  truth: () => app.lastTruthNow,
  marks: () => app.marks,
  frames: () => Object.fromEntries(Object.entries(app.frames).map(([k, f]) => [k, { frame_id: f.frame_id, t: f.t, px: f.truth.px[k], visible: f.visible }])),
  info: () => ({ scene: app.sceneLabel, stub: app.stub, renderer: app.rendererInfo, renderer_kind: app.kind, fallback_from: app.fallbackFrom ?? null,
    rendererError: app.rendererError ?? null, sceneRef: app.sceneRef, cameras: app.scene.cameras.map((c) => c.name), views: app.views?.available ?? [],
    socket: app.socket?.state, jpeg: app.jpegTables?.source, selected: app.controls?.selected, mode: app.mode, live: app.live }),
  /** The view state, and actions on it ({type: 'toggleGrid' | 'pick' | 'escape' | 'toggleHelp' | 'setMain', name}). */
  views: () => structuredClone(app.views),
  ui: (action) => structuredClone(dispatch(action)),
  /** The main view's screen basis now (right, up in D): what drags and arrows act along. */
  screenBasis: () => app.controls?.screenBasis?.(),
  live(on) { app.live = !!on; if (on) { app.mode = 'manual'; app.sched?.resetLive(); } return app.live; },
  /** Live autoplay of a script (the P key), and its stop; for the perf scenarios (web/twin/tests/ui_perf_*.py). */
  autoplay(name = app.autoName, seed = app.autoSeed) { if (app.mode === 'autoplay') stopAutoplay('restarted'); app.autoName = name; app.autoSeed = seed; startAutoplay(name, seed); return { name, seed, duration: app.script.duration }; },
  stopAutoplay() { stopAutoplay('stopped'); return app.mode; },
  qualityTables,
};
window.twinJudge = twinJudge;

boot().then(() => readyResolve(true)).catch((err) => {
  console.error(err);
  setBoot(`Could not start: ${err?.message || err}`, true);
  readyReject(err);
});
