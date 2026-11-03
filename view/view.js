// The one-view page (docs/WEB_TWIN.md §18.2), from scratch: one big laparoscope view, "4 views" for SCOPE + CAM1-3,
// nothing else on screen. Reuses the one kinematics, the limits, the controller and the autoplay scripts (control/),
// the GPU blit (ui/present.js) when a renderer has no present(), and the model socket (model/socket.js, ui/feeds.js).
//
// Display: requestAnimationFrame, one frame in flight (the next frame starts only after the GPU finished the last),
// only the visible views render: the main view at <= 1280x720, grid tiles at 640x360, scaled to the screen by the
// compositor. A still scene renders at ?stillfps= (10) so sensor noise and AE stay alive while the GPU mostly idles.
// No readback for display. CAM1-CAM3 render offscreen at 20 fps only while a model is attached (view/model_feed.js).
//
// URL: ?renderer=lite|lite_nodes|lite_pt|lite_prt|lite_rt|trace|stub  ?view=SCOPE|CAM1..  ?grid=1  ?help=1  ?autoplay=task_set
//      ?seed=0  ?stillfps=10  ?scene=<base>/  ?aim=0 (the export's scope pose)  ?autoscope=1 (the script drives the
//      scope)  ?try=lite,lite_pt (the default search order of lite candidates)
import { loadScene } from '../gpu/scene_gpu.js';
import * as K from '../control/kinematics.js';
import { Limits } from '../control/limits.js';
import { Controls } from '../control/input.js';
import { makeScript, SCRIPTS } from '../control/scripts.js';
import { viewBasis, cameraBasis } from '../control/view_basis.js';
import * as S from './state.js';
import { aimScope, followStep } from './aim.js';

const Q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const EXPOSURE_S = 0.010;
const STILL_FPS = Number(Q.get('stillfps')) || 10;
const MOVING_MS = 1000;             // a change keeps the display at full rate this long (progressive renderers converge)
const TOOL_NAME = { L: 'Left grasper (fenestrated)', R: 'Right grasper (Maryland)', scope: 'Scope' };

const V = {
  scene: null, bundle: null, physics: null, r: null, kind: null, fallback: false, limits: null,
  state: null, controls: null, views: null, slots: {}, presenter: null, ctxs: new WeakMap(),
  mode: 'manual', script: null, scriptT0: 0, scriptWall: 0, t0: performance.now(), t: 0,
  stateKey: '', lastMotion: performance.now(), prevPoses: null,
  disp: { busy: false, at: -Infinity, dirty: true, frames: 0, err: null, presentBroken: false },
  perf: { raf: [], frame: [], views: {}, lastRaf: null },
  feed: null, problems: {}, hintTimer: 0, seed: Number(Q.get('seed')) || 0,
};
window.twinView = V;

const push = (a, v, max = 900) => { a.push(v); if (a.length > max) a.splice(0, a.length - max); };
const val = (x) => (x && typeof x === 'object' && !Array.isArray(x) && 'value' in x ? x.value : x);

// ------------------------------------------------------------------ boot

async function fetchJSON(url) {
  try { const r = await fetch(url, { cache: 'no-store' }); return r.ok ? await r.json() : null; } catch { return null; }
}

async function loadAll() {
  const want = Q.get('scene');
  for (const base of want ? [want] : ['../assets/', '../tests/stub/']) {
    try {
      const head = await fetch(base + 'scene.json', { cache: 'no-store' });
      if (!head.ok) continue;
      V.bundle = await loadScene(base);
      V.scene = V.bundle.scene;
      break;
    } catch { /* try the next */ }
  }
  if (!V.scene) throw new Error('no scene could be loaded');
  V.physics = (await fetchJSON('/veridien-twin/configs/twin/physics.json')) || {};
}

function withTimeout(p, ms, what) {
  return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${what} took over ${ms / 1000} s`)), ms))]);
}

async function createKind(kind) {
  const spec = S.RENDERERS[kind];
  const mod = await import(spec.path);
  const Cls = spec.exports.map((n) => mod[n]).find((c) => c && typeof c.create === 'function');
  if (!Cls) throw new Error(`${spec.path} exports no renderer class with create()`);
  const log = (...a) => console.log(`[view-${kind}]`, ...a);
  const opts = { scene: V.bundle, physics: V.physics, preset: Q.get('preset') || 'live', log };
  return withTimeout(Cls.create(opts), kind === 'trace' ? 60000 : 30000, kind);
}

async function makeRenderer(skip = new Set()) {
  const asked = Q.get('renderer');
  const tried = V.tried ?? [];
  for (const { kind, needScope } of S.rendererPlan(asked, Q.get('try')?.split(','))) {
    if (skip.has(kind)) continue;
    const t0 = performance.now();
    boot(kind.startsWith('lite') ? 'Preparing the renderer' : kind === 'trace' ? 'Preparing the path tracer' : 'Starting the stub renderer');
    try {
      const r = await createKind(kind);
      const cams = S.rendererCameras(r, V.scene.cameras.map((c) => c.name));
      if (needScope && !cams.includes('SCOPE')) {
        tried.push(`${kind}: no SCOPE`);
        try { r.destroy?.(); } catch { /* best effort */ }
        continue;
      }
      V.createMs = performance.now() - t0;
      V.tried = tried;
      // a fallback: the asked-for renderer did not start, or (by default) no lite renderer with SCOPE did
      V.fallback = asked ? asked !== kind : !S.LITE_ORDER.includes(kind);
      return { r, kind, cams };
    } catch (err) {
      const msg = String(err?.message || err).split('\n')[0];
      tried.push(`${kind}: ${msg}`);
      console.warn(`[view] renderer ${kind} unavailable:`, msg);
    }
  }
  throw new Error(`no renderer could start (${tried.join('; ')})`);
}

function boot(text, err = false) {
  const b = $('boot');
  b.textContent = text; b.classList.toggle('err', err); b.hidden = false;
}

async function start() {
  const tStart = performance.now();
  await loadAll();
  adopt(await makeRenderer());
  V.limits = new Limits(V.scene, V.bundle.meshes);
  V.state = V.limits.clampState(K.defaultState(V.scene)).state;
  if (Q.get('aim') !== '0') {           // the scope looks at the graspers on load (view/aim.js)
    const a = aimScope(V.scene, V.limits, V.state);
    V.state = a.state; V.aim = { before_deg: a.before * 180 / Math.PI, after_deg: a.after * 180 / Math.PI };
  }
  const available = availableViews();
  const requested = Q.get('view') || (V.scopeMode === 'renderer' ? 'SCOPE' : 'CAM1');
  V.views = S.initialState({ available, requested, grid: Q.get('grid') === '1', help: Q.get('help') === '1' });
  buildPage();
  const cam1 = V.scene.cameras.find((c) => c.name === 'CAM1') ?? V.scene.cameras[0];
  const fallbackBasis = cameraBasis(cam1);
  V.controls = new Controls({
    surface: $('main-view'), scene: V.scene,
    // drags and arrows act in the main view's image directions (the renderer's own basis when it publishes one)
    screenBasis: () => viewBasis(V.scene, V.state, V.views.main, { renderer: V.r, physics: V.physics }) ?? fallbackBasis,
    orbit: () => {},
  });
  V.controls.addEventListener('select', (e) => hint(TOOL_NAME[e.detail.tool] ?? e.detail.tool));
  window.addEventListener('keydown', onKey);
  setInterval(controlTick, 8);
  setInterval(pollProblems, 1000);
  // the model socket path: loaded lazily, quiet unless a model attaches
  import('./model_feed.js').then((m) => {
    V.feed = new m.ModelFeed(V);
    V.feed.addEventListener('change', () => {
      if (!V.feed.attached && V.views.markers) { const prev = V.views; V.views = { ...prev, markers: false }; applyViews(prev); }
      V.disp.dirty = true; paint();
    });
  })
    .catch((e) => console.warn('[view] model feed unavailable', e));
  $('boot').hidden = true;
  const ap = Q.get('autoplay');
  if (ap && ap !== '0') startAutoplay(SCRIPTS.includes(ap) ? ap : 'task_set');
  paint();
  requestAnimationFrame(tick);
  V.readyMs = performance.now() - tStart;
  V.firstFrame = new Promise((res) => { V._firstFrame = res; });
}

/** Take a renderer into use: its cameras, how it draws, its scope view, and a watch on its GPU errors. */
function adopt({ r, kind, cams }) {
  V.r = r; V.kind = kind; V.cams = new Set(cams); V.presentMode = S.presentMode(kind, r);
  // SCOPE: the renderer's own laparoscope camera; else a plain pinhole from the kinematics (the tracer's overview
  // pass, the stub's pinhole), shown in the grid only: the main view then defaults to CAM1 (§18.2)
  V.scopeMode = V.cams.has('SCOPE') ? 'renderer' : (r.renderPinhole || r.renderOverview) ? 'pinhole' : null;
  V.gpuErr = { n: 0, last: null, at: 0 };
  V.prevPoses = null; V.presenter = null; V.disp.frames = 0; V.disp.presentBroken = false; V.disp.dirty = true;
  r.device?.lost?.then?.((i) => { if (V.r === r) { V.problems.deviceLost = i?.message || String(i?.reason || 'lost'); paint(); } });
  r.device?.addEventListener?.('uncapturederror', (e) => {
    if (V.r !== r) return;
    V.gpuErr.n++; V.gpuErr.at = performance.now(); V.gpuErr.last = String(e?.error?.message || e).split('\n')[0];
  });
}

function availableViews() {
  return S.GRID.filter((n) => (n === 'SCOPE' ? !!V.scopeMode : V.cams.has(n)));
}

/**
 * A default-chosen renderer that reports GPU validation errors in its first frames draws garbage or nothing: drop it
 * and take the next in the plan (fresh canvases, since a context keeps its device). An asked-for renderer (?renderer=)
 * is kept, with the status line saying so.
 */
async function switchRenderer() {
  if (V.switching) return;
  V.switching = true;
  const old = V.r, oldKind = V.kind;
  V.r = null;
  (V.dropped ??= new Set()).add(oldKind);
  V.tried = [...(V.tried ?? []), `${oldKind}: GPU errors (${V.gpuErr.last})`];
  console.warn(`[view] ${oldKind} reported ${V.gpuErr.n} GPU errors in its first frames; trying the next renderer`);
  await waitIdle();
  try { old.destroy?.(); } catch { /* best effort */ }
  try {
    adopt(await makeRenderer(V.dropped));
    $('boot').hidden = true;
    for (const sl of Object.values(V.slots)) {
      const c = document.createElement('canvas');
      c.className = 'img';
      sl.canvas.replaceWith(c);
      sl.canvas = c;
    }
    const prev = V.views;
    V.views = S.reduce(prev, { type: 'available', available: availableViews() });
    applyViews(prev);
    V.problems.note = { text: `${oldKind} renderer had GPU errors; showing ${V.kind}.`, until: performance.now() + 8000 };
  } catch (err) {
    V.problems.fatal = String(err?.message || err);
  } finally {
    V.switching = false;
    paint();
  }
}

function waitIdle() {
  return new Promise((res) => { const w = () => (V.disp.busy || V.feed?.busy ? setTimeout(w, 5) : res()); w(); });
}

// ------------------------------------------------------------------ the page

function buildPage() {
  const main = $('main-view');
  V.slots.main = slotOf('main', main);
  const grid = $('grid');
  S.GRID.forEach((name, i) => {
    const t = document.createElement('button');
    t.type = 'button'; t.className = 'tile'; t.dataset.view = name;
    t.innerHTML = `<span class="label">${S.viewLabel(name)}</span><span class="frame"><canvas class="img"></canvas><canvas class="ov"></canvas><span class="note" hidden></span></span>`;
    t.addEventListener('click', () => dispatch({ type: 'pick', name }));
    grid.appendChild(t);
    V.slots[`tile${i}`] = slotOf(`tile${i}`, t.querySelector('.frame'), t);
  });
  $('keys').innerHTML = S.KEYS.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  $('grid-btn').addEventListener('click', (e) => { e.currentTarget.blur(); dispatch({ type: 'toggleGrid' }); });
  $('help-btn').addEventListener('click', (e) => { e.currentTarget.blur(); dispatch({ type: 'toggleHelp' }); });
  document.addEventListener('pointerdown', (e) => { if (V.views.help && !e.target.closest('#help, #help-btn')) dispatch({ type: 'closeHelp' }); });
  main.addEventListener('pointerdown', (e) => { if (e.button === 0) main.classList.add('dragging'); });
  for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) main.addEventListener(ev, () => main.classList.remove('dragging'));
  const ro = new ResizeObserver(() => layout());
  for (const s of Object.values(V.slots)) ro.observe(s.el);
  window.addEventListener('resize', layout);
  applyViews();
}

function slotOf(id, el, tile = null) {
  return { id, el, tile, canvas: el.querySelector('canvas.img'), ov: el.querySelector('canvas.ov'), note: el.querySelector('.note'), size: null, rect: null, drawn: null };
}

/** Each visible slot: its 16:9 image rectangle in CSS px (the canvas sits there) and its render size (capped). */
function layout() {
  const dpr = window.devicePixelRatio || 1;
  for (const s of Object.values(V.slots)) {
    const w = s.el.clientWidth, h = s.el.clientHeight;
    if (!(w > 0 && h > 0)) { s.size = null; continue; }
    const rect = S.letterbox(w, h);
    const cap = s.id === 'main' ? S.MAIN_CAP : S.TILE_CAP;
    const size = S.renderSize(rect.w, rect.h, dpr, cap);
    for (const c of [s.canvas, s.ov]) Object.assign(c.style, { left: `${rect.x}px`, top: `${rect.y}px`, width: `${rect.w}px`, height: `${rect.h}px` });
    if (s.canvas.width !== size[0] || s.canvas.height !== size[1]) { s.canvas.width = size[0]; s.canvas.height = size[1]; s.drawn = null; }
    s.rect = rect; s.size = size;
  }
  V.disp.dirty = true;
}

function dispatch(a) {
  const prev = V.views;
  if (a.type === 'toggleMarkers') a = { ...a, modelAttached: !!V.feed?.attached };
  V.views = S.reduce(prev, a);
  if (V.views !== prev) applyViews(prev);
}

function applyViews(prev = null) {
  const v = V.views;
  $('main-view').hidden = v.grid;
  $('grid').hidden = !v.grid;
  $('help').hidden = !v.help;
  $('grid-btn').setAttribute('aria-pressed', String(v.grid));
  $('help-btn').setAttribute('aria-expanded', String(v.help));
  for (const [id, s] of Object.entries(V.slots)) {
    const name = id === 'main' ? v.main : s.tile.dataset.view;
    s.name = name;
    const drawable = v.available.includes(name);
    s.note.hidden = drawable;
    if (!drawable) s.note.textContent = name === 'SCOPE' ? 'No laparoscope view in this renderer' : `${S.viewLabel(name)} is not available`;
    s.canvas.style.visibility = drawable ? 'visible' : 'hidden';
    if (s.tile) s.tile.classList.toggle('main', name === v.main);
    if (!v.markers) clearOv(s);
  }
  $('main-view').setAttribute('aria-label', `Main view: ${S.viewLabel(v.main)}`);
  if (prev && prev.main !== v.main && !v.grid) hint(`${S.viewLabel(v.main)}`);
  if (prev && prev.markers !== v.markers) hint(v.markers ? 'Model markers on' : 'Model markers off');
  layout();
}

function clearOv(s) {
  if (!s.ov.width) return;
  const c = s.ov.getContext('2d');
  c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, s.ov.width, s.ov.height);
}

function onKey(e) {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName ?? '')) return;
  const a = S.keyAction(e);
  if (!a) return;
  e.preventDefault();
  if (a.type === 'autoplay') { if (V.mode === 'autoplay') stopAutoplay('off'); else startAutoplay('task_set'); return; }
  if (a.type === 'toggleMarkers' && !V.feed?.attached) { hint('Markers need a model attached'); return; }
  dispatch(a);
}

function hint(text, ms = 1600) {
  const h = $('hint');
  h.textContent = text; h.classList.add('on');
  clearTimeout(V.hintTimer);
  V.hintTimer = setTimeout(() => h.classList.remove('on'), ms);
}

/** The one quiet status line, only while a problem lasts. */
function paint() {
  const p = V.problems;
  const text = S.statusText({
    fatal: p.fatal, deviceLost: p.deviceLost, trip: p.trip, kind: V.kind, fallback: V.fallback, asked: Q.get('renderer'),
    modelWanted: !!V.feed?.everAttached, socket: V.feed?.socketState, drawError: V.disp.err,
    gpuErrors: V.gpuErr && performance.now() - V.gpuErr.at < 3000 ? V.gpuErr.n : 0,
    note: p.note && performance.now() < p.note.until ? p.note.text : null,
  });
  const el = $('status');
  el.textContent = text; el.hidden = !text;
}

function pollProblems() {
  const tm = V.r?.timings?.() ?? {};
  const trip = tm.trip ?? tm.tripped ?? V.r?.tripped ?? V.r?.stopped ?? null;
  V.problems.trip = trip ? (typeof trip === 'string' ? trip : trip.reason || 'budget') : null;
  paint();
}

// ------------------------------------------------------------------ input, autoplay

let lastControl = performance.now();

function controlTick() {
  const now = performance.now();
  const dt = Math.min(0.1, Math.max(0, (now - lastControl) / 1000));
  lastControl = now;
  V.t = (now - V.t0) / 1000;
  if (V.mode === 'autoplay') {
    if (V.controls.activity > V.scriptWall) stopAutoplay('you took over');
    else {
      let ts = V.t - V.scriptT0;
      if (ts > V.script.duration) { V.scriptT0 = V.t; ts = 0; }          // loop
      const st = V.script.stateAt(ts);
      // the script drives the graspers; the scope follows them like a camera assistant (yaw/pitch only, smoothed),
      // so the laparoscope view keeps the tools; ?autoscope=1 lets the script drive the scope instead
      if (V.holdScope && st.scope) {
        if (now - (V.aimAt ?? 0) > 100) {      // re-aim at the tools at 10 Hz (warm start: a few Gauss-Newton steps)
          V.aimAt = now;
          V.aimGoal = aimScope(V.scene, V.limits, { ...st, scope: { ...V.holdScope } }, null, 6).state.scope;
        }
        if (V.aimGoal) V.holdScope = followStep(V.holdScope, V.aimGoal, dt);
        st.scope = { ...V.holdScope };
      }
      V.state = V.limits.clampState(st).state;
    }
  }
  if (V.mode === 'manual') V.state = V.limits.clampState(V.controls.step(V.state, dt)).state;
  else V.controls.step(V.state, dt);
  V.state = { ...V.state, t: V.t };
  const key = JSON.stringify({ ...V.state, t: 0 });
  if (key !== V.stateKey) { V.stateKey = key; V.lastMotion = now; }
}

function startAutoplay(name) {
  V.script = makeScript(name, V.scene, V.seed, { limits: V.limits });
  V.scriptT0 = V.t; V.scriptWall = performance.now(); V.mode = 'autoplay';
  V.holdScope = Q.get('autoscope') === '1' || !V.state.scope ? null : { ...V.state.scope };
  V.aimGoal = null; V.aimAt = 0;
  hint('Autoplay on: any input takes over');
}

function stopAutoplay(why) {
  if (V.mode !== 'autoplay') return;
  V.mode = 'manual';
  V.controls.adopt(V.state);
  hint(why === 'off' ? 'Autoplay off' : `Autoplay off: ${why}`);
}

// ------------------------------------------------------------------ display

function tick(now) {
  requestAnimationFrame(tick);
  const P = V.perf;
  if (P.lastRaf != null) push(P.raf, now - P.lastRaf);
  P.lastRaf = now;
  const D = V.disp;
  if (!V.r || document.hidden || V.switching) return;
  if (V.problems.deviceLost) return;
  const moving = now - V.lastMotion < MOVING_MS;
  if (!S.frameDue({ now, lastAt: D.at, busy: D.busy, moving, dirty: D.dirty, stillFps: STILL_FPS })) return;
  if (V.kind === 'trace' && paceTrace(now)) return;
  D.busy = true; D.at = now; D.dirty = false;
  drawFrame().then(() => { D.err = null; }, (err) => {
    D.err = String(err?.message || err).split('\n')[0];
    console.error('[view] frame', err);
    paint();
  }).finally(() => { D.busy = false; });
}

/** §12 for the path tracer: never more than 1.8 s of back-to-back frames without a 50 ms idle gap. */
function paceTrace(now) {
  const D = V.disp;
  D.runFrom ??= now;
  if (now - D.runFrom < 1800) return false;
  if (D.gapFrom == null) D.gapFrom = now;
  if (now - D.gapFrom < 50) return true;
  D.runFrom = now; D.gapFrom = null;
  return false;
}

function posesKey(P) {
  let s = '';
  for (const [k, T] of Object.entries(P.bodies)) { s += k; if (T) for (let i = 12; i < 15; i++) s += T[i].toFixed(7); if (T) for (let i = 0; i < 11; i++) s += T[i].toFixed(6); }
  return s;
}

async function drawFrame() {
  const r = V.r, D = V.disp, t0 = performance.now();
  const P = K.posesFromState(V.scene, V.state);
  const pk = posesKey(P);
  if (pk !== V.prevPoses || V.feed?.posesDirty) { r.setPoses(P); V.prevPoses = pk; if (V.feed) V.feed.posesDirty = false; }
  const frame = ++D.frames;
  for (const v of S.visibleViews(V.views)) {
    if (!v.drawable) continue;
    const s = V.slots[v.slot];
    if (!s?.size) continue;
    const tv = performance.now();
    await drawView(v.name, s, frame);
    push(V.perf.views[v.name] ??= [], performance.now() - tv, 300);
    if (V.views.markers && V.feed?.attached) V.feed.drawMarkers(v.name, s);
  }
  await r.device?.queue?.onSubmittedWorkDone?.();
  push(V.perf.frame, performance.now() - t0);
  if (frame <= 30 && V.gpuErr.n >= 3 && !Q.get('renderer') && !V.switching) setTimeout(switchRenderer, 0);
  if (frame === 1) { V.firstFrameMs = performance.now() - V.t0; V._firstFrame?.(V.firstFrameMs); }
}

function ctxFor(canvas) {
  let ctx = V.ctxs.get(canvas);
  if (!ctx) {
    ctx = canvas.getContext('webgpu');
    ctx.configure({ device: V.r.device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' });
    V.ctxs.set(canvas, ctx);
  }
  return ctx;
}

/** The scope view of a renderer without SCOPE: a pinhole through the kinematics' scope basis (§18.1 HFOV; no sensor
 * model, no image circle). The stub draws straight into the canvas; the tracer renders its overview pass. */
async function drawScopePinhole(s) {
  const r = V.r;
  const sc = K.truthFromState(V.scene, V.state).scope;
  const b = viewBasis(V.scene, V.state, 'SCOPE', { physics: V.physics });
  if (!sc || !b) return;
  const hfov = Number(val(V.physics?.scope?.camera?.hfov_deg) ?? 75) * Math.PI / 180;
  const [W, H] = s.size;
  const fovy = 2 * Math.atan(Math.tan(hfov / 2) * H / W) * 180 / Math.PI;
  const view = { eye: sc.lens_D, target: K.add(sc.lens_D, b.forward), up: b.up, fovy_deg: fovy, size: [W, H] };
  if (r.renderPinhole) { r.renderPinhole(s.canvas, view, { aid: false, cut: null }); s.drawn = V.disp.frames; return; }
  let tex = r.renderOverview(view, { aid: false, cut: null, spp: 1, background: null, present: false });
  if (tex?.then) tex = await tex;
  if (tex) await blit(tex, s);
}

async function blit(tex, s) {
  if (!V.presenter) {
    const { Presenter } = await import('../ui/present.js');
    V.presenter = new Presenter(V.r.device);
  }
  V.presenter.present(tex, s.canvas);
  s.drawn = V.disp.frames;
}

async function drawView(name, s, frame) {
  const r = V.r;
  if (name === 'SCOPE' && V.scopeMode === 'pinhole') return drawScopePinhole(s);
  const idx = S.GRID.indexOf(name);
  const seed = (frame * 7919 + (idx + 1) * 104729) >>> 0;
  // the tracer keeps its static caches with a fixed seed per camera; lite renderers run their own AE and accumulation
  const ropts = S.RENDERERS[V.kind]?.seed ? { seed: 1009 * (idx + 1) } : { size: s.size };
  const out = r.renderCamera(name, ropts);
  if (out?.then) await out;
  const mode = V.disp.presentBroken ? 'blit' : V.presentMode;
  if (mode !== 'blit') {
    try {
      if (mode === 'sense+present') r.sensor(name, { seed, frame });
      const p = r.present(name, ctxFor(s.canvas), {});
      if (p?.then) await p;
      s.drawn = frame;
      return;
    } catch (err) {
      console.warn('[view] renderer present() failed, using the page blit', err);
      V.disp.presentBroken = true;
    }
  }
  const tex = r.sensor(name, { seed, frame });
  if (tex) await blit(tex, s);
}

// ------------------------------------------------------------------ numbers for the bench (window.twinView.report())

V.report = () => {
  const P = V.perf;
  const views = Object.fromEntries(Object.entries(P.views).map(([k, xs]) => [k, S.stats(xs)]));
  let timings = null;
  try { timings = V.r?.timings?.() ?? null; } catch { timings = null; }
  return {
    renderer: V.kind, tried: V.tried, aim: V.aim, gpuErrors: V.gpuErr?.n ?? 0, dropped: [...(V.dropped ?? [])], fallback: V.fallback, createMs: V.createMs, readyMs: V.readyMs, firstFrameMs: V.firstFrameMs,
    views: V.views, mode: V.mode, frames: V.disp.frames, raf: S.stats(P.raf), frame: S.stats(P.frame), perView: views,
    sizes: Object.fromEntries(Object.entries(V.slots).map(([k, s]) => [k, s.size])), status: $('status').textContent,
    presentPath: V.disp.presentBroken ? 'blit (present() failed)' : V.presentMode,
    timings, feed: V.feed?.report?.() ?? null,
  };
};
V.resetPerf = () => { V.perf.raf = []; V.perf.frame = []; V.perf.views = {}; };
V.dispatch = dispatch;
V.startAutoplay = startAutoplay;
V.stopAutoplay = stopAutoplay;
V.ready = start().catch((err) => {
  console.error('[view] boot', err);
  V.problems.fatal = String(err?.message || err);
  boot(V.problems.fatal, true);
  paint();
  throw err;
});
V.ready.catch(() => {});
