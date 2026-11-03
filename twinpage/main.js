// The twin page (docs/WEB_TWIN.md §19, §21, ADR-137, ADR-132): the god's-eye 3-D view of the whole rig is the main view;
// the REAL camera images (CAM1-CAM3 from the CV server, with its keypoints drawn over them, twinpage/real_cams.js) sit
// in a column on the right while that server is reachable. The column's Real / Sim switch (T) shows instead SCOPE and
// CAM1-CAM3 simulated by the god view's own raster (godview_pt3 renderCameraView: the scope LED alone inside the closed
// dome, KB4 fisheye; Sim is the default while the server does not answer); ?cams=<dir> draws Sim with a camera-sim
// renderer instead (§18). Nothing else: a "?" key sheet and one quiet line while a problem lasts.
//
//   ?god=three|pt|<dir>|placeholder   god's-eye renderer (twinpage/loader.js; default: the first that loads)
//   ?cams=<dir>                       Sim tiles from a camera-sim renderer (default: the god view's raster; e.g. ?cams=lite_pt2)
//   ?real=0                           no real camera images (default: shown while the CV server answers)
//   ?camsWorker=0                     run that renderer on the page thread instead of its worker (cams_worker.js)
//   ?extras=0                         the god view without the handles, eyepiece, light post and cable
//   ?view=V1_surgeon|V2_left|...      the starting viewpoint (godview.yaml)     ?side=0  start with the column hidden
//   ?autoplay=task_set&seed=0         start the scripted autoplay
//   ?rig=1 | ?rig=<url> | ?rig=0      rig mode: L and R follow the real rig's tips (twinpage/rig_link.js; R toggles).
//                                     Default: on when the CV server (http://127.0.0.1:8833, or <url>) answers; 0 forces off
//   ?tracker=auto|server|browser      where the tips come from (twinpage/tracker_source.js): the CV server, or this page
//                                     itself (cameras + TipNet + RCM solve in a worker, twinpage/browser_source.js; no
//                                     server). auto: the server when it answers at boot, else the page (which reloads onto
//                                     the server once it answers while its own cameras are off). Default: server here, auto
//                                     on the static site (its veridien-tracker meta); ?replay=<session url>|synthetic replays.
//
// Threads: input integrates on a 125 Hz timer (never on rAF); the god's-eye view renders on requestAnimationFrame,
// only when something changed (or every frame for a progressive renderer); the camera tiles render on their own
// timer, one camera per slot, at <= 15 Hz in total (Sim from the god view: <= 10 Hz, ~2-3 ms GPU each, its scope-only
// bake in <= 3 ms chunks on a 30 Hz idle timer), so they never slow the 3-D view.
import { loadScene } from '../gpu/scene_gpu.js';
import * as K from '../control/kinematics.js';
import { Limits } from '../control/limits.js';
import { Controls } from '../control/input.js';
import { makeScript, SCRIPTS } from '../control/scripts.js';
import { aimScope, toolsTarget } from '../ui/scope_aim.js';
import { Presenter } from '../ui/present.js';
import { viewLabel } from '../ui/views.js';
import { normaliseGodview, cutawayDefault, DEFAULT_VIEWPOINT, zoomCutOffset, zoomFixedShown } from './config.js';
import { cutModeFromQuery, nextCutMode, cutModeNote } from '../godview_caps.js';
import { OrbitCamera } from './orbit.js';
import { godOrder, godModuleUrl, camsOrder, camsModuleUrl, rendererClass, godLabel, PLACEHOLDER } from './loader.js';
import { TILE_ORDER, TILE_SIZE, LARGE_CAP, nextTile, slotPeriodMs } from './tiles.js';
import { resolveCamsMode, toggleCamsMode, loadCamsMode, saveCamsMode } from './cams_mode.js';
import { ModelLink } from './model_link.js';
import { CamsHost } from './cams_host.js';
import { RealCams } from './real_cams.js';
import { feedHub, TipSmoother, applyPredictions, solveTip, rigLine, rigUrlFromQuery, lostIds, lostAfter, pivotOffsets, RIG_DEFAULT_URL } from './rig_link.js';
import { makeSource, watchServer, ServerSource, LOCAL_BASE } from './tracker_source.js';
import { SnapshotInterp } from './rig_smooth.js';

const Q = new URLSearchParams(location.search);
// startup marks (performance.mark 'twin:*', godview_pt3 'gv:*'); ?perf=1 loads the harness that reports them (twinpage/perf.js)
const mark = (n) => { try { performance.mark(n); } catch { /* old browser */ } };
mark('twin:main:eval');
if (Q.has('perf')) import('./perf.js').then((m) => m.install()).catch((e) => console.warn('[twin] perf harness', e));
const $ = (id) => document.getElementById(id);
const ROOT = new URL('../', import.meta.url);           // web/twin/
const TOOL_NAMES = { L: 'Fenestrated grasper (L)', R: 'Maryland (R)', scope: 'Scope (C)' };

export const KEYS = Object.freeze([
  ['View'],
  ['drag', 'orbit'],
  ['right or Shift drag', 'pan'],
  ['wheel', 'zoom'],
  ['F', 'back to the surgeon\u2019s view'],
  ['C', 'cutaway: follows the view, fixed section, off'],
  ['V', 'camera column on or off'],
  ['T', 'camera column: real or simulated images'],
  ['click a camera', 'show it large (Esc returns)'],
  ['Tools'],
  ['1 / 2 / 3', 'fenestrated (L), Maryland (R), scope'],
  ['Alt drag', 'move the selected tool on screen'],
  ['arrows', 'swing it'],
  ['W / S', 'insert, withdraw'],
  ['Q / E', 'roll'],
  ['J / K', 'open, close the jaws'],
  ['Space', 'hold to grasp'],
  ['Shift', 'fine, with the tool keys'],
  ['A', 'autoplay on or off'],
  ['R', 'real rig: the tools follow its tips'],
  ['M', 'model\u2019s predictions as ghosts'],
]);

const page = window.twinPage = {
  scene: null, meshes: null, bundle: null, physics: null, gv: null, gvSource: null,
  god: null, godDir: null, godFailed: [], cams: null, camsDir: null, camsFailed: [], presenter: null,
  limits: null, state: null, controls: null, orbit: null, cutaway: 'view',
  mode: 'manual', script: null, scriptT0: 0, scriptWall: 0, autoName: 'task_set', autoSeed: 0,
  posesV: 0, drawnPosesV: -1, godDirty: true, camsPosesV: -1, lastMotion: 0, stateKey: '',
  large: null, collapsed: Q.get('side') === '0', help: false, ghostsOn: false, ghosts: null,
  tiles: {}, tileLast: null, tileAt: {}, tileCount: {}, tileBusy: false, tileFrames: 0, tileMs: [],
  frameTimes: [], rafMs: [], notes: [], problem: null, model: null, bootMs: null, firstFrameMs: null,
  rig: null, real: null, rigAuto: false, rigTouched: false, source: null,
  camsMode: null, camsChosen: loadCamsMode(), godCamNames: new Set(), simBakeV: null,
};

// ------------------------------------------------------------------ boot

async function fetchJson(url) {
  try { const r = await fetch(url, { cache: 'no-store' }); return r.ok ? await r.json() : null; } catch { return null; }
}

async function loadAll() {
  const bases = Q.get('scene') ? [Q.get('scene')] : ['assets/', 'tests/stub/'];
  for (const b of bases) {
    const base = new URL(b, ROOT).href;
    try {
      const r = await fetch(base + 'scene.json', { cache: 'no-store', method: 'HEAD' });
      if (!r.ok) continue;
      page.bundle = await loadScene(base);
      mark('twin:scene:fetched');
      break;
    } catch { /* next */ }
  }
  if (!page.bundle) throw new Error('no scene: neither assets/scene.json nor tests/stub/scene.json could be loaded');
  page.scene = page.bundle.scene; page.meshes = page.bundle.meshes;
  const [phys, gv] = await Promise.all([fetchJson('/veridien-twin/configs/twin/physics.json'), fetchJson('/veridien-twin/configs/twin/godview.json')]);
  page.physics = phys || {};
  page.gv = normaliseGodview(gv);
  mark('twin:configs');
  page.gvSource = gv ? 'configs/twin/godview.yaml' : 'built-in copy of godview.yaml (the server did not serve it)';
}

function freshCanvas(id) {
  const old = $(id), c = document.createElement('canvas');
  c.id = id; c.className = old.className; c.tabIndex = old.tabIndex;
  for (const a of ['aria-label', 'hidden']) if (old.hasAttribute(a)) c.setAttribute(a, old.getAttribute(a));
  c.width = old.width; c.height = old.height;
  old.replaceWith(c);
  return c;
}

async function makeGod() {
  for (const dir of godOrder(Q.get('god'))) {
    const t0 = performance.now();
    const canvas = page.godFailed.length ? freshCanvas('god') : $('god');
    sizeCanvas(canvas);
    try {
      const mod = await import(new URL(godModuleUrl(dir), ROOT).href);
      const G = rendererClass(mod, ['GodView']);
      mark('twin:god:module');
      if (!G) throw new Error(`${dir}/godview.js exports no GodView`);
      // the god view also draws the grasper handles and the scope's eyepiece, light post, cable and camera head
      // (assets/godview_extras.json, scripts/build_godview_extras.py); the camera simulations keep page.bundle
      page.godBundle ??= Q.get('extras') === '0' ? page.bundle : await import(new URL('godview_extras.js', ROOT).href)
        .then((m) => m.mergeExtras(page.bundle, page.bundle.base)).catch(() => page.bundle);
      mark('twin:god:extras');
      const god = await G.create({ canvas, scene: page.godBundle, physics: page.physics, godview: page.gv,
        log: (...a) => console.log(`[god ${dir}]`, ...a) });
      mark('twin:god:created');
      god.setCamera(page.orbit.camera());
      god.setCutaway(page.cutaway);
      // movable props (the practice module, body "prop/module"): every pose the page sends gets the prop's pose too
      propInit(page.godBundle);
      const setPoses0 = god.setPoses.bind(god);
      god.setPoses = (poses) => setPoses0(withProps(poses));
      god.setPoses(K.posesFromState(page.scene, page.state));
      god.render();
      mark('twin:god:firstRender');
      (god.device ?? god.r?.backend?.device)?.queue?.onSubmittedWorkDone?.().then(() => mark('twin:god:firstFrameGpu'));
      page.god = god; page.godDir = dir; page.godLoadMs = performance.now() - t0;
      // GPU validation errors are asynchronous: listen on the renderer's device when it exposes one
      const dev = god.device ?? god.r?.backend?.device ?? null;
      dev?.addEventListener?.('uncapturederror', (e) => { page.godGpuErr = { text: String(e.error?.message || e.error || 'error').split('\n')[0], at: performance.now() }; });
      console.log(`[twin] god view: ${dir} in ${page.godLoadMs.toFixed(0)} ms`);
      return canvas;
    } catch (e) {
      if (dir !== PLACEHOLDER || Q.get('god')) console.warn(`[twin] god view ${dir} unavailable:`, e);
      page.godFailed.push({ dir, error: String(e?.message || e) });
    }
  }
  throw new Error(`no god-view renderer could start: ${page.godFailed.map((f) => `${f.dir}: ${f.error}`).join('; ')}`);
}

async function makeCams() {
  const order = camsOrder(Q.get('cams'));
  // the renderer runs in a worker when the browser can (ADR-137): its seconds of set-up never stall the 3-D view
  if (order.length && Q.get('camsWorker') !== '0' && CamsHost.supported()) {
    const t0 = performance.now();
    try {
      const h = await CamsHost.create({ order, base: page.bundle.base ?? new URL('assets/', ROOT).href, physics: page.physics,
        onLost: (m) => { page.camsLost = m; }, onGpuError: (m) => { page.tileErr = `GPU error: ${m}`; } });
      page.camsFailed.push(...(h.failed || []));
      if (h.dir) {
        page.cams = h; page.camsDir = h.dir; page.camsLoadMs = performance.now() - t0; page.camsWorker = true;
        page.camNames = new Set(h.cameras);
        console.log(`[twin] camera sims: ${h.dir} (worker) in ${page.camsLoadMs.toFixed(0)} ms`);
        return;
      }
      return;                                                  // nothing in the order starts there: none would here either
    } catch (e) {
      console.warn('[twin] camera worker unavailable; the renderer runs on the page thread:', e);
      page.camsFailed.push({ dir: 'worker', error: String(e?.message || e) });
    }
  }
  for (const dir of order) {
    const t0 = performance.now();
    try {
      const mod = await import(new URL(camsModuleUrl(dir), ROOT).href);
      const R = rendererClass(mod, dir === 'trace' ? ['TwinRenderer'] : dir === 'stub' ? ['StubRenderer'] : ['LiteRenderer']);
      if (!R) throw new Error(`${dir} exports no renderer`);
      const r = await R.create({ scene: page.bundle, physics: page.physics, preset: 'live', log: (...a) => console.log(`[cams ${dir}]`, ...a) });
      page.cams = r; page.camsDir = dir; page.camsLoadMs = performance.now() - t0;
      page.camNames = new Set(Array.isArray(r.cameras) ? r.cameras : Object.keys(r.cameras || {}));
      try { page.presenter = new Presenter(r.device); } catch (e) { console.warn('[twin] presenter', e); }
      r.device?.lost?.then?.((i) => { page.camsLost = i?.message || 'lost'; });
      console.log(`[twin] camera sims: ${dir} in ${page.camsLoadMs.toFixed(0)} ms`);
      return;
    } catch (e) {
      console.warn(`[twin] camera renderer ${dir} unavailable:`, e);
      page.camsFailed.push({ dir, error: String(e?.message || e) });
    }
  }
}

async function boot() {
  // the source first and in parallel with the 3-D load: calib_embed.js mounts its dock (in server mode, the camera page
  // that captures the cameras) as soon as the source is known, never after the scene, the god view or a failure of either
  page.sourceP = sourceInit();
  await loadAll();
  page.limits = new Limits(page.scene, page.meshes);
  mark('twin:boot:limits');
  page.state = page.limits.clampState(K.defaultState(page.scene)).state;
  if (Q.get('aim') !== '0') {
    const clamp = (st) => page.limits.clampState(st).state;
    page.state = aimScope(page.scene, page.state, toolsTarget(page.scene, page.state), { clamp }).state;
  }
  mark('twin:boot:aimed');
  const vp = page.gv.viewpoints[Q.get('view')] ? Q.get('view') : DEFAULT_VIEWPOINT;
  page.orbit = OrbitCamera.fromViewpoint(page.gv, vp);
  // 'view' (the plane follows the eye; the default), 'fixed' (a fixed vertical section with solid caps) or false
  page.cutaway = cutModeFromQuery(Q.get('cut')) ?? (cutawayDefault(page.gv) ? 'view' : false);
  buildPage();
  mark('twin:boot:pageBuilt');
  await makeGod();
  page.godCamNames = new Set(page.god?.cameraViewNames?.() ?? []);
  syncSide();                                            // Sim is possible now (the god view draws the cameras)
  bindGodInput($('god'));
  $('boot').hidden = true;
  // headless Chromium on Vulkan left a stale 6 px tile under the "?" button after the first WebGPU frame; one repaint
  // of the button clears it (harmless elsewhere)
  requestAnimationFrame(() => requestAnimationFrame(() => { const b = $('help-btn'); b.style.transform = 'translateZ(0)'; requestAnimationFrame(() => { b.style.transform = ''; }); }));
  page.bootMs = performance.now();
  mark('twin:boot:godUp');
  page.controls = new Controls({ surface: null, scene: page.scene, screenBasis: () => page.orbit.basis(), orbit: () => {} });
  page.controls.addEventListener('select', (e) => flashNote(`${TOOL_NAMES[e.detail.tool] ?? e.detail.tool} selected`));
  page.controls.addEventListener('toggle', (e) => {
    if (e.detail.what === 'reset-view') { page.orbit.reset(); page.godDirty = true; }
    else if (e.detail.what === 'cutaway') {
      // C cycles view -> fixed -> off (a fallback renderer without the fixed section: view <-> off)
      page.cutaway = nextCutMode(page.cutaway, page.god?.cutawayMode !== undefined);
      page.god.setCutaway(page.cutaway); page.godDirty = true; flashNote(cutModeNote(page.cutaway));
    }
  });
  wireAutoplay();
  await page.sourceP;
  mark('twin:boot:sourceAwaited');
  if (page.sourceNote) flashNote(page.sourceNote, 6000);
  rigInit();
  realInit();
  setInterval(controlTick, 8);
  requestAnimationFrame(frame);
  paintLine();
  setInterval(paintLine, 250);
  // the camera column after the 3-D view is up: its renderer's set-up (seconds of CPU for some) never delays the
  // first frame, and starts once the first frames are on screen
  await new Promise((res) => setTimeout(res, Number(Q.get('camsDelay') ?? 800)));
  mark('twin:cams:start');
  makeCams().then(() => {
    mark('twin:cams:made');
    applyTiles();
    setTimeout(tileSlot, 50);
    // Sim from the god view: its scope-only bake fills in idle time, one <= 3 ms GPU chunk per 33 ms while Sim shows
    setInterval(() => { if (!page.cams && (simShown() || simLarge())) page.god?.stepCameraBake?.(); }, 33);
    // the model socket is the local twin server's (/ws/twin); the static site has none (?modelSocket=1 tries anyway)
    if (!document.querySelector('meta[name="veridien-tracker"]') || Q.get('modelSocket') === '1') page.model = new ModelLink(page);
  }).catch((e) => console.warn('[twin] cams', e));
}

// ------------------------------------------------------------------ the page

function sizeCanvas(c) {
  const r = c.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; return true; }
  return false;
}

function buildPage() {
  syncSide();
  $('keys').innerHTML = KEYS.map(([k, v]) => (v == null ? `<dt class="group">${k}</dt>` : `<dt>${k}</dt><dd>${v}</dd>`)).join('');
  const side = $('side');
  // the column's header: the Real / Sim switch (T); "?" sits at its right while the column shows
  const head = document.createElement('div');
  head.className = 'col-head';
  head.innerHTML = '<div class="seg" role="group" aria-label="Camera images">'
    + '<button type="button" data-mode="real" aria-pressed="false" title="The CV server\u2019s camera images (T)">Real</button>'
    + '<button type="button" data-mode="sim" aria-pressed="false" title="The twin\u2019s simulated cameras (T)">Sim</button></div>';
  for (const b of head.querySelectorAll('button')) b.addEventListener('click', (e) => { e.currentTarget.blur(); setCamsMode(b.dataset.mode); });
  side.appendChild(head);
  // the simulated tiles (Sim): SCOPE first, then CAM1-CAM3; ADR-132: Real stays the default while the server answers
  const simSec = document.createElement('div');
  simSec.className = 'sim-cams';
  side.appendChild(simSec);
  for (const name of TILE_ORDER) {
    const t = document.createElement('button');
    t.type = 'button'; t.className = 'tile'; t.dataset.view = name;
    t.innerHTML = `<span class="label">${viewLabel(name)} · sim</span><span class="frame"><canvas></canvas><span class="note">Loading</span></span>`;
    t.addEventListener('click', (e) => { e.currentTarget.blur(); setLarge(page.large === name ? null : name); });
    simSec.appendChild(t);
    page.tiles[name] = { el: t, canvas: t.querySelector('canvas'), note: t.querySelector('.note'), shown: false };
  }
  $('help-btn').addEventListener('click', (e) => { e.currentTarget.blur(); setHelp(!page.help); });
  document.addEventListener('pointerdown', (e) => { if (page.help && !e.target.closest('#help, #help-btn')) setHelp(false); });
  $('large').addEventListener('click', () => setLarge(null));
  syncSide();                                            // the sim tiles (if any) exist now
  window.addEventListener('keydown', onPageKey);
  window.addEventListener('resize', () => { page.godDirty = true; });
  new ResizeObserver(() => { page.godDirty = true; page.largeShown = false; for (const t of Object.values(page.tiles)) t.shown = false; }).observe($('stage'));
}

function setHelp(on) {
  page.help = on;
  $('help').hidden = !on;
  $('help-btn').setAttribute('aria-expanded', String(on));
  if (on) paintInfo();
}

function setLarge(name) {
  if (name && !tileAvailable(name)) return;
  if (name) page.real?.setLarge(null);
  page.large = name;
  $('large').hidden = !name;
  $('god').hidden = !!name;
  $('large-label').hidden = !name;
  $('large-label').textContent = name ? `${viewLabel(name)} · sim · Esc returns` : '';
  page.largeShown = false;
  for (const [n, t] of Object.entries(page.tiles)) t.el.classList.toggle('current', n === name);
  if (!name) { page.godDirty = true; $('god').focus({ preventScroll: true }); }
  page.tileLast = null;
}

function setCollapsed(on) {
  if (!on && !columnHasViews()) { flashNote('No camera views: the CV server is not reachable'); return; }
  page.collapsed = on;
  syncSide();
}

/** A renderer can draw the simulated cameras: ?cams=<dir> or the god view itself (godview_pt3 renderCameraView). */
function simOk() { return !!page.cams || (typeof page.god?.renderCameraView === 'function' && page.godCamNames.size > 0); }
function realUp() { return !!page.real?.available(); }

/** The column has something to show: the simulated tiles or the real cameras (CV server reachable). */
function columnHasViews() { return simOk() || realUp(); }
function simShown() { return page.camsMode === 'sim' && !document.body.classList.contains('collapsed'); }
function simLarge() { return !!page.large; }

/** The column is shown when it has views and V has not hidden it; otherwise the 3-D view takes the full width. Its mode
 *  (cams_mode.js): the remembered choice while it can be shown, else Real while the server answers, else Sim. */
function syncSide() {
  const shown = !page.collapsed && columnHasViews();
  document.body.classList.toggle('collapsed', !shown);
  const mode = resolveCamsMode({ chosen: page.camsChosen, realUp: realUp(), simOk: simOk() });
  if (mode !== page.camsMode) {
    page.camsMode = mode;
    if (mode !== 'sim' && page.large) setLarge(null);
    if (mode !== 'real' && page.real?.large) page.real.setLarge(null);
    page.tileLast = null;
  }
  const side = $('side');
  if (mode) side.dataset.mode = mode; else delete side.dataset.mode;
  for (const b of side.querySelectorAll('.col-head button')) {
    b.setAttribute('aria-pressed', String(b.dataset.mode === mode));
    b.classList.toggle('unavailable', b.dataset.mode === 'real' ? !realUp() : !simOk());
  }
  page.real?.setVisible(shown && mode === 'real');
  page.godDirty = true;
}

/** Real or Sim, from the switch or T: remembered (localStorage) when it can be shown, else a note says why not. */
function setCamsMode(m) {
  if (m === 'real' && !realUp()) { flashNote('Real camera images need the CV server; it is not answering'); return; }
  if (m === 'sim' && !simOk()) { flashNote('No renderer can draw the simulated cameras'); return; }
  page.camsChosen = m;
  saveCamsMode(m);
  if (page.collapsed) page.collapsed = false;
  syncSide();
  flashNote(m === 'sim' ? 'Simulated cameras: the twin seen by SCOPE and CAM1-CAM3, scope light only' : 'Real camera images from the CV server');
}

// a camera's first WARM_FRAMES frames are rendered (its auto-exposure and white balance converge) but not shown: the
// tile keeps saying "Starting" instead of flashing an overexposed, green-tinted image
const WARM_FRAMES = 12;
function warm(name) {
  const t = page.tiles[name], n = page.tileCount[name] ?? 0;
  const on = n < (page.cams ? WARM_FRAMES : 2);           // the god view's AE settles from its second frame
  t.el.classList.toggle('warming', on);
  t.note.hidden = !on;
  if (on) t.note.textContent = 'Starting';
}

function tileAvailable(name) { return page.cams ? !!page.camNames?.has(name) : page.godCamNames.has(name) && simOk(); }

function applyTiles() {
  for (const [name, t] of Object.entries(page.tiles)) {
    const ok = tileAvailable(name);
    t.el.classList.toggle('off', !ok);
    t.el.disabled = !ok;
    t.note.hidden = ok;
    t.note.textContent = ok ? '' : !simOk() ? 'No camera renderer' : name === 'SCOPE' ? 'No scope view in this renderer' : 'Not rendered';
    if (ok) warm(name);
  }
}

function onPageKey(e) {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName ?? '') || e.metaKey || e.ctrlKey) return;
  if (e.key === '?' || (e.code === 'Slash' && e.shiftKey)) { e.preventDefault(); setHelp(!page.help); return; }
  if (e.repeat || e.altKey) return;
  if (e.code === 'Escape' && page.propMode) { setPropMode(false); return; }
  if (e.code === 'Escape') { if (page.help) setHelp(false); else if (page.real?.large) page.real.setLarge(null); else if (page.large) setLarge(null); }
  else if (e.code === 'KeyV') setCollapsed(!page.collapsed);
  else if (e.code === 'KeyT') { const m = toggleCamsMode(page.camsMode, { realUp: realUp(), simOk: simOk() }); if (m !== page.camsMode) setCamsMode(m); else flashNote(page.camsMode === 'sim' ? 'Real camera images need the CV server; it is not answering' : 'No renderer can draw the simulated cameras'); }
  else if (e.code === 'KeyR') toggleRig();
  else if (e.code === 'KeyP') setPropMode(!page.propMode);
  else if (e.code === 'KeyA') { if (page.mode === 'autoplay') stopAutoplay('stopped'); else startAutoplay(page.autoName, page.autoSeed); }
  else if (e.code === 'KeyM') {
    if (!page.model?.attached()) { flashNote('Ghosts need a model attached'); page.ghostsOn = false; }
    else { page.ghostsOn = !page.ghostsOn; flashNote(page.ghostsOn ? 'Predicted tools shown as ghosts' : 'Ghosts off'); }
    page.godDirty = true;
  }
}

// ------------------------------------------------------------------ god's-eye input: orbit, pan, zoom; Alt-drag moves the tool

function bindGodInput(c) {
  c.addEventListener('contextmenu', (e) => e.preventDefault());
  let mode = null, lx = 0, ly = 0, pid = null;
  c.addEventListener('pointerdown', (e) => {
    c.focus({ preventScroll: true });
    if (page.help) setHelp(false);
    mode = e.altKey && e.button === 0 ? 'tool' : (e.button === 2 || e.button === 1 || (e.button === 0 && e.shiftKey)) ? 'pan'
      : e.button === 0 ? (page.propMode ? 'prop' : 'orbit') : null;
    if (!mode) return;
    pid = e.pointerId; lx = e.clientX; ly = e.clientY;
    try { c.setPointerCapture(pid); } catch { /* plain events still work */ }
    c.classList.toggle('dragging', mode !== 'tool');
    c.classList.toggle('tool', mode === 'tool');
    e.preventDefault();
  });
  c.addEventListener('pointermove', (e) => {
    if (!mode || e.pointerId !== pid) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    if (mode === 'orbit') page.orbit.orbitBy(dx, dy);
    else if (mode === 'pan') page.orbit.panBy(dx, dy, c.clientHeight);
    else if (mode === 'prop') propDrag(dx, dy, c.clientHeight);
    else if (page.controls) {
      // the tool follows the cursor in the view's plane: Controls' drag gain is per px, scaled here so the tip
      // moves about as far as the cursor does at the target's depth
      const mPerPx = (2 * page.orbit.d * Math.tan(page.orbit.vfov_deg * Math.PI / 360)) / Math.max(1, c.clientHeight);
      const k = mPerPx / 0.00008;
      page.controls.drag.dx += dx * k; page.controls.drag.dy += dy * k;
      page.controls.shift = false;
      page.controls._touch();
    }
  });
  const end = (e) => { if (e.pointerId === pid) { mode = null; pid = null; c.classList.remove('dragging', 'tool'); } };
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);
  c.addEventListener('wheel', (e) => {
    e.preventDefault();
    const k = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1;
    if (e.altKey && page.controls) { page.controls.wheel += e.deltaY * k; page.controls._touch(); }
    else if (page.propMode) propTurn(e.deltaY * k);
    else page.orbit.zoomBy(e.deltaY * k);
  }, { passive: false });
}

// ------------------------------------------------------------------ movable props (P): the practice module on the floor
// Body "prop/module" (assets/godview_extras.json, its vertices in D at the kit placement): its pose turns the prop by
// yaw about its own centre O and slides it by (dx, dy) on the floor. Remembered in localStorage.

const PROP_KEY = 'veridien.twin.prop';
function propInit(bundle) {
  const ms = (bundle?.scene?.meshes ?? bundle?.meshes ?? []).filter((m) => m?.body === 'prop/module' && m.bbox);
  page.propO = null;
  if (!ms.length) return;
  const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const m of ms) for (let i = 0; i < 2; i++) { lo[i] = Math.min(lo[i], m.bbox[0][i]); hi[i] = Math.max(hi[i], m.bbox[1][i]); }
  page.propO = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
  page.prop = { dx: 0, dy: 0, yaw: 0 };
  try { Object.assign(page.prop, JSON.parse(localStorage.getItem(PROP_KEY) || '{}')); } catch { /* no storage */ }
}
function withProps(poses) {
  if (!page.propO || !poses?.bodies) return poses;
  const { dx, dy, yaw } = page.prop, [ox, oy] = page.propO, c = Math.cos(yaw), s = Math.sin(yaw);
  const tx = ox + dx - (c * ox - s * oy), ty = oy + dy - (s * ox + c * oy);
  return { ...poses, bodies: { ...poses.bodies, 'prop/module': [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, tx, ty, 0, 1] } };
}
function propChanged() {
  page.posesV = (page.posesV ?? 0) + 1; page.godDirty = true;
  try { localStorage.setItem(PROP_KEY, JSON.stringify(page.prop)); } catch { /* no storage */ }
  const p = page.prop;
  page.flash = { text: `Practice base: ${(p.dx * 1000).toFixed(0)}, ${(p.dy * 1000).toFixed(0)} mm, ${(p.yaw * 180 / Math.PI).toFixed(0)}° · P when done`, until: performance.now() + 4000 };
}
function setPropMode(on) {
  if (on && !page.propO) { flashNote('No practice module in this view'); return; }
  page.propMode = !!on;
  flashNote(on ? 'Moving the practice base: drag to slide it, scroll to turn it, P when done' : 'Practice base placed', on ? 4000 : 1600);
}
function propDrag(dx, dy, h) {
  // the cursor's motion mapped onto the floor: screen right and screen up projected onto x-y, at the target's depth
  const { right, up } = page.orbit.basis();
  const mPerPx = (2 * page.orbit.d * Math.tan(page.orbit.vfov_deg * Math.PI / 360)) / Math.max(1, h);
  const r = [right[0], right[1]], u = [up[0], up[1]];
  const nr = Math.hypot(...r) || 1, nu = Math.hypot(...u) || 1;
  page.prop.dx += (r[0] / nr * dx - u[0] / nu * dy) * mPerPx;
  page.prop.dy += (r[1] / nr * dx - u[1] / nu * dy) * mPerPx;
  propChanged();
}
function propTurn(delta) { page.prop.yaw += delta * 0.002; propChanged(); }

// ------------------------------------------------------------------ input, autoplay, limits (125 Hz)

let lastControl = performance.now();
function controlTick() {
  const now = performance.now();
  const dt = Math.min(0.1, Math.max(0, (now - lastControl) / 1000));
  lastControl = now;
  if (!page.controls) return;
  const t = now / 1000;
  if (page.mode === 'autoplay') {
    if (page.controls.activity > page.scriptWall) stopAutoplay('you took over');
    else {
      const ts = t - page.scriptT0;
      if (ts > page.script.duration) stopAutoplay('finished');
      else page.state = page.limits.clampState(page.script.stateAt(ts)).state;
    }
    page.controls.step(page.state, dt);       // drain the input accumulators
  } else if (page.rig?.on) {
    page.state = page.limits.clampState(rigMerge(page.controls.step(page.state, dt))).state;   // the scope stays manual
  } else {
    page.state = page.limits.clampState(page.controls.step(page.state, dt)).state;
  }
  page.state = { ...page.state, t };
  const key = JSON.stringify({ ...page.state, t: 0 });
  if (key !== page.stateKey) { page.stateKey = key; page.posesV++; page.lastMotion = now; }
}

function wireAutoplay() {
  const ap = Q.get('autoplay');
  page.autoName = SCRIPTS.includes(ap) ? ap : 'task_set';
  page.autoSeed = Number(Q.get('seed')) || 0;
  if (page.autoName.startsWith('charuco') && !(page.scene.boards || []).length) page.autoName = 'task_set';
  if (ap && ap !== '0') setTimeout(() => startAutoplay(page.autoName, page.autoSeed), 300);
}

function startAutoplay(name, seed) {
  if (page.rig?.on) { flashNote('The real rig drives the tools; R hands them back'); return; }
  page.script = makeScript(name, page.scene, seed, { limits: page.limits });
  page.scriptT0 = performance.now() / 1000; page.scriptWall = performance.now(); page.mode = 'autoplay';
  paintLine();
}

function stopAutoplay(why) {
  if (page.mode !== 'autoplay') return;
  page.mode = 'manual';
  page.controls.adopt(page.state);
  flashNote(`Autoplay ${why}`);
}

// ------------------------------------------------------------------ the god's-eye view (rAF)

let lastFrame = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  rigFrame(now);                // rig mode: the interpolated tips for this frame (twinpage/rig_smooth.js)
  const dt = Math.min(0.1, Math.max(0, (now - lastFrame) / 1000));
  page.rafMs.push(now - lastFrame); if (page.rafMs.length > 240) page.rafMs.shift();
  lastFrame = now;
  const god = page.god;
  if (!god || page.large || page.real?.large) return;
  // the GodView owns its canvas's backing size (three.js resizes its own targets in render()); the page only notices
  // that the CSS box changed
  const c = $('god');
  const box = `${c.clientWidth}x${c.clientHeight}`;
  const resized = box !== page.godBox;
  page.godBox = box;
  const moving = page.orbit.update(dt);
  const poses = page.posesV !== page.drawnPosesV;
  // render while anything changes, for KEEP_MS after the last change (a renderer that refines when still gets its
  // frames), and for as long as a progressive renderer says so (progressive, needsRender(), or timings().samples rising)
  const changed = moving || poses || resized || page.godDirty;
  if (changed) page.godChangedAt = now;
  if (!(changed || now - (page.godChangedAt ?? 0) < KEEP_MS || refining(god, now))) return;
  const before = [c.width, c.height];
  const t0 = performance.now();
  try {
    const cam = page.orbit.camera();
    const camKey = JSON.stringify(cam);
    if (camKey !== page.godCamKey) { god.setCamera(cam); page.godCamKey = camKey; }    // a refining renderer restarts on every setCamera
    zoomCut(god);
    if (poses || page.godDirty) {
      god.setPoses(K.posesFromState(page.scene, page.state));
      page.drawnPosesV = page.posesV;
      if (typeof god.setGhosts === 'function') god.setGhosts(page.ghostsOn ? page.ghosts : null);
    }
    god.render();
    // a GodView that does not size its own canvas (three.js does) gets the CSS box x devicePixelRatio (<= 2)
    if (resized && c.width === before[0] && c.height === before[1] && sizeCanvas(c)) page.godDirty = true;
    drawGhostOverlay(god);
    page.godDirty = false;
    if (page.firstFrameMs == null) mark('twin:god:firstRafFrame');
    page.firstFrameMs ??= performance.now();
    page.godErr = null;
  } catch (e) {
    page.godErr = String(e?.message || e);
    if (!page.godErrLogged) { console.error('[twin] god render', e); page.godErrLogged = true; }
  }
  page.frameTimes.push(performance.now() - t0); if (page.frameTimes.length > 240) page.frameTimes.shift();
}

// the cutaway follows the zoom (godview.yaml cutaway.zoom): the dome closes as the orbit backs off and opens coming in.
// 'view': the plane's offset slides (smoothstep in distance); 'fixed' (caps built for one plane): shown or closed with
// hysteresis; off: untouched. page.cutaway stays the C key's choice; page.zoomCut is what the renderer was given.
function zoomCut(god) {
  const z = page.zoomCut;
  if (!z || z.god !== god || z.mode !== page.cutaway) page.zoomCut = { god, mode: page.cutaway, shown: null, off: null };
  const Z = page.zoomCut, d = page.orbit.d;
  if (page.cutaway === 'fixed') {
    const shown = zoomFixedShown(page.gv, d, Z.shown);
    if (shown !== Z.shown) { god.setCutaway(shown ? 'fixed' : false); Z.shown = shown; }
  } else if (page.cutaway && typeof god.setCutawayOffset === 'function') {
    const off = zoomCutOffset(page.gv, d);
    if (off !== Z.off) { god.setCutawayOffset(off); Z.off = off; }
  }
}

const KEEP_MS = 3000;
function refining(god, now) {
  if (god.progressive === true) return true;
  if (typeof god.needsRender === 'function') return !!god.needsRender();
  const n = god.timings?.()?.samples;
  if (typeof n !== 'number') return false;
  if (n !== page.godSamples) { page.godSamples = n; page.godSamplesAt = now; }
  return now - (page.godSamplesAt ?? 0) < 1500;
}

/** Ghosts for a renderer without setGhosts: the predicted shafts and jaws as lines on a 2-D overlay. */
function drawGhostOverlay(god) {
  const ov = $('ghosts');
  const want = page.ghostsOn && page.ghosts?.lines && typeof god.setGhosts !== 'function';
  ov.hidden = !want;
  if (!want) return;
  sizeCanvas(ov);
  const g = ov.getContext('2d');
  const w = ov.width, h = ov.height, s = w / Math.max(1, ov.clientWidth);
  g.clearRect(0, 0, w, h);
  g.strokeStyle = 'rgba(47, 91, 211, 0.75)'; g.lineCap = 'round';
  for (const seg of page.ghosts.lines) {
    const a = page.orbit.project(seg[0], ov.clientWidth, ov.clientHeight), b = page.orbit.project(seg[1], ov.clientWidth, ov.clientHeight);
    if (!a || !b) continue;
    g.lineWidth = (seg[2] || 3) * s;
    g.beginPath(); g.moveTo(a[0] * s, a[1] * s); g.lineTo(b[0] * s, b[1] * s); g.stroke();
  }
}

// ------------------------------------------------------------------ the camera column (its own timer)

async function tileSlot() {
  const t0 = performance.now();
  let wall = 0;
  try { wall = await tileOnce(); } finally {
    // <= 15 Hz in total, and the tiles never take more than ~15 % of the wall clock (a slow camera renderer slows the
    // column, never the 3-D view); a camera shown large (the 3-D view hidden) gets 30 Hz
    const period = page.large ? slotPeriodMs(page.large) : Math.max(slotPeriodMs(null, !page.cams), 6.5 * wall);
    setTimeout(tileSlot, Math.max(0, period - (performance.now() - t0)));
  }
}

async function tileOnce() {
  const r = page.cams;
  const large = page.large;
  if (!r) return godTileOnce();
  if (page.tileBusy || page.camsLost || page.model?.busy) return 0;
  const now = performance.now();
  const still = now - page.lastMotion > 1500 && page.camsPosesV === page.posesV;
  // a camera's first frames come at the full rate (its auto-exposure converges), then a still scene re-renders at 1 Hz
  const lastAt = Object.fromEntries(Object.entries(page.tileAt).map(([n, t]) => [n, (page.tileCount[n] ?? 0) < 12 ? -Infinity : t]));
  const name = nextTile({ available: TILE_ORDER.filter(tileAvailable), large, collapsed: page.collapsed || page.camsMode !== 'sim', last: page.tileLast, now, lastAt, still: still && (!large || page.largeShown), stillMs: 1000 });
  if (!name) return 0;
  page.tileBusy = true;
  const t0 = performance.now();
  if (page.camsWorker) {
    try {
      const t = page.tiles[name];
      const targets = [{ canvas: t.canvas, ...backing(t.canvas) }];
      if (name === large) targets.push({ canvas: $('large'), ...backing($('large')) });
      const poses = page.camsPosesV !== page.posesV ? K.posesFromState(page.scene, page.state) : null;
      page.camsPosesV = page.posesV;
      const frame = ++page.camFrame || (page.camFrame = 1);
      await r.tile({ name, size: name === large ? LARGE_CAP : TILE_SIZE, poses, seed: 1 + TILE_ORDER.indexOf(name), frame, canvases: targets });
      if (name === large) page.largeShown = true;
      t.shown = true;
      page.tileLast = name; page.tileAt[name] = now; page.tileFrames++; page.tileCount[name] = (page.tileCount[name] ?? 0) + 1; warm(name);
      page.tileErr = null;
    } catch (e) {
      page.tileErr = `${viewLabel(name)}: ${e?.message || e}`;
      if (!page.tileErrLogged) { console.error('[twin] tile', e); page.tileErrLogged = true; }
    } finally {
      page.tileBusy = false;
      page.tileMs.push(performance.now() - t0); if (page.tileMs.length > 120) page.tileMs.shift();
    }
    return performance.now() - t0;
  }
  try {
    const tex = await renderCamera(name, name === large ? LARGE_CAP : TILE_SIZE);
    const t = page.tiles[name];
    if (tex !== undefined) {
      if (name === large) { presentTo(name, tex, $('large')); page.largeShown = true; }
      presentTo(name, tex, t.canvas);
      t.shown = true;
    }
    await r.device?.queue?.onSubmittedWorkDone?.();          // the slot's wall time includes its GPU work (duty cap)
    page.tileLast = name; page.tileAt[name] = now; page.tileFrames++; page.tileCount[name] = (page.tileCount[name] ?? 0) + 1; warm(name);
    page.tileErr = null;
  } catch (e) {
    page.tileErr = `${viewLabel(name)}: ${e?.message || e}`;
    if (!page.tileErrLogged) { console.error('[twin] tile', e); page.tileErrLogged = true; }
  } finally {
    page.tileBusy = false;
    page.tileMs.push(performance.now() - t0); if (page.tileMs.length > 120) page.tileMs.shift();
  }
  return performance.now() - t0;
}

/** Sim from the god view (godview_pt3 renderCameraView): one camera per slot, round-robin, raster only; never waits on
 *  the GPU (its cost is the view's own GPU timestamps, page.god.cameraViewTimings()). */
function godTileOnce() {
  const god = page.god, large = page.large;
  if (typeof god?.renderCameraView !== 'function' || god.lostInfo || page.model?.busy) return 0;
  const now = performance.now();
  const still = now - page.lastMotion > 1500 && page.camsPosesV === page.posesV;
  const bakeV = god.cameraViewTimings?.()?.bake?.version ?? 0;
  if (bakeV !== page.simBakeV) { page.simBakeV = bakeV; page.tileAt = {}; }        // a finished bake: draw every view again
  const lastAt = Object.fromEntries(Object.entries(page.tileAt).map(([n, t]) => [n, (page.tileCount[n] ?? 0) < 3 ? -Infinity : t]));
  const name = nextTile({ available: TILE_ORDER.filter(tileAvailable), large, collapsed: page.collapsed || page.camsMode !== 'sim', last: page.tileLast, now, lastAt, still: still && (!large || page.largeShown), stillMs: 1000 });
  if (!name) return 0;
  const t0 = performance.now();
  try {
    // the god view and the camera views share the instance poses: bring them up to date (the 3-D view redraws too)
    if (page.drawnPosesV !== page.posesV) { god.setPoses(K.posesFromState(page.scene, page.state)); page.drawnPosesV = page.posesV; page.godDirty = true; }
    page.camsPosesV = page.posesV;
    const t = page.tiles[name];
    const targets = [];
    if (name === large) targets.push(godTarget($('large')));
    targets.push(godTarget(t.canvas));
    if (god.renderCameraView(targets, name, { large: name === large })) {
      if (name === large) page.largeShown = true;
      t.shown = true;
      if (!page.tileFrames) mark('twin:sim:firstTile');
      page.tileAt[name] = now; page.tileFrames++; page.tileCount[name] = (page.tileCount[name] ?? 0) + 1;
    }
    page.tileLast = name;
    warm(name);
    page.tileErr = null;
  } catch (e) {
    page.tileErr = `${viewLabel(name)}: ${e?.message || e}`;
    if (!page.tileErrLogged) { console.error('[twin] sim tile', e); page.tileErrLogged = true; }
  }
  page.tileMs.push(performance.now() - t0); if (page.tileMs.length > 120) page.tileMs.shift();
  return performance.now() - t0;
}

/** A canvas as a target of the god view's camera views: its backing at CSS x dpr, a context on the god view's device. */
function godTarget(canvas) {
  sizeCanvas(canvas);
  const dev = page.god.device;
  if (!canvas.__gctx || canvas.__gdev !== dev) {
    const ctx = canvas.getContext('webgpu');
    ctx.configure({ device: dev, format: page.god.canvasFormat || navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' });
    canvas.__gctx = ctx; canvas.__gdev = dev;
  }
  return { ctx: canvas.__gctx, w: canvas.width, h: canvas.height };
}

/** Render + sense one camera at `size` (the §18.1 API). Returns the sensor texture (or null when the renderer drew it). */
export async function renderCamera(name, size) {
  const r = page.cams;
  if (r.setViewSize) {
    const cur = page.camSize?.[name];
    if (!cur || cur[0] !== size[0] || cur[1] !== size[1]) { r.setViewSize(name, size); (page.camSize ??= {})[name] = [...size]; }
  }
  if (page.camsPosesV !== page.posesV) { r.setPoses(K.posesFromState(page.scene, page.state)); page.camsPosesV = page.posesV; }
  const frame = ++page.camFrame || (page.camFrame = 1);
  const lt = r.renderCamera(name, { await: false, wait: false, seed: 1 + TILE_ORDER.indexOf(name) });
  if (lt?.then) await lt;
  const tex = r.sensor(name, { seed: (frame * 7919) >>> 0, frame });
  return tex?.then ? await tex : tex;
}

/** The backing size a canvas should have: its CSS box x devicePixelRatio (<= 2), as sizeCanvas sets it. */
function backing(c) {
  const r = c.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  return { w: Math.max(1, Math.round(r.width * dpr)), h: Math.max(1, Math.round(r.height * dpr)) };
}

function presentTo(name, tex, canvas) {
  sizeCanvas(canvas);
  const r = page.cams;
  if (typeof r.present === 'function') {
    // configure the context on the renderer's device ourselves: some renderers (lite_pt) configure only a canvas they
    // are handed and draw into a context as given; the others reconfigure it to the same preferred format
    let ctx = canvas.__ctx;
    if (!ctx) {
      ctx = canvas.__ctx = canvas.getContext('webgpu');
      if (r.device) ctx.configure({ device: r.device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' });
    }
    r.present(name, ctx, { fit: 'contain' });
  } else if (tex && page.presenter) page.presenter.present(tex, canvas);
}

// ------------------------------------------------------------------ rig mode (twinpage/rig_link.js): L and R follow the real rig
// ?rig=1 (http://127.0.0.1:8833) or ?rig=<url>; R toggles. Each new feed frame becomes the twin's L/R state (tip exact,
// shaft through the CAD pivot, limits.js clamps); autoplay and the keyboard tool moves for L and R are suspended, the
// view and the scope stay manual. When the feed is down the tools hold their last pose and the quiet line says so.
// Smoothing (ADR-133): snapshot interpolation, as networked games draw remote players. Each frame is buffered with its
// sender t and drawn ~2 feed intervals late on a curve through every snapshot at its own t (rigFrame, once per rAF).
// ?rigSmooth=0 applies each snapshot as it arrives; ?rigSmooth=euro is the old one-euro filter; ?rigDelay=<ms> fixes
// the delay. The tips are PUSHED (EventSource on <url>/api/tips/stream, ~28 Hz, ADR-133); the 30 Hz poll of /api/tips is
// the fallback (?rigStream=0 forces it). The feed comes from rig_link.js's feedHub, shared with the camera overlay.
// While the feed is live the god view's path tracer is capped (rigThrottle; ?rigThrottle=0 keeps full quality).

// ------------------------------------------------------------------ the tracker source (twinpage/tracker_source.js)
// 'server': the CV server, exactly as before. 'browser': this page's own tracker; its feed is registered in the shared
// feedHub under LOCAL_BASE, so rig mode and the camera column take it through the same code as a server's. calib_embed.js
// waits for the 'twin-source' event to know which dock to build.

async function sourceInit() {
  // the static build (scripts/build_twin_site.py) writes <meta name="veridien-tracker" content="auto"> into twin.html
  const siteDefault = document.querySelector('meta[name="veridien-tracker"]')?.content || null;
  try {
    page.source = await makeSource(Q, {
      siteDefault,
      makeBrowser: async () => { const m = await import('./browser_source.js'); return new m.BrowserSource({ q: Q }).init(); },
    });
  } catch (e) {
    console.error('[twin] tracker source', e);
    page.sourceNote = `The in-page tracker could not start: ${e?.message || e}`;
    page.source = new ServerSource({ base: rigUrlFromQuery(Q) ?? RIG_DEFAULT_URL });
  }
  if (page.source.kind === 'browser') {
    const src = page.source, make = feedHub.make;
    feedHub.make = (base, opts) => (base === LOCAL_BASE ? src.feed : make(base, opts));
    src.on((what) => { if (what === 'status' && page.bootMs) paintLine(); });
    // auto fell back to this page's tracker: once the CV server answers while this page's cameras are off (and nothing
    // is recording or replaying), reload onto it, as the server feed reconnects by itself (tracker_source.watchServer)
    if (src.asked === 'auto' && !src.replay) {
      page.serverWatch = watchServer(rigUrlFromQuery(Q) ?? RIG_DEFAULT_URL, {
        when: () => !document.hidden && !['starting', 'running'].includes(src.status) && !src.record.active(),
        onUp: () => { console.log('[twin] the camera server answers: reloading onto it'); location.reload(); },
      });
    }
  }
  console.log(`[twin] tracker: ${page.source.kind}${page.source.asked !== page.source.kind ? ` (asked ${page.source.asked})` : ''}`);
  mark('twin:source:made');
  window.__twinSource = page.source;
  window.dispatchEvent(new CustomEvent('twin-source', { detail: page.source }));
  return page.source;
}

function rigInit() {
  const url = rigUrlFromQuery(Q);
  const sm = Q.get('rigSmooth'), mode = sm === '0' ? 'raw' : sm === 'euro' ? 'euro' : 'interp';
  const fixed = Number(Q.get('rigDelay'));
  page.rig = { on: false, url: page.source?.kind === 'browser' ? LOCAL_BASE : (url ?? RIG_DEFAULT_URL), feed: null, q: {}, residual_mm: {}, fit_mm: {}, limited: {}, seen: {}, applied: 0, fed: null,
    pivots_mm: null, mode,
    smoother: new TipSmoother({ smooth: mode === 'euro' }),
    interp: new SnapshotInterp({ fixedDelay: Q.has('rigDelay') && Number.isFinite(fixed) ? fixed / 1000 : null }) };
  // feeds made from now on (rig mode's and the camera overlay's are one): push first unless ?rigStream=0
  feedHub.configure({ stream: Q.get('rigStream') !== '0', timeoutMs: 400, staleS: 0.6 });
  if (url) setRig(true);
}

function setRig(on, quiet = false) {
  const r = page.rig;
  if (!r || r.on === on) return;
  r.on = on;
  if (on) {
    if (page.mode === 'autoplay') stopAutoplay('handed to the real rig');
    r.q = Object.fromEntries((page.state?.instruments ?? []).map((q) => [q.id, { ...q }]));   // held until the first frame
    r.smoother.reset(); r.interp.reset(); r.seen = {}; r.residual_mm = {}; r.fit_mm = {};
    // pushed (SSE); the fallback polls at 30 Hz (GET, never overlapping) since a poll sees only the newest frame
    r.sub = feedHub.acquire(r.url, rigResult, { hz: 30 });
    r.feed = r.sub.feed;
    rigCalibrationOnce();
  } else {
    r.sub?.release(); r.sub = null; r.feed = null;      // the stream stays open only while the camera overlay holds it
    rigThrottle(false);
    page.controls?.adopt(page.state);
    if (!quiet) flashNote('Real rig off; the keyboard has the tools');
  }
  paintLine();
}
function toggleRig() { page.rigTouched = true; page.rigAuto = false; setRig(!page.rig?.on); }

function rigResult(pred) {
  const r = page.rig;
  if (!r?.on || !page.scene) return;
  // ?rigStale=1 also applies frames the feed marks stale (default: they hold the last pose)
  const interp = r.mode === 'interp';
  const out = applyPredictions(page.scene, page.state, pred, { smoother: interp ? null : r.smoother, limits: page.limits, holdStale: Q.get('rigStale') !== '1' });
  const now = performance.now() / 1000;
  if (interp) r.interp.pushFrame(pred, now, out.used);   // drawn by rigFrame; the residuals below are the raw frame's
  for (const id of out.present) r.seen[id] = now;
  for (const id of out.used) {
    if (!interp) r.q[id] = out.state.instruments.find((q) => q.id === id);
    r.residual_mm[id] = out.residual_mm[id];
    r.limited[id] = out.limited[id];
    // the feed's own fit: residual_px (tipnet's RCM reprojection RMS) or the older residual_mm (classical axis-to-pivot)
    const rec = (pred.instruments || []).find((x) => x?.id === id);
    const fit = Number.isFinite(rec?.residual_px) ? rec.residual_px : rec?.residual_mm;
    r.fit_mm[id] = Number.isFinite(fit) ? fit : undefined;
    r.fit_unit = Number.isFinite(rec?.residual_px) ? 'px' : 'mm';
  }
  // an instrument the feed omits holds its last pose; after 0.6 s its numbers leave the line and it shows as not seen
  const after = lostAfter(r.feed?.rate());
  for (const id of Object.keys(r.residual_mm)) if (!out.present.includes(id) && now - (r.seen[id] ?? 0) > after) { delete r.residual_mm[id]; delete r.fit_mm[id]; }
  r.fed = { frame_id: pred.frame_id, t: pred.t, tips: Object.fromEntries((pred.instruments || []).filter((x) => x?.tip_D).map((x) => [x.id, x.tip_D])) };
  r.applied++;
}

// The live tracker shares this Mac: while rig mode is on and the feed is live (or was within 2 s), the god view's path
// tracer is capped at RIG_TRACE (128 spp for a still view, one band every other frame); rig off restores the options it
// had. godview_pt3 reads opts.liveSamples (needsRender), opts.maxSamples and opts.maxTilesPerFrame (render) each frame.
const RIG_TRACE = { liveSamples: 128, maxSamples: 128, maxTilesPerFrame: 1 };
function rigThrottle(on, nowMs = performance.now()) {
  const god = page.god;
  if (!god?.opts || typeof god.needsRender !== 'function') return;
  const r = page.rig;
  if (on && r?.feed?.status() === 'live') r.liveAt = nowMs;
  const want = on && Q.get('rigThrottle') !== '0' && nowMs - (r?.liveAt ?? -Infinity) < 2000;
  if (want && !god.__rigSaved) {
    god.__rigSaved = Object.fromEntries(Object.keys(RIG_TRACE).map((k) => [k, god.opts[k]]));
    Object.assign(god.opts, RIG_TRACE);
  } else if (!want && god.__rigSaved) {
    for (const [k, v] of Object.entries(god.__rigSaved)) { if (v === undefined) delete god.opts[k]; else god.opts[k] = v; }
    god.__rigSaved = null;                                 // needsRender() asks again: the trace refines on from where it is
  }
  if (god.__rigSaved) god.opts.maxTilesPerFrame = (page.rigTick = (page.rigTick ?? 0) ^ 1);   // trace every other frame
}

function rigMerge(state) {
  const q = page.rig.q;
  return { ...state, instruments: state.instruments.map((s) => (q[s.id] ? { ...s, ...q[s.id] } : s)) };
}

/** Once per rendered frame (rAF time, ms): sample the interpolated tips, solve them about the CAD pivots, clamp. */
function rigFrame(nowMs) {
  const r = page.rig;
  rigThrottle(!!(r?.on && r.feed), nowMs);
  if (!r?.on || r.mode !== 'interp' || !page.scene || !page.state || !page.limits) return;
  const S = r.interp.sampleAll(nowMs / 1000);
  let any = false;
  for (const [id, s] of Object.entries(S)) {
    if (!page.scene.instruments.some((i) => i.id === id)) continue;
    const res = solveTip(page.scene, id, s.tip, s.dir, page.state, { roll: s.roll, jaw: s.jaw, limits: page.limits });
    r.q[id] = { ...(r.q[id] || {}), ...res.q };
    any = true;
  }
  if (!any) return;
  page.state = page.limits.clampState(rigMerge(page.state)).state;
  const key = JSON.stringify({ ...page.state, t: 0 });
  if (key !== page.stateKey) { page.stateKey = key; page.posesV++; page.lastMotion = performance.now(); page.godDirty = true; }
}

function rigText() {
  const r = page.rig, f = r.feed;
  if (!f) return null;
  const ids = page.scene.instruments.map((i) => i.id);
  const lost = r.applied ? lostIds(r.seen, performance.now() / 1000, ids, lostAfter(f.rate())) : [];
  const smooth_s = r.mode === 'interp' && r.interp.ready() ? r.interp.delayS : undefined;
  const src = page.source?.kind === 'browser' ? page.source : null;
  const waiting = src ? browserWaiting(src) : undefined;
  return rigLine({ label: f.label, waiting, status: f.status(), host: f.host(), rate: f.rate(), smooth_s, residual_mm: r.residual_mm, fit_mm: r.fit_mm, fit_unit: r.fit_unit, lost, model: f.last?.model });
}

/** The browser tracker's line while no tips come: what it waits for. */
function browserWaiting(src) {
  const t = src.workerInfo;
  if (src.status === 'idle') return 'Browser tracker: cameras off (Start cameras, bottom right)';
  if (src.status === 'starting') return 'Browser tracker: opening the cameras';
  if (src.status === 'error' || src.status === 'moved') return `Browser tracker: ${src.error}`;
  if (!t) return 'Browser tracker: loading the model';
  return 'Browser tracker: waiting for frames';
}

// ------------------------------------------------------------------ the real cameras (twinpage/real_cams.js, ADR-132)
// The CV server (the rig feed's base: ?rig=<url>, else http://127.0.0.1:8833) is probed at boot and every 10 s. While it
// answers, the column shows CAM1-CAM3 as it sees them with its keypoints, and rig mode turns on by itself (unless ?rig=0
// or R was pressed); when it stops answering, the column goes and an automatic rig mode turns off quietly. No error.

function realInit() {
  if (Q.get('real') === '0') return;
  const local = page.source?.kind === 'browser' ? page.source : null;
  const base = local ? LOCAL_BASE : (rigUrlFromQuery(Q) ?? RIG_DEFAULT_URL);
  page.real = new RealCams({
    base, side: $('side'), stage: $('stage'), frames: local?.frames ?? null, calib: local?.calib ?? null,
    tips: () => (page.rig?.on && page.rig.feed?.last ? { payload: page.rig.feed.last, at: page.rig.feed.lastAt } : null),
    onAvail: (up) => {
      syncSide();
      if (up && !page.rig.on && Q.get('rig') !== '0' && !page.rigTouched) { page.rigAuto = true; setRig(true); }
      if (!up && page.rigAuto && page.rig.on) { page.rigAuto = false; setRig(false, true); }
    },
    onLarge: (cam) => {
      if (cam && page.large) setLarge(null);
      $('god').hidden = !!(cam || page.large);
      if (!cam) { page.godDirty = true; $('god').focus({ preventScroll: true }); }
    },
    log: (m) => console.log(`[twin] ${m}`),
  });
  page.real.start();
}

// /api/calibration, once per rig session: the calibrated pivots vs the twin's CAD ones (informational; changes nothing)
async function rigCalibrationOnce() {
  const r = page.rig;
  if (r.pivots_mm) return;
  try {
    const calib = await page.source?.calibration();
    if (!calib) return;
    const off = pivotOffsets(page.scene, calib);
    if (!Object.keys(off).length) return;
    r.pivots_mm = off;
    const txt = Object.entries(off).map(([id, mm]) => `${id} ${mm.toFixed(1)} mm`).join(', ');
    console.info(`[twin] rig calibrated pivots vs CAD: ${txt}`);
    flashNote(`Calibrated pivots vs the twin's: ${txt}`, 4000);
  } catch { /* the calibration endpoint is optional */ }
}

// ------------------------------------------------------------------ the quiet line

function flashNote(text, ms = 1600) { page.flash = { text, until: performance.now() + ms }; paintLine(); }

function problems() {
  const p = [];
  if (page.god?.lostInfo) p.push(`The 3-D view's GPU device was lost (${page.god.lostInfo}); reload the page`);
  if (page.godErr) p.push(`The 3-D view failed to draw: ${page.godErr}`);
  if (page.godGpuErr && performance.now() - page.godGpuErr.at < 4000) p.push(`The 3-D view reported a GPU error: ${page.godGpuErr.text}`);
  if (page.godDir === PLACEHOLDER) p.push(page.godFailed.length && Q.get('god') ? `${page.godFailed[0].dir} did not load; showing the placeholder 3-D renderer` : 'Placeholder 3-D renderer: no god-view renderer is installed yet');
  else if (page.godFailed.length && Q.get('god')) p.push(`${page.godFailed[0].dir} did not load; showing ${godLabel(page.godDir)}`);
  if (page.camsLost) p.push(`The camera views' GPU device was lost (${page.camsLost})`);
  if (page.cams === null && page.camsFailed.length >= camsOrder(Q.get('cams')).length && camsOrder(Q.get('cams')).length) p.push('No camera renderer could start');
  if (page.tileErr) p.push(`Camera view failed: ${page.tileErr}`);
  if (page.model?.problem()) p.push(page.model.problem());
  if (page.gvSource?.startsWith('built-in')) p.push('Using the built-in copy of godview.yaml: restart veridien webtwin serve to read the file');
  return p;
}

function paintLine() {
  const el = $('line');
  const p = problems();
  let text = p[0] ?? null, problem = !!text;
  if (!text && page.rig?.on) text = rigText();
  if (!text && page.mode === 'autoplay') text = `Autoplay, ${page.autoName.replace(/_/g, ' ')}; any input takes over`;
  if (!text && page.flash && performance.now() < page.flash.until) text = page.flash.text;
  if (page.flash && performance.now() < page.flash.until && problem) text = page.flash.text;
  el.hidden = !text;
  if (text && el.textContent !== text) el.textContent = text;
  el.classList.toggle('problem', problem);
}

function paintInfo() {
  const t = page.god?.timings?.() ?? {};
  const ms = (x) => (x == null ? '–' : `${x.toFixed(1)} ms`);
  const ct = page.god?.cameraViewTimings?.();
  $('help-info').textContent = `3-D view: ${godLabel(page.godDir)} (GPU ${ms(t.gpu_ms_median)}). Camera column: ${page.camsMode ?? 'none'}. Camera sims: ${page.camsDir ?? (ct ? `the 3-D view's raster (GPU ${ms(ct.tile_gpu_ms_median)} a view)` : 'none')}. Real cameras: ${page.real?.available() ? `${page.real.probe.cams.join(', ')} from ${page.real.base} (${page.real.probe.mode})` : 'none'}. Config: ${page.gvSource}.`;
}

// ------------------------------------------------------------------ test hooks (web/twin/tests/twinpage_shots.py)

window.twinPageApi = {
  ready: null,
  info: () => ({ god: page.godDir, godFailed: page.godFailed, cams: page.camsDir, camsFailed: page.camsFailed, gv: page.gvSource,
    large: page.large, collapsed: page.collapsed, help: page.help, cutaway: page.cutaway, mode: page.mode, line: $('line').hidden ? null : $('line').textContent,
    bootMs: page.bootMs, firstFrameMs: page.firstFrameMs, godLoadMs: page.godLoadMs, camsLoadMs: page.camsLoadMs, camsWorker: !!page.camsWorker,
    sideShown: !document.body.classList.contains('collapsed'), simTiles: Object.keys(page.tiles), realLarge: page.real?.large ?? null,
    camsMode: page.camsMode, camsChosen: page.camsChosen, simOk: simOk(), realUp: realUp() }),
  camsMode: () => page.camsMode,
  setCamsMode: (m) => setCamsMode(m),
  setLarge: (name) => setLarge(name),
  simCams: () => ({ timings: page.god?.cameraViewTimings?.() ?? null, tileCount: { ...page.tileCount }, tileMs: [...page.tileMs] }),
  realCams: () => page.real?.info() ?? null,
  setRealLarge: (cam) => page.real?.setLarge(cam),
  setCollapsed: (on) => setCollapsed(!!on),
  timings: () => {
    const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
    return { god: page.god?.timings?.() ?? null, page_cpu_ms_median: med(page.frameTimes), raf_ms_median: med(page.rafMs),
      tile_ms_median: med(page.tileMs), tiles: page.tileFrames, cams: page.cams?.timings?.() ?? null };
  },
  orbit: (dAz, dEl) => { page.orbit.orbitBy(dAz, dEl); },
  setView: (name) => { page.orbit = OrbitCamera.fromViewpoint(page.gv, name); page.godDirty = true; },
  state: () => page.state,
  setState: (s) => { page.state = page.limits.clampState(s).state; page.posesV++; },
  forceRender: () => { page.godDirty = true; },
  zoomCut: () => ({ d: page.orbit.d, cutaway: page.cutaway, renderer: page.god?.cutawayMode ?? null, offset: page.god?.cutawayOffset ?? null }),
  rig: () => (page.rig ? { on: page.rig.on, url: page.rig.url, status: page.rig.feed?.status() ?? 'off', rate: page.rig.feed?.rate() ?? 0,
    feed: page.rig.feed?.info() ?? null, throttled: !!page.god?.__rigSaved,
    residual_mm: page.rig.residual_mm, fit_mm: page.rig.fit_mm, pivots_mm: page.rig.pivots_mm, limited: page.rig.limited, last: page.rig.feed?.last ?? null, applied: page.rig.applied,
    fed: page.rig.fed, line: page.rig.on ? rigText() : null, smooth: { mode: page.rig.mode, ...page.rig.interp.info() } } : null),
  setRig: (on) => setRig(!!on),
  /** Where the tips come from: {mode: 'server', base} or the browser source's status, tracker, timing, cameras, names. */
  tracker: () => page.source?.info() ?? null,
  /** Rig mode's tip match: the last fed tip_D vs the twin's tip (truthFromState of the drawn state), in mm and in the
   * 3-D view's CSS px (web/twin/tests/rig_shots.py). */
  rigCheck: () => {
    const r = page.rig; if (!r?.fed) return null;
    const c = $('god'), W = c.clientWidth, H = c.clientHeight;
    const tr = K.truthFromState(page.scene, page.state).instruments;
    const out = { frame_id: r.fed.frame_id, t: r.fed.t, W, H, tools: {} };
    for (const [id, fed] of Object.entries(r.fed.tips)) {
      const tw = tr.find((x) => x.id === id); if (!tw) continue;
      const e = Math.hypot(fed[0] - tw.tip_D[0], fed[1] - tw.tip_D[1], fed[2] - tw.tip_D[2]);
      out.tools[id] = { fed_tip_D: fed, twin_tip_D: tw.tip_D, err_mm: e * 1000, fed_px: page.orbit.project(fed, W, H), twin_px: page.orbit.project(tw.tip_D, W, H),
        hinge_px: page.orbit.project(tw.hinge_D, W, H) };
    }
    return out;
  },
  /** The selected-or-given tool's hinge and tip in the 3-D view (CSS px), for the Alt-drag check. */
  toolScreen: (id) => {
    const tr = K.truthFromState(page.scene, page.state).instruments.find((i) => i.id === id);
    const c = $('god');
    return tr ? { hinge: page.orbit.project(tr.hinge_D, c.clientWidth, c.clientHeight), tip: page.orbit.project(tr.tip_D, c.clientWidth, c.clientHeight) } : null;
  },
};
window.twinPageApi.ready = boot().catch((e) => {
  console.error(e);
  const b = $('boot'); b.hidden = false; b.classList.add('err'); b.textContent = `The page could not start.\n${e?.stack || e}`;
  throw e;
});
