// The one-view page's state, as pure functions (docs/WEB_TWIN.md §18.2). No DOM, no GPU: view/tests/ runs them in node.
//
//   main     the view that fills the window: SCOPE by default (CAM1 when the renderer has no SCOPE)
//   grid     the "4 views" 2x2 grid of SCOPE, CAM1, CAM2, CAM3 (V toggles, Esc closes, a click makes a tile the main view)
//   help     the small "?" key list
//   markers  model-vs-truth markers (M), only while a model is attached

export const GRID = Object.freeze(['SCOPE', 'CAM1', 'CAM2', 'CAM3']);
export const MAIN_CAP = Object.freeze([1280, 720]);
export const TILE_CAP = Object.freeze([640, 360]);
export const ASPECT = 16 / 9;

/** ?renderer= names -> module path and export name. Order of the default search: the lite candidates that have SCOPE. */
// present: how the renderer draws a view on a canvas.
//   'sense+present'  r.sensor(name) runs the display look, then r.present(name, ctx) blits it (lite_nodes, lite_pt, lite_prt)
//   'present'        r.present(name, ctx) runs the display look itself (lite_rt; any renderer with presentSenses = true)
//   'blit'           r.sensor(name) -> GPUTexture, drawn by ui/present.js (the tracer's own present() takes a texture)
// seed: the tracer keeps its static caches with a fixed seed per camera; the lite tracers accumulate only without one.
export const RENDERERS = Object.freeze({
  lite_nodes: { path: '../lite_nodes/renderer.js', exports: ['LiteNodesRenderer', 'LiteRenderer', 'Renderer', 'default'], present: 'sense+present' },
  lite: { path: '../lite/renderer.js', exports: ['LiteRenderer', 'Renderer', 'default'], present: 'sense+present' },
  lite_pt: { path: '../lite_pt/renderer.js', exports: ['LitePTRenderer', 'LiteRenderer', 'Renderer', 'default'], present: 'sense+present' },
  lite_prt: { path: '../lite_prt/renderer.js', exports: ['LitePRTRenderer', 'LiteRenderer', 'Renderer', 'default'], present: 'sense+present' },
  lite_rt: { path: '../lite_rt/renderer.js', exports: ['LiteRTRenderer', 'LiteRenderer', 'Renderer', 'default'], present: 'present' },
  trace: { path: '../gpu/renderer.js', exports: ['TwinRenderer', 'default'], present: 'blit', seed: true },
  stub: { path: '../ui/stub_renderer.js', exports: ['StubRenderer', 'default'], present: 'blit' },
});
export const LITE_ORDER = Object.freeze(['lite_nodes', 'lite', 'lite_pt', 'lite_prt', 'lite_rt']);

/**
 * The renderers to try, in order, for a ?renderer= value. An asked-for renderer comes first, then the default search
 * as the fallback. Default: the lite candidates (each must have SCOPE), then the path tracer, then the stub.
 * Returns [{kind, needScope}].
 */
export function rendererPlan(asked, order = null) {
  const out = [];
  const add = (kind, needScope) => { if (!out.some((x) => x.kind === kind)) out.push({ kind, needScope }); };
  if (asked && RENDERERS[asked]) add(asked, false);
  const lite = order?.length ? order.filter((k) => RENDERERS[k] && k !== 'trace' && k !== 'stub') : LITE_ORDER;
  for (const k of lite) add(k, true);
  add('trace', false);
  add('stub', false);
  return out;
}

/** The camera names a renderer can render: r.cameras as a list or an object keyed by name, or r.cams (the stub). */
export function rendererCameras(r, fallback = []) {
  if (!r) return [];
  if (Array.isArray(r.cameras)) return r.cameras.map((c) => (typeof c === 'string' ? c : c.name));
  if (r.cameras && typeof r.cameras === 'object') return Object.keys(r.cameras);
  if (r.cams && typeof r.cams === 'object') return Object.keys(r.cams);
  return fallback;
}

export function viewLabel(name) {
  if (name === 'SCOPE') return 'Laparoscope';
  const m = /^CAM(\d+)$/.exec(name ?? '');
  return m ? `Camera ${m[1]}` : String(name ?? '');
}

export function initialState({ available = [], requested = null, grid = false, help = false } = {}) {
  const main = available.includes(requested) ? requested : defaultMain(available);
  return { available: [...available], main, grid: !!grid, help: !!help, markers: false };
}

export function defaultMain(available) {
  if (available.includes('SCOPE')) return 'SCOPE';
  if (available.includes('CAM1')) return 'CAM1';
  return available[0] ?? 'CAM1';
}

/** The reducer. Unknown actions return the same object (so callers can test for a change by identity). */
export function reduce(s, a) {
  switch (a?.type) {
    case 'toggleGrid': return { ...s, grid: !s.grid, help: false };
    case 'openGrid': return s.grid ? s : { ...s, grid: true, help: false };
    case 'closeGrid': return s.grid ? { ...s, grid: false } : s;
    case 'pick': return s.available.includes(a.name) ? { ...s, main: a.name, grid: false } : s;
    case 'toggleHelp': return { ...s, help: !s.help };
    case 'closeHelp': return s.help ? { ...s, help: false } : s;
    case 'escape': return s.help ? { ...s, help: false } : s.grid ? { ...s, grid: false } : s;
    case 'toggleMarkers': return a.modelAttached ? { ...s, markers: !s.markers } : s;
    case 'available': {
      const available = [...a.available];
      const main = available.includes(s.main) ? s.main : defaultMain(available);
      return { ...s, available, main };
    }
    default: return s;
  }
}

/** The views on screen: the main view alone, or the four grid tiles (only the ones the renderer can draw). */
export function visibleViews(s) {
  if (s.grid) return GRID.map((name, i) => ({ slot: `tile${i}`, name, drawable: s.available.includes(name) }));
  return [{ slot: 'main', name: s.main, drawable: s.available.includes(s.main) }];
}

/**
 * The page's own keys (the tool keys 1/2/3, arrows, W/S, Q/E, Space, J/K, Shift are control/input.js's).
 * Returns an action for reduce(), {type: 'autoplay'}, or null.
 */
export function keyAction(e) {
  if (!e || e.metaKey || e.ctrlKey || e.altKey) return null;
  if (e.repeat) return null;
  switch (e.code) {
    case 'KeyV': return { type: 'toggleGrid' };
    case 'Escape': return { type: 'escape' };
    case 'KeyM': return { type: 'toggleMarkers' };
    case 'KeyA': return { type: 'autoplay' };
    case 'Slash': return e.shiftKey || e.key === '?' ? { type: 'toggleHelp' } : null;
    case 'KeyH': return { type: 'toggleHelp' };
    default: return e.key === '?' ? { type: 'toggleHelp' } : null;
  }
}

/** The key list behind "?". Sentence case, short. */
export const KEYS = Object.freeze([
  ['1 2 3', 'select left grasper, right grasper, scope'],
  ['drag', 'swing the tool about its port'],
  ['arrows', 'swing'],
  ['wheel, W S', 'insert, withdraw'],
  ['Q E', 'roll'],
  ['Space', 'hold to grasp'],
  ['J K', 'open, close the jaws'],
  ['Shift', 'fine'],
  ['A', 'autoplay on, off'],
  ['V', '4 views'],
  ['Esc', 'close'],
  ['M', 'model markers (with a model)'],
]);

/** The largest w x h of the given aspect that fits cap (both integers, even). */
export function fitCap(aspect, [cw, ch]) {
  let w = cw, h = Math.round(cw / aspect);
  if (h > ch) { h = ch; w = Math.round(ch * aspect); }
  return [w - (w % 2), h - (h % 2)];
}

/** The 16:9 letterbox rectangle of a frame of css size (w, h): {x, y, w, h}, centred, integers. */
export function letterbox(fw, fh, aspect = ASPECT) {
  let w = fw, h = fw / aspect;
  if (h > fh) { h = fh; w = fh * aspect; }
  w = Math.floor(w); h = Math.floor(h);
  return { x: Math.floor((fw - w) / 2), y: Math.floor((fh - h) / 2), w, h };
}

/**
 * The render size of a view: its on-screen image size in device px, capped (main 1280x720, tiles 640x360), 16:9,
 * never below 64 px wide. The display scales it to the canvas.
 */
export function renderSize(cssW, cssH, dpr, cap) {
  const d = Math.min(Math.max(dpr || 1, 1), 2);
  const want = [Math.max(64, Math.round(cssW * d)), Math.max(36, Math.round(cssH * d))];
  const [w, h] = fitCap(ASPECT, [Math.min(want[0], cap[0]), Math.min(want[1], cap[1])]);
  return [Math.max(64, w), Math.max(36, h)];
}

/**
 * Whether a display frame is due. Moving (or just changed): every display frame, one in flight. Still: stillFps, so
 * sensor noise and auto-exposure stay alive while the GPU mostly idles.
 */
export function frameDue({ now, lastAt, busy, moving, dirty, stillFps = 10 }) {
  if (busy) return false;
  if (moving || dirty) return true;
  return now - lastAt >= 1000 / Math.max(0.5, stillFps);
}

/** The one quiet status line: the most important problem, or '' while there is none. */
export function statusText(p = {}) {
  if (p.fatal) return p.fatal;
  if (p.deviceLost) return `The GPU device was lost (${p.deviceLost}). Reload the page.`;
  if (p.trip) return `GPU work stopped to protect the machine: ${p.trip}`;
  if (p.kind === 'stub') return p.fallback ? 'Stub renderer: no lite renderer or path tracer could start.' : 'Stub renderer: a stand-in, not the realistic twin.';
  if (p.kind === 'trace' && p.fallback) return 'Path tracer: no lite renderer with a scope view could start.';
  if (p.fallback && p.asked) return `The ${p.asked} renderer could not start; showing ${p.kind}.`;
  if (p.gpuErrors) return `The ${p.kind} renderer reports GPU errors (${p.gpuErrors}); see the console.`;
  if (p.note) return p.note;
  if (p.modelWanted && p.socket && p.socket !== 'open') return 'Model socket offline, reconnecting.';
  if (p.drawError) return `Display: ${p.drawError}`;
  return '';
}

/** How to draw a view with renderer r of this kind (see RENDERERS). */
export function presentMode(kind, r) {
  if (r && typeof r.present !== 'function') return 'blit';
  if (r && r.presentSenses === true) return 'present';
  return RENDERERS[kind]?.present ?? 'sense+present';
}

/** Median and p95 of a list (null when empty). */
export function stats(xs) {
  if (!xs?.length) return { n: 0, p50: null, p95: null, mean: null };
  const s = [...xs].sort((a, b) => a - b);
  const q = (f) => s[Math.min(s.length - 1, Math.floor(f * (s.length - 1) + 0.5))];
  return { n: s.length, p50: q(0.5), p95: q(0.95), mean: s.reduce((a, b) => a + b, 0) / s.length };
}
