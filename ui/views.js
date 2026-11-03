// The one-view page's state and layout (docs/WEB_TWIN.md §18.2), as pure functions: which views exist, which one
// is the main view, whether the 2x2 grid or the key help is open, what each visible view renders at, and where the
// camera image lands inside its frame. No DOM and no GPU (web/twin/tests/ui_views.test.mjs).

/** The grid's views, in reading order. */
export const GRID = Object.freeze(['SCOPE', 'CAM1', 'CAM2', 'CAM3']);

/** Render caps (§18.2): the main view at most 1280x720, a grid tile at most 640x360; the display scales them up. */
export const MAIN_CAP = Object.freeze([1280, 720]);
export const TILE_CAP = Object.freeze([640, 360]);

/** The keys, as the "?" panel lists them (sentence case). */
export const KEYS = Object.freeze([
  ['1 / 2 / 3', 'Select the fenestrated grasper, the Maryland, the scope'],
  ['Drag', 'Swing the selected tool in the view’s directions (button = clutch)'],
  ['Arrows', 'Swing'],
  ['Wheel, W / S', 'Insert, withdraw'],
  ['Q / E', 'Roll'],
  ['Space', 'Hold to close the jaws'],
  ['J / K', 'Open, close the jaws'],
  ['Shift', 'Fine'],
  ['V', 'Four views; click one to make it the main view'],
  ['Esc', 'Close the four views or this panel'],
  ['M', 'Model against truth markers (model attached)'],
  ['P', 'Start or stop autoplay'],
  ['Shift R', 'Record the autoplay script (twin server only)'],
  ['?', 'These keys'],
]);

/** "Scope", "Camera 1", ...: the label of a view. */
export function viewLabel(name) {
  if (name === 'SCOPE') return 'Scope';
  const m = /^CAM(\d+)$/.exec(name);
  return m ? `Camera ${m[1]}` : name;
}

/**
 * The initial view state.
 *   available: names the page can show (the renderer's cameras, plus 'SCOPE' when it or a fallback can draw it)
 *   requested: ?view=... (ignored when not available)
 *   grid:      ?grid=1 opens the grid at load
 */
export function initialViewState({ available, requested = null, grid = false, help = false } = {}) {
  const av = [...new Set(available ?? [])];
  const main = requested && av.includes(requested) ? requested : defaultMain(av);
  return { mode: grid ? 'grid' : 'single', main, help: !!help, markers: false, available: av };
}

/** SCOPE when available, else CAM1, else the first available view. */
export function defaultMain(available) {
  if (available.includes('SCOPE')) return 'SCOPE';
  if (available.includes('CAM1')) return 'CAM1';
  return available[0] ?? null;
}

/**
 * The view state's one reducer. Actions:
 *   {type: 'toggleGrid'}            V and the "4 views" control
 *   {type: 'pick', name}            a grid tile clicked: it becomes the main view and the grid closes
 *   {type: 'escape'}                Esc: closes the help, else the grid
 *   {type: 'toggleHelp'} / {type: 'closeHelp'}
 *   {type: 'toggleMarkers', modelAttached}   M: only while a model is attached
 *   {type: 'modelDetached'}         markers go off with the model
 *   {type: 'setMain', name}         ?view=, scripts
 */
export function viewReducer(s, a) {
  switch (a?.type) {
    case 'toggleGrid': return { ...s, mode: s.mode === 'grid' ? 'single' : 'grid', help: false };
    case 'pick':
      if (!s.available.includes(a.name)) return s.mode === 'grid' ? { ...s, mode: 'single' } : s;
      return { ...s, main: a.name, mode: 'single' };
    case 'setMain': return s.available.includes(a.name) ? { ...s, main: a.name } : s;
    case 'escape':
      if (s.help) return { ...s, help: false };
      if (s.mode === 'grid') return { ...s, mode: 'single' };
      return s;
    case 'toggleHelp': return { ...s, help: !s.help };
    case 'closeHelp': return s.help ? { ...s, help: false } : s;
    case 'toggleMarkers': return a.modelAttached ? { ...s, markers: !s.markers } : (s.markers ? { ...s, markers: false } : s);
    case 'modelDetached': return s.markers ? { ...s, markers: false } : s;
    default: return s;
  }
}

/** The views on screen now: the main view, or the four grid tiles (each {name, slot, available, cap}). */
export function visibleViews(s) {
  if (s.mode === 'grid') return GRID.map((name, i) => ({ name, slot: `tile${i}`, available: s.available.includes(name), cap: TILE_CAP }));
  return s.main ? [{ name: s.main, slot: 'main', available: true, cap: MAIN_CAP }] : [];
}

/** Largest w x h of the source aspect that fits a cap box (integers, even). */
export function fitCap(aspect, cap) {
  let w = cap[0], h = Math.round(w / aspect);
  if (h > cap[1]) { h = cap[1]; w = Math.round(h * aspect); }
  return [Math.max(2, w - (w % 2)), Math.max(2, h - (h % 2))];
}

/** A canvas's backing size for a CSS box: devicePixelRatio, at most 2, at least 1 px. */
export function backingSize(cssW, cssH, dpr = 1) {
  const k = Math.max(1, Math.min(2, dpr || 1));
  return [Math.max(1, Math.round(cssW * k)), Math.max(1, Math.round(cssH * k))];
}

/**
 * 'contain' letterbox: the rectangle {x, y, w, h} (canvas px) a srcW x srcH image takes inside a dstW x dstH frame,
 * centred, aspect kept, edges on whole pixels. The rest of the frame is near-black.
 */
export function letterbox(srcW, srcH, dstW, dstH) {
  if (!(srcW > 0 && srcH > 0 && dstW > 0 && dstH > 0)) return { x: 0, y: 0, w: Math.max(0, dstW), h: Math.max(0, dstH) };
  const k = Math.min(dstW / srcW, dstH / srcH);
  const w = Math.min(dstW, Math.round(srcW * k)), h = Math.min(dstH, Math.round(srcH * k));
  return { x: Math.floor((dstW - w) / 2), y: Math.floor((dstH - h) / 2), w, h };
}

/** Image px (u, v of a srcW x srcH camera image) -> CSS px inside a frame, through the letterbox (for markers). */
export function imageToFrame(uv, src, rect) {
  return [rect.x + (uv[0] + 0.5) * rect.w / src[0], rect.y + (uv[1] + 0.5) * rect.h / src[1]];
}

/**
 * Display pacing for the direct (rAF) path. A frame is due when the scene or the views changed, or, in a still
 * scene, at stillFps (sensor noise and auto-exposure stay alive at a low cost). The period never drops below
 * 2x the last frame's GPU time, so the GPU idles at least as long as it works (§12: never back-to-back).
 */
export function displayDue({ now, lastAt, lastMs = 0, dirty = false, moving = false, stillFps = 10, minPeriodMs = 0 }) {
  const floor = Math.max(minPeriodMs, 2 * Math.max(0, lastMs));
  const since = now - lastAt;
  if (since < floor) return false;
  if (dirty || moving) return true;
  return since >= 1000 / stillFps;
}

/** The status line (§18.2): one quiet line, only while a problem lasts; the first problem wins. */
export function statusLine(p) {
  if (p.deviceLost) return `The GPU device was lost (${p.deviceLost}). Reload the page to restart.`;
  if (p.tripped) return `Rendering stopped by the GPU budget guard: ${p.tripped}. Reload the page to restart.`;
  if (p.socketLost) return 'Lost the connection to the twin server; retrying.';
  if (p.renderer === 'stub') return `Stub renderer: flat shading for plumbing tests, not the twin’s light${p.fallbackFrom ? ` (${p.fallbackFrom} did not load)` : ''}.`;
  if (p.fallbackFrom && p.renderer === 'trace') return `The lite renderer did not load, so the path tracer is drawing (slower).`;
  if (p.error) return p.error;
  return '';
}
