// Which renderers the twin page loads (docs/WEB_TWIN.md §19). Pure choice of names here (node tests); the page imports.
//
//   ?god=three|pt|<dir>   the god's-eye renderer, <dir>/godview.js exporting class GodView (web/twin/<dir>/).
//                         Default (ADR-137): godview_pt3 (godview_pt, the Mitsuba judge's winner, plus the fixes: two-sided
//                         cut bake so the moving raster passes 8/8 unmasked, clip-distance cut, cached static shadows,
//                         bounded first submits), then godview_pt, godview_three, else the placeholder
//                         (twinpage/placeholder_godview.js). ?god=pt means godview_pt3.
//   ?cams=<dir>           the camera-simulation renderer for the side column (§18.1 API, <dir>/renderer.js).
//                         Default (ADR-132, 2026-11-01): none. The column shows the REAL camera images instead
//                         (twinpage/real_cams.js); ?cams=<dir> turns the simulations on, <dir> first, then the order
//                         below. That order (ADR-137): the first that loads of lite_pt2 (lite_pt's light transport, byte-identical
//                         renderCamera/sensor, with a prepared-scene cache, worker CPU work and async pipelines: create
//                         0.4-1 s warm / ~6 s cold on the M4 instead of 7.5 s), lite_pt (fewest failing judge cells and
//                         fastest on the M4 of the realistic ones), lite_prt, lite_nodes, lite_rt, lite, then the path
//                         tracer (gpu/, no SCOPE), then the stub.

export const GOD_DEFAULT_ORDER = Object.freeze(['godview_pt3', 'godview_pt', 'godview_three']);
export const PLACEHOLDER = 'placeholder';
export const CAMS_DEFAULT_ORDER = Object.freeze(['lite_pt2', 'lite_pt', 'lite_prt', 'lite_nodes', 'lite_rt', 'lite', 'trace', 'stub']);
const SAFE_DIR = /^[a-z][a-z0-9_]{0,40}$/;

/** The god-view candidates to try, in order. `asked`: the ?god= value (null for the default). */
export function godOrder(asked) {
  if (asked) {
    const dir = asked === 'three' ? 'godview_three' : asked === 'pt' ? 'godview_pt3' : asked === 'placeholder' ? PLACEHOLDER : asked;
    if (dir !== PLACEHOLDER && !SAFE_DIR.test(dir)) return [PLACEHOLDER];
    return dir === PLACEHOLDER ? [PLACEHOLDER] : [dir, PLACEHOLDER];
  }
  return [...GOD_DEFAULT_ORDER, PLACEHOLDER];
}

/** The module URL of a god-view candidate, relative to web/twin/. */
export function godModuleUrl(dir) {
  return dir === PLACEHOLDER ? './twinpage/placeholder_godview.js' : `./${dir}/godview.js`;
}

/** The camera-sim candidates to try, in order: none by default; ?cams=<dir> puts <dir> first, then CAMS_DEFAULT_ORDER. */
export function camsOrder(asked) {
  if (asked) {
    if (asked === 'none') return [];
    if (!SAFE_DIR.test(asked)) return [...CAMS_DEFAULT_ORDER];
    return [asked, ...CAMS_DEFAULT_ORDER.filter((d) => d !== asked)];
  }
  return [];                                             // ADR-132: no simulated cameras unless ?cams= asks
}

export function camsModuleUrl(dir) {
  if (dir === 'trace') return './gpu/renderer.js';
  if (dir === 'stub') return './ui/stub_renderer.js';
  return `./${dir}/renderer.js`;
}

/** The renderer class a module exports: a named class with create(), the default export, or the first such value. */
export function rendererClass(mod, preferred = []) {
  for (const n of preferred) if (typeof mod?.[n]?.create === 'function') return mod[n];
  if (typeof mod?.default?.create === 'function') return mod.default;
  return Object.values(mod || {}).find((v) => typeof v === 'function' && typeof v.create === 'function') || null;
}

/** Labels for the quiet status line and the help sheet. */
export function godLabel(dir) {
  return dir === PLACEHOLDER ? 'placeholder raster' : dir === 'godview_three' ? 'three.js god view' : dir.startsWith('godview_pt') ? 'path-traced god view' : dir;
}
