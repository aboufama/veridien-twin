// The side column's camera tiles (docs/WEB_TWIN.md §19): SCOPE, CAM1, CAM2, CAM3 at 480x270, round-robin at <= 15 Hz in
// total so they never slow the god's-eye view; a camera shown large renders at <= 30 Hz at the large size, and its
// tile shows that image. Pure scheduling (node tests); the page does the GPU work.

export const TILE_ORDER = Object.freeze(['SCOPE', 'CAM1', 'CAM2', 'CAM3']);
export const TILE_SIZE = Object.freeze([480, 270]);
export const LARGE_CAP = Object.freeze([1280, 720]);
export const TILE_HZ = 15;
export const LARGE_HZ = 30;
export const SIM_HZ = 10;               // the god view's simulated cameras (Sim mode): round-robin, <= 10 Hz in total

// The column's width (twin.html --col): about a third of the page so the camera views balance the 3-D view (32 % at
// 1280-1440 px), never under COL_MIN_PX; the god view keeps the rest. Keep in step with the CSS (tests check both).
export const COL_FRAC = 0.32, COL_MIN_PX = 360, COL_MAX_PX = 560;
export function columnWidthPx(viewportW) { return Math.min(COL_MAX_PX, Math.max(COL_MIN_PX, COL_FRAC * viewportW)); }

/**
 * The next camera to render, or null. s: {available: names, large: name|null, collapsed, last: name|null,
 * now, lastAt: {name: ms}, still: bool}. A still scene re-renders a tile at most every `stillMs` (noise, AE).
 */
export function nextTile({ available, large = null, collapsed = false, last = null, now = 0, lastAt = {}, still = false, stillMs = 1000 }) {
  const order = TILE_ORDER.filter((n) => available.includes(n));
  if (large && available.includes(large)) return large;               // the large view owns the slot (paced by the caller)
  if (collapsed || !order.length) return null;
  const start = last && order.includes(last) ? order.indexOf(last) + 1 : 0;
  for (let k = 0; k < order.length; k++) {
    const n = order[(start + k) % order.length];
    if (!still || now - (lastAt[n] ?? -Infinity) >= stillMs) return n;
  }
  return null;
}

/** Period in ms of the camera slot: the tiles share TILE_HZ; a large camera gets LARGE_HZ. */
export function slotPeriodMs(large, sim = false) { return 1000 / (large ? LARGE_HZ : sim ? SIM_HZ : TILE_HZ); }

/** Render size of a view: the large view fits the canvas (aspect 16:9) under LARGE_CAP; tiles TILE_SIZE. */
export function viewSizeFor(name, large, canvasPx = null) {
  if (name !== large) return [...TILE_SIZE];
  if (!canvasPx) return [...LARGE_CAP];
  const [w, h] = canvasPx;
  const s = Math.min(LARGE_CAP[0] / 16, Math.max(w / 16, h / 9));   // cover the canvas's longer side, at most the cap
  return [Math.round(16 * Math.min(s, LARGE_CAP[0] / 16)), Math.round(9 * Math.min(s, LARGE_CAP[0] / 16))];
}
