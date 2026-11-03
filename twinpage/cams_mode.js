// The side column's Real / Sim switch (docs/WEB_TWIN.md §21): pure state (node tests, tests/twinpage_pure.test.mjs).
//   real: the CV server's images with its keypoints (twinpage/real_cams.js)
//   sim:  CAM1-CAM3 and SCOPE drawn by the god view's own raster (godview_pt3 renderCameraView)
// The choice is remembered in localStorage (guarded: a private window or blocked storage just forgets it).

export const CAMS_MODES = Object.freeze(['real', 'sim']);
export const CAMS_MODE_KEY = 'veridien.twin.camsMode';

/**
 * The mode the column shows. chosen: the remembered or clicked choice (or null); realUp: the CV server answers;
 * simOk: a renderer can draw the simulated views. A choice holds while it can be shown; otherwise Real when the
 * server answers, else Sim (better than an empty column), else null (no column).
 */
export function resolveCamsMode({ chosen = null, realUp = false, simOk = false } = {}) {
  if (chosen === 'sim' && simOk) return 'sim';
  if (chosen === 'real' && realUp) return 'real';
  if (realUp) return 'real';
  if (simOk) return 'sim';
  return null;
}

/** The other mode for the key (T), when it can be shown; else the current one. */
export function toggleCamsMode(current, { realUp = false, simOk = false } = {}) {
  const want = current === 'sim' ? 'real' : 'sim';
  return (want === 'real' ? realUp : simOk) ? want : current;
}

export function loadCamsMode(storage = globalThis.localStorage) {
  try { const v = storage?.getItem?.(CAMS_MODE_KEY); return CAMS_MODES.includes(v) ? v : null; } catch { return null; }
}

export function saveCamsMode(mode, storage = globalThis.localStorage) {
  try { if (CAMS_MODES.includes(mode)) storage?.setItem?.(CAMS_MODE_KEY, mode); return true; } catch { return false; }
}
