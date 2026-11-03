// Tip-net output decode for the in-browser tracker: the JS twin of tipnet.py's forward sigmoid, decode_heatmap and
// CellHysteresis (ADR-144), with numpy's float32 rules (NEP 50: a float32 compared with or multiplied by a Python
// float stays float32). Net output y: Float32Array [N, 6, 72, 128] = per image (tip logit, hinge logit, dx_tip,
// dy_tip, dx_hinge, dy_hinge), offsets in heatmap cells. Candidates come out in full-resolution fisheye px.

import { f32 } from './math.js';

export const CHANNELS = ['tip', 'hinge'];
export const STRIDE = 4;

/** In place: channels 0-1 of every image -> float32 sigmoid heat (1 / (1 + exp(-y)), each op rounded to float32). */
export function sigmoidHeat(y, n, hw = 72 * 128) {
  for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) {
    const o = (i * 6 + c) * hw;
    for (let p = 0; p < hw; p++) y[o + p] = f32(1 / f32(1 + f32(Math.exp(-y[o + p]))));
  }
  return y;
}

/** 3x3 non-maximum suppression peaks above `thresh` (heat > float32(thresh)), top k by conf (ties: row-major order,
 *  numpy's argsort is stable for <= 16 peaks). heat: Float32Array view of one h x w plane. -> [{iy, ix, conf}]. */
export function peaks(heat, h, w, k = 4, thresh = 0.2) {
  const th = f32(thresh);
  const out = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = heat[y * w + x];
    if (!(v > th)) continue;
    let ok = true;
    for (let dy = -1; dy <= 1 && ok; dy++) {
      const yy = y + dy;
      if (yy < 0 || yy >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx;
        if (xx < 0 || xx >= w) continue;
        if (heat[yy * w + xx] > v) { ok = false; break; }
      }
    }
    if (ok) out.push({ iy: y, ix: x, conf: v });
  }
  out.sort((a, b) => b.conf - a.conf);          // stable: equal conf keeps row-major order
  return out.slice(0, k);
}

/** Cell coordinates -> full-res px: from_cells ((c + 0.5) * stride - 0.5) then scale_px (src net input -> dst). */
export function cellToPx(cx, cy, src = [512, 288], dst = [1280, 720], stride = STRIDE) {
  const u = (cx + 0.5) * stride - 0.5, v = (cy + 0.5) * stride - 0.5;
  return [(u + 0.5) * (dst[0] / src[0]) - 0.5, (v + 0.5) * (dst[1] / src[1]) - 0.5];
}

/** Sticky peak cells (tipnet.CellHysteresis): a peak keeps decoding from the cell the last solve used when that cell
 *  is adjacent and its heat is >= ratio x the peak's; memory holds only the cells the solved ports used. */
export class CellHysteresis {
  constructor({ ratio = 0.8, proj_px = 15, channels = ['tip', 'hinge'] } = {}) {
    this.ratio = ratio; this.proj_px = proj_px; this.channels = channels;
    this.reset();
  }

  reset() {
    this.mem = new Map();      // `${camera}|${channel}` -> [[row, col], ...]
    this.last = new Map();     // cells of this set's candidates, in candidate order
    this.lastUv = new Map();   // their full-res px
  }

  /** One channel's candidates in cells: [{cx, cy (cells, float), conf}] plus the chosen cells. */
  decode(key, heat, offX, offY, h, w, k = 4, thresh = 0.2) {
    const pk = peaks(heat, h, w, k, thresh);
    if (!pk.length) {
      this.mem.set(key, []); this.last.set(key, []); this.lastUv.set(key, []);
      return [];
    }
    const prev = this.mem.get(key) ?? [];
    const lim = f32(this.ratio);
    const cells = [], out = [];
    for (const p of pk) {
      let best = null;
      const need = f32(lim * p.conf);
      for (const [py, px] of prev) {
        if ((py !== p.iy || px !== p.ix) && Math.abs(py - p.iy) <= 1 && Math.abs(px - p.ix) <= 1
            && heat[py * w + px] >= need && (best === null || heat[py * w + px] > heat[best[0] * w + best[1]])) best = [py, px];
      }
      const [cy, cx] = best ?? [p.iy, p.ix];
      cells.push([cy, cx]);
      out.push({ cx: cx + offX[cy * w + cx], cy: cy + offY[cy * w + cx], conf: p.conf });
    }
    this.mem.set(key, cells); this.last.set(key, cells);
    return out;
  }

  /** Keep only the cells the solve used. res: solveFrame output; names / cams indexed like its views. */
  afterSolve(res, names, cams, projectPx, jaw = 0.018) {
    const keep = new Map();
    const push = (key, cell) => { if (!keep.has(key)) keep.set(key, []); keep.get(key).push(cell); };
    for (const pt of Object.values(res)) {
      if (!pt) continue;
      const hingeX = [pt.tip[0] - jaw * pt.u[0], pt.tip[1] - jaw * pt.u[1], pt.tip[2] - jaw * pt.u[2]];
      for (const [ch, used, X] of [['tip', pt.views, pt.tip], ['hinge', pt.hinges, hingeX]]) {
        for (const [k, i] of used) {
          const key = `${names[k]}|${ch}`;
          const cells = this.last.get(key) ?? [];
          if (i < cells.length) push(key, cells[i]);
        }
        if (this.proj_px <= 0) continue;
        const usedK = new Set(used.map(([k]) => k));
        for (let k = 0; k < names.length; k++) {
          const key = `${names[k]}|${ch}`;
          const uvs = this.lastUv.get(key);
          if (usedK.has(k) || !uvs || !uvs.length) continue;
          const q = projectPx(cams[k], X);
          if (!Number.isFinite(q[0]) || !Number.isFinite(q[1])) continue;
          let j = 0, bd = Infinity;
          uvs.forEach((uv, n) => { const dd = Math.hypot(uv[0] - q[0], uv[1] - q[1]); if (dd < bd) { bd = dd; j = n; } });
          if (bd < this.proj_px) push(key, this.last.get(key)[j]);
        }
      }
    }
    this.mem = keep;
  }
}

/** Decode one frame set's net output (after sigmoidHeat) into per-camera candidates in full-res px:
 *  {name: {tip: [{uv: [u, v], conf}], hinge: [...]}}. names[i] is batch slot i; sizes[name] = [w, h] of the full
 *  frame (the px frame of the result). Uses `hyst` (CellHysteresis) when given, as live. */
export function decodeSet(y, names, sizes, { hyst = null, k = 4, thresh = 0.2, h = 72, w = 128, inSize = [512, 288] } = {}) {
  const hw = h * w, out = {};
  names.forEach((n, i) => {
    const d = {};
    CHANNELS.forEach((ch, ci) => {
      const heat = y.subarray((i * 6 + ci) * hw, (i * 6 + ci + 1) * hw);
      const offX = y.subarray((i * 6 + 2 + 2 * ci) * hw, (i * 6 + 3 + 2 * ci) * hw);
      const offY = y.subarray((i * 6 + 3 + 2 * ci) * hw, (i * 6 + 4 + 2 * ci) * hw);
      const key = `${n}|${ch}`;
      let cells;
      if (hyst && hyst.channels.includes(ch)) cells = hyst.decode(key, heat, offX, offY, h, w, k, thresh);
      else cells = peaks(heat, h, w, k, thresh).map((p) => ({ cx: p.ix + offX[p.iy * w + p.ix], cy: p.iy + offY[p.iy * w + p.ix], conf: p.conf }));
      const c = cells.map((q) => ({ uv: cellToPx(q.cx, q.cy, inSize, sizes[n]), conf: q.conf }));
      if (hyst && hyst.channels.includes(ch)) hyst.lastUv.set(key, c.map((q) => q.uv));
      d[ch] = c;
    });
    out[n] = d;
  });
  return out;
}
