// Overlays of truth (ink) against the model's prediction (blue) on the camera feeds and in the 3-D overview.
// Truth: tip ring, hinge square, jaw-tip dots, shaft axis. Hollow grey when the renderer says the point is occluded
// in that camera. Prediction: a cross at the predicted tip (and hinge when given). Drawn on 2-D canvases stacked
// over the WebGPU canvases; every mark is also recorded in `marks` so the page can be checked by numbers.
import { add, sub, scale, projectD } from '../control/kinematics.js';

export const INK = '#111211', GREY = '#8b8c87', PRED = '#2f5bd3';

/** Prepare a 2-D overlay canvas at device resolution for a CSS size; returns the context scaled to CSS px. */
export function fit(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}

function ring(ctx, x, y, r, color, hollow = false, dash = false) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.setLineDash(dash ? [2, 2] : []);
  ctx.strokeStyle = color; ctx.lineWidth = hollow ? 1 : 1.5; ctx.stroke();
  ctx.setLineDash([]);
}
function dotAt(ctx, x, y, r, color) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill(); }
function square(ctx, x, y, s, color) { ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.strokeRect(x - s, y - s, 2 * s, 2 * s); }
function crossAt(ctx, x, y, s, color) {
  ctx.strokeStyle = color; ctx.lineWidth = 1.8;
  ctx.beginPath(); ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s); ctx.moveTo(x - s, y + s); ctx.lineTo(x + s, y - s); ctx.stroke();
}
function label(ctx, x, y, text, color) {
  ctx.font = '11px ui-sans-serif, system-ui, -apple-system, "Helvetica Neue", sans-serif';
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}

/**
 * Draw one camera feed's overlay.
 * frame: {cam (kinematics.cameras record), truth (truthFromState at this camera's mid-exposure), visible: {L: {tip,..}},
 *         prediction?: {instruments: [{id, tip_D, hinge_D}], keypoints_px?}, pivots: {L: [x,y,z]}, stub}
 * Returns marks [{kind, id, x, y}] in image pixels (full sensor resolution), for DOM/numeric checks.
 */
export function drawFeed(canvas, frame) {
  const { ctx, w, h } = fit(canvas);
  const marks = [];
  if (!frame) return marks;
  const [W, H] = frame.cam.size;
  const sx = w / W, sy = h / H;
  const px = frame.truth.px[frame.cam.name] || {};
  for (const ins of frame.truth.instruments) {
    const id = ins.id;
    const P = px[id];
    if (!P) continue;
    const vis = frame.visible?.[id] || {};
    // shaft axis: pivot -> hinge, sampled (straight in D, curved in the fisheye image)
    const piv = frame.pivots[id];
    if (piv) {
      ctx.beginPath(); let pen = false;
      for (let k = 0; k <= 40; k++) {
        const p = add(piv, scale(sub(ins.hinge_D, piv), k / 40));
        const uv = projectD(frame.cam, p);
        if (!uv || uv[0] < -W || uv[0] > 2 * W || uv[1] < -H || uv[1] > 2 * H) { pen = false; continue; }
        if (!pen) { ctx.moveTo(uv[0] * sx, uv[1] * sy); pen = true; } else ctx.lineTo(uv[0] * sx, uv[1] * sy);
      }
      ctx.strokeStyle = 'rgba(17,18,17,0.55)'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([]);
    }
    const inside = (uv) => uv && uv[0] >= 0 && uv[1] >= 0 && uv[0] < W && uv[1] < H;
    if (inside(P.hinge)) { square(ctx, P.hinge[0] * sx, P.hinge[1] * sy, 3.5, vis.hinge === false ? GREY : INK); marks.push({ kind: 'truth_hinge', id, x: P.hinge[0], y: P.hinge[1] }); }
    for (const k of ['jaw_a', 'jaw_b']) if (inside(P[k])) { dotAt(ctx, P[k][0] * sx, P[k][1] * sy, 2.2, vis[k] === false ? GREY : INK); marks.push({ kind: `truth_${k}`, id, x: P[k][0], y: P[k][1] }); }
    if (inside(P.tip)) {
      const occ = vis.tip === false;
      ring(ctx, P.tip[0] * sx, P.tip[1] * sy, 7, occ ? GREY : INK, occ, occ);
      label(ctx, P.tip[0] * sx + 9, P.tip[1] * sy - 8, id + (occ ? ' (hidden)' : ''), occ ? GREY : INK);
      marks.push({ kind: 'truth_tip', id, x: P.tip[0], y: P.tip[1], visible: !occ });
    }
  }
  // prediction
  const pr = frame.prediction;
  if (pr) {
    const kp = pr.keypoints_px?.[frame.cam.name] || {};
    for (const pi of pr.instruments || []) {
      let uv = kp[pi.id]?.tip ?? (pi.tip_D ? projectD(frame.cam, pi.tip_D) : null);
      if (uv && uv[0] >= 0 && uv[1] >= 0 && uv[0] < W && uv[1] < H) {
        crossAt(ctx, uv[0] * sx, uv[1] * sy, 5, PRED);
        marks.push({ kind: 'pred_tip', id: pi.id, x: uv[0], y: uv[1] });
      }
      const hv = kp[pi.id]?.hinge ?? (pi.hinge_D ? projectD(frame.cam, pi.hinge_D) : null);
      if (hv && hv[0] >= 0 && hv[1] >= 0 && hv[0] < W && hv[1] < H) { crossAt(ctx, hv[0] * sx, hv[1] * sy, 3, PRED); marks.push({ kind: 'pred_hinge', id: pi.id, x: hv[0], y: hv[1] }); }
    }
  }
  if (frame.stub) {
    ctx.font = '600 12px ui-sans-serif, system-ui, -apple-system, sans-serif';
    ctx.fillStyle = 'rgba(17,18,17,0.8)';
    ctx.fillText('STUB render', 8, h - 8);
  }
  return marks;
}

/** 3-D overview overlay: truth markers and shaft axes in ink, prediction in blue, tool selection. */
export function drawOverview(canvas, view, { truth, prediction, pivots, selected, scope }) {
  const { ctx, w, h } = fit(canvas);
  const marks = [];
  if (!truth) return marks;
  view.width = w; view.height = h;
  const P = (p) => view.project(p, w, h);
  for (const ins of truth.instruments) {
    const piv = pivots[ins.id];
    const a = P(piv), b = P(ins.hinge_D), t = P(ins.tip_D);
    const sel = selected === ins.id;
    if (a && b) { ctx.strokeStyle = sel ? INK : 'rgba(17,18,17,0.5)'; ctx.lineWidth = sel ? 1.4 : 1; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); ctx.setLineDash([]); }
    if (b) square(ctx, b[0], b[1], 3, INK);
    for (const jt of ins.jaw_tips_D) { const q = P(jt); if (q) dotAt(ctx, q[0], q[1], 2, INK); }
    if (t) {
      ring(ctx, t[0], t[1], sel ? 8 : 6, INK);
      label(ctx, t[0] + 10, t[1] - 8, ins.id, INK);
      marks.push({ kind: 'truth_tip', id: ins.id, x: t[0], y: t[1] });
    }
  }
  if (scope) {
    const l = P(scope.lens_D), v = P(add(scope.lens_D, scale(scope.view_dir_D, 0.03)));
    if (l && v) {
      ctx.strokeStyle = selected === 'scope' ? INK : GREY; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(...l); ctx.lineTo(...v); ctx.stroke();
      label(ctx, l[0] + 8, l[1] - 6, 'scope', selected === 'scope' ? INK : GREY);
    }
  }
  if (prediction) {
    for (const pi of prediction.instruments || []) {
      const q = pi.tip_D ? P(pi.tip_D) : null;
      if (q) { crossAt(ctx, q[0], q[1], 6, PRED); marks.push({ kind: 'pred_tip', id: pi.id, x: q[0], y: q[1] }); }
    }
  }
  return marks;
}
