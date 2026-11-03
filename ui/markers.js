// Model against truth markers (docs/WEB_TWIN.md §18.2, key M, only while a model is attached): on a tracking camera's
// view, a small ring at each true tip and a small cross at the model's predicted tip, nothing else. Drawn on a 2-D
// canvas over the view, through the same letterbox as the image. Returns the marks (image px) for checks.
import { projectD } from '../control/kinematics.js';
import { imageToFrame } from './views.js';

const TRUTH = 'rgba(255, 255, 255, 0.95)', HALO = 'rgba(17, 18, 17, 0.55)', PRED = '#2f5bd3';

/** Size a 2-D overlay to its CSS box at device resolution; returns {ctx, w, h} in CSS px, cleared. */
export function fitOverlay(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height, dpr };
}

export function clearOverlay(canvas) {
  if (!canvas.width) return;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * frame: {cam (kinematics.cameras record), truth (truthFromState), visible: {L: {tip}}, prediction?: {instruments:
 * [{id, tip_D}], keypoints_px?: {CAM1: {L: {tip: [u, v]}}}}}; rectCss: the image rectangle in CSS px.
 */
export function drawMarkers(canvas, frame, rectCss) {
  const { ctx } = fitOverlay(canvas);
  const marks = [];
  if (!frame || !rectCss) return marks;
  const src = frame.cam.size, name = frame.cam.name;
  const inside = (uv) => uv && uv[0] >= 0 && uv[1] >= 0 && uv[0] < src[0] && uv[1] < src[1];
  const px = frame.truth?.px?.[name] ?? {};
  for (const ins of frame.truth?.instruments ?? []) {
    const uv = px[ins.id]?.tip;
    if (!inside(uv)) continue;
    const [x, y] = imageToFrame(uv, src, rectCss);
    const seen = frame.visible?.[ins.id]?.tip !== false;
    ctx.setLineDash(seen ? [] : [2, 2]);
    ctx.lineWidth = 3; ctx.strokeStyle = HALO; ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
    ctx.lineWidth = 1.5; ctx.strokeStyle = TRUTH; ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([]);
    marks.push({ kind: 'truth_tip', id: ins.id, x: uv[0], y: uv[1], visible: seen });
  }
  const pr = frame.prediction;
  for (const pi of pr?.instruments ?? []) {
    const uv = pr.keypoints_px?.[name]?.[pi.id]?.tip ?? (pi.tip_D ? projectD(frame.cam, pi.tip_D) : null);
    if (!inside(uv)) continue;
    const [x, y] = imageToFrame(uv, src, rectCss);
    for (const [w, c] of [[3.5, 'rgba(255,255,255,0.85)'], [1.6, PRED]]) {
      ctx.lineWidth = w; ctx.strokeStyle = c;
      ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x - 5, y + 5); ctx.lineTo(x + 5, y - 5); ctx.stroke();
    }
    marks.push({ kind: 'pred_tip', id: pi.id, x: uv[0], y: uv[1] });
  }
  return marks;
}
