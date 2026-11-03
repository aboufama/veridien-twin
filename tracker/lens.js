// KB4 (Kannala-Brandt, cv2.fisheye) camera model and the rig camera models of the in-browser tracker: the JS twin of
// src/veridien/multicam/lens.py (Lens.project / Lens.normalized, i.e. cv2.fisheye.projectPoints / undistortPoints with
// criteria COUNT|EPS 50, 1e-10) and tipnet.py (cam_models, project_px, lens_jacobian). Same operation order as OpenCV,
// so projections agree to ~1e-12 px. Frames: rig R (= CAM1's frame), camera C (OpenCV: x right, y down, z forward),
// pixels in the full-resolution fisheye image with pixel centres at integers. Metres.

import { mulMv, mulMtv } from './math.js';

/** A lens from rig.json's camera.lens ({model: "kb4", K, D, ...}) plus its image size [w, h]. */
export function makeLens(l, size) {
  if (l.model !== 'kb4') throw new Error(`lens model ${l.model}: only kb4 is supported`);
  const K = l.K;
  return { model: 'kb4', fx: K[0][0], fy: K[1][1], cx: K[0][2], cy: K[1][2], alpha: K[0][1] / K[0][0],
    k: l.D.slice(0, 4), size: size.slice(0, 2), K };
}

/** cv2.fisheye.projectPoints of one camera-frame point -> [u, v]. The sign of z is lost (callers mask z). */
export function project(lens, X) {
  let z = X[2];
  if (Math.abs(z) < 2.2250738585072014e-308) z = 1;
  const a = X[0] / z, b = X[1] / z;
  const r2 = a * a + b * b, r = Math.sqrt(r2);
  const th = Math.atan(r);
  const th2 = th * th, th3 = th2 * th, th4 = th2 * th2, th5 = th4 * th;
  const th6 = th3 * th3, th7 = th6 * th, th8 = th4 * th4, th9 = th8 * th;
  const k = lens.k;
  const thd = th + k[0] * th3 + k[1] * th5 + k[2] * th7 + k[3] * th9;
  const invR = r > 1e-8 ? 1.0 / r : 1;
  const cdist = r > 1e-8 ? thd * invR : 1;
  const x0 = a * cdist, x1 = b * cdist;
  return [(x0 + lens.alpha * x1) * lens.fx + lens.cx, x1 * lens.fy + lens.cy];
}

/** cv2.fisheye.undistortPoints (no R, no P) of one pixel -> ideal pinhole [x/z, y/z]; [-1e6, -1e6] when the
 *  iteration fails or flips (as OpenCV): the pixel gates then reject it. */
export function normalized(lens, uv) {
  const px = (uv[0] - lens.cx) / lens.fx, py = (uv[1] - lens.cy) / lens.fy;
  let thd = Math.sqrt(px * px + py * py);
  thd = Math.min(Math.max(-Math.PI / 2, thd), Math.PI / 2);
  let converged = false, th = thd, s = 0;
  const k = lens.k;
  if (thd > 1e-8) {
    for (let j = 0; j < 50; j++) {
      const t2 = th * th, t4 = t2 * t2, t6 = t4 * t2, t8 = t6 * t2;
      const k0 = k[0] * t2, k1 = k[1] * t4, k2 = k[2] * t6, k3 = k[3] * t8;
      const fix = (th * (1 + k0 + k1 + k2 + k3) - thd) / (1 + 3 * k0 + 5 * k1 + 7 * k2 + 9 * k3);
      th -= fix;
      if (Math.abs(fix) < 1e-10) { converged = true; break; }
    }
    s = Math.tan(th) / thd;
  } else {
    converged = true;
    s = 1;
  }
  const flipped = (thd < 0 && th > 0) || (thd > 0 && th < 0);
  if (converged && !flipped) return [px * s, py * s];
  return [-1e6, -1e6];
}

/** d pixel / d normalised coordinate at xn: [[du/dx, du/dy], [dv/dx, dv/dy]], central differences (eps 1e-5). */
export function jacobian(lens, xn, eps = 1e-5) {
  const a = project(lens, [xn[0] + eps, xn[1], 1]), b = project(lens, [xn[0] - eps, xn[1], 1]);
  const c = project(lens, [xn[0], xn[1] + eps, 1]), d = project(lens, [xn[0], xn[1] - eps, 1]);
  return [[(a[0] - b[0]) / (2 * eps), (c[0] - d[0]) / (2 * eps)], [(a[1] - b[1]) / (2 * eps), (c[1] - d[1]) / (2 * eps)]];
}

/** Rig camera models from rig.json cameras[]: {name, lens, R (rig -> camera), t, C (centre in the rig frame)}. */
export function camModels(rig) {
  return rig.cameras.map((c) => {
    const R = c.T_cam_R.slice(0, 3).map((r) => r.slice(0, 3));
    const t = c.T_cam_R.slice(0, 3).map((r) => r[3]);
    const C = mulMtv(R, t).map((x) => -x);
    return { name: c.name, index: c.index, lens: makeLens(c.lens, c.size), R, t, C, size: c.size.slice(0, 2) };
  });
}

/** Rig point -> camera frame. */
export const toCam = (cam, X) => {
  const p = mulMv(cam.R, X);
  return [p[0] + cam.t[0], p[1] + cam.t[1], p[2] + cam.t[2]];
};

/** tipnet.project_px: rig point -> fisheye px, [NaN, NaN] when z <= 1e-6 (behind the camera). */
export function projectPx(cam, X) {
  const Xc = toCam(cam, X);
  if (!(Xc[2] > 1e-6)) return [NaN, NaN];
  return project(cam.lens, Xc);
}
