// Kannala-Brandt (KB4) lens, exactly as cv2.fisheye: projection of camera-frame points to pixels and the
// inverse, pixels to unit rays, by Newton on theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8).
// Pure ES module, float64. The WGSL twin of these functions is wgsl/lens.wgsl.js (judged against this file on
// the GPU by tests/gpu_test.html). Camera frame C is OpenCV's: x right, y down, z forward; pixel centres at
// integer coordinates, so pixel (i, j) covers [i - 0.5, i + 0.5] x [j - 0.5, j + 0.5].

/** Normalise a scene lens block {model, K, D, size} into flat numbers. */
export function lensParams(lens) {
  if (!lens || (lens.model && lens.model !== 'kb4')) throw new Error(`lens model ${lens?.model} is not kb4`);
  const K = lens.K;
  const fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];
  const alpha = K[0][1] / fx;                     // cv2.fisheye skew: u = fx (x + alpha y) + cx
  const D = (lens.D.flat ? lens.D.flat() : lens.D).map(Number);
  if (D.length !== 4) throw new Error('kb4 needs 4 coefficients');
  return { fx, fy, cx, cy, alpha, k: D, size: lens.size ? [...lens.size] : null };
}

/** theta -> theta_d. */
export function distort(L, th) {
  const t2 = th * th;
  return th * (1 + t2 * (L.k[0] + t2 * (L.k[1] + t2 * (L.k[2] + t2 * L.k[3]))));
}

/** d theta_d / d theta. */
export function distortDeriv(L, th) {
  const t2 = th * th;
  return 1 + t2 * (3 * L.k[0] + t2 * (5 * L.k[1] + t2 * (7 * L.k[2] + t2 * 9 * L.k[3])));
}

/** theta_d -> theta by Newton (cv2.fisheye.undistortPoints iterates the same map). */
export function undistortTheta(L, thd, iters = 20) {
  let th = thd;
  for (let i = 0; i < iters; i++) {
    const f = distort(L, th) - thd;
    const step = f / distortDeriv(L, th);
    th -= step;
    if (Math.abs(step) < 1e-15) break;
  }
  return th;
}

/** Camera-frame point [x, y, z] -> pixel [u, v]. theta = atan2(rho, z), equal to cv2's atan(rho / z) for z > 0. */
export function project(L, X) {
  const x = X[0], y = X[1], z = X[2];
  const rho = Math.hypot(x, y);
  if (rho < 1e-300) return [L.cx, L.cy];
  const th = Math.atan2(rho, z);
  const s = distort(L, th) / rho;                 // cv2: (theta_d / r) * (x/z) with r = rho / z -> theta_d * x / rho
  const xd = x * s, yd = y * s;
  return [L.fx * (xd + L.alpha * yd) + L.cx, L.fy * yd + L.cy];
}

/** Pixel [u, v] -> unit ray [x, y, z] in the camera frame. */
export function unproject(L, uv) {
  const yd = (uv[1] - L.cy) / L.fy;
  const xd = (uv[0] - L.cx) / L.fx - L.alpha * yd;
  const thd = Math.hypot(xd, yd);
  if (thd < 1e-300) return [0, 0, 1];
  const th = undistortTheta(L, thd);
  const s = Math.sin(th) / thd;
  return [xd * s, yd * s, Math.cos(th)];
}

/** Largest theta (rad) at which the KB4 map is still increasing, i.e. where the inverse is unique. */
export function maxMonotoneTheta(L, limit = Math.PI) {
  const n = 4096;
  for (let i = 1; i <= n; i++) {
    const th = (i / n) * limit;
    if (distortDeriv(L, th) <= 0) return ((i - 1) / n) * limit;
  }
  return limit;
}

/** Field angle (rad) of every image corner and edge midpoint; the image never needs rays past the largest. */
export function maxImageTheta(L, size = L.size) {
  const [w, h] = size;
  let m = 0;
  for (const u of [-0.5, (w - 1) / 2, w - 0.5]) for (const v of [-0.5, (h - 1) / 2, h - 0.5]) {
    const r = unproject(L, [u, v]);
    m = Math.max(m, Math.acos(Math.max(-1, Math.min(1, r[2]))));
  }
  return m;
}
