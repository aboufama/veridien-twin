// The pure parts of the god view's simulated camera views (godview_pt3/camviews.js, docs/WEB_TWIN.md §19, §21): each
// camera's pose and lens from scene.json (CAM1-CAM3: KB4 fisheye with T_D_C; SCOPE: the 75 deg pinhole on the scope
// body, as lite_pt2/cams.js scopeCameraT, §18.1), the symmetric pinhole frustum that covers a fisheye image (the raster
// draws that pinhole, the display warps it to KB4 per pixel), and the per-view auto-exposure. Node tests:
// web/twin/tests/godview_camview.test.mjs.

export const SCOPE_HFOV_DEG = 75;           // §18.1 default (physics.yaml scope.camera.hfov_deg is absent)
export const SCOPE_CIRCLE_FRAC = 0.95;      // §18.1: the scope image circle's diameter / image width
export const SCOPE_SIZE = Object.freeze([1280, 720]);
export const THETA_MAX_CAP = 1.4;           // rad: no pinhole can cover 90 deg; every dome camera folds well before (<= 1.34)
export const NEAR = Object.freeze({ kb4: 0.008, pinhole: 0.001 });   // m: past the camera's own module / the scope's tip face

const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const mulRv = (R, v) => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];

/** KB4 distortion: theta_d = theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8). */
export function kb4Distort(theta, D) {
  const t2 = theta * theta;
  return theta * (1 + t2 * (D[0] + t2 * (D[1] + t2 * (D[2] + t2 * D[3]))));
}

/** Inverse of kb4Distort by Newton (6 steps from theta_d; the dome lenses are near-equidistant). */
export function kb4Undistort(thetaD, D) {
  let th = thetaD;
  for (let i = 0; i < 8; i++) {
    const t2 = th * th;
    const f = th * (1 + t2 * (D[0] + t2 * (D[1] + t2 * (D[2] + t2 * D[3])))) - thetaD;
    const df = 1 + t2 * (3 * D[0] + t2 * (5 * D[1] + t2 * (7 * D[2] + t2 * 9 * D[3])));
    th -= f / (Math.abs(df) > 1e-9 ? df : 1e-9);
  }
  return th;
}

/** Project a camera-frame point (OpenCV: x right, y down, z forward) to pixels. */
export function projectCam(cam, p) {
  const [fx, fy, cx, cy] = cam.K;
  if (cam.model === 'pinhole') return p[2] > 0 ? [fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy] : null;
  const r = Math.hypot(p[0], p[1]);
  const th = Math.atan2(r, p[2]);
  if (th > cam.thetaMax) return null;
  const td = kb4Distort(th, cam.D);
  const s = r > 1e-12 ? td / r : 1 / Math.max(p[2], 1e-12);
  return [fx * p[0] * s + cx, fy * p[1] * s + cy];
}

/** Unit ray (camera frame) through pixel (u, v), or null past the lens's fold. */
export function unprojectCam(cam, u, v) {
  const [fx, fy, cx, cy] = cam.K;
  const mx = (u - cx) / fx, my = (v - cy) / fy;
  if (cam.model === 'pinhole') return norm3([mx, my, 1]);
  const td = Math.hypot(mx, my);
  if (td < 1e-12) return [0, 0, 1];
  const th = kb4Undistort(td, cam.D);
  if (!(th >= 0) || th > cam.thetaMax) return null;
  const s = Math.sin(th) / td;
  return [mx * s, my * s, Math.cos(th)];
}

/** T_D_C (row-major 4x4) of the scope camera from the scope body's rotation R (rows) and position p (lite_pt2/cams.js). */
export function scopeCameraT(scene, R, p) {
  const sc = scene.scope || {};
  const oK = sc.objective_centre_K || [0, 0, 0];
  const vK = sc.view_dir_K || [0, Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)];
  const c = mulRv(R, oK).map((x, i) => x + p[i]);
  const z = norm3(mulRv(R, vK));
  const yK = mulRv(R, [0, 1, 0]);
  const up = norm3(yK.map((x, i) => x - dot3(yK, z) * z[i]));   // image up: the side the oblique view tilts toward
  const y = up.map((x) => -x);                                   // OpenCV y is down
  const x = cross3(y, z);
  return [[x[0], y[0], z[0], c[0]], [x[1], y[1], z[1], c[1]], [x[2], y[2], z[2], c[2]], [0, 0, 0, 1]];
}

/** The pinhole SCOPE lens (§18.1): f from the 75 deg HFOV, principal point at the centre. */
export function scopeK(size = SCOPE_SIZE, hfovDeg = SCOPE_HFOV_DEG) {
  const [w, h] = size;
  const f = (w / 2) / Math.tan((hfovDeg * Math.PI) / 360);
  return [f, f, w / 2 - 0.5, h / 2 - 0.5];
}

/**
 * One camera's model: {name, model: 'kb4' | 'pinhole', K: [fx, fy, cx, cy], D, size, thetaMax, T (row-major T_D_C),
 * centre, axes: {x, y, z} (camera axes in D), near, maskR (px; SCOPE's image circle, else 0)}.
 * scopePose: {R (rows), p} of the scope body (the god view's instance), needed for SCOPE only.
 */
export function cameraModel(scene, name, scopePose = null) {
  let T, model, K, D = [0, 0, 0, 0], size, thetaMax, maskR = 0;
  if (name === 'SCOPE') {
    if (!scopePose || !scene.scope) return null;
    T = scopeCameraT(scene, scopePose.R, scopePose.p);
    model = 'pinhole'; size = [...SCOPE_SIZE]; K = scopeK(size); thetaMax = Math.PI / 2;
    maskR = SCOPE_CIRCLE_FRAC * size[0] / 2;
  } else {
    const c = (scene.cameras || []).find((x) => x.name === name);
    if (!c?.T_D_C || !c.lens) return null;
    T = c.T_D_C;
    const L = c.lens, Km = L.K;
    size = [...(c.size || L.size)];
    const ls = L.size || size, sx = size[0] / ls[0], sy = size[1] / ls[1];
    K = [Km[0][0] * sx, Km[1][1] * sy, (Km[0][2] + 0.5) * sx - 0.5, (Km[1][2] + 0.5) * sy - 0.5];
    model = L.model === 'kb4' ? 'kb4' : 'pinhole';
    D = model === 'kb4' ? [...L.D] : D;
    thetaMax = Math.min(THETA_MAX_CAP, Number(c.theta_fold_rad) || THETA_MAX_CAP);
  }
  const axes = { x: [T[0][0], T[1][0], T[2][0]], y: [T[0][1], T[1][1], T[2][1]], z: [T[0][2], T[1][2], T[2][2]] };
  return { name, model, K, D, size, thetaMax, T, centre: [T[0][3], T[1][3], T[2][3]], axes, near: NEAR[model], maskR };
}

/**
 * The symmetric pinhole (about the optical axis) that covers every pixel of `cam`'s image: half extents tx = max |X/Z|,
 * ty = max |Y/Z| over the image border (directions past thetaMax clamp to it), plus `margin`.
 */
export function coveringPinhole(cam, { perEdge = 48, margin = 1.02 } = {}) {
  const [w, h] = cam.size;
  let tx = 0, ty = 0;
  const visit = (u, v) => {
    let d = unprojectCam(cam, u, v);
    if (!d) {                                                  // past the fold: clamp to thetaMax in that azimuth
      const [fx, fy, cx, cy] = cam.K;
      const phi = Math.atan2((v - cy) / fy, (u - cx) / fx), th = cam.thetaMax;
      d = [Math.sin(th) * Math.cos(phi), Math.sin(th) * Math.sin(phi), Math.cos(th)];
    }
    const z = Math.max(d[2], Math.cos(THETA_MAX_CAP));
    tx = Math.max(tx, Math.abs(d[0] / z)); ty = Math.max(ty, Math.abs(d[1] / z));
  };
  for (let i = 0; i <= perEdge; i++) {
    const a = i / perEdge;
    visit(-0.5 + a * w, -0.5); visit(-0.5 + a * w, h - 0.5); visit(-0.5, -0.5 + a * h); visit(w - 0.5, -0.5 + a * h);
  }
  return { tx: tx * margin, ty: ty * margin };
}

/**
 * Pixel size of the pinhole render for an output whose image is drawn `outW` px wide: the centre's sampling density
 * matches the output's (2 tan-half-extent x f x scale), capped at maxPx pixels (aspect kept), at least 64 x 36.
 */
export function pinholeSize(cam, pin, outW, maxPx = 600000) {
  const s = outW / cam.size[0];
  let w = 2 * pin.tx * cam.K[0] * s, h = 2 * pin.ty * cam.K[1] * s;
  const k = Math.min(1, Math.sqrt(maxPx / Math.max(1, w * h)));
  w = Math.max(64, Math.round(w * k)); h = Math.max(36, Math.round(h * k));
  return [w, h];
}

/**
 * Auto-exposure of one view from luminance samples of its linear image (zeros = no geometry, ignored): the geometric
 * mean goes to `key` (a white dome lit by the scope reads light grey, as the camera's own AE makes it), and the
 * `pHi` quantile at most to `hiCap` (a close, bright surface does not clip). Returns the linear multiplier, or null.
 */
export function exposureFrom(lums, { key = 0.32, hiCap = 1.8, pHi = 0.98, floor = 1e-7 } = {}) {
  const v = [];
  for (const x of lums) if (x > floor && Number.isFinite(x)) v.push(x);
  if (v.length < 8) return null;
  let sl = 0;
  for (const x of v) sl += Math.log(x);
  const gm = Math.exp(sl / v.length);
  v.sort((a, b) => a - b);
  const hi = v[Math.min(v.length - 1, Math.floor(pHi * v.length))];
  return Math.min(key / gm, hiCap / hi);
}

/** One step of the exposure's settling, in stops: the first measurement jumps, then `rate` of the gap per frame. */
export function settleExposure(prev, target, rate = 0.5) {
  if (!(target > 0)) return prev ?? null;
  if (!(prev > 0)) return target;
  return Math.exp(Math.log(prev) + rate * (Math.log(target) - Math.log(prev)));
}

/** Static vertices inside the dome (its bounding ellipsoid, a little enlarged, and the floor under it): the only ones
 *  a dome camera sees, so the scope-only bake covers just these. verts: 8 floats per vertex (position, normal). */
export function interiorVertices(verts, nStatic, ell, { grow = 1.08, below = 0.03 } = {}) {
  const out = [];
  if (!ell) { for (let i = 0; i < nStatic; i++) out.push(i); return Uint32Array.from(out); }
  const [c, h] = ell;
  for (let i = 0; i < nStatic; i++) {
    const x = verts[8 * i], y = verts[8 * i + 1], z = verts[8 * i + 2];
    if (z < c[2] - below) continue;
    const qx = (x - c[0]) / h[0], qy = (y - c[1]) / h[1], qz = Math.max(0, z - c[2]) / h[2];
    if (qx * qx + qy * qy + qz * qz <= grow * grow) out.push(i);
  }
  return Uint32Array.from(out);
}
