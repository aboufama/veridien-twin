// Rigid poses for colliders: {R: row-major 9, t: [3]}, from column-major 4x4 (control/kinematics.js makeT) and
// interpolated across substeps (quaternion slerp + linear translation). Pure.

export function poseFromT(T) {
  if (T && T.R && T.t) return { R: Array.from(T.R), t: Array.from(T.t) };
  if (!T) return { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] };
  return { R: [T[0], T[4], T[8], T[1], T[5], T[9], T[2], T[6], T[10]], t: [T[12], T[13], T[14]] };
}

export function quatFromR(R) {
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = R;
  const tr = m00 + m11 + m22;
  let w, x, y, z;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = s / 4; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; w = (m21 - m12) / s; x = s / 4; y = (m01 + m10) / s; z = (m02 + m20) / s; }
  else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; w = (m02 - m20) / s; x = (m01 + m10) / s; y = s / 4; z = (m12 + m21) / s; }
  else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = s / 4; }
  const n = Math.hypot(w, x, y, z);
  return [w / n, x / n, y / n, z / n];
}
export function rFromQuat([w, x, y, z]) {
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}
export function slerp(a, b, u) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (d < 0) { d = -d; bb = b.map((v) => -v); }
  if (d > 0.9995) { const r = a.map((v, i) => v + u * (bb[i] - v)); const n = Math.hypot(...r); return r.map((v) => v / n); }
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - u) * th) / s, wb = Math.sin(u * th) / s;
  return a.map((v, i) => wa * v + wb * bb[i]);
}
export function lerpPose(p0, p1, u) {
  if (u <= 0) return p0; if (u >= 1) return p1;
  const q = slerp(quatFromR(p0.R), quatFromR(p1.R), u);
  return { R: rFromQuat(q), t: [0, 1, 2].map((i) => p0.t[i] + u * (p1.t[i] - p0.t[i])) };
}
export const toLocal = (P, x, y, z, out) => {
  const dx = x - P.t[0], dy = y - P.t[1], dz = z - P.t[2], R = P.R;
  out[0] = R[0] * dx + R[3] * dy + R[6] * dz; out[1] = R[1] * dx + R[4] * dy + R[7] * dz; out[2] = R[2] * dx + R[5] * dy + R[8] * dz;
  return out;
};
export const toWorld = (P, a, b, c, out) => {
  const R = P.R;
  out[0] = R[0] * a + R[1] * b + R[2] * c + P.t[0]; out[1] = R[3] * a + R[4] * b + R[5] * c + P.t[1]; out[2] = R[6] * a + R[7] * b + R[8] * c + P.t[2];
  return out;
};
export const rotWorld = (P, a, b, c, out) => {
  const R = P.R;
  out[0] = R[0] * a + R[1] * b + R[2] * c; out[1] = R[3] * a + R[4] * b + R[5] * c; out[2] = R[6] * a + R[7] * b + R[8] * c;
  return out;
};
