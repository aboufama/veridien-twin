// WGSL KB4 lens: the float32 twin of gpu/lens.js (cv2.fisheye). Lens = {a: fx fy cx cy, b: k1..k4, c: alpha}.
// The inverse is Newton on theta_d with a fixed iteration count, so every pixel takes the same path.
export const LENS = /* wgsl */ `
fn kb4Distort(L: Lens, th: f32) -> f32 {
  let t2 = th * th;
  return th * (1.0 + t2 * (L.b.x + t2 * (L.b.y + t2 * (L.b.z + t2 * L.b.w))));
}

fn kb4Theta(L: Lens, thd: f32) -> f32 {
  var th = thd;
  for (var i = 0; i < 8; i++) {
    let t2 = th * th;
    let f = th * (1.0 + t2 * (L.b.x + t2 * (L.b.y + t2 * (L.b.z + t2 * L.b.w)))) - thd;
    let fp = 1.0 + t2 * (3.0 * L.b.x + t2 * (5.0 * L.b.y + t2 * (7.0 * L.b.z + t2 * 9.0 * L.b.w)));
    th = th - f / fp;
  }
  return th;
}

// pixel -> unit ray in the camera frame (OpenCV axes)
fn kb4Unproject(L: Lens, uv: vec2f) -> vec3f {
  let yd = (uv.y - L.a.w) / L.a.y;
  let xd = (uv.x - L.a.z) / L.a.x - L.c.x * yd;
  let thd = sqrt(xd * xd + yd * yd);
  if (thd < 1e-20) { return vec3f(0.0, 0.0, 1.0); }
  let th = kb4Theta(L, thd);
  let s = sin(th) / thd;
  return vec3f(xd * s, yd * s, cos(th));
}

// camera-frame point -> pixel
fn kb4Project(L: Lens, X: vec3f) -> vec2f {
  let rho = sqrt(X.x * X.x + X.y * X.y);
  if (rho < 1e-30) { return L.a.zw; }
  let th = atan2(rho, X.z);
  let s = kb4Distort(L, th) / rho;
  let xd = X.x * s;
  let yd = X.y * s;
  return vec2f(L.a.x * (xd + L.c.x * yd) + L.a.z, L.a.y * yd + L.a.w);
}
`;
