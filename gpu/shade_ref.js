// JS ports of the WGSL shading helpers that are not in bsdf.js (wgsl/shade.wgsl.js), float64, for the tests only:
// the emitter profile p(theta), the analytic capsule occlusion of indirect light, and the disc-emitter irradiance
// reference used by the direct-light judge.

export function profile(theta, knots, p) {
  const n = knots.length;
  if (theta >= knots[n - 1]) return theta === knots[n - 1] ? p[n - 1] : 0;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (knots[m] <= theta) lo = m; else hi = m; }
  const f = Math.min(1, Math.max(0, (theta - knots[lo]) / Math.max(knots[hi] - knots[lo], 1e-12)));
  return p[lo] + (p[hi] - p[lo]) * f;
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

/**
 * Cosine-weighted fraction of the hemisphere (p, n) hidden by a thin cylinder of radius r on segment a-b:
 * O = (1/pi) int (2r/h) cos(phi) max(0, n . omega(phi)) dphi over the axis angle phi, h = sqrt(dist^2 + r^2).
 */
export function capsuleOcc(p, n, a, b, r) {
  const ab = sub(b, a), L = Math.hypot(...ab);
  if (L < 1e-9 || r <= 0) return 0;
  const e = ab.map((x) => x / L);
  const ap = sub(p, a), s0 = dot(ap, e);
  const perp = ap.map((x, k) => x - s0 * e[k]);
  const pl = Math.hypot(...perp);
  const h = Math.sqrt(pl * pl + r * r);
  let ep = perp.map((x) => -x / Math.max(pl, 1e-12));
  if (pl < 1e-12) { const t = Math.abs(e[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]; ep = [e[1] * t[2] - e[2] * t[1], e[2] * t[0] - e[0] * t[2], e[0] * t[1] - e[1] * t[0]]; const l = Math.hypot(...ep); ep = ep.map((x) => x / l); }
  const np = dot(n, ep), na = dot(n, e);
  let f1 = Math.atan((0 - s0) / h), f2 = Math.atan((L - s0) / h);
  const f0 = Math.atan2(na, np);
  f1 = Math.max(f1, f0 - Math.PI / 2); f2 = Math.min(f2, f0 + Math.PI / 2);
  if (f2 <= f1) return 0;
  const I = np * (0.5 * (f2 - f1) + 0.25 * (Math.sin(2 * f2) - Math.sin(2 * f1))) + na * 0.5 * (Math.sin(f2) ** 2 - Math.sin(f1) ** 2);
  return Math.min(1, Math.max(0, (2 * r / h) * I / Math.PI));
}

/**
 * Irradiance at point x (normal n) from a Lambertian disc (centre c, unit normal m facing x, radius a, radiance L),
 * unoccluded, by Gauss-Legendre quadrature in polar coordinates on the disc (exact to ~1e-10 for smooth cases).
 */
export function discIrradiance(x, n, c, m, a, L, nr = 48, nphi = 96) {
  const [gx, gw] = gaussLegendre(nr);
  const t = Math.abs(m[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  let u = [m[1] * t[2] - m[2] * t[1], m[2] * t[0] - m[0] * t[2], m[0] * t[1] - m[1] * t[0]];
  const ul = Math.hypot(...u); u = u.map((v) => v / ul);
  const w = [m[1] * u[2] - m[2] * u[1], m[2] * u[0] - m[0] * u[2], m[0] * u[1] - m[1] * u[0]];
  let E = 0;
  for (let i = 0; i < nr; i++) {
    const rho = 0.5 * a * (gx[i] + 1), wr = 0.5 * a * gw[i];
    for (let j = 0; j < nphi; j++) {
      const ph = (2 * Math.PI * (j + 0.5)) / nphi;
      const q = [0, 1, 2].map((k) => c[k] + rho * (Math.cos(ph) * u[k] + Math.sin(ph) * w[k]));
      const d = sub(q, x), d2 = dot(d, d), dl = Math.sqrt(d2);
      const cosX = dot(n, d) / dl, cosQ = -dot(m, d) / dl;
      if (cosX > 0 && cosQ > 0) E += L * cosX * cosQ / d2 * rho * wr * (2 * Math.PI / nphi);
    }
  }
  return E;
}

export function gaussLegendre(n) {
  const x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5)), pp = 0;
    for (let it = 0; it < 100; it++) {
      let p1 = 1, p2 = 0;
      for (let j = 1; j <= n; j++) { const p3 = p2; p2 = p1; p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j; }
      pp = n * (z * p1 - p2) / (z * z - 1);
      const z1 = z; z = z1 - p1 / pp;
      if (Math.abs(z - z1) < 1e-15) break;
    }
    x[i] = -z; w[i] = 2 / ((1 - z * z) * pp * pp);
  }
  return [x, w];
}
