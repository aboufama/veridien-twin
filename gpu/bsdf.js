// The twin's BSDF, word for word as docs/WEB_TWIN.md §8 states it, in float64 JS. This is the reference for the
// WGSL copy in wgsl/bsdf.wgsl.js (the GPU test compares the two) and for the Mitsuba plugin (mitsuba_ref.py):
//
//   f(l, v) = rho_d/pi * (1 - F(n.l)) (1 - F(n.v)) / (1 - Fbar)                    (dielectrics only)
//           + D_GGX(h; alpha) * G2(l, v; alpha) * F(v.h) / (4 |n.l| |n.v|)
//   alpha = roughness^2. G2 = 1 / (1 + Lambda(l) + Lambda(v)), the height-correlated Smith term for GGX,
//   Lambda(w) = (-1 + sqrt(1 + alpha^2 tan^2 theta_w)) / 2.
//   Dielectric F: the exact unpolarised Fresnel reflectance from air into IOR n.
//   Fbar = 2 * integral_0^1 F(mu) mu dmu, the cosine-weighted hemispherical average of F (one constant per material).
//   Conductors (metallic 1): no diffuse term, F = Schlick with RGB F0: F0 + (1 - F0)(1 - v.h)^5.
//   Reflection only: f = 0 unless n.l > 0 and n.v > 0 (n = shading normal).
//
// coated_conductor (metallic 2; WEB_TWIN_PHYSICS §3.7, the trocar cannula sheath), word for word:
//   f = D(h; a_c) G2(l, v; a_c) F_c(v.h) / (4 |n.l| |n.v|)
//     + (1 - F_c(n.l)) (1 - F_c(n.v)) tau^(1/|n.l| + 1/|n.v|) D(h; a_m) G2(l, v; a_m) F_s(v.h; f0) / (4 |n.l| |n.v|)
//   a_c = coat_roughness^2, a_m = roughness^2, F_c exact Fresnel for coat_ior, F_s Schlick RGB, tau per normal pass.
//   Sampling: coat lobe with P = F_c(n.v) / (F_c(n.v) + (1 - F_c(n.v)) mean(f0)), else the metal lobe (VNDF each);
//   pdf = P pdf_c + (1 - P) pdf_m.
//
// Also: importance sampling (visible-normal GGX, Heitz 2018, mixed with cosine sampling for the diffuse term),
// its pdf, directional-albedo tables used by the radiosity, and the FDM layer-line normal perturbation.

export const INV_PI = 1 / Math.PI;

export function fresnelDielectric(cosi, eta) {
  const c = Math.min(1, Math.max(0, cosi));
  const s2 = (1 - c * c) / (eta * eta);
  if (s2 >= 1) return 1;
  const ct = Math.sqrt(1 - s2);
  const rs = (c - eta * ct) / (c + eta * ct), rp = (eta * c - ct) / (eta * c + ct);
  return 0.5 * (rs * rs + rp * rp);
}

/** Fbar = 2 int_0^1 F(mu) mu dmu (composite Simpson, 4096 intervals; exact to ~1e-12). */
export function fresnelAverage(eta, n = 4096) {
  let s = 0;
  for (let i = 0; i <= n; i++) {
    const mu = i / n, w = i === 0 || i === n ? 1 : i % 2 ? 4 : 2;
    s += w * fresnelDielectric(mu, eta) * mu;
  }
  return 2 * s / (3 * n);
}

export const schlick = (f0, c) => { const m = Math.pow(1 - Math.min(1, Math.max(0, c)), 5); return f0 + (1 - f0) * m; };

export function ggxD(nh, a) {
  if (nh <= 0) return 0;
  const a2 = a * a, d = nh * nh * (a2 - 1) + 1;
  return a2 / (Math.PI * d * d);
}

export function ggxLambda(c, a) {
  const c2 = Math.max(c * c, 1e-12);
  const t2 = (1 - c2) / c2;
  return 0.5 * (-1 + Math.sqrt(1 + a * a * t2));
}

export const ggxG1 = (c, a) => 1 / (1 + ggxLambda(c, a));
export const ggxG2 = (nl, nv, a) => 1 / (1 + ggxLambda(nl, a) + ggxLambda(nv, a));

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const normalize = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

/**
 * Material as the tracer sees it, from a physics.yaml entry (values unwrapped):
 * {metallic: 0|1, rho: [r,g,b] (rho_d, dielectrics), f0: [r,g,b] (conductors), roughness, ior}
 */
export function prepareMaterial(m) {
  if (m.coated) {
    const roughness = Math.max(m.roughness ?? 0.3, 0.02), rc = Math.max(m.coatRoughness ?? 0.05, 0.02);
    return {
      metallic: 2, coated: true, roughness, alpha: roughness * roughness, f0: [...(m.f0 ?? [0.56, 0.57, 0.58])], rho: [0, 0, 0],
      ior: m.coatIor ?? 1.5, coatRoughness: rc, alphaC: rc * rc, tau: [...(m.tau ?? [1, 1, 1])], fbar: 0, fbarComputed: 0,
    };
  }
  const metallic = m.metallic ? 1 : 0;
  const roughness = Math.max(m.roughness ?? 0.5, 0.02);      // alpha >= 4e-4: keeps D finite in float32
  const out = {
    metallic, roughness, alpha: roughness * roughness,
    rho: metallic ? [0, 0, 0] : [...(m.rho ?? m.rho_d ?? [0.5, 0.5, 0.5])],
    f0: metallic ? [...(m.f0 ?? [0.56, 0.57, 0.58])] : null,
    ior: m.ior ?? 1.5,
  };
  // Fbar: physics.yaml materials.<name>.fresnel_avg when given (shared with the Mitsuba plugin), else computed
  out.fbarComputed = metallic ? 0 : fresnelAverage(out.ior);
  out.fbar = metallic ? 0 : (Number.isFinite(m.fbar) ? m.fbar : out.fbarComputed);
  return out;
}

/** Fresnel term as RGB for the material at cosine c. */
export function fresnelRGB(M, c) {
  if (M.metallic) return [schlick(M.f0[0], c), schlick(M.f0[1], c), schlick(M.f0[2], c)];
  const F = fresnelDielectric(c, M.ior);
  return [F, F, F];
}

/** f(l, v) (RGB, no cosine). n, l, v unit, l and v pointing away from the surface. */
export function evalBSDF(M, n, l, v) {
  const nl = dot(n, l), nv = dot(n, v);
  if (nl <= 0 || nv <= 0) return [0, 0, 0];
  const h = normalize([l[0] + v[0], l[1] + v[1], l[2] + v[2]]);
  const nh = dot(n, h), vh = Math.max(0, dot(v, h));
  if (M.coated) {
    const coat = ggxD(nh, M.alphaC) * ggxG2(nl, nv, M.alphaC) * fresnelDielectric(vh, M.ior) / (4 * nl * nv);
    const t = (1 - fresnelDielectric(nl, M.ior)) * (1 - fresnelDielectric(nv, M.ior));
    const metal = ggxD(nh, M.alpha) * ggxG2(nl, nv, M.alpha) / (4 * nl * nv);
    return [0, 1, 2].map((c) => coat + t * Math.pow(M.tau[c], 1 / nl + 1 / nv) * metal * schlick(M.f0[c], vh));
  }
  const spec = ggxD(nh, M.alpha) * ggxG2(nl, nv, M.alpha) / (4 * nl * nv);
  const F = fresnelRGB(M, vh);
  const out = [spec * F[0], spec * F[1], spec * F[2]];
  if (!M.metallic) {
    const kd = INV_PI * (1 - fresnelDielectric(nl, M.ior)) * (1 - fresnelDielectric(nv, M.ior)) / (1 - M.fbar);
    for (let c = 0; c < 3; c++) out[c] += M.rho[c] * kd;
  }
  return out;
}

// ------------------------------------------------------------------ sampling (local frame: n = +z)

/** Orthonormal basis around n (Duff et al. 2017), returns [t, b]. */
export function basis(n) {
  const s = n[2] >= 0 ? 1 : -1;
  const a = -1 / (s + n[2]), b = n[0] * n[1] * a;
  return [[1 + s * n[0] * n[0] * a, s * b, -s * n[0]], [b, s + n[1] * n[1] * a, -n[1]]];
}
export const toWorld = (t, b, n, x) => [t[0] * x[0] + b[0] * x[1] + n[0] * x[2], t[1] * x[0] + b[1] * x[1] + n[1] * x[2], t[2] * x[0] + b[2] * x[1] + n[2] * x[2]];
export const toLocal = (t, b, n, w) => [dot(t, w), dot(b, w), dot(n, w)];

/** Heitz 2018: sample a visible GGX normal for local view ve (ve.z > 0). */
export function sampleVNDF(ve, a, u1, u2) {
  const vh = normalize([a * ve[0], a * ve[1], ve[2]]);
  const lensq = vh[0] * vh[0] + vh[1] * vh[1];
  const T1 = lensq > 0 ? [-vh[1] / Math.sqrt(lensq), vh[0] / Math.sqrt(lensq), 0] : [1, 0, 0];
  const T2 = [vh[1] * T1[2] - vh[2] * T1[1], vh[2] * T1[0] - vh[0] * T1[2], vh[0] * T1[1] - vh[1] * T1[0]];
  const r = Math.sqrt(u1), phi = 2 * Math.PI * u2;
  const t1 = r * Math.cos(phi);
  let t2 = r * Math.sin(phi);
  const s = 0.5 * (1 + vh[2]);
  t2 = (1 - s) * Math.sqrt(Math.max(0, 1 - t1 * t1)) + s * t2;
  const t3 = Math.sqrt(Math.max(0, 1 - t1 * t1 - t2 * t2));
  const nh = [t1 * T1[0] + t2 * T2[0] + t3 * vh[0], t1 * T1[1] + t2 * T2[1] + t3 * vh[1], t1 * T1[2] + t2 * T2[2] + t3 * vh[2]];
  return normalize([a * nh[0], a * nh[1], Math.max(1e-7, nh[2])]);
}

/** Probability of choosing the specular lobe (coated: the coat lobe) (shared with WGSL). */
export function specProb(M, nv) {
  if (M.coated) {
    const Fc = fresnelDielectric(nv, M.ior), mf = (M.f0[0] + M.f0[1] + M.f0[2]) / 3;
    return Fc / (Fc + (1 - Fc) * mf);
  }
  if (M.metallic) return 1;
  const F = fresnelDielectric(nv, M.ior);
  const d = (M.rho[0] * 0.2126 + M.rho[1] * 0.7152 + M.rho[2] * 0.0722) * (1 - F);
  const s = Math.max(F, 0.04);
  return Math.min(0.95, Math.max(0.05, s / (s + d)));
}

/** pdf (solid angle) of sampleBSDF for direction l. */
export function pdfBSDF(M, n, v, l) {
  const nl = dot(n, l), nv = dot(n, v);
  if (nl <= 0 || nv <= 0) return 0;
  const h = normalize([l[0] + v[0], l[1] + v[1], l[2] + v[2]]);
  const nh = dot(n, h), vh = dot(v, h);
  const ps = specProb(M, nv);
  if (M.coated) {
    if (!(vh > 0)) return 0;
    const pc = ggxG1(nv, M.alphaC) * ggxD(nh, M.alphaC) / (4 * nv), pm = ggxG1(nv, M.alpha) * ggxD(nh, M.alpha) / (4 * nv);
    return ps * pc + (1 - ps) * pm;
  }
  const pdfSpec = vh > 0 ? ggxG1(nv, M.alpha) * Math.max(0, vh) * ggxD(nh, M.alpha) / nv / (4 * vh) : 0;
  return ps * pdfSpec + (1 - ps) * nl * INV_PI;
}

/** Sample l given v. Returns {l, pdf} (l in world space) or null. u = [u0 (lobe), u1, u2]. */
export function sampleBSDF(M, n, v, u) {
  const nv = dot(n, v);
  if (nv <= 0) return null;
  const [t, b] = basis(n);
  const ps = specProb(M, nv);
  let l;
  if (M.coated) {
    const hl = sampleVNDF(toLocal(t, b, n, v), u[0] < ps ? M.alphaC : M.alpha, u[1], u[2]);
    const h = toWorld(t, b, n, hl);
    const vh = dot(v, h);
    l = [2 * vh * h[0] - v[0], 2 * vh * h[1] - v[1], 2 * vh * h[2] - v[2]];
  } else if (u[0] < ps) {
    const hl = sampleVNDF(toLocal(t, b, n, v), M.alpha, u[1], u[2]);
    const h = toWorld(t, b, n, hl);
    const vh = dot(v, h);
    l = [2 * vh * h[0] - v[0], 2 * vh * h[1] - v[1], 2 * vh * h[2] - v[2]];
  } else {
    const r = Math.sqrt(u[1]), phi = 2 * Math.PI * u[2];
    l = toWorld(t, b, n, [r * Math.cos(phi), r * Math.sin(phi), Math.sqrt(Math.max(0, 1 - u[1]))]);
  }
  if (dot(n, l) <= 0) return null;
  return { l, pdf: pdfBSDF(M, n, v, l) };
}

// ------------------------------------------------------------------ albedo tables (radiosity)

/** Radical inverse base 2 and a (0,2) pair, deterministic. */
export const radinv2 = (i) => { let b = 0, f = 0.5; while (i) { if (i & 1) b += f; i >>>= 1; f *= 0.5; } return b; };

/**
 * Directional albedo A(mu) = int f(l, v) (n.l) dl, split into the specular part (RGB) and the diffuse part (RGB),
 * at `bins` view cosines mu_k = (k + 0.5)/bins. Deterministic quadrature: specular by VNDF importance sampling
 * with a Hammersley set, diffuse analytically: A_d(mu) = rho_d (1 - F(mu)).
 */
export function albedoTable(M, bins = 32, samples = 2048) {
  const spec = new Float64Array(bins * 3), diff = new Float64Array(bins * 3);
  const n = [0, 0, 1];
  for (let k = 0; k < bins; k++) {
    const mu = (k + 0.5) / bins, v = [Math.sqrt(1 - mu * mu), 0, mu];
    const acc = [0, 0, 0];
    if (M.coated) {
      // both lobes are specular: estimate with the material's own sampler (stratified lobe choice)
      for (let i = 0; i < samples; i++) {
        const s = sampleBSDF(M, n, v, [(i + 0.5) / samples, radinv2(i), radinv2(i ^ 0x5555) % 1]);
        if (!s || !(s.pdf > 0)) continue;
        const f = evalBSDF(M, n, s.l, v);
        for (let c = 0; c < 3; c++) acc[c] += f[c] * s.l[2] / s.pdf;
      }
      for (let c = 0; c < 3; c++) { spec[3 * k + c] = acc[c] / samples; diff[3 * k + c] = 0; }
      continue;
    }
    for (let i = 0; i < samples; i++) {
      const u1 = (i + 0.5) / samples, u2 = radinv2(i);
      const h = sampleVNDF(v, M.alpha, u1, u2);
      const vh = dot(v, h);
      const l = [2 * vh * h[0] - v[0], 2 * vh * h[1] - v[1], 2 * vh * h[2] - v[2]];
      if (l[2] <= 0) continue;
      // estimator of the specular term with VNDF sampling: F(v.h) G2/G1(v)
      const w = ggxG2(l[2], mu, M.alpha) / ggxG1(mu, M.alpha);
      const F = fresnelRGB(M, vh);
      for (let c = 0; c < 3; c++) acc[c] += F[c] * w;
    }
    for (let c = 0; c < 3; c++) {
      spec[3 * k + c] = acc[c] / samples;
      // int rho/pi (1-F(l))(1-F(v))/(1-Fbar) cos dl = rho (1-F(mu)) (1-Fbar_exact)/(1-Fbar)
      diff[3 * k + c] = M.metallic ? 0 : M.rho[c] * (1 - fresnelDielectric(mu, M.ior)) * (1 - (M.fbarComputed ?? M.fbar)) / (1 - M.fbar);
    }
  }
  // hemispherical (cosine-weighted) averages: A_hh = 2 int A(mu) mu dmu (midpoint rule on the same bins)
  const hh = [0, 0, 0], hhSpec = [0, 0, 0];
  for (let k = 0; k < bins; k++) {
    const mu = (k + 0.5) / bins;
    for (let c = 0; c < 3; c++) {
      hh[c] += 2 * mu * (spec[3 * k + c] + diff[3 * k + c]) / bins;
      hhSpec[c] += 2 * mu * spec[3 * k + c] / bins;
    }
  }
  return { bins, spec, diff, hh, hhSpec };
}

/** Linear lookup of a table (Float64Array, 3 per bin) at cosine mu. */
export function lookupAlbedo(tab, mu, which = 'all') {
  const x = Math.min(tab.bins - 1, Math.max(0, mu * tab.bins - 0.5));
  const i = Math.floor(x), f = x - i, j = Math.min(tab.bins - 1, i + 1);
  const out = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const g = (arr) => arr[3 * i + c] * (1 - f) + arr[3 * j + c] * f;
    out[c] = which === 'spec' ? g(tab.spec) : which === 'diff' ? g(tab.diff) : g(tab.spec) + g(tab.diff);
  }
  return out;
}

// ------------------------------------------------------------------ FDM layer lines

/**
 * Height field h(s) = (depth/2) cos(2 pi s / period), s = p . build_dir (peak-to-valley = depth). The perturbed
 * normal is n' = normalize(n - h'(s) b_t), b_t = build_dir - (build_dir . n) n (the in-surface part of build_dir).
 */
export function layerNormal(n, p, buildDir, period, depth) {
  const s = dot(p, buildDir);
  const dh = -(depth / 2) * (2 * Math.PI / period) * Math.sin(2 * Math.PI * s / period);
  const bn = dot(buildDir, n);
  const bt = [buildDir[0] - bn * n[0], buildDir[1] - bn * n[1], buildDir[2] - bn * n[2]];
  return normalize([n[0] - dh * bt[0], n[1] - dh * bt[1], n[2] - dh * bt[2]]);
}
