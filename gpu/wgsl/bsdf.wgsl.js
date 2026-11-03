// WGSL BSDF: the float32 copy of gpu/bsdf.js, which states the formula (docs/WEB_TWIN.md §8). Any change here
// must be made there too; tests/gpu_test.html compares the two on random inputs.
export const BSDF = /* wgsl */ `
struct Mat {
  c: vec3f,          // rho_d (dielectric) or F0 (conductor, coated conductor)
  alpha: f32,        // GGX alpha (the metal's for a coated conductor)
  ior: f32,          // dielectric IOR, or the coat's
  metal: f32,        // 0 dielectric, 1 conductor, 2 coated conductor
  fbar: f32,
  rough: f32,
  hh: vec3f,         // hemispherical albedo (all lobes)
  hhSpec: vec3f,
  x: vec4f,          // coated: (tau rgb, alpha_c); textured (charuco_print): (ink rho_d rgb, ink roughness)
  idx: u32,
}

const MAT_STRIDE: u32 = 5u;

fn loadMat(m: u32) -> Mat {
  let a = params[MAT_OFF + MAT_STRIDE * m];
  let b = params[MAT_OFF + MAT_STRIDE * m + 1u];
  let c = params[MAT_OFF + MAT_STRIDE * m + 2u];
  let d = params[MAT_OFF + MAT_STRIDE * m + 3u];
  let e = params[MAT_OFF + MAT_STRIDE * m + 4u];
  return Mat(a.xyz, a.w, b.x, b.y, b.z, b.w, c.xyz, d.xyz, e, m);
}

fn fresnelDielectric(cosi: f32, eta: f32) -> f32 {
  let c = clamp(cosi, 0.0, 1.0);
  let s2 = (1.0 - c * c) / (eta * eta);
  if (s2 >= 1.0) { return 1.0; }
  let ct = sqrt(1.0 - s2);
  let rs = (c - eta * ct) / (c + eta * ct);
  let rp = (eta * c - ct) / (eta * c + ct);
  return 0.5 * (rs * rs + rp * rp);
}

fn schlick3(f0: vec3f, c: f32) -> vec3f {
  let m = pow(1.0 - clamp(c, 0.0, 1.0), 5.0);
  return f0 + (vec3f(1.0) - f0) * m;
}

fn fresnelRGB(M: Mat, c: f32) -> vec3f {
  if (M.metal > 0.5) { return schlick3(M.c, c); }
  return vec3f(fresnelDielectric(c, M.ior));
}

fn ggxD(nh: f32, a: f32) -> f32 {
  if (nh <= 0.0) { return 0.0; }
  let a2 = a * a;
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

fn ggxLambda(c: f32, a: f32) -> f32 {
  let c2 = max(c * c, 1e-12);
  let t2 = (1.0 - c2) / c2;
  return 0.5 * (-1.0 + sqrt(1.0 + a * a * t2));
}
fn ggxG1(c: f32, a: f32) -> f32 { return 1.0 / (1.0 + ggxLambda(c, a)); }
fn ggxG2(nl: f32, nv: f32, a: f32) -> f32 { return 1.0 / (1.0 + ggxLambda(nl, a) + ggxLambda(nv, a)); }

fn evalBSDF(M: Mat, n: vec3f, l: vec3f, v: vec3f) -> vec3f {
  let nl = dot(n, l);
  let nv = dot(n, v);
  if (nl <= 0.0 || nv <= 0.0) { return vec3f(0.0); }
  let h = normalize(l + v);
  let nh = dot(n, h);
  let vh = max(0.0, dot(v, h));
  if (M.metal > 1.5) {
    // coated conductor (WEB_TWIN_PHYSICS §3.7)
    let coat = ggxD(nh, M.x.w) * ggxG2(nl, nv, M.x.w) * fresnelDielectric(vh, M.ior) / (4.0 * nl * nv);
    let t = (1.0 - fresnelDielectric(nl, M.ior)) * (1.0 - fresnelDielectric(nv, M.ior));
    let metal = ggxD(nh, M.alpha) * ggxG2(nl, nv, M.alpha) / (4.0 * nl * nv);
    return vec3f(coat) + t * pow(M.x.xyz, vec3f(1.0 / nl + 1.0 / nv)) * metal * schlick3(M.c, vh);
  }
  let spec = ggxD(nh, M.alpha) * ggxG2(nl, nv, M.alpha) / (4.0 * nl * nv);
  var out = spec * fresnelRGB(M, vh);
  if (M.metal < 0.5) {
    let kd = INV_PI * (1.0 - fresnelDielectric(nl, M.ior)) * (1.0 - fresnelDielectric(nv, M.ior)) / (1.0 - M.fbar);
    out = out + M.c * kd;
  }
  return out;
}

fn sampleVNDF(ve: vec3f, a: f32, u1: f32, u2: f32) -> vec3f {
  let vh = normalize(vec3f(a * ve.x, a * ve.y, ve.z));
  let lensq = vh.x * vh.x + vh.y * vh.y;
  var T1 = vec3f(1.0, 0.0, 0.0);
  if (lensq > 0.0) { T1 = vec3f(-vh.y, vh.x, 0.0) / sqrt(lensq); }
  let T2 = cross(vh, T1);
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let t1 = r * cos(phi);
  var t2 = r * sin(phi);
  let s = 0.5 * (1.0 + vh.z);
  t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
  let t3 = sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2));
  let nh = t1 * T1 + t2 * T2 + t3 * vh;
  return normalize(vec3f(a * nh.x, a * nh.y, max(1e-7, nh.z)));
}

fn specProb(M: Mat, nv: f32) -> f32 {
  if (M.metal > 1.5) {
    let Fc = fresnelDielectric(nv, M.ior);
    return Fc / (Fc + (1.0 - Fc) * (M.c.x + M.c.y + M.c.z) / 3.0);
  }
  if (M.metal > 0.5) { return 1.0; }
  let F = fresnelDielectric(nv, M.ior);
  let d = luminance(M.c) * (1.0 - F);
  let s = max(F, 0.04);
  return clamp(s / (s + d), 0.05, 0.95);
}

fn pdfBSDF(M: Mat, n: vec3f, v: vec3f, l: vec3f) -> f32 {
  let nl = dot(n, l);
  let nv = dot(n, v);
  if (nl <= 0.0 || nv <= 0.0) { return 0.0; }
  let h = normalize(l + v);
  let nh = dot(n, h);
  let vh = dot(v, h);
  let ps = specProb(M, nv);
  if (M.metal > 1.5) {
    if (vh <= 0.0) { return 0.0; }
    return ps * ggxG1(nv, M.x.w) * ggxD(nh, M.x.w) / (4.0 * nv) + (1.0 - ps) * ggxG1(nv, M.alpha) * ggxD(nh, M.alpha) / (4.0 * nv);
  }
  var pdfSpec = 0.0;
  if (vh > 0.0) { pdfSpec = ggxG1(nv, M.alpha) * ggxD(nh, M.alpha) / (4.0 * nv); }
  return ps * pdfSpec + (1.0 - ps) * nl * INV_PI;
}

// returns (l.xyz, pdf); pdf = 0 when the sample is invalid
fn sampleBSDF(M: Mat, n: vec3f, v: vec3f, u: vec3f) -> vec4f {
  let nv = dot(n, v);
  if (nv <= 0.0) { return vec4f(0.0); }
  let t = basisT(n);
  let b = basisB(n);
  let ps = specProb(M, nv);
  var l: vec3f;
  if (M.metal > 1.5) {
    let hl = sampleVNDF(vec3f(dot(t, v), dot(b, v), nv), select(M.alpha, M.x.w, u.x < ps), u.y, u.z);
    let h = t * hl.x + b * hl.y + n * hl.z;
    l = 2.0 * dot(v, h) * h - v;
  } else if (u.x < ps) {
    let hl = sampleVNDF(vec3f(dot(t, v), dot(b, v), nv), M.alpha, u.y, u.z);
    let h = t * hl.x + b * hl.y + n * hl.z;
    l = 2.0 * dot(v, h) * h - v;
  } else {
    let r = sqrt(u.y);
    let phi = 2.0 * PI * u.z;
    l = t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u.y));
  }
  if (dot(n, l) <= 0.0) { return vec4f(0.0); }
  return vec4f(l, pdfBSDF(M, n, v, l));
}

// directional albedo at view cosine mu: x = specular part, y = diffuse part (rgb each), from the tables
fn albedoSpec(m: u32, mu: f32) -> vec3f {
  let x = clamp(mu * f32(ALB_BINS) - 0.5, 0.0, f32(ALB_BINS - 1u));
  let i = u32(floor(x));
  let j = min(ALB_BINS - 1u, i + 1u);
  let f = x - f32(i);
  let base = ALB_OFF + m * ALB_BINS * 2u;
  return mix(params[base + 2u * i].xyz, params[base + 2u * j].xyz, f);
}
fn albedoDiff(m: u32, mu: f32) -> vec3f {
  let x = clamp(mu * f32(ALB_BINS) - 0.5, 0.0, f32(ALB_BINS - 1u));
  let i = u32(floor(x));
  let j = min(ALB_BINS - 1u, i + 1u);
  let f = x - f32(i);
  let base = ALB_OFF + m * ALB_BINS * 2u;
  return mix(params[base + 2u * i + 1u].xyz, params[base + 2u * j + 1u].xyz, f);
}

// FDM layer lines: h(s) = depth/2 cos(2 pi s / period), s = p . buildDir
fn layerNormal(n: vec3f, p: vec3f, bd: vec3f, period: f32, depth: f32) -> vec3f {
  let ph = fract(dot(p, bd) / period);          // phase first: keeps sin's argument small in float32
  let dh = -(depth * 0.5) * (2.0 * PI / period) * sin(2.0 * PI * ph);
  let bt = bd - dot(bd, n) * n;
  return normalize(n - dh * bt);
}
`;
