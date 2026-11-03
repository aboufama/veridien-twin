// WGSL of the lite renderer (lite/renderer.js). Every loop has a constant cap (docs/WEB_TWIN.md §12).
// BSDF: the §8 formula, copied from gpu/wgsl/bsdf.wgsl.js (evaluation only; judged by the same Mitsuba compare).

export const MAX_EMIT = 16, MAX_CAPS = 16, ALB_BINS = 32;

const COMMON = /* wgsl */ `
const PI = 3.14159265358979;
const INV_PI = 0.318309886183791;
const ALB_BINS = ${ALB_BINS}u;

struct Light {
  pts: array<vec4f, ${MAX_EMIT}>,   // emitter sample points: xyz (D), w = area (m^2)
  nE: vec4f,                        // outlet normal (D), w = point count
  axis: vec4f,                      // emission axis (D), w = profile step (rad)
  L0: vec4f,                        // axial radiance rgb (W sr^-1 m^-2), w = profile entries
  prof: array<vec4f, 4>,            // p(theta) at k * step, linear, 0 past the end
  cen: vec4f,                       // outlet centroid (D), w = shadow-disk radius (m)
}
struct Caps {
  n: vec4u,                         // x = count
  a: array<vec4f, ${MAX_CAPS}>,     // end a (D), w = radius
  b: array<vec4f, ${MAX_CAPS}>,     // end b (D), w = mean albedo of the body (indirect)
  g: array<vec4u, ${MAX_CAPS / 4}>, // shadow group (0: casts on all; a fragment skips its own group); bit 8: no direct shadow
}
struct Mat {
  c: vec4f,      // rho_d (dielectric) or F0 (conductor / coated), w = alpha
  p: vec4f,      // ior (coat ior), metal (0, 1, 2 coated), fbar, kind (0 surface, 1 emitter, 2 printed board)
  x: vec4f,      // coated: tau rgb, alpha_c; board: ink alpha
  hh: vec4f,     // hemispherical albedo rgb
  ink: vec4f,    // board ink rho_d rgb
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
fn schlick3(f0: vec3f, c: f32) -> vec3f { return f0 + (vec3f(1.0) - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0); }
fn ggxD(nh: f32, a: f32) -> f32 {
  if (nh <= 0.0) { return 0.0; }
  let a2 = a * a;
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
fn ggxLambda(c: f32, a: f32) -> f32 {
  let c2 = max(c * c, 1e-12);
  return 0.5 * (-1.0 + sqrt(1.0 + a * a * (1.0 - c2) / c2));
}
fn ggxG2(nl: f32, nv: f32, a: f32) -> f32 { return 1.0 / (1.0 + ggxLambda(nl, a) + ggxLambda(nv, a)); }

// f(l, v) of docs/WEB_TWIN.md §8 (no cosine); rho / alpha passed in (the board mixes them from its texture)
fn evalBSDF(M: Mat, rho: vec3f, alpha: f32, n: vec3f, l: vec3f, v: vec3f) -> vec3f {
  let nl = dot(n, l);
  let nv = dot(n, v);
  if (nl <= 0.0 || nv <= 0.0) { return vec3f(0.0); }
  let h = normalize(l + v);
  let nh = dot(n, h);
  let vh = max(0.0, dot(v, h));
  if (M.p.y > 1.5) {
    let coat = ggxD(nh, M.x.w) * ggxG2(nl, nv, M.x.w) * fresnelDielectric(vh, M.p.x) / (4.0 * nl * nv);
    let t = (1.0 - fresnelDielectric(nl, M.p.x)) * (1.0 - fresnelDielectric(nv, M.p.x));
    let metal = ggxD(nh, alpha) * ggxG2(nl, nv, alpha) / (4.0 * nl * nv);
    return vec3f(coat) + t * pow(M.x.xyz, vec3f(1.0 / nl + 1.0 / nv)) * metal * schlick3(M.c.xyz, vh);
  }
  let spec = ggxD(nh, alpha) * ggxG2(nl, nv, alpha) / (4.0 * nl * nv);
  if (M.p.y > 0.5) { return spec * schlick3(M.c.xyz, vh); }
  let kd = INV_PI * (1.0 - fresnelDielectric(nl, M.p.x)) * (1.0 - fresnelDielectric(nv, M.p.x)) / (1.0 - M.p.z);
  return vec3f(spec * fresnelDielectric(vh, M.p.x)) + rho * kd;
}

fn profileAt(th: f32) -> f32 {
  let x = th / LI.axis.w;
  let n = LI.L0.w;
  if (x < 0.0 || x > n - 1.0) { return 0.0; }
  let i = u32(floor(x));
  let j = min(i + 1u, u32(n) - 1u);
  let f = x - f32(i);
  return mix(LI.prof[i / 4u][i % 4u], LI.prof[j / 4u][j % 4u], f);
}

// closest points of segments p0-p1 and q0-q1: (s, t, distance) (Ericson, Real-Time Collision Detection 5.1.9)
fn segSeg(p0: vec3f, p1: vec3f, q0: vec3f, q1: vec3f) -> vec3f {
  let d1 = p1 - p0; let d2 = q1 - q0; let r = p0 - q0;
  let a = dot(d1, d1); let e = dot(d2, d2); let f = dot(d2, r);
  var s = 0.0; var t = 0.0;
  if (a <= 1e-14) { t = clamp(f / max(e, 1e-14), 0.0, 1.0); }
  else {
    let c = dot(d1, r);
    if (e <= 1e-14) { s = clamp(-c / a, 0.0, 1.0); }
    else {
      let b = dot(d1, d2);
      let den = a * e - b * b;
      if (den > 1e-14) { s = clamp((b * f - c * e) / den, 0.0, 1.0); }
      t = (b * s + f) / e;
      if (t < 0.0) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); }
      else if (t > 1.0) { t = 1.0; s = clamp((b - c) / a, 0.0, 1.0); }
    }
  }
  return vec3f(s, t, length((p0 + d1 * s) - (q0 + d2 * t)));
}

// fraction of a disk of radius rho below the chord at signed offset u * rho
fn diskBelow(u: f32) -> f32 {
  let x = clamp(u, -1.0, 1.0);
  return 0.5 + (asin(x) + x * sqrt(max(0.0, 1.0 - x * x))) * INV_PI;
}
// fraction of a disk of radius rho covered by a circle of radius r at centre distance h
fn diskDisk(h: f32, r: f32, rho: f32) -> f32 {
  if (h >= r + rho) { return 0.0; }
  if (h <= abs(r - rho)) { return min(1.0, (r * r) / (rho * rho)); }
  let a1 = acos(clamp((h * h + rho * rho - r * r) / (2.0 * h * rho), -1.0, 1.0));
  let a2 = acos(clamp((h * h + r * r - rho * rho) / (2.0 * h * r), -1.0, 1.0));
  let k = 0.5 * sqrt(max(0.0, (-h + r + rho) * (h + r - rho) * (h - r + rho) * (h + r + rho)));
  return (rho * rho * a1 + r * r * a2 - k) / (PI * rho * rho);
}
fn capGroup(i: u32) -> u32 { return CP.g[i / 4u][i % 4u]; }

// soft shadow of the posed capsules on the light outlet (a disk of radius LI.cen.w at its centroid) seen from p:
// each capsule crossing the cone from p to the disk covers a band (its middle) or a disk (its end) of the cone's
// cross-section there; visibilities multiply
fn softVis(p: vec3f, group: u32) -> f32 {
  let c = LI.cen.xyz;
  let R = LI.cen.w;
  var vis = 1.0;
  for (var i = 0u; i < min(CP.n.x, ${MAX_CAPS}u); i++) {
    let g = capGroup(i);
    if ((g & 256u) != 0u) { continue; }
    if (group != 0u && (g & 255u) == group) { continue; }
    let A = CP.a[i]; let B = CP.b[i];
    let st = segSeg(p, c, A.xyz, B.xyz);
    let r = A.w;
    let rho = max(R * st.x, 1e-5);
    var F = 0.0;
    if (st.y > 0.0 && st.y < 1.0) {
      F = diskBelow((st.z + r) / rho) - diskBelow((st.z - r) / rho);
    } else {
      F = diskDisk(st.z, r, rho);
    }
    vis = vis * (1.0 - clamp(F, 0.0, 1.0));
  }
  return vis;
}

fn albSpec(m: u32, mu: f32) -> vec3f {
  let x = clamp(mu * f32(ALB_BINS) - 0.5, 0.0, f32(ALB_BINS - 1u));
  let i = u32(floor(x)); let j = min(ALB_BINS - 1u, i + 1u); let f = x - f32(i);
  return mix(alb[(m * ALB_BINS + i) * 2u].xyz, alb[(m * ALB_BINS + j) * 2u].xyz, f);
}
fn albDiff(m: u32, mu: f32) -> vec3f {
  let x = clamp(mu * f32(ALB_BINS) - 0.5, 0.0, f32(ALB_BINS - 1u));
  let i = u32(floor(x)); let j = min(ALB_BINS - 1u, i + 1u); let f = x - f32(i);
  return mix(alb[(m * ALB_BINS + i) * 2u + 1u].xyz, alb[(m * ALB_BINS + j) * 2u + 1u].xyz, f);
}

// direct light from the M outlet points (area light, L0 p(theta)), BSDF per point, soft capsule shadows
fn directLight(M: Mat, rho: vec3f, alpha: f32, p: vec3f, n: vec3f, wo: vec3f, group: u32) -> vec3f {
  var acc = vec3f(0.0);
  for (var m = 0u; m < min(u32(LI.nE.w), ${MAX_EMIT}u); m++) {
    let s = LI.pts[m];
    let wv = s.xyz - p;
    let d2 = max(dot(wv, wv), 1e-10);
    let l = wv * inverseSqrt(d2);
    if (dot(n, l) <= 0.0) { continue; }
    let ce = -dot(l, LI.nE.xyz);
    if (ce <= 0.0) { continue; }
    let pr = profileAt(acos(clamp(-dot(l, LI.axis.xyz), -1.0, 1.0)));
    if (pr <= 0.0) { continue; }
    acc = acc + evalBSDF(M, rho, alpha, n, l, wo) * (pr * ce * s.w * dot(n, l) / d2);
  }
  if (all(acc == vec3f(0.0))) { return acc; }
  return acc * LI.L0.xyz * softVis(p, group);
}
`;

// ------------------------------------------------------------------ radiosity (compute)
export const RADIOSITY = /* wgsl */ `
${COMMON}
struct RadU { n: vec4u }          // x: nodes, y: K, z: first node (chunk), w: -
@group(0) @binding(0) var<uniform> LI: Light;
@group(0) @binding(1) var<uniform> CP: Caps;
@group(0) @binding(2) var<storage, read> mats: array<Mat>;
@group(0) @binding(3) var<storage, read> alb: array<vec4f>;
@group(0) @binding(4) var<storage, read> nodeA: array<vec4f>;     // pos, area
@group(0) @binding(5) var<storage, read> nodeN: array<vec4f>;     // normal, mesh (bits)
@group(0) @binding(6) var<uniform> RU: RadU;

@group(1) @binding(0) var<storage, read_write> edir: array<vec4f>;   // reflected direct flux / area (E_dir A(mu_l))

// direct irradiance at every node, times the directional albedo at the light's incidence (what it re-emits)
@compute @workgroup_size(64)
fn nodeDirect(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + RU.n.z;
  if (i >= RU.n.x) { return; }
  let p = nodeA[i].xyz;
  let n = nodeN[i].xyz;
  let mesh = bitcast<u32>(nodeN[i].w);
  let M = mats[mesh];
  var acc = vec3f(0.0);
  for (var m = 0u; m < min(u32(LI.nE.w), ${MAX_EMIT}u); m++) {
    let s = LI.pts[m];
    let wv = s.xyz - p;
    let d2 = max(dot(wv, wv), 1e-10);
    let l = wv * inverseSqrt(d2);
    let nl = dot(n, l);
    if (nl <= 0.0) { continue; }
    let ce = -dot(l, LI.nE.xyz);
    if (ce <= 0.0) { continue; }
    let pr = profileAt(acos(clamp(-dot(l, LI.axis.xyz), -1.0, 1.0)));
    acc = acc + (albSpec(mesh, nl) + albDiff(mesh, nl)) * (pr * ce * s.w * nl / d2);
  }
  edir[i] = vec4f(acc * LI.L0.xyz * softVis(p + 1e-3 * n, 0u), 0.0);
}
`;

export const GATHER = /* wgsl */ `
${COMMON}
struct RadU { n: vec4u }          // x: nodes, y: K, z: first node, w: -
@group(0) @binding(0) var<uniform> LI: Light;
@group(0) @binding(1) var<uniform> CP: Caps;
@group(0) @binding(2) var<storage, read> mats: array<Mat>;
@group(0) @binding(3) var<storage, read> alb: array<vec4f>;
@group(0) @binding(4) var<storage, read> nodeA: array<vec4f>;
@group(0) @binding(5) var<storage, read> nodeN: array<vec4f>;
@group(0) @binding(6) var<uniform> RU: RadU;
@group(1) @binding(0) var<storage, read> gather: array<u32>;         // 2 x u16 per word: hit node, 0xffff = none
@group(1) @binding(1) var<storage, read_write> occ: array<u32>;      // 4 x u8 per word: transmission x 255

// per gather ray: 255 when no posed body's capsule cuts the segment node -> hit node, else 255 x that body's mean
// albedo (the light it would have brought is replaced by what the body re-emits, ~ its albedo x the same radiance)
@compute @workgroup_size(64)
fn gatherOcc(@builtin(global_invocation_id) gid: vec3u) {
  let K = RU.n.y;
  let w = gid.x + RU.n.z;                 // word index: 4 rays
  if (w >= RU.n.x * K / 4u) { return; }
  let node = (w * 4u) / K;
  let p = nodeA[node].xyz + 1e-3 * nodeN[node].xyz;
  var word = 0u;
  for (var q = 0u; q < 4u; q++) {
    let k = w * 4u + q;
    let hw = gather[k / 2u];
    let hit = (hw >> (16u * (k & 1u))) & 0xffffu;
    var t = 255u;
    if (hit != 0xffffu) {
      let e = nodeA[hit].xyz;
      for (var i = 0u; i < min(CP.n.x, ${MAX_CAPS}u); i++) {
        let A = CP.a[i];
        let st = segSeg(p, e, A.xyz, CP.b[i].xyz);
        if (st.z < A.w) { t = min(t, u32(round(255.0 * clamp(CP.b[i].w, 0.0, 1.0)))); }
      }
    }
    word = word | (t << (8u * q));
  }
  occ[w] = word;
}

@group(2) @binding(0) var<storage, read> bIn: array<vec4f>;
@group(2) @binding(1) var<storage, read_write> bOut: array<vec4f>;
@group(2) @binding(2) var<storage, read_write> eind: array<vec4f>;
@group(2) @binding(3) var<storage, read> edirR: array<vec4f>;

// one Jacobi step: E_ind = pi/K sum_k t_k B(hit_k); B = (E_dir A(mu_l) + hh E_ind) / pi
@compute @workgroup_size(64)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + RU.n.z;
  if (i >= RU.n.x) { return; }
  let K = RU.n.y;
  var acc = vec3f(0.0);
  for (var k = 0u; k < min(K, 1024u); k++) {
    let gi = i * K + k;
    let hit = (gather[gi / 2u] >> (16u * (gi & 1u))) & 0xffffu;
    if (hit == 0xffffu) { continue; }
    let t = f32((occ[gi / 4u] >> (8u * (gi & 3u))) & 255u) * (1.0 / 255.0);
    acc = acc + t * bIn[hit].xyz;
  }
  let E = acc * (PI / f32(K));
  let mesh = bitcast<u32>(nodeN[i].w);
  eind[i] = vec4f(E, 0.0);
  bOut[i] = vec4f((edirR[i].xyz + mats[mesh].hh.xyz * E) * INV_PI, 0.0);
}
`;

// ------------------------------------------------------------------ the reflection cube (node mesh)
export const ENV = /* wgsl */ `
struct EnvU { probe: vec4f, face: vec4u, nf: vec4f }      // nf: near, far
@group(0) @binding(0) var<uniform> EU: EnvU;
@group(0) @binding(1) var<storage, read> nodeA: array<vec4f>;
@group(0) @binding(2) var<storage, read> bCur: array<vec4f>;
struct EOut { @builtin(position) pos: vec4f, @location(0) c: vec4f };
@vertex fn vsEnv(@builtin(vertex_index) vid: u32) -> EOut {
  let d = nodeA[vid].xyz - EU.probe.xyz;
  var cl: vec3f;            // (sc, -tc, |ma|) per cube face (Vulkan / WebGPU face addressing)
  switch (EU.face.x) {
    case 0u: { cl = vec3f(-d.z, d.y, d.x); }
    case 1u: { cl = vec3f(d.z, d.y, -d.x); }
    case 2u: { cl = vec3f(d.x, -d.z, d.y); }
    case 3u: { cl = vec3f(d.x, d.z, -d.y); }
    case 4u: { cl = vec3f(d.x, d.y, d.z); }
    default: { cl = vec3f(-d.x, d.y, -d.z); }
  }
  let n = EU.nf.x; let f = EU.nf.y;
  var o: EOut;
  o.pos = vec4f(cl.x, cl.y, (cl.z - n) * f / (f - n), cl.z);
  o.c = vec4f(bCur[vid].xyz, length(d));
  return o;
}
@fragment fn fsEnv(v: EOut) -> @location(0) vec4f { return v.c; }
`;

export const ENV_MIP = /* wgsl */ `
@group(0) @binding(0) var srcMip: texture_2d_array<f32>;
@group(0) @binding(1) var dstMip: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn envDown(@builtin(global_invocation_id) gid: vec3u) {
  let d = textureDimensions(dstMip);
  if (gid.x >= d.x || gid.y >= d.y) { return; }
  let q = vec2i(gid.xy) * 2;
  let f = i32(gid.z);
  let c = textureLoad(srcMip, q, f, 0) + textureLoad(srcMip, q + vec2i(1, 0), f, 0) + textureLoad(srcMip, q + vec2i(0, 1), f, 0) + textureLoad(srcMip, q + vec2i(1, 1), f, 0);
  textureStore(dstMip, vec2i(gid.xy), f, c * 0.25);
}
`;

// ------------------------------------------------------------------ camera pass (raster)
export const CAMERA = /* wgsl */ `
${COMMON}
struct View {
  T_C_D: mat4x4f,
  K: vec4f,          // fx fy cx cy
  D: vec4f,          // KB4 k1..k4
  size: vec4f,       // W H near far
  mode: vec4f,       // x: 0 KB4, 1 pinhole; y: theta max (KB4); z: skew alpha; w: -
  cam: vec4f,        // camera centre (D)
  circ: vec4f,       // image circle cx, cy, r (px); r <= 0: none
}
struct SceneU { sMin: vec4f, sExt: vec4f, dMin: vec4f, dExt: vec4f, probe: vec4f, opts: vec4f }  // opts.x: indirect on, y: direct on
@group(0) @binding(0) var<uniform> LI: Light;
@group(0) @binding(1) var<uniform> CP: Caps;
@group(0) @binding(2) var<storage, read> mats: array<Mat>;
@group(0) @binding(3) var<storage, read> alb: array<vec4f>;
@group(0) @binding(4) var<uniform> SU: SceneU;
@group(0) @binding(5) var<storage, read> bodies: array<mat4x4f>;
@group(0) @binding(6) var<storage, read> bodyGroup: array<vec4u>;
@group(0) @binding(7) var<storage, read> eind: array<vec4f>;
@group(0) @binding(8) var envTex: texture_cube<f32>;
@group(0) @binding(9) var envSamp: sampler;
@group(0) @binding(10) var boardTex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> V: View;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) pD: vec3f,
  @location(1) nD: vec3f,
  @location(2) ei: vec4f,                         // static: indirect irradiance (w = 1); dynamic: w = 0
  @location(3) uv: vec2f,
  @location(4) @interpolate(flat) meta: vec2u,    // mesh, shadow group
};

fn projectD(pD: vec3f) -> vec4f {
  let pc = (V.T_C_D * vec4f(pD, 1.0)).xyz;
  let near = V.size.z; let far = V.size.w;
  if (V.mode.x < 0.5) {
    let rho = length(pc.xy);
    let th = atan2(rho, pc.z);
    let t2 = th * th;
    let thd = th * (1.0 + t2 * (V.D.x + t2 * (V.D.y + t2 * (V.D.z + t2 * V.D.w))));
    var uv = V.K.zw;
    if (rho > 1e-12) {
      let s = thd / rho;
      let xd = pc.x * s; let yd = pc.y * s;
      uv = vec2f(V.K.x * (xd + V.mode.z * yd) + V.K.z, V.K.y * yd + V.K.w);
    }
    let ndc = vec2f((uv.x + 0.5) / V.size.x * 2.0 - 1.0, 1.0 - (uv.y + 0.5) / V.size.y * 2.0);
    var z = clamp((length(pc) - near) / (far - near), 0.0, 1.0);
    if (th > V.mode.y) { z = 2.0; }                // past the lens's monotone field: clipped
    return vec4f(ndc, z, 1.0);
  }
  let w = pc.z;
  let u = V.K.x * pc.x + V.K.z * w;
  let v = V.K.y * pc.y + V.K.w * w;
  return vec4f((u + 0.5 * w) / V.size.x * 2.0 - w, w - (v + 0.5 * w) / V.size.y * 2.0, (w - near) * far / (far - near), w);
}

@vertex fn vsStatic(@location(0) q: vec4u, @location(1) n8: vec4f, @location(2) nd: vec4u, @location(3) wt: vec4f) -> VOut {
  let p = SU.sMin.xyz + vec3f(q.xyz) * (1.0 / 65535.0) * SU.sExt.xyz;
  var o: VOut;
  o.pos = projectD(p);
  o.pD = p;
  o.nD = n8.xyz;
  let e = wt.x * eind[nd.x].xyz + wt.y * eind[nd.y].xyz + wt.z * eind[nd.z].xyz + wt.w * eind[nd.w].xyz;
  o.ei = vec4f(e, 1.0);
  o.uv = vec2f(0.0);
  o.meta = vec2u(q.w, 0u);
  return o;
}

@vertex fn vsDyn(@location(0) q: vec4u, @location(1) n16: vec4i, @location(2) uv: vec2f) -> VOut {
  let b = u32(n16.w);
  let M = bodies[b];
  let pB = SU.dMin.xyz + vec3f(q.xyz) * (1.0 / 65535.0) * SU.dExt.xyz;
  let p = (M * vec4f(pB, 1.0)).xyz;
  var o: VOut;
  o.pos = projectD(p);
  o.pD = p;
  o.nD = (M * vec4f(vec3f(n16.xyz) * (1.0 / 32767.0), 0.0)).xyz;
  o.ei = vec4f(0.0);
  o.uv = uv;
  o.meta = vec2u(q.w, bodyGroup[b].x);
  return o;
}

// radiance arriving at p from direction r, from the cube at the probe, parallax-corrected by its distance channel
fn envLookup(p: vec3f, r: vec3f, lod: f32) -> vec3f {
  let d0 = p - SU.probe.xyz;
  let b = dot(d0, r);
  let c0 = dot(d0, d0);
  var dir = r;
  for (var it = 0; it < 3; it++) {
    let R = textureSampleLevel(envTex, envSamp, dir, 2.0).w;
    let disc = b * b - c0 + R * R;
    let t = select(R, -b + sqrt(max(disc, 0.0)), disc >= 0.0);
    dir = normalize(d0 + t * r);
  }
  return textureSampleLevel(envTex, envSamp, dir, lod).xyz;
}

@fragment fn fsMain(v: VOut) -> @location(0) vec4f {
  if (V.circ.z > 0.0) {
    let d = v.pos.xy - 0.5 - V.circ.xy;
    if (dot(d, d) > V.circ.z * V.circ.z) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  }
  let mesh = v.meta.x;
  let M = mats[mesh];
  let wo = normalize(V.cam.xyz - v.pD);
  let n = normalize(v.nD);
  let nv = dot(n, wo);
  let kind = u32(M.p.w);
  if (kind == 1u) {                                // the light outlet, one-sided
    if (nv <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    return vec4f(LI.L0.xyz * profileAt(acos(clamp(dot(wo, LI.axis.xyz), -1.0, 1.0))), 1.0);
  }
  if (nv <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }   // one-sided BSDF (the Mitsuba plugin's convention)
  var rho = M.c.xyz;
  var alpha = M.c.w;
  if (kind == 2u) {
    let m = textureSampleLevel(boardTex, envSamp, v.uv, 0.0).x;
    rho = mix(M.ink.xyz, M.c.xyz, m);
    alpha = mix(M.x.w, M.c.w, m);
  }
  var L = vec3f(0.0);
  if (SU.opts.y > 0.5) { L = directLight(M, rho, alpha, v.pD, n, wo, v.meta.y); }
  if (SU.opts.x > 0.5) {
    let mu = clamp(nv, 1e-4, 1.0);
    if (v.ei.w > 0.5) {
      // static: interpolated node irradiance, re-emitted with the directional albedo at the view angle
      L = L + v.ei.xyz * INV_PI * (albSpec(mesh, mu) + albDiff(mesh, mu));
    } else {
      // posed bodies: the cube's radiance in the mirror direction (lobe width -> mip) and around the normal
      let lod = clamp(log2(max(alpha * 128.0 / 1.2, 1.0)), 0.0, 7.0);
      let r = reflect(-wo, n);
      L = L + albSpec(mesh, mu) * envLookup(v.pD, r, lod);
      let dif = albDiff(mesh, mu);
      if (any(dif > vec3f(0.0))) { L = L + dif * envLookup(v.pD, n, 6.0) * select(vec3f(1.0), rho / max(M.c.xyz, vec3f(1e-4)), kind == 2u); }
    }
  }
  return vec4f(L, 1.0);
}

`;

// linear radiance (resolved rgba16float) -> float32 rows, for readback(name, 'linear')
export const LINEAR = /* wgsl */ `
@group(0) @binding(0) var linIn: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> linOut: array<vec4f>;
@compute @workgroup_size(8, 8)
fn toF32(@builtin(global_invocation_id) gid: vec3u) {
  let d = textureDimensions(linIn);
  if (gid.x >= d.x || gid.y >= d.y) { return; }
  linOut[gid.y * d.x + gid.x] = textureLoad(linIn, vec2i(gid.xy), 0);
}
`;

// ------------------------------------------------------------------ light sensor (display) + camera MJPG planes
export const SENSOR = /* wgsl */ `
struct SensU {
  size: vec4u,       // w, h, frame, flags (1: KB4 relative illumination, 2: noise, 4: glare, 8: 4:2:0)
  K: vec4f,          // KB4 fx fy cx cy (relative illumination)
  D: vec4f,          // KB4 k
  ri: vec4f,         // skew alpha, pupil exponent q, -, -
  optics: vec4f,     // pi T / (4 N^2), exposure (s), analogue gain, glare fraction k
  m0: vec4f, m1: vec4f, m2: vec4f,   // response matrix rows (e-/s per W m^-2)
  noise: vec4f,      // read noise (e-), full well (e-), dark (e-/s), conversion gain (DN/e-)
  adc: vec4f,        // max DN, black level (DN), seed (bits), -
  wb: vec4f,
  c0: vec4f, c1: vec4f, c2: vec4f,
  tone: vec4f,       // mode (0 sRGB, 1 gamma, 2 linear, 3 sRGB + S-curve), gamma, S-curve c, black offset
  glare: vec4f,      // level factor F, sigma at the level (px), level w, level h
}
@group(0) @binding(0) var<uniform> su: SensU;
@group(0) @binding(1) var linTex: texture_2d<f32>;
@group(0) @binding(2) var gIn: texture_2d<f32>;
@group(0) @binding(3) var gOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read_write> yuv: array<u32>;
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>>;

fn riAt(x: f32, y: f32) -> f32 {
  if ((su.size.w & 1u) == 0u) { return 1.0; }
  let yd = (y - su.K.w) / su.K.y;
  let xd = (x - su.K.z) / su.K.x - su.ri.x * yd;
  let thd = sqrt(xd * xd + yd * yd);
  if (thd < 1e-6) { return 1.0; }
  var th = thd;
  for (var i = 0; i < 8; i++) {
    let t2 = th * th;
    let f = th * (1.0 + t2 * (su.D.x + t2 * (su.D.y + t2 * (su.D.z + t2 * su.D.w)))) - thd;
    let df = 1.0 + t2 * (3.0 * su.D.x + t2 * (5.0 * su.D.y + t2 * (7.0 * su.D.z + t2 * 9.0 * su.D.w)));
    th = th - f / df;
  }
  let t2 = th * th;
  let dd = 1.0 + t2 * (3.0 * su.D.x + t2 * (5.0 * su.D.y + t2 * (7.0 * su.D.z + t2 * 9.0 * su.D.w)));
  return max(0.0, pow(cos(th), su.ri.y) * sin(th) / (thd * dd));
}

@compute @workgroup_size(8, 8)
fn glareDown(@builtin(global_invocation_id) gid: vec3u) {
  let F = i32(su.glare.x);
  let gw = i32(su.glare.z); let gh = i32(su.glare.w);
  if (i32(gid.x) >= gw || i32(gid.y) >= gh) { return; }
  let W = i32(su.size.x); let H = i32(su.size.y);
  var acc = vec3f(0.0); var n = 0.0;
  for (var j = 0; j < min(F, 32); j += 2) {
    for (var i = 0; i < min(F, 32); i += 2) {
      let x = i32(gid.x) * F + i; let y = i32(gid.y) * F + j;
      if (x >= W || y >= H) { continue; }
      acc = acc + textureLoad(linTex, vec2i(x, y), 0).xyz * riAt(f32(x), f32(y));
      n = n + 1.0;
    }
  }
  textureStore(gOut, vec2i(gid.xy), vec4f(acc * su.optics.x / max(n, 1.0), 1.0));
}
fn blur(gid: vec3u, dir: u32) {
  let gw = i32(su.glare.z); let gh = i32(su.glare.w);
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= gw || y >= gh) { return; }
  let sg = max(su.glare.y, 1e-3);
  let n = select(gh, gw, dir == 0u);
  var acc = vec3f(0.0);
  for (var k = 0; k < min(n, 256); k++) {
    let d = f32(k - select(y, x, dir == 0u));
    acc = acc + exp(-0.5 * d * d / (sg * sg)) * textureLoad(gIn, select(vec2i(x, k), vec2i(k, y), dir == 0u), 0).xyz;
  }
  textureStore(gOut, vec2i(x, y), vec4f(acc / (sg * 2.5066282746), 1.0));
}
@compute @workgroup_size(8, 8)
fn glareH(@builtin(global_invocation_id) gid: vec3u) { blur(gid, 0u); }
@compute @workgroup_size(8, 8)
fn glareV(@builtin(global_invocation_id) gid: vec3u) { blur(gid, 1u); }

fn glareAt(x: f32, y: f32) -> vec3f {
  let F = su.glare.x;
  let gw = i32(su.glare.z); let gh = i32(su.glare.w);
  let p = vec2f((x + 0.5) / F - 0.5, (y + 0.5) / F - 0.5);
  let p0 = vec2i(floor(p)); let f = p - floor(p);
  let mx = vec2i(gw - 1, gh - 1);
  let a = textureLoad(gIn, clamp(p0, vec2i(0), mx), 0).xyz;
  let b = textureLoad(gIn, clamp(p0 + vec2i(1, 0), vec2i(0), mx), 0).xyz;
  let c = textureLoad(gIn, clamp(p0 + vec2i(0, 1), vec2i(0), mx), 0).xyz;
  let d = textureLoad(gIn, clamp(p0 + vec2i(1, 1), vec2i(0), mx), 0).xyz;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
fn pcg(v: u32) -> u32 { let s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
fn gaussH(h: u32) -> f32 {
  let u1 = (f32(pcg(h) >> 8u) + 0.5) / 16777216.0;
  let u2 = (f32(pcg(h ^ 0x68bc21ebu) >> 8u) + 0.5) / 16777216.0;
  return sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
}
fn toneC(x: f32) -> f32 {
  let m = su.tone.x;
  if (m > 2.5) {
    let b = su.tone.w;
    let v = clamp((x - b) / (1.0 - b), 0.0, 1.0);
    let u = select(1.055 * pow(v, 1.0 / 2.4) - 0.055, 12.92 * v, v <= 0.0031308);
    let a = pow(u, su.tone.z);
    return a / max(a + pow(1.0 - u, su.tone.z), 1e-12);
  }
  let c = clamp(x, 0.0, 1.0);
  if (m < 0.5) { return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308); }
  if (m < 1.5) { return pow(c, 1.0 / max(su.tone.y, 1e-3)); }
  return c;
}
var<workgroup> wsum: array<vec4u, 64>;
// one thread per 2x2 block: irradiance (RI, glare), electrons (+ optional shot/read noise), DN, white balance, CCM,
// tone; BT.601 full-range Y and 4:2:0 Cb/Cr (the camera MJPG stage reads the planes); sums for AE and AWB
@compute @workgroup_size(8, 8)
fn sensorMain(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let bx = i32(gid.x) * 2; let by = i32(gid.y) * 2;
  var ls = vec4u(0u);
  if (bx < W && by < H) {
    var cb = 0.0; var cr = 0.0; var nb = 0.0;
    var Ys: array<f32, 4>;
    for (var q = 0; q < 4; q++) {
      let x = bx + (q & 1); let y = by + (q >> 1);
      if (x >= W || y >= H) { continue; }
      var E = textureLoad(linTex, vec2i(x, y), 0).xyz * riAt(f32(x), f32(y)) * su.optics.x;
      if ((su.size.w & 4u) != 0u) { E = (1.0 - su.optics.w) * E + su.optics.w * glareAt(f32(x), f32(y)); }
      var e = vec3f(dot(su.m0.xyz, E), dot(su.m1.xyz, E), dot(su.m2.xyz, E)) * su.optics.y + vec3f(su.noise.z * su.optics.y);
      if ((su.size.w & 2u) != 0u) {
        let h = pcg(u32(y * W + x) * 0x9E3779B9u ^ pcg(su.size.z * 0x85ebca6bu ^ bitcast<u32>(su.adc.z)));
        let s = sqrt(max(e, vec3f(0.0)) + vec3f(su.noise.x * su.noise.x));
        e = e + s * vec3f(gaussH(h), gaussH(h ^ 0x1b873593u), gaussH(h ^ 0xcc9e2d51u));
      }
      e = clamp(e, vec3f(0.0), vec3f(su.noise.y));
      var c = e * su.noise.w * su.optics.z / max(su.adc.x - su.adc.y, 1.0);
      ls = ls + vec4u(0u, vec3u(clamp(c, vec3f(0.0), vec3f(1.0)) * 1023.0));
      c = c * su.wb.xyz;
      c = vec3f(dot(su.c0.xyz, c), dot(su.c1.xyz, c), dot(su.c2.xyz, c));
      let c8 = clamp(vec3f(toneC(c.x), toneC(c.y), toneC(c.z)), vec3f(0.0), vec3f(1.0)) * 255.0;
      Ys[q] = 0.299 * c8.x + 0.587 * c8.y + 0.114 * c8.z;
      cb += 128.0 - 0.168736 * c8.x - 0.331264 * c8.y + 0.5 * c8.z;
      cr += 128.0 + 0.5 * c8.x - 0.418688 * c8.y - 0.081312 * c8.z;
      nb += 1.0;
      if ((su.size.w & 8u) == 0u) {
        textureStore(outTex, vec2i(x, y), vec4f(round(c8) / 255.0, 1.0));
        ls.x += u32(clamp(round(Ys[q]), 0.0, 255.0));
      }
    }
    if ((su.size.w & 8u) != 0u) {
      let Cb = clamp(round(cb / nb), 0.0, 255.0);
      let Cr = clamp(round(cr / nb), 0.0, 255.0);
      let cw = (u32(W) + 1u) / 2u;
      let bi = u32(gid.y) * cw + u32(gid.x);
      for (var q = 0; q < 4; q++) {
        let x = bx + (q & 1); let y = by + (q >> 1);
        if (x >= W || y >= H) { continue; }
        let y8 = clamp(round(Ys[q]), 0.0, 255.0);
        let rgb = vec3f(y8 + 1.402 * (Cr - 128.0), y8 - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0), y8 + 1.772 * (Cb - 128.0));
        textureStore(outTex, vec2i(x, y), vec4f(clamp(round(rgb), vec3f(0.0), vec3f(255.0)) / 255.0, 1.0));
        yuv[u32(y * W + x)] = u32(y8);
        ls.x += u32(y8);
      }
      yuv[u32(W * H) + bi] = u32(Cb);
      yuv[u32(W * H) + cw * ((u32(H) + 1u) / 2u) + bi] = u32(Cr);
    }
  }
  wsum[li] = ls;
  workgroupBarrier();
  if (li == 0u) {
    var t = vec4u(0u);
    for (var k = 0u; k < 64u; k++) { t += wsum[k]; }
    atomicAdd(&stats[0], t.x); atomicAdd(&stats[1], t.y); atomicAdd(&stats[2], t.z); atomicAdd(&stats[3], t.w);
  }
}
`;

// ------------------------------------------------------------------ present (sensor texture -> canvas)
export const PRESENT = /* wgsl */ `
struct PU { rect: vec4f }    // destination rect in the canvas (x, y, w, h) px
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> pu: PU;
struct POut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vsP(@builtin(vertex_index) i: u32) -> POut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: POut;
  o.pos = vec4f(p[i], 0.0, 1.0);
  o.uv = vec2f(0.5 * (p[i].x + 1.0), 0.5 * (1.0 - p[i].y));
  return o;
}
@fragment fn fsP(v: POut) -> @location(0) vec4f { return vec4f(textureSampleLevel(src, smp, v.uv, 0.0).xyz, 1.0); }
`;
