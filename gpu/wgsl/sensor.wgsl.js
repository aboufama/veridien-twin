// WGSL sensor chain (docs/WEB_TWIN.md §9), radiance -> 8-bit camera RGB, every number a uniform (gpu/sensor.js):
//   sensorOptics    image irradiance E = L pi T / (4 N^2) RI(theta) V(theta) (RI map per camera), horizontal half of
//                   the optics blur (Gaussian, sigma = sensor.lens.esf_sigma_px)
//   glareDown/BlurH/BlurV  veiling glare on a 1/F pyramid level: E (x) G, G Gaussian of sigma = 0.25 x the diagonal
//                   (zero outside the sensor: scattered light that leaves the array is lost)
//   sensorExpose    vertical blur, E' = (1 - k) E + k (E (x) G), electrons = M E' t_exp + dark per CFA site, shot noise
//                   (Poisson; normal above 64 e-), read noise, full-well clip, gain, black level, ADC -> raw DN
//   demosaicG       green plane: edge-aware (as OpenCV's EA: interpolate along the smaller green gradient), Hamilton-
//                   Adams (the same with the site's own second-derivative correction), or bilinear
//   demosaicRB      red and blue: bilinear (bilinear and edge-aware modes, as OpenCV's EA) or by colour-difference
//                   interpolation on the green (Hamilton-Adams mode); white balance, CCM; AWB sums
//   sensorNR        edge-preserving noise reduction: 3x3 bilateral, spatial sigma (isp.noise_reduction_sigma_px), range
//                   sigma = scale x the modelled shot + read noise at the pixel's level
//   sensorOutput    tone (sRGB, or sRGB + contrast S-curve with black offset, or gamma), optional unsharp mask,
//                   BT.601 full-range YCbCr, 4:2:0 chroma (2x2 mean), 8-bit RGB (nearest chroma) + Y/Cb/Cr planes,
//                   luma sum for AE
// The noise is a hash of (pixel, frame, seed, camera): deterministic per frame, independent between frames.
// Every loop is bounded by a constant (docs/WEB_TWIN.md §12).
export const SENSOR = /* wgsl */ `
struct SensU {
  size: vec4u,          // w, h, CFA code (0 RGGB, 1 GRBG, 2 GBRG, 3 BGGR), frame
  optics: vec4f,        // pi T / (4 N^2), ESF sigma (px), exposure (s), analogue gain
  m0: vec4f, m1: vec4f, m2: vec4f,   // response matrix rows: e-/s per W m^-2 (camera channel = M @ linear sRGB)
  noise: vec4f,         // read noise (e-), full well (e-), dark current (e-/s), conversion gain (LSB/e-)
  adc: vec4f,           // max DN, black level (DN), seed, noise on (1) / off (0)
  wb: vec4f,            // white-balance gains rgb, -
  c0: vec4f, c1: vec4f, c2: vec4f,   // CCM rows
  tone: vec4f,          // mode (0 sRGB, 1 gamma, 2 linear, 3 sRGB + S-curve), gamma, sharpening amount, radius (px)
  flags: vec4u,         // 4:2:0 on, camera index, demosaic (0 bilinear, 1 edge-aware (OpenCV EA), 2 Hamilton-Adams), NR on
  glare: vec4f,         // fraction k, sigma at the pyramid level (px), level factor F, -
  tone2: vec4f,         // S-curve c, black offset b (fraction of white), NR spatial sigma (px), NR range scale
}

@group(0) @binding(0) var<uniform> su: SensU;
@group(0) @binding(1) var<storage, read> riMap: array<f32>;
@group(0) @binding(2) var srcTex: texture_2d<f32>;
@group(0) @binding(3) var tmpA: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> raw: array<f32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(6) var dstTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var<storage, read_write> yuv: array<u32>;
@group(0) @binding(8) var<storage, read_write> gpl: array<f32>;
@group(0) @binding(9) var glareIn: texture_2d<f32>;
@group(0) @binding(10) var glareOut: texture_storage_2d<rgba32float, write>;

fn pcgS(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn uS(x: u32) -> f32 { return (f32(x >> 8u) + 0.5) * (1.0 / 16777216.0); }

fn gaussTaps(sigma: f32) -> i32 { return clamp(i32(ceil(3.0 * sigma + 0.5)), 0, 12); }

// weight of tap k of a Gaussian integrated over the pixel [k - 1/2, k + 1/2] (a sampled Gaussian of sigma < 0.5 px
// would be almost a delta); erf by Abramowitz-Stegun 7.1.26
fn erfA(z: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * abs(z));
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return select(-y, y, z >= 0.0);
}
fn tapW(k: i32, sg: f32) -> f32 {
  if (sg <= 0.0) { return select(0.0, 1.0, k == 0); }
  let s2 = sg * 1.41421356;
  return 0.5 * (erfA((f32(k) + 0.5) / s2) - erfA((f32(k) - 0.5) / s2));
}

@compute @workgroup_size(8, 8)
fn sensorOptics(@builtin(global_invocation_id) gid: vec3u) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  let sg = su.optics.y;
  let r = gaussTaps(sg);
  var acc = vec3f(0.0);
  var ws = 0.0;
  for (var k = -r; k <= r; k++) {
    let xx = clamp(x + k, 0, W - 1);
    let w = tapW(k, sg);
    acc = acc + w * textureLoad(srcTex, vec2i(xx, y), 0).xyz * riMap[u32(y * W + xx)];
    ws = ws + w;
  }
  textureStore(tmpA, vec2i(x, y), vec4f(acc * (su.optics.x / ws), 1.0));
}

// ---------------------------------------------------------------- veiling glare (pyramid level 1/F)
@compute @workgroup_size(8, 8)
fn glareDown(@builtin(global_invocation_id) gid: vec3u) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let F = i32(su.glare.z);
  let gw = (W + F - 1) / F; let gh = (H + F - 1) / F;
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= gw || y >= gh) { return; }
  var acc = vec3f(0.0);
  var n = 0.0;
  for (var j = 0; j < min(F, 32); j++) {
    for (var i = 0; i < min(F, 32); i++) {
      let xx = x * F + i; let yy = y * F + j;
      if (xx >= W || yy >= H) { continue; }
      acc = acc + textureLoad(srcTex, vec2i(xx, yy), 0).xyz * riMap[u32(yy * W + xx)];
      n = n + 1.0;
    }
  }
  textureStore(glareOut, vec2i(x, y), vec4f(acc * (su.optics.x / max(n, 1.0)), 1.0));
}

// a Gaussian over the whole row (dir 0) or column (dir 1) of the level, zero outside, normalised on the infinite line
fn glareBlur(gid: vec3u, dir: u32) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let F = i32(su.glare.z);
  let gw = (W + F - 1) / F; let gh = (H + F - 1) / F;
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= gw || y >= gh) { return; }
  let sg = max(su.glare.y, 1e-3);
  let n = select(gh, gw, dir == 0u);
  var acc = vec3f(0.0);
  for (var k = 0; k < min(n, 256); k++) {
    let d = f32(k - select(y, x, dir == 0u));
    let w = exp(-0.5 * d * d / (sg * sg));
    acc = acc + w * textureLoad(glareIn, select(vec2i(x, k), vec2i(k, y), dir == 0u), 0).xyz;
  }
  textureStore(glareOut, vec2i(x, y), vec4f(acc / (sg * 2.5066282746), 1.0));
}
@compute @workgroup_size(8, 8)
fn glareBlurH(@builtin(global_invocation_id) gid: vec3u) { glareBlur(gid, 0u); }
@compute @workgroup_size(8, 8)
fn glareBlurV(@builtin(global_invocation_id) gid: vec3u) { glareBlur(gid, 1u); }

fn glareAt(x: i32, y: i32) -> vec3f {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let F = su.glare.z;
  let gw = (W + i32(F) - 1) / i32(F); let gh = (H + i32(F) - 1) / i32(F);
  let p = vec2f((f32(x) + 0.5) / F - 0.5, (f32(y) + 0.5) / F - 0.5);
  let p0 = vec2i(floor(p));
  let f = p - floor(p);
  let a = textureLoad(glareIn, clamp(p0, vec2i(0), vec2i(gw - 1, gh - 1)), 0).xyz;
  let b = textureLoad(glareIn, clamp(p0 + vec2i(1, 0), vec2i(0), vec2i(gw - 1, gh - 1)), 0).xyz;
  let c = textureLoad(glareIn, clamp(p0 + vec2i(0, 1), vec2i(0), vec2i(gw - 1, gh - 1)), 0).xyz;
  let d = textureLoad(glareIn, clamp(p0 + vec2i(1, 1), vec2i(0), vec2i(gw - 1, gh - 1)), 0).xyz;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn cfaChannel(x: u32, y: u32) -> u32 {
  // 2x2 pattern codes: RGGB, GRBG, GBRG, BGGR (row-major); 0 = R, 1 = G, 2 = B
  let k = (y & 1u) * 2u + (x & 1u);
  let c = su.size.z;
  var pat = vec4u(0u, 1u, 1u, 2u);
  if (c == 1u) { pat = vec4u(1u, 0u, 2u, 1u); }
  if (c == 2u) { pat = vec4u(1u, 2u, 0u, 1u); }
  if (c == 3u) { pat = vec4u(2u, 1u, 1u, 0u); }
  return pat[k];
}

fn gauss(h: u32) -> f32 {
  let u1 = uS(pcgS(h));
  let u2 = uS(pcgS(h ^ 0x68bc21ebu));
  return sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
}

fn poisson(lam: f32, h: u32) -> f32 {
  if (lam <= 0.0) { return 0.0; }
  if (lam > 64.0) { return max(0.0, floor(lam + sqrt(lam) * gauss(h) + 0.5)); }
  // inversion by sequential search (exact), bounded
  let u = uS(pcgS(h ^ 0x1b873593u));
  var p = exp(-lam);
  var F = p;
  var k = 0.0;
  for (var i = 0; i < 200; i++) {
    if (u <= F) { break; }
    k = k + 1.0;
    p = p * lam / k;
    F = F + p;
  }
  return k;
}

@compute @workgroup_size(8, 8)
fn sensorExpose(@builtin(global_invocation_id) gid: vec3u) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  let sg = su.optics.y;
  let r = gaussTaps(sg);
  var E = vec3f(0.0);
  var ws = 0.0;
  for (var k = -r; k <= r; k++) {
    let yy = clamp(y + k, 0, H - 1);
    let w = tapW(k, sg);
    E = E + w * textureLoad(srcTex, vec2i(x, yy), 0).xyz;
    ws = ws + w;
  }
  E = E / ws;
  if (su.glare.x > 0.0) { E = (1.0 - su.glare.x) * E + su.glare.x * glareAt(x, y); }
  let ch = cfaChannel(u32(x), u32(y));
  var row = su.m0;
  if (ch == 1u) { row = su.m1; }
  if (ch == 2u) { row = su.m2; }
  let texp = su.optics.z;
  let lam = max(0.0, dot(row.xyz, E)) * texp + su.noise.z * texp;
  let h = pcgS((u32(y * W + x) * 0x9E3779B9u) ^ pcgS((su.size.w * 0x85ebca6bu) ^ pcgS(bitcast<u32>(su.adc.z) ^ (su.flags.y * 0xc2b2ae35u))));
  var e = lam;
  if (su.adc.w > 0.5) {
    e = poisson(lam, h) + su.noise.x * gauss(h ^ 0x27d4eb2fu);
  }
  e = min(e, su.noise.y);
  let dn = clamp(floor(e * su.noise.w * su.optics.w + su.adc.y + 0.5), 0.0, su.adc.x);
  raw[u32(y * W + x)] = dn;
}

fn rawAt(x: i32, y: i32) -> f32 {
  let W = i32(su.size.x); let H = i32(su.size.y);
  // mirror at the borders, keeping the CFA phase (reflect by 2)
  var xx = x; var yy = y;
  if (xx < 0) { xx = -xx; } if (xx >= W) { xx = 2 * (W - 1) - xx; }
  if (yy < 0) { yy = -yy; } if (yy >= H) { yy = 2 * (H - 1) - yy; }
  return raw[u32(clamp(yy, 0, H - 1) * W + clamp(xx, 0, W - 1))];
}
fn gAt(x: i32, y: i32) -> f32 {
  let W = i32(su.size.x); let H = i32(su.size.y);
  return gpl[u32(clamp(y, 0, H - 1) * W + clamp(x, 0, W - 1))];
}

// green plane (DN): at G sites the sample; elsewhere Hamilton-Adams (edge-aware) or the 4-neighbour mean (bilinear)
@compute @workgroup_size(8, 8)
fn demosaicG(@builtin(global_invocation_id) gid: vec3u) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  let ch = cfaChannel(u32(x), u32(y));
  var g = rawAt(x, y);
  if (ch != 1u) {
    let gl = rawAt(x - 1, y); let gr = rawAt(x + 1, y); let gu = rawAt(x, y - 1); let gd = rawAt(x, y + 1);
    if (su.flags.z == 0u) { g = 0.25 * (gl + gr + gu + gd); }
    else if (su.flags.z == 1u) {
      // OpenCV-style edge-aware (cv2 COLOR_Bayer*_EA): average along the direction with the smaller green gradient
      let dh = abs(gl - gr);
      let dv = abs(gu - gd);
      if (dh < dv) { g = 0.5 * (gl + gr); } else if (dv < dh) { g = 0.5 * (gu + gd); } else { g = 0.25 * (gl + gr + gu + gd); }
    } else {
      // Hamilton-Adams: the same choice with the site's own second derivative (sharper)
      let c = rawAt(x, y);
      let lh = 2.0 * c - rawAt(x - 2, y) - rawAt(x + 2, y);
      let lv = 2.0 * c - rawAt(x, y - 2) - rawAt(x, y + 2);
      let dh = abs(gl - gr) + abs(lh);
      let dv = abs(gu - gd) + abs(lv);
      let gh = 0.5 * (gl + gr) + 0.25 * lh;
      let gv = 0.5 * (gu + gd) + 0.25 * lv;
      if (dh < dv) { g = gh; } else if (dv < dh) { g = gv; } else { g = 0.5 * (gh + gv); }
      g = clamp(g, 0.0, su.adc.x);
    }
  }
  gpl[u32(y * W + x)] = g;
}

var<workgroup> wR: array<u32, 64>;
var<workgroup> wG: array<u32, 64>;
var<workgroup> wB: array<u32, 64>;

// red and blue by colour-difference interpolation (c - G) over their nearest sites, then normalise, AWB sums
// (x 1023, integers: deterministic), white balance and CCM -> linear camera RGB
@compute @workgroup_size(8, 8)
fn demosaicRB(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let x = i32(gid.x); let y = i32(gid.y);
  wR[li] = 0u; wG[li] = 0u; wB[li] = 0u;
  if (x < W && y < H) {
    let own = cfaChannel(u32(x), u32(y));
    let g = gAt(x, y);
    var sum = vec2f(0.0);         // (R - G), (B - G)
    var cnt = vec2f(0.0);
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let xx = clamp(x + dx, 0, W - 1); let yy = clamp(y + dy, 0, H - 1);
        let ch = cfaChannel(u32(xx), u32(yy));
        if (ch == 1u) { continue; }
        if (ch == own && !(dx == 0 && dy == 0)) { continue; }
        let d = rawAt(xx, yy) - gAt(xx, yy);
        if (ch == 0u) { sum.x += d; cnt.x += 1.0; } else { sum.y += d; cnt.y += 1.0; }
      }
    }
    var rgb = vec3f(g + sum.x / max(cnt.x, 1.0), g, g + sum.y / max(cnt.y, 1.0));
    if (su.flags.z <= 1u) {
      // bilinear R and B (plain means of each channel's sites); mode 1 keeps the edge-aware green (as OpenCV EA)
      var s3 = vec3f(0.0); var c3 = vec3f(0.0);
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let xx = clamp(x + dx, 0, W - 1); let yy = clamp(y + dy, 0, H - 1);
          let ch = cfaChannel(u32(xx), u32(yy));
          if (ch == own && !(dx == 0 && dy == 0)) { continue; }
          if (ch == 1u && own == 1u && !(dx == 0 && dy == 0)) { continue; }
          let v = rawAt(xx, yy);
          if (ch == 0u) { s3.x += v; c3.x += 1.0; } else if (ch == 1u) { s3.y += v; c3.y += 1.0; } else { s3.z += v; c3.z += 1.0; }
        }
      }
      rgb = s3 / max(c3, vec3f(1.0));
      if (su.flags.z == 1u) { rgb.y = g; }
    }
    let full = max(su.adc.x - su.adc.y, 1.0);
    var c = (rgb - vec3f(su.adc.y)) / full;
    let q = vec3u(round(clamp(c, vec3f(0.0), vec3f(1.0)) * 1023.0));
    wR[li] = q.x; wG[li] = q.y; wB[li] = q.z;
    c = c * su.wb.xyz;
    c = vec3f(dot(su.c0.xyz, c), dot(su.c1.xyz, c), dot(su.c2.xyz, c));
    textureStore(tmpA, vec2i(x, y), vec4f(c, 1.0));
  }
  workgroupBarrier();
  if (li == 0u) {
    var r = 0u; var g = 0u; var b = 0u;
    for (var k = 0u; k < 64u; k++) { r += wR[k]; g += wG[k]; b += wB[k]; }
    atomicAdd(&stats[1], r); atomicAdd(&stats[2], g); atomicAdd(&stats[3], b);
  }
}

// edge-preserving noise reduction: 3x3 bilateral on linear RGB (range on luma, sigma = scale x modelled noise)
@compute @workgroup_size(8, 8)
fn sensorNR(@builtin(global_invocation_id) gid: vec3u) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  let c0 = textureLoad(srcTex, vec2i(x, y), 0).xyz;
  if (su.flags.w == 0u || su.tone2.z <= 0.0) { textureStore(tmpA, vec2i(x, y), vec4f(c0, 1.0)); return; }
  let Y0 = max(dot(c0, vec3f(0.2126, 0.7152, 0.0722)), 0.0);
  // electrons at full scale (normalised value 1) and the modelled noise there
  let fullE = max(su.adc.x - su.adc.y, 1.0) / max(su.noise.w * su.optics.w, 1e-9);
  let sn = sqrt(Y0 * fullE + su.noise.x * su.noise.x) / fullE;
  let rs = max(su.tone2.w * sn, 1e-6);
  let ss = su.tone2.z;
  var acc = vec3f(0.0);
  var ws = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let c = textureLoad(srcTex, vec2i(clamp(x + dx, 0, W - 1), clamp(y + dy, 0, H - 1)), 0).xyz;
      let dY = dot(c - c0, vec3f(0.2126, 0.7152, 0.0722));
      let w = exp(-0.5 * f32(dx * dx + dy * dy) / (ss * ss) - 0.5 * dY * dY / (rs * rs));
      acc = acc + w * c;
      ws = ws + w;
    }
  }
  textureStore(tmpA, vec2i(x, y), vec4f(acc / ws, 1.0));
}

fn toneC(x: f32) -> f32 {
  let m = su.tone.x;
  if (m > 2.5) {
    // sRGB + contrast S-curve: S_c(OETF((x - b)/(1 - b))), S_c(u) = u^c / (u^c + (1 - u)^c)
    let b = su.tone2.y;
    let v = clamp((x - b) / (1.0 - b), 0.0, 1.0);
    let u = select(1.055 * pow(v, 1.0 / 2.4) - 0.055, 12.92 * v, v <= 0.0031308);
    let cc = su.tone2.x;
    let a = pow(u, cc);
    return a / max(a + pow(1.0 - u, cc), 1e-12);
  }
  let c = clamp(x, 0.0, 1.0);
  if (m < 0.5) { return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308); }
  if (m < 1.5) { return pow(c, 1.0 / max(su.tone.y, 1e-3)); }
  return c;
}
fn toned(x: i32, y: i32) -> vec3f {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let c = textureLoad(srcTex, vec2i(clamp(x, 0, W - 1), clamp(y, 0, H - 1)), 0).xyz;
  return vec3f(toneC(c.x), toneC(c.y), toneC(c.z));
}

var<workgroup> wsum: array<u32, 64>;

// one thread per 2x2 block
@compute @workgroup_size(8, 8)
fn sensorOutput(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = i32(su.size.x); let H = i32(su.size.y);
  let bx = i32(gid.x) * 2; let by = i32(gid.y) * 2;
  var lsum = 0u;
  if (bx < W && by < H) {
    var Ys: array<f32, 4>;
    var cb = 0.0; var cr = 0.0; var nb = 0.0;
    let amt = su.tone.z;
    let rad = clamp(i32(round(su.tone.w)), 1, 4);
    for (var q = 0; q < 4; q++) {
      let x = bx + (q & 1); let y = by + (q >> 1);
      if (x >= W || y >= H) { Ys[q] = 0.0; continue; }
      var c = toned(x, y);
      if (amt > 0.0) {
        // unsharp mask on luma: Y += amt (Y - box mean)
        var m = 0.0; var n = 0.0;
        for (var dy = -rad; dy <= rad; dy++) { for (var dx = -rad; dx <= rad; dx++) {
          m += dot(toned(x + dx, y + dy), vec3f(0.299, 0.587, 0.114)); n += 1.0; } }
        let Y0 = dot(c, vec3f(0.299, 0.587, 0.114));
        c = c + vec3f(amt * (Y0 - m / n));
      }
      let c8 = clamp(c, vec3f(0.0), vec3f(1.0)) * 255.0;
      Ys[q] = 0.299 * c8.x + 0.587 * c8.y + 0.114 * c8.z;
      cb += 128.0 - 0.168736 * c8.x - 0.331264 * c8.y + 0.5 * c8.z;
      cr += 128.0 + 0.5 * c8.x - 0.418688 * c8.y - 0.081312 * c8.z;
      nb += 1.0;
      if (su.flags.x == 0u) {
        // 4:4:4: write this pixel's own chroma
        let y8 = clamp(round(Ys[q]), 0.0, 255.0);
        textureStore(dstTex, vec2i(x, y), vec4f(round(c8) / 255.0, 1.0));
        lsum += u32(y8);
      }
    }
    if (su.flags.x != 0u) {
      let Cb = clamp(round(cb / nb), 0.0, 255.0);
      let Cr = clamp(round(cr / nb), 0.0, 255.0);
      let cw = (u32(W) + 1u) / 2u;
      let bi = u32(gid.y) * cw + u32(gid.x);
      let planeY = u32(W * H);
      for (var q = 0; q < 4; q++) {
        let x = bx + (q & 1); let y = by + (q >> 1);
        if (x >= W || y >= H) { continue; }
        let y8 = clamp(round(Ys[q]), 0.0, 255.0);
        let rgb = vec3f(y8 + 1.402 * (Cr - 128.0), y8 - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0), y8 + 1.772 * (Cb - 128.0));
        textureStore(dstTex, vec2i(x, y), vec4f(clamp(round(rgb), vec3f(0.0), vec3f(255.0)) / 255.0, 1.0));
        yuv[u32(y * W + x)] = u32(y8);
        lsum += u32(y8);
      }
      yuv[planeY + bi] = u32(Cb);
      yuv[planeY + cw * ((u32(H) + 1u) / 2u) + bi] = u32(Cr);
    }
  }
  wsum[li] = lsum;
  workgroupBarrier();
  if (li == 0u) {
    var t = 0u;
    for (var k = 0u; k < 64u; k++) { t += wsum[k]; }
    atomicAdd(&stats[0], t);
  }
}
`;
