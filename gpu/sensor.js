// Sensor model parameters (docs/WEB_TWIN.md §9) from physics.yaml -> `sensor`, `isp`, plus the UVC-like auto-
// exposure loop and the per-camera relative-illumination map. Pure ES module; the GPU stages are in
// wgsl/sensor.wgsl.js. Every number is a parameter: `sensorParams(physics, overrides)` reads physics.yaml and lets the
// caller override any field; `describe()` reports each value with its status and source.

import { lensParams, unproject, distort, distortDeriv } from './lens.js';

const unwrap = (x) => (x && typeof x === 'object' && !Array.isArray(x) && 'value' in x ? x.value : x);
const meta = (x) => (x && typeof x === 'object' && 'value' in x ? { status: x.status, source: x.source } : { status: 'default', source: 'gpu/sensor.js' });

const H = 6.62607015e-34, C = 299792458;
const CHANNEL_NM = [600, 540, 460];                 // representative wavelengths of the R, G, B channels
const CFA = { RGGB: 0, GRBG: 1, GBRG: 2, BGGR: 3 };
export const ANALOG_GAIN_MAX = 8;                   // assumed: UVC gain 0..100 maps linearly onto 1..8x

/** Sensor + ISP parameters with provenance. `over` overrides any field of the result. */
export function sensorParams(physics = {}, over = {}) {
  const s = physics.sensor || {}, isp = physics.isp || {}, lens = s.lens || {};
  const prov = {};
  const take = (key, x, dflt) => { prov[key] = meta(x); const v = unwrap(x); return v === undefined || v === null ? dflt : v; };
  const pitch = take('pixel_pitch', s.pixel_pitch, 3e-6);
  const qe = take('qe_peak', s.qe_peak, [0.45, 0.55, 0.5]);
  let M = take('response_matrix', s.response_matrix?.M, null);
  const placeholder = !M || /placeholder/i.test(String(s.response_matrix?.M?.source || '')) || Math.max(...M.flat().map(Math.abs)) < 1e3;
  if (placeholder) {
    // photoelectrons per second per pixel per W m^-2: pixel area x QE x photons per joule at the channel wavelength
    const g = CHANNEL_NM.map((nm, c) => pitch * pitch * qe[c] * (nm * 1e-9) / (H * C));
    M = [[g[0], 0, 0], [0, g[1], 0], [0, 0, g[2]]];
    prov.response_matrix = { status: 'derived', source: 'gpu/sensor.js: pixel_pitch^2 x qe_peak x lambda/(h c) (physics M is a placeholder)' };
  }
  const uvc = s.uvc || {}, ae = s.auto_exposure || {};
  const p = {
    responseMatrix: M,
    fNumber: take('f_number', lens.f_number, 2.0),
    transmission: take('transmission', lens.transmission, 0.9),
    pupilExponent: take('pupil_obliquity_exponent', lens.relative_illumination?.pupil_obliquity_exponent, 1.0),
    vignettingThetaDeg: take('vignetting_theta_deg', lens.relative_illumination?.vignetting_theta_deg, [0, 90]),
    vignetting: take('vignetting', lens.relative_illumination?.vignetting, [1, 1]),
    esfSigmaPx: take('esf_sigma_px', lens.esf_sigma_px, 0.7),
    esfIncludesAperture: true,
    readNoise: take('read_noise', s.read_noise, 3.0),
    fullWell: take('full_well', s.full_well, 10000),
    darkCurrent: take('dark_current', s.dark_current, 0),
    conversionGain: take('conversion_gain_base', s.conversion_gain_base, 0.33),
    adcBits: take('adc_bits', s.adc_bits, 12),
    blackLevel: take('black_level', s.black_level, 0),
    cfa: take('bayer_order', s.bayer_order, 'GRBG'),
    exposureUnit: take('exposure_unit', uvc.exposure_unit, 1e-4),
    exposureRange: take('exposure_range', uvc.exposure_range, [1, 5000]),
    gainRange: take('gain_range', uvc.gain_range, [0, 100]),
    fps: take('frame_rate', uvc.frame_rate, 20),
    aeTarget: take('target_mean', ae.target_mean, 0.45),
    aeDamping: take('damping', ae.damping, 0.3),
    wb: take('white_balance_gains', isp.white_balance_gains, [1, 1, 1]),
    ccm: take('ccm', isp.ccm, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    tone: take('tone_curve', isp.tone_curve, 'srgb'),
    sCurveC: take('s_curve_c', isp.s_curve_c, null),
    blackOffset: take('black_offset_linear', isp.black_offset_linear, 0),
    demosaic: take('demosaic', isp.demosaic, 'bilinear'),
    noiseReduction: take('noise_reduction', isp.noise_reduction, 'none'),
    nrSigmaPx: take('noise_reduction_sigma_px', isp.noise_reduction_sigma_px, 0.4),
    nrRangeScale: 2.0,
    glareFraction: take('veiling_glare.fraction', lens.veiling_glare?.fraction, 0),
    glareSigmaFrac: take('veiling_glare.sigma_frac_diag', lens.veiling_glare?.sigma_frac_diag, 0.25),
    glareLevel: 16,
    gamma: take('gamma', isp.gamma, 2.2),
    sharpenAmount: take('sharpening.amount', isp.sharpening?.amount, 0),
    sharpenRadius: take('sharpening.radius_px', isp.sharpening?.radius_px, 1),
    chroma: take('chroma_subsampling', isp.chroma_subsampling, '4:2:0'),
    aeMaxStep: take('max_step', ae.max_step, 2.0),
    awb: !!take('white_balance_auto_default', uvc.white_balance_auto_default, true),
    awbDamping: 0.3,
    noise: true,
    analogGainMax: (take('analog_gain_range', s.analog_gain_range, [1, ANALOG_GAIN_MAX]) || [1, ANALOG_GAIN_MAX])[1],
    ...over,
  };
  prov.nrRangeScale = { status: 'assumed', source: 'gpu/sensor.js: bilateral range sigma = 2 x modelled noise (the fit gives only the spatial sigma)' };
  prov.glareLevel = { status: 'assumed', source: 'gpu/sensor.js: glare convolved on a 1/16 pyramid level (sigma 0.25 x diagonal = 367 px is smooth at that scale)' };
  prov.gainMapping = { status: 'assumed', source: 'gpu/sensor.js: UVC gain 0..100 -> analog_gain_range linear (no camera setting yet)' };
  prov.awbMethod = { status: 'assumed', source: 'gpu/sensor.js: grey-world auto white balance on the raw channel means, damped 0.3/frame (UVC auto WB on)' };
  p.exposureUnits0 = unwrap(uvc.exposure_default) ?? 10;
  p.provenance = prov;
  return p;
}

/** Status counts, for the judge's report. */
export function describe(p) {
  const counts = {};
  for (const v of Object.values(p.provenance)) counts[v.status || 'unknown'] = (counts[v.status || 'unknown'] || 0) + 1;
  return { counts, provenance: p.provenance };
}

export const analogGain = (p, uvcGain) => {
  const [g0, g1] = p.gainRange;
  return 1 + (Math.min(g1, Math.max(g0, uvcGain)) - g0) / Math.max(1, g1 - g0) * (p.analogGainMax - 1);
};

/**
 * Relative illumination x vignetting per pixel (Float32Array w*h), from the KB4 lens:
 * RI = cos^q(theta) sin(theta) / (theta_d dtheta_d/dtheta) (physics.yaml lens.relative_illumination, = cos^4 for a
 * pinhole mapping), times the vignetting table (linear in theta).
 */
export function relativeIlluminationMap(lens, p, size = lens.size) {
  const L = lens.fx ? lens : lensParams(lens);
  const [w, h] = size;
  const out = new Float32Array(w * h);
  const vt = p.vignettingThetaDeg.map((d) => (d * Math.PI) / 180), vv = p.vignetting;
  const vig = (th) => {
    if (th <= vt[0]) return vv[0];
    for (let i = 1; i < vt.length; i++) if (th <= vt[i]) return vv[i - 1] + (vv[i] - vv[i - 1]) * (th - vt[i - 1]) / (vt[i] - vt[i - 1]);
    return vv[vv.length - 1];
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const r = unproject(L, [x, y]);
    const th = Math.acos(Math.max(-1, Math.min(1, r[2])));
    let ri = 1;
    if (th > 1e-6) {
      const td = distort(L, th), dd = distortDeriv(L, th);
      ri = Math.pow(Math.cos(th), p.pupilExponent) * Math.sin(th) / (td * dd);
    }
    out[y * w + x] = Math.max(0, ri) * vig(th);
  }
  return out;
}

/** UVC-style auto exposure: exposure time first (whole 100 us units, up to the frame time), then analogue gain. */
export class AutoExposure {
  constructor(p, { exposureUnits = 10, gainUnits = 0, enabled = true } = {}) {
    this.p = p; this.units = exposureUnits; this.gainUnits = gainUnits; this.enabled = enabled;
    this.product = this.units * p.exposureUnit * analogGain(p, gainUnits);
  }
  get exposure() { return this.units * this.p.exposureUnit; }
  get gain() { return analogGain(this.p, this.gainUnits); }
  /** Feed the mean output luma (0..1) of the last frame; returns the new {exposure, gain}. */
  update(meanLuma) {
    if (!this.enabled) return { exposure: this.exposure, gain: this.gain };
    const p = this.p;
    const gam = p.tone === 'linear' ? 1 : 2.2;
    let ratio = meanLuma > 1e-4 ? Math.pow(p.aeTarget / meanLuma, gam) : 8;
    ratio = Math.min(8, Math.max(1 / 8, ratio));
    const step = Math.min(p.aeMaxStep || 2, Math.max(1 / (p.aeMaxStep || 2), Math.pow(ratio, p.aeDamping)));
    this.product *= step;
    const unit = p.exposureUnit;
    const maxUnits = Math.min(p.exposureRange[1], Math.floor(1 / (p.fps * unit)));
    const want = this.product / unit;
    this.units = Math.max(p.exposureRange[0], Math.min(maxUnits, Math.round(want)));
    const g = this.product / (this.units * unit);
    // invert the linear gain map
    const [g0, g1] = p.gainRange;
    const gu = g0 + (Math.max(1, Math.min(p.analogGainMax, g)) - 1) / (p.analogGainMax - 1) * (g1 - g0);
    this.gainUnits = Math.round(gu);
    this.product = Math.max(p.exposureRange[0] * unit, Math.min(maxUnits * unit * p.analogGainMax, this.product));
    return { exposure: this.exposure, gain: this.gain };
  }
}

/** Pack the sensor uniform (Float32/Uint32 views over 16 vec4 = 256 bytes). */
export function packSensorUniform(p, { width, height, frame = 0, seed = 0, exposure, gain, camIndex = 0 }) {
  const buf = new ArrayBuffer(256), f = new Float32Array(buf), u = new Uint32Array(buf);
  u[0] = width; u[1] = height; u[2] = CFA[p.cfa] ?? 1; u[3] = frame >>> 0;
  // physics' esf_sigma_px is optics + pixel aperture (doc §8.4); the tracer already box-integrates each pixel
  // (aperture variance 1/12 px^2), so the optics stage adds only the remainder
  const sOpt = p.esfIncludesAperture ? Math.sqrt(Math.max(0, p.esfSigmaPx * p.esfSigmaPx - 1 / 12)) : p.esfSigmaPx;
  f[4] = Math.PI * p.transmission / (4 * p.fNumber * p.fNumber); f[5] = sOpt; f[6] = exposure; f[7] = gain;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) f[8 + 4 * r + c] = p.responseMatrix[r][c];
  f[20] = p.readNoise; f[21] = p.fullWell; f[22] = p.darkCurrent; f[23] = p.conversionGain;
  f[24] = Math.pow(2, p.adcBits) - 1; f[25] = p.blackLevel; u[26] = seed >>> 0; f[27] = p.noise ? 1 : 0;
  f[28] = p.wb[0]; f[29] = p.wb[1]; f[30] = p.wb[2];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) f[32 + 4 * r + c] = p.ccm[r][c];
  f[44] = p.tone === 'srgb' ? (p.sCurveC ? 3 : 0) : p.tone === 'linear' ? 2 : 1; f[45] = p.gamma; f[46] = p.sharpenAmount; f[47] = p.sharpenRadius;
  u[48] = String(p.chroma).startsWith('4:2:0') ? 1 : 0; u[49] = camIndex;
  u[50] = /hamilton/i.test(String(p.demosaic)) ? 2 : /edge/.test(String(p.demosaic)) ? 1 : 0; u[51] = /edge|bilateral|on/.test(String(p.noiseReduction)) ? 1 : 0;
  const F = p.glareLevel || 16;
  f[52] = p.glareFraction || 0; f[53] = (p.glareSigmaFrac * Math.hypot(width, height)) / F; f[54] = F;
  f[56] = p.sCurveC || 1; f[57] = p.blackOffset || 0; f[58] = p.nrSigmaPx || 0; f[59] = p.nrRangeScale || 2;
  return buf;
}

/** Modelled noise std (in DN) of one raw pixel at a mean signal of `e` electrons: shot + read, times gain. */
export function rawNoiseDN(p, e, gain) {
  return Math.sqrt(Math.max(0, e) + p.readNoise * p.readNoise) * p.conversionGain * gain;
}

/** Grey-world auto white balance: gains that equalise the raw channel means, damped. */
export function updateAWB(prev, meanRGB, damping = 0.3) {
  const g = prev || [1, 1, 1];
  if (!(meanRGB[0] > 1e-4 && meanRGB[1] > 1e-4 && meanRGB[2] > 1e-4)) return g;
  const target = [meanRGB[1] / meanRGB[0], 1, meanRGB[1] / meanRGB[2]];
  return g.map((x, c) => Math.exp(Math.log(x) + damping * (Math.log(target[c]) - Math.log(x))));
}
