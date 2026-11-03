// The in-page ONNX slot (docs/WEB_TWIN.md §7, last paragraph): a model that runs in the page itself, takes the same
// camera images as a socket model and returns the same `predictions` object, so the server's scoring is identical.
//
// onnxruntime-web is imported lazily, only when a model is loaded, from jsDelivr at a pinned version (ORT_URL; MIT
// licence). The WebGPU execution provider is tried first, then WASM; `model.ep` records which one actually runs.
// Pass {ortUrl} to load it from elsewhere (offline mirror), or {ort} to inject a module (tests).
//
// ---------------------------------------------------------------------------------------------------------------
// MAPPING: how the model's tensors relate to the twin (DEFAULT_MAPPING below; override any field per model)
// ---------------------------------------------------------------------------------------------------------------
// Inputs
//   cameras     camera order. Default: calib.cameras order (the calibration message), else the images' keys sorted.
//   input.layout
//     "stacked"     one tensor named input.name, float32 [n_cams, 3, H, W] (NCHW), cameras in `cameras` order.
//     "per_camera"  one tensor per camera, named by the camera ("CAM1", ...), float32 [1, 3, H, W].
//   input.size  [W, H] the model expects. Default: the model's declared static input dims when ORT reports them,
//               else the camera image size (no resize).
//   Each image (rgb8, row-major, RGB order) is resized bilinearly with half-pixel centres (cv2.INTER_LINEAR on
//   float), scaled to [0, 1], then normalised per channel: (x/255 - mean[c]) / std[c]. input.order "BGR" swaps the
//   channel order (for models trained on cv2 images). A camera missing from `images` is fed as zeros.
// Outputs (names configurable in mapping.outputs; every output except tip_D is optional)
//   tip_D         float [n_instr, 3]   tool tip (midpoint of the jaw tips), metres, frame D (§2)
//   hinge_D       float [n_instr, 3]   jaw hinge, metres, frame D
//   shaft_dir_D   float [n_instr, 3]   unit shaft direction toward the tip, frame D (re-normalised here)
//   jaw           float [n_instr] or [n_instr, 1]   full jaw opening, radians
//   conf          float [n_instr] or [n_instr, 1]   confidence in [0, 1]
//   keypoints_px  float [n_cams, n_instr, n_kp, 2]  pixels; keypoint slots named by mapping.keypoints
//                 ("tip", "hinge", "jaw_a", "jaw_b"); keypoints_space "full" = full-resolution camera pixels,
//                 "input" = model-input pixels, scaled back here by the camera size / input size.
//                 Non-finite coordinates mean "not seen" and are left out.
//   instruments   instrument order of the n_instr rows. Default ["L", "R"] (fenestrated in port L, Maryland in R).
// Result: {type: "predictions", model: "onnx:<name>@<version>", frame_id?, instruments: [{id, tip_D, hinge_D?,
//          shaft_dir_D?, jaw?, conf?}], keypoints_px?: {CAM1: {L: {tip: [u, v], ...}}}}, exactly §7.3.
// ---------------------------------------------------------------------------------------------------------------

export const ORT_VERSION = '1.20.1';
export const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
export const ORT_URL = `${ORT_DIST}ort.webgpu.min.mjs`;

export const DEFAULT_MAPPING = Object.freeze({
  name: null,                 // default: the model file's base name
  version: '0',
  instruments: ['L', 'R'],
  cameras: null,
  input: { layout: 'stacked', name: 'images', size: null, mean: [0, 0, 0], std: [1, 1, 1], order: 'RGB' },
  outputs: {
    tip_D: 'tip_D', hinge_D: 'hinge_D', shaft_dir_D: 'shaft_dir_D', jaw: 'jaw', conf: 'conf', keypoints_px: 'keypoints_px',
  },
  keypoints: ['tip', 'hinge', 'jaw_a', 'jaw_b'],
  keypoints_space: 'full',
});

/** DEFAULT_MAPPING with `m` merged over it (one level deep for input/outputs). */
export function resolveMapping(m = {}) {
  return {
    ...DEFAULT_MAPPING, ...m,
    input: { ...DEFAULT_MAPPING.input, ...(m.input ?? {}) },
    outputs: { ...DEFAULT_MAPPING.outputs, ...(m.outputs ?? {}) },
  };
}

/** Camera order: mapping.cameras, else the calibration's, else the images' keys sorted. */
export function cameraOrder(mapping, calib, images) {
  if (mapping.cameras) return [...mapping.cameras];
  if (calib?.cameras?.length) return calib.cameras.map((c) => c.name);
  return Object.keys(images ?? {}).sort();
}

/**
 * One rgb8 image -> normalised CHW float32, bilinear with half-pixel centres (cv2.INTER_LINEAR on float images).
 * Writes 3*H*W values into `out` at `offset` and returns `out`.
 */
export function toCHW(rgb, w, h, W, H, { mean = [0, 0, 0], std = [1, 1, 1], order = 'RGB' } = {}, out = new Float32Array(3 * W * H), offset = 0) {
  const plane = W * H;
  const sx = w / W, sy = h / H;
  const ch = order === 'BGR' ? [2, 1, 0] : [0, 1, 2];
  // output channel k reads source channel ch[k] and is normalised with mean[k] / std[k]
  const scale = [0, 1, 2].map((k) => 1 / (255 * std[k]));
  const bias = [0, 1, 2].map((k) => -mean[k] / std[k]);
  for (let y = 0; y < H; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0;
    let y0 = Math.floor(fy);
    if (y0 > h - 1) y0 = h - 1;
    const y1 = Math.min(y0 + 1, h - 1);
    const ty = Math.min(1, fy - y0);
    for (let x = 0; x < W; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0;
      let x0 = Math.floor(fx);
      if (x0 > w - 1) x0 = w - 1;
      const x1 = Math.min(x0 + 1, w - 1);
      const tx = Math.min(1, fx - x0);
      const i00 = (y0 * w + x0) * 3, i01 = (y0 * w + x1) * 3, i10 = (y1 * w + x0) * 3, i11 = (y1 * w + x1) * 3;
      const o = offset + y * W + x;
      for (let k = 0; k < 3; k++) {
        const c = ch[k];
        const top = rgb[i00 + c] + (rgb[i01 + c] - rgb[i00 + c]) * tx;
        const bot = rgb[i10 + c] + (rgb[i11 + c] - rgb[i10 + c]) * tx;
        out[o + k * plane] = (top + (bot - top) * ty) * scale[k] + bias[k];
      }
    }
  }
  return out;
}

/**
 * Images -> input tensors as plain {name: {data: Float32Array, dims}} (wrapped in ort.Tensor by the caller).
 * @param {Object<string, {rgb: Uint8Array, width: number, height: number}>} images
 * @param {[number, number]} size  [W, H] the model expects
 */
export function prepareInputs(images, mapping, calib, size) {
  const m = resolveMapping(mapping);
  const cams = cameraOrder(m, calib, images);
  const [W, H] = size;
  const per = 3 * W * H;
  const fill = (img, out, off) => {
    if (img) toCHW(img.rgb, img.width, img.height, W, H, m.input, out, off);
    else out.fill(0, off, off + per);              // camera missing from this bundle: zeros
  };
  if (m.input.layout === 'stacked') {
    const data = new Float32Array(cams.length * per);
    cams.forEach((c, i) => fill(images[c], data, i * per));
    return { [m.input.name]: { data, dims: [cams.length, 3, H, W] } };
  }
  if (m.input.layout === 'per_camera') {
    const out = {};
    for (const c of cams) {
      const data = new Float32Array(per);
      fill(images[c], data, 0);
      out[c] = { data, dims: [1, 3, H, W] };
    }
    return out;
  }
  throw new RangeError(`unknown input layout ${m.input.layout}`);
}

const vec3 = (d, i) => [d[3 * i], d[3 * i + 1], d[3 * i + 2]];

function perInstrument(t, n, name) {
  if (!t) return null;
  const ok = (t.dims.length === 1 && t.dims[0] === n) || (t.dims.length === 2 && t.dims[0] === n && t.dims[1] === 1);
  if (!ok) throw new RangeError(`output ${name}: dims [${t.dims}] are not [${n}] or [${n}, 1]`);
  return t.data;
}

function vecs(t, n, name) {
  if (!t) return null;
  if (t.dims.length !== 2 || t.dims[0] !== n || t.dims[1] !== 3) throw new RangeError(`output ${name}: dims [${t.dims}] are not [${n}, 3]`);
  return t.data;
}

/**
 * The model's raw outputs -> the §7.3 predictions object. Pure: no onnxruntime needed.
 * @param {Object<string, {data: ArrayLike<number>, dims: number[]}>} outputs
 * @param {object} mapping  (merged over DEFAULT_MAPPING)
 * @param {object} calib    the calibration message (camera order and sizes)
 * @param {{inputSize?: [number, number], images?: object}} ctx  input size, for keypoints_space "input"
 */
export function mapOutputs(outputs, mapping, calib, ctx = {}) {
  const m = resolveMapping(mapping);
  const o = m.outputs;
  const tip = outputs[o.tip_D];
  if (!tip) throw new RangeError(`the model has no ${o.tip_D} output (outputs: ${Object.keys(outputs).join(', ')})`);
  if (tip.dims.length !== 2 || tip.dims[1] !== 3) throw new RangeError(`output ${o.tip_D}: dims [${tip.dims}] are not [n_instr, 3]`);
  const n = tip.dims[0];
  if (n > m.instruments.length) throw new RangeError(`${n} instruments out but the mapping names ${m.instruments.length}`);
  const hinge = vecs(outputs[o.hinge_D], n, o.hinge_D);
  const shaft = vecs(outputs[o.shaft_dir_D], n, o.shaft_dir_D);
  const jaw = perInstrument(outputs[o.jaw], n, o.jaw);
  const conf = perInstrument(outputs[o.conf], n, o.conf);
  const instruments = [];
  for (let i = 0; i < n; i++) {
    const r = { id: m.instruments[i], tip_D: vec3(tip.data, i) };
    if (hinge) r.hinge_D = vec3(hinge, i);
    if (shaft) {
      const v = vec3(shaft, i);
      const nn = Math.hypot(...v);
      r.shaft_dir_D = nn > 0 ? v.map((x) => x / nn) : v;
    }
    if (jaw) r.jaw = jaw[i];
    if (conf) r.conf = conf[i];
    instruments.push(r);
  }
  const pred = { type: 'predictions', model: `onnx:${m.name ?? 'model'}@${m.version}`, instruments };
  const kp = outputs[o.keypoints_px];
  if (kp) {
    const cams = cameraOrder(m, calib, ctx.images);
    const [nc, ni, nk, two] = kp.dims;
    if (kp.dims.length !== 4 || two !== 2 || ni !== n || nc !== cams.length || nk > m.keypoints.length) {
      throw new RangeError(`output ${o.keypoints_px}: dims [${kp.dims}] are not [${cams.length}, ${n}, n_kp<=${m.keypoints.length}, 2]`);
    }
    const out = {};
    cams.forEach((cam, c) => {
      let sx = 1, sy = 1;
      if (m.keypoints_space === 'input') {
        const size = calib?.cameras?.find((k) => k.name === cam)?.size ?? [ctx.images?.[cam]?.width, ctx.images?.[cam]?.height];
        if (!ctx.inputSize || !size?.[0]) throw new RangeError('keypoints_space "input" needs the input size and the camera size');
        sx = size[0] / ctx.inputSize[0];
        sy = size[1] / ctx.inputSize[1];
      }
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < nk; k++) {
          const j = ((c * ni + i) * nk + k) * 2;
          const u = kp.data[j], v = kp.data[j + 1];
          if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
          ((out[cam] ??= {})[m.instruments[i]] ??= {})[m.keypoints[k]] = [u * sx, v * sy];
        }
      }
    });
    pred.keypoints_px = out;
  }
  return pred;
}

let _ortPromise = null;
let _ortUrlLoaded = null;

/** Import onnxruntime-web once (lazily) and point its WASM files at the same pinned dist folder. */
export async function loadOrt(url = ORT_URL) {
  if (_ortPromise && _ortUrlLoaded === url) return _ortPromise;
  _ortUrlLoaded = url;
  _ortPromise = import(/* @vite-ignore */ url).then((mod) => {
    const ort = mod.default ?? mod;
    if (ort.env?.wasm) ort.env.wasm.wasmPaths = url.slice(0, url.lastIndexOf('/') + 1);
    return ort;
  });
  _ortPromise.catch(() => { _ortPromise = null; _ortUrlLoaded = null; });
  return _ortPromise;
}

export class OnnxModel {
  constructor(ort, session, mapping, ep) {
    this.ort = ort;
    this.session = session;
    this.mapping = mapping;
    this.ep = ep;                      // the execution provider that runs: "webgpu" | "wasm"
    this.lastRaw = null;               // the last raw outputs, {name: {data, dims}}, for debugging
    this.lastMs = null;                // the last session.run time
  }

  /**
   * @param {string|URL|Uint8Array|ArrayBuffer} urlOrBytes  the .onnx file
   * @param {{mapping?: object, ort?: object, ortUrl?: string, executionProviders?: string[]}} opts
   */
  static async load(urlOrBytes, { mapping = {}, ort, ortUrl, executionProviders } = {}) {
    ort = ort ?? await loadOrt(ortUrl);
    const m = resolveMapping(mapping);
    if (!m.name) {
      const s = typeof urlOrBytes === 'string' || urlOrBytes instanceof URL ? String(urlOrBytes) : 'model';
      m.name = s.split('/').pop().replace(/\?.*$/, '').replace(/\.onnx$/i, '') || 'model';
    }
    let bytes = urlOrBytes;
    if (typeof urlOrBytes === 'string' || urlOrBytes instanceof URL) {
      const r = await fetch(urlOrBytes);
      if (!r.ok) throw new Error(`fetch ${urlOrBytes}: HTTP ${r.status}`);
      bytes = new Uint8Array(await r.arrayBuffer());
    } else if (urlOrBytes instanceof ArrayBuffer) {
      bytes = new Uint8Array(urlOrBytes);
    }
    const eps = executionProviders ?? (globalThis.navigator?.gpu ? ['webgpu', 'wasm'] : ['wasm']);
    const errors = [];
    for (const ep of eps) {
      try {
        const session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
        return new OnnxModel(ort, session, m, ep);
      } catch (e) {
        errors.push(`${ep}: ${e?.message ?? e}`);
      }
    }
    throw new Error(`no execution provider could load the model (${errors.join('; ')})`);
  }

  /** The [W, H] the model takes: mapping.input.size, else static dims ORT reports, else the first image's size. */
  inputSize(images) {
    if (this.mapping.input.size) return this.mapping.input.size;
    const meta = this.session.inputMetadata;
    const md = Array.isArray(meta) ? meta[0] : meta?.[this.mapping.input.name];
    const dims = md?.shape ?? md?.dims;
    if (dims && Number.isInteger(dims[2]) && Number.isInteger(dims[3])) return [dims[3], dims[2]];
    const first = Object.values(images)[0];
    return [first.width, first.height];
  }

  /**
   * @param {Object<string, {rgb: Uint8Array, width: number, height: number}>} images  rgb8 per camera
   * @param {object} calib  the calibration message
   * @param {{frame_id?: number}} opts
   */
  async predict(images, calib, { frame_id } = {}) {
    const size = this.inputSize(images);
    const raw = prepareInputs(images, this.mapping, calib, size);
    const feeds = {};
    for (const [k, t] of Object.entries(raw)) feeds[k] = new this.ort.Tensor('float32', t.data, t.dims);
    const t0 = globalThis.performance?.now() ?? Date.now();
    const res = await this.session.run(feeds);
    this.lastMs = (globalThis.performance?.now() ?? Date.now()) - t0;
    const outputs = {};
    for (const [k, t] of Object.entries(res)) {
      const data = typeof t.getData === 'function' ? await t.getData() : t.data;
      outputs[k] = { data, dims: [...t.dims] };
    }
    this.lastRaw = outputs;
    const pred = mapOutputs(outputs, this.mapping, calib, { inputSize: size, images });
    if (frame_id !== undefined) pred.frame_id = frame_id;
    return pred;
  }
}
