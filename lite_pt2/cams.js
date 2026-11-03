// Pure scene helpers of lite_pt2 (no WebGPU; tools/build_cache.mjs imports them in node): the SCOPE camera, the
// '<cam>@lo' cameras the tracer renders, and the raster G-buffer meshes. Unchanged from lite_pt/renderer.js.
import { weld, subdivide } from '../gpu/scene_gpu.js';

export const SCOPE_HFOV_DEG = 75;       // physics.yaml scope.camera.hfov_deg is absent (2026-10-31): §18.1 default, assumed
export const SCOPE_CIRCLE_FRAC = 0.95;  // physics.yaml scope.camera.image_circle_frac is absent: §18.1 default, assumed
export const SCOPE_SIZE = [1280, 720];
// pinhole as KB4: theta_d = tan(theta) = theta (1 + t2/3 + 2 t2^2/15 + 17 t2^3/315 + 62 t2^4/2835) (+ O(theta^11),
// < 0.05 px inside the 36 deg image circle)
const TAN_KB4 = [1 / 3, 2 / 15, 17 / 315, 62 / 2835];

const rowMajor4 = (T) => (Array.isArray(T[0]) ? T : [[T[0], T[4], T[8], T[12]], [T[1], T[5], T[9], T[13]], [T[2], T[6], T[10], T[14]], [T[3], T[7], T[11], T[15]]]);
const mulRv = (R, v) => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** T_D_C (row-major 4x4) of the scope camera from the scope body pose T_D_K (column-major 16). */
export function scopeCameraT(scene, T_D_K) {
  const T = rowMajor4(T_D_K);
  const R = [[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]];
  const sc = scene.scope || {};
  const oK = sc.objective_centre_K || [0, 0, 0];
  const vK = sc.view_dir_K || [0, Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)];
  const c = mulRv(R, oK).map((x, i) => x + T[i][3]);
  const z = norm3(mulRv(R, vK));
  // image up = the side the oblique view tilts toward (+y_K, the light post side on the tower monitor)
  const yK = mulRv(R, [0, 1, 0]);
  const up = norm3(yK.map((x, i) => x - dot3(yK, z) * z[i]));
  const y = up.map((x) => -x);                    // OpenCV y is down
  const x = cross3(y, z);
  return [[x[0], y[0], z[0], c[0]], [x[1], y[1], z[1], c[1]], [x[2], y[2], z[2], c[2]], [0, 0, 0, 1]];
}

/** Scene lens block of a pinhole SCOPE camera as KB4 (tan series). */
export function scopeLens(size = SCOPE_SIZE, hfovDeg = SCOPE_HFOV_DEG) {
  const [w, h] = size;
  const f = (w / 2) / Math.tan((hfovDeg * Math.PI) / 360);
  return { model: 'kb4', K: [[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1]], D: [...TAN_KB4], size: [w, h] };
}

/** The same lens at 1/F of the pixels (pixel centres at integer coordinates). */
export function scaleLens(lens, F) {
  const K = lens.K, [w, h] = lens.size;
  const W = Math.round(w / F), H = Math.round(h / F);
  const sx = W / w, sy = H / h;
  return { ...lens, K: [[K[0][0] * sx, K[0][1] * sx, (K[0][2] + 0.5) * sx - 0.5], [0, K[1][1] * sy, (K[1][2] + 0.5) * sy - 0.5], [0, 0, 1]], size: [W, H] };
}

/** The tracer's scene: the tracking cameras plus SCOPE, each at full res (targets) and at 1/F ('<cam>@lo', traced). */
export function innerScene(scene, F) {
  const camMeta = scene.cameras.map((c) => ({ ...c }));
  const scopeCam = { name: 'SCOPE', size: [...SCOPE_SIZE], lens: scopeLens(SCOPE_SIZE), T_D_C: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], scope: true };
  if (scene.scope) camMeta.unshift(scopeCam);
  const cams = [];
  for (const c of camMeta) {
    const size = c.size || c.lens.size;
    const lens = { ...c.lens, size: [...size] };
    cams.push({ ...c, size, lens });
    const lo = scaleLens(lens, F);
    cams.push({ ...c, name: `${c.name}@lo`, size: lo.size, lens: lo });
  }
  return { names: camMeta.map((c) => c.name), scene: { ...scene, cameras: cams } };
}

/**
 * The raster G-buffer meshes: per scene mesh the welded vertices and the index list with every edge <= edgeMax
 * (per-vertex KB4 bends only at vertices), and the bench as a 4 mm grid. -> [{name, body, owner, scopePart, P, N, I, orig}]
 * where orig = true means the pinhole SCOPE draws the exported mesh itself (meshData[name]).
 */
export function rasterGeometry(scene, meshData, edgeMax) {
  const out = [];
  const scopeMeshes = new Set([...(scene.scope?.mesh_parts || []), scene.scope?.emitter_mesh].filter(Boolean));
  for (const m of scene.meshes) {
    const md = meshData[m.name];
    if (!md || m.overview_only) continue;
    const body = m.dynamic ? (m.body || (scopeMeshes.has(m.name) ? 'scope' : null)) : null;
    const w = weld(md.P, md.N, md.I);
    const I2 = subdivide(w.P, w.N, w.I, edgeMax);
    out.push({ name: m.name, body, owner: m.owner_camera || null, scopePart: scopeMeshes.has(m.name) || body === 'scope',
      P: Float32Array.from(w.P), N: Float32Array.from(w.N), I: Uint32Array.from(I2), orig: true });
  }
  if (scene.bench) {
    const bz = scene.bench.z ?? 0, ex = scene.bench.extent || [[-0.5, -0.5], [0.5, 0.5]];
    const nx = Math.min(400, Math.ceil((ex[1][0] - ex[0][0]) / 0.004)), ny = Math.min(400, Math.ceil((ex[1][1] - ex[0][1]) / 0.004));
    const P = new Float32Array((nx + 1) * (ny + 1) * 3), N = new Float32Array((nx + 1) * (ny + 1) * 3), I = new Uint32Array(nx * ny * 6);
    for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
      const k = 3 * (j * (nx + 1) + i);
      P[k] = ex[0][0] + (ex[1][0] - ex[0][0]) * i / nx; P[k + 1] = ex[0][1] + (ex[1][1] - ex[0][1]) * j / ny; P[k + 2] = bz; N[k + 2] = 1;
    }
    let q = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
      I[q++] = a; I[q++] = b; I[q++] = d; I[q++] = a; I[q++] = d; I[q++] = c;
    }
    out.push({ name: 'bench', body: null, owner: null, scopePart: false, P, N, I, orig: false });
  }
  return out;
}
