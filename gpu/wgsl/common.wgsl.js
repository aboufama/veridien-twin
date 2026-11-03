// WGSL shared by every tracer kernel: constants, the per-frame scene uniform (instances with their exposure-start
// and exposure-end poses), the per-dispatch camera uniform, hashing, low-discrepancy sequences and quaternions.
// Scene-dependent sizes and table offsets are compiled in as constants (see renderer.js `defines`).

export const MAX_INST = 24;

/** Constant block for a prepared scene. */
export function defines(c) {
  return /* wgsl */ `
const MAX_INST: u32 = ${MAX_INST}u;
const N_MESH: u32 = ${c.nMesh}u;
const N_MAT: u32 = ${c.nMat}u;
const MESH_OFF: u32 = ${c.meshOff}u;
const MAT_OFF: u32 = ${c.matOff}u;
const ALB_OFF: u32 = ${c.albOff}u;
const ALB_BINS: u32 = ${c.albBins}u;
const PROF_OFF: u32 = ${c.profOff}u;
const PROF_N: u32 = ${c.profN}u;
const EMIT_OFF: u32 = ${c.emitOff}u;
const EMIT_N: u32 = ${c.emitN}u;
const N_STATIC_VERTS: u32 = ${c.nStaticVerts}u;
const GATHER_K: u32 = ${c.gatherK}u;
const NODE_SAMPLES: u32 = ${c.nodeSamples}u;
const NBR_K: u32 = ${c.nbrK}u;
`;
}

export const COMMON = /* wgsl */ `
const PI: f32 = 3.14159265358979;
const INV_PI: f32 = 0.318309886183791;
const BIG: f32 = 1e30;
const MISS: u32 = 0xffffffffu;
const EPS_OFF: f32 = 2e-6;               // ray origin offset along the geometric normal (m)

// mesh flags (scene_gpu.js MESH_FLAG)
const F_STATIC: u32 = 1u;
const F_EMITTER: u32 = 2u;
const F_LAYER: u32 = 4u;
const F_BOARD: u32 = 8u;
const F_INSTRUMENT: u32 = 16u;
const F_SCOPE: u32 = 32u;
const F_TEXTURED: u32 = 128u;

// camera option bits (CamU.flags.y)
const O_CUT: u32 = 1u;
const O_AID: u32 = 2u;
const O_NO_INDIRECT: u32 = 4u;
const O_DIRECT_ONLY: u32 = 8u;
const O_LAYER: u32 = 16u;
const O_NO_CAPSULE: u32 = 32u;
const O_NORMALS: u32 = 64u;
const O_NO_MIS: u32 = 128u;
const O_SPEC_STATIC: u32 = 256u;
const O_PACKET: u32 = 512u;
const O_REFINE_SHADOW: u32 = 1024u;
const O_CACHE_WRITE: u32 = 2048u;
const O_CACHE_READ: u32 = 4096u;
const O_SHADOW_CACHE: u32 = 8192u;

struct Inst {
  q0: vec4f, q1: vec4f,          // rotation at exposure start / end (x, y, z, w)
  p0: vec4f, p1: vec4f,          // translation; p0.w = BLAS root (bits), p1.w = flags (bits: 1 moving, 2 hidden)
  wmin: vec4f, wmax: vec4f,      // world box swept over the exposure
  capA: vec4f, capB: vec4f,      // world occlusion capsule at mid-exposure, in-cavity part only (scene_gpu.js cavityCapsule): A.xyz r, B.xyz rho_eff
  bcA: vec4f, bcB: vec4f,        // world bounding capsule of the whole body over the exposure: A.xyz r, B.xyz
  pcA: vec4f, pcB: vec4f,        // the same capsule when the node direct light was last computed (incremental update)
}

struct SceneU {
  L0: vec4f,                     // emitter radiance scale (rgb), w = emitter area (m^2)
  axis: vec4f,                   // emission axis in the scope frame, w = shadow-ray skip mask (bits)
  eAvg: vec4f,                   // mean dome irradiance (rgb), for instrument-to-instrument bounces
  cut: vec4f,                    // overview cutaway plane: points with dot(n, p) > w are removed
  info: vec4u,                   // nInst, scope instance, latest radiosity slot, frame
  emitC: vec4f,                  // emitter centroid in the scope frame, w = emitter radius about it (m)
  inst: array<Inst, ${MAX_INST}>,
}

struct CamU {
  rx: vec4f, ry: vec4f, rz: vec4f,   // camera axes in D (columns of R_D_C)
  pos: vec4f,                        // centre in D; w = mode (0 kb4, 1 pinhole)
  lensA: vec4f,                      // fx, fy, cx, cy
  lensB: vec4f,                      // k1..k4
  lensC: vec4f,                      // skew alpha, static root (bits), light samples (bits), -
  size: vec4u,                       // w, h, spp, seed
  flags: vec4u,                      // camera index, options, frame, skip mask (bodies its primary rays ignore)
  aid: vec4f,                        // viewing-aid light: direction toward the light, intensity
  expo: vec4f,                       // time at shutter open, at close (0..1 between poses0 and poses1), output scale, refine
  tile: vec4u,                       // this dispatch's tile: x0, y0, width, height (pixels)
}

struct Lens { a: vec4f, b: vec4f, c: vec4f }

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hash3(a: u32, b: u32, c: u32) -> u32 { return pcg(a ^ pcg(b ^ pcg(c))); }
fn u01(x: u32) -> f32 { return f32(x >> 8u) * (1.0 / 16777216.0); }

// Kronecker / R-sequence constants: R2 for the sub-pixel position, the golden ratio for time, and
// fractional parts of square roots of primes for the light and BSDF dimensions.
const R2A: f32 = 0.7548776662466927;
const R2B: f32 = 0.5698402909980532;
const GOLD: f32 = 0.6180339887498949;
const KL0: f32 = 0.41421356237;    // sqrt 2
const KL1: f32 = 0.73205080757;    // sqrt 3
const KL2: f32 = 0.23606797750;    // sqrt 5
const KB0: f32 = 0.64575131106;    // sqrt 7
const KB1: f32 = 0.31662479036;    // sqrt 11
const KB2: f32 = 0.60555127546;    // sqrt 13
fn kr(i: u32, a: f32, shift: f32) -> f32 { return fract(shift + f32(i) * a); }

fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

fn quatToMat(q: vec4f) -> mat3x3f {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
    vec3f(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
    vec3f(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)));
}

fn slerp(a: vec4f, b0: vec4f, t: f32) -> vec4f {
  var b = b0;
  var c = dot(a, b);
  if (c < 0.0) { b = -b; c = -c; }
  if (c > 0.9995) { return normalize(mix(a, b, t)); }
  let th = acos(c);
  let s = sin(th);
  return (a * sin((1.0 - t) * th) + b * sin(t * th)) / s;
}

struct Pose { r: mat3x3f, p: vec3f }

fn instPose(i: u32, time: f32) -> Pose {
  let I = scn.inst[i];
  if ((bitcast<u32>(I.p1.w) & 1u) != 0u) {
    return Pose(quatToMat(slerp(I.q0, I.q1, time)), mix(I.p0.xyz, I.p1.xyz, time));
  }
  return Pose(quatToMat(I.q0), I.p0.xyz);
}

// Duff et al. 2017 orthonormal basis; returns the tangent t and bitangent b of n.
fn basisT(n: vec3f) -> vec3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
}
fn basisB(n: vec3f) -> vec3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return vec3f(b, s + n.y * n.y * a, -n.y);
}
`;
