// Stub scene for renderer development, in the exact docs/WEB_TWIN.md §4 format, until the CAD export exists:
// an ellipsoidal white dome (the analytic inner surface of veridien.design.dome) with three camera holes and a
// skirt down to the bench, two 5 mm graspers (shaft, collar, two jaws) in ports L and R, and a 10 mm 30° scope
// whose light outlet is a disc emitter on its oblique tip face. Lenses are the measured dome_v1 KB4 lenses.
// Deterministic. Usage: node web/twin/tools/make_stub_scene.mjs [outDir=web/twin/tests/stub] [--fine]

import { mkdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const OUT = args.find((a) => !a.startsWith('--')) || join(here, '..', 'tests', 'stub');
const FINE = args.includes('--fine');

// --------------------------------------------------------------- tiny vector helpers
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm = (a) => scl(a, 1 / Math.hypot(...a));

class Mesh {
  constructor() { this.P = []; this.N = []; this.I = []; }
  v(p, n) { this.P.push(...p); this.N.push(...nrm(n)); return this.P.length / 3 - 1; }
  t(a, b, c) { this.I.push(a, b, c); }
  get nv() { return this.P.length / 3; }
}

// --------------------------------------------------------------- dome (veridien.design.dome, inner surface)
const IC = [-0.048180, 0.027324, 0.038736], IA = [0.228, 0.183, 0.183];
const BENCH_Z = 0.0037357;                 // skirt bottom = lowest point of the shell rim
const AIM = [0.0543, 0.026, 0.0986];       // dome_camera_install.AIM_TARGET (m)

const innerZ = (x, y) => IC[2] + IA[2] * Math.sqrt(Math.max(0, 1 - ((x - IC[0]) / IA[0]) ** 2 - ((y - IC[1]) / IA[1]) ** 2));
const innerNormalIn = (p) => nrm([-(p[0] - IC[0]) / IA[0] ** 2, -(p[1] - IC[1]) / IA[1] ** 2, -(p[2] - IC[2]) / IA[2] ** 2]);

const camSpots = [
  { name: 'CAM1', tag: 'CAM 1', role: 'APEX-L', xy: [0.000, IC[1] + 0.024] },
  { name: 'CAM2', tag: 'CAM 2', role: 'APEX-R', xy: [0.000, IC[1] - 0.024] },
  { name: 'CAM3', tag: 'CAM 3', role: 'SIDE', xy: [0.060, IC[1] - 0.150] },
];
for (const c of camSpots) {
  const p = [c.xy[0], c.xy[1], innerZ(c.xy[0], c.xy[1])];
  c.surface = p;
  c.centre = add(p, scl(innerNormalIn(p), -0.0003));      // lens 0.3 mm behind the surface (LENS_RECESS)
}
const HOLE_R = 0.0089;

function shellMesh() {
  const m = new Mesh();
  const nT = FINE ? 96 : 48, nP = FINE ? 320 : 160, nS = FINE ? 12 : 6;
  const grid = [];
  for (let i = 0; i <= nT; i++) {
    const th = (i / nT) * (Math.PI / 2), row = [];
    for (let j = 0; j < nP; j++) {
      const ph = (j / nP) * 2 * Math.PI;
      const p = [IC[0] + IA[0] * Math.sin(th) * Math.cos(ph), IC[1] + IA[1] * Math.sin(th) * Math.sin(ph), IC[2] + IA[2] * Math.cos(th)];
      row.push(m.v(p, i === 0 ? [0, 0, -1] : innerNormalIn(p)));
    }
    grid.push(row);
  }
  // skirt rows from the spring line down to the bench
  for (let s = 1; s <= nS; s++) {
    const z = IC[2] - (s / nS) * (IC[2] - BENCH_Z), row = [];
    for (let j = 0; j < nP; j++) {
      const ph = (j / nP) * 2 * Math.PI;
      const p = [IC[0] + IA[0] * Math.cos(ph), IC[1] + IA[1] * Math.sin(ph), z];
      row.push(m.v(p, [-Math.cos(ph) / IA[0], -Math.sin(ph) / IA[1], 0]));
    }
    grid.push(row);
  }
  const P = (k) => [m.P[3 * k], m.P[3 * k + 1], m.P[3 * k + 2]];
  const inHole = (a, b, c) => {
    const g = scl(add(add(P(a), P(b)), P(c)), 1 / 3);
    return camSpots.some((cs) => Math.hypot(...sub(g, cs.surface)) < HOLE_R);
  };
  for (let i = 0; i < grid.length - 1; i++) for (let j = 0; j < nP; j++) {
    const a = grid[i][j], b = grid[i][(j + 1) % nP], c = grid[i + 1][j], d = grid[i + 1][(j + 1) % nP];
    if (i > 0 && !inHole(a, c, b)) m.t(a, c, b);
    if (!inHole(b, c, d)) m.t(b, c, d);
  }
  return m;
}

// --------------------------------------------------------------- instruments and scope (body frames)
function cylinder(m, r, z0, z1, seg, caps = true) {
  const base = m.nv;
  for (let k = 0; k <= 1; k++) for (let j = 0; j < seg; j++) {
    const a = (j / seg) * 2 * Math.PI;
    m.v([r * Math.cos(a), r * Math.sin(a), k ? z1 : z0], [Math.cos(a), Math.sin(a), 0]);
  }
  for (let j = 0; j < seg; j++) {
    const a = base + j, b = base + (j + 1) % seg, c = a + seg, d = b + seg;
    m.t(a, b, d); m.t(a, d, c);
  }
  if (caps) for (const [z, s] of [[z0, -1], [z1, 1]]) {
    const c0 = m.v([0, 0, z], [0, 0, s]), ring = [];
    for (let j = 0; j < seg; j++) { const a = (j / seg) * 2 * Math.PI; ring.push(m.v([r * Math.cos(a), r * Math.sin(a), z], [0, 0, s])); }
    for (let j = 0; j < seg; j++) s > 0 ? m.t(c0, ring[j], ring[(j + 1) % seg]) : m.t(c0, ring[(j + 1) % seg], ring[j]);
  }
  return m;
}

function box(m, lo, hi) {
  const faces = [
    [[1, 0, 0], [[hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [hi[0], hi[1], hi[2]], [hi[0], lo[1], hi[2]]]],
    [[-1, 0, 0], [[lo[0], lo[1], lo[2]], [lo[0], lo[1], hi[2]], [lo[0], hi[1], hi[2]], [lo[0], hi[1], lo[2]]]],
    [[0, 1, 0], [[lo[0], hi[1], lo[2]], [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]], [hi[0], hi[1], lo[2]]]],
    [[0, -1, 0], [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]], [lo[0], lo[1], hi[2]]]],
    [[0, 0, 1], [[lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]]],
    [[0, 0, -1], [[lo[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]], [hi[0], lo[1], lo[2]]]],
  ];
  for (const [n, q] of faces) {
    const k = q.map((p) => m.v(p, n));
    m.t(k[0], k[1], k[2]); m.t(k[0], k[2], k[3]);
  }
  return m;
}

/** Flat disc of radius r centred at c with normal n (seg triangles). */
function disc(m, c, n, r, seg) {
  const [t, b] = (() => { const a = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]; const t = nrm(cross(a, n)); return [t, cross(n, t)]; })();
  const c0 = m.v(c, n), ring = [];
  for (let j = 0; j < seg; j++) {
    const a = (j / seg) * 2 * Math.PI;
    ring.push(m.v(add(c, add(scl(t, r * Math.cos(a)), scl(b, r * Math.sin(a)))), n));
  }
  for (let j = 0; j < seg; j++) {
    // winding so that (p1 - p0) x (p2 - p0) points along n
    const p1 = ring[j], p2 = ring[(j + 1) % seg];
    const g = cross(sub(m.P.slice(3 * p1, 3 * p1 + 3), c), sub(m.P.slice(3 * p2, 3 * p2 + 3), c));
    dot(g, n) > 0 ? m.t(c0, p1, p2) : m.t(c0, p2, p1);
  }
  return m;
}

const SEG = FINE ? 48 : 24;
const meshes = {};
meshes.static_shell = { mesh: shellMesh(), material: 'pla_matte_white', group: 'shell', dynamic: false, build_dir: [0, 0, 1] };
for (const id of ['L', 'R']) {
  meshes[`${id}_shaft`] = { mesh: cylinder(new Mesh(), 0.0025, -0.36, -0.007, SEG), material: 'insulation_black', group: 'instrument_part', dynamic: true };
  meshes[`${id}_collar`] = { mesh: cylinder(new Mesh(), 0.0025, -0.007, 0, SEG), material: 'steel_satin', group: 'instrument_part', dynamic: true };
  meshes[`${id}_jaw_a`] = { mesh: box(new Mesh(), [-0.0018, 0.0, 0.0], [0.0018, 0.0012, 0.018]), material: 'steel_jaw', group: 'instrument_part', dynamic: true };
  meshes[`${id}_jaw_b`] = { mesh: box(new Mesh(), [-0.0018, -0.0012, 0.0], [0.0018, 0.0, 0.018]), material: 'steel_jaw', group: 'instrument_part', dynamic: true };
}
const DOV = 30 * Math.PI / 180, nView = [0, Math.sin(DOV), Math.cos(DOV)];
const TIP_C = [0, 0, -0.0005], OUT_C = [0, -0.0025, 0];
meshes.scope_shaft = { mesh: cylinder(new Mesh(), 0.005, -0.31, -0.0035, SEG * 2), material: 'scope_steel', group: 'scope', dynamic: true };
meshes.scope_tip_face = { mesh: disc(new Mesh(), TIP_C, nView, 0.0055, SEG * 2), material: 'scope_glass', group: 'scope', dynamic: true };
// the outlet disc sits 0.1 mm in front of the tip face (along the view normal), so shadow rays reach it
const outC = add(OUT_C, scl(nView, dot(sub(TIP_C, OUT_C), nView) + 0.0001));
meshes.scope_light = { mesh: disc(new Mesh(), outC, nView, 0.002, SEG), material: 'emitter', group: 'scope', dynamic: true };

// --------------------------------------------------------------- cameras, ports, scope
const DOME_V1 = {
  CAM1: { K: [[745.386252, 0, 633.927484], [0, 747.207117, 380.042876], [0, 0, 1]], D: [-0.012300643, -0.016677738, 0.035329677, -0.033581606], uid: '0x113100032e40144' },
  CAM2: { K: [[742.233415, 0, 624.055587], [0, 746.5229, 408.659744], [0, 0, 1]], D: [-0.02167181, 0.005458033, 0.006324293, -0.013590434], uid: '0x113200032e40144' },
  CAM3: { K: [[753.707994, 0, 539.674967], [0, 756.773813, 387.364381], [0, 0, 1]], D: [-0.011875441, -0.003513882, -0.006539589, 0.001768382], uid: '0x14000032e40144' },
};
function lookAt(c, target, upHint = [0, 0, 1]) {
  const z = nrm(sub(target, c));
  let y = scl(upHint, -1); y = nrm(sub(y, scl(z, dot(y, z))));
  const x = cross(y, z);
  return [[x[0], y[0], z[0], c[0]], [x[1], y[1], z[1], c[1]], [x[2], y[2], z[2], c[2]], [0, 0, 0, 1]];
}
const cameras = camSpots.map((c) => ({
  name: c.name, tag: c.tag, role: c.role, unique_id: DOME_V1[c.name].uid, size: [1280, 720], fps: 20.0, phase_s: 0.0,
  lens: { model: 'kb4', K: DOME_V1[c.name].K, D: DOME_V1[c.name].D, size: [1280, 720] },
  T_D_C: lookAt(c.centre, AIM, c.role === 'SIDE' ? [0, 0, 1] : [1, 0, 0]),
  lens_front_D: c.surface, source: { pose: 'stub: inner ellipsoid, aimed at AIM_TARGET', lens: 'dome_v1.yaml' },
}));

function portAt(id, role, x, y) {
  const p = [x, y, innerZ(x, y)];
  const axisOut = scl(innerNormalIn(p), -1);
  const z = scl(axisOut, -1);                            // port z: into the body
  let xp = nrm(cross([0, 0, 1], z)); if (!Number.isFinite(xp[0])) xp = [1, 0, 0];
  const yp = cross(z, xp);
  return { id, role, pivot: p, axis_out: axisOut, R_D_P: [[xp[0], yp[0], z[0]], [xp[1], yp[1], z[1]], [xp[2], yp[2], z[2]]], cone_half_deg: 30.0, trocar_d: id === 'C' ? 0.010 : 0.005 };
}
const ports = [portAt('L', 'working', -0.170, IC[1] + 0.075), portAt('R', 'working', -0.170, IC[1] - 0.075), portAt('C', 'scope', -0.120, IC[1])];
const scopeDir = nrm(sub([0.02, IC[1], 0.06], ports[2].pivot));

// --------------------------------------------------------------- write
mkdirSync(join(OUT, 'meshes'), { recursive: true });
const meshList = [];
for (const [name, e] of Object.entries(meshes)) {
  const m = e.mesh, nv = m.nv, nt = m.I.length / 3;
  const P = new Float32Array(m.P), N = new Float32Array(m.N), I = new Uint32Array(m.I);
  const buf = Buffer.alloc(P.byteLength + N.byteLength + I.byteLength);
  Buffer.from(P.buffer).copy(buf, 0); Buffer.from(N.buffer).copy(buf, P.byteLength); Buffer.from(I.buffer).copy(buf, P.byteLength + N.byteLength);
  writeFileSync(join(OUT, 'meshes', `${name}.bin`), buf);
  meshList.push({
    name, file: `meshes/${name}.bin`, material: e.material, group: e.group, dynamic: e.dynamic, build_dir: e.build_dir ?? null,
    vertex_count: nv, triangle_count: nt, max_chord_error_m: null,
    layout: { positions: [0, nv], normals: [P.byteLength, nv], indices: [P.byteLength + N.byteLength, nt * 3] },
  });
}
const parts = (id) => [
  { mesh: `${id}_shaft`, body: 'shaft' }, { mesh: `${id}_collar`, body: 'shaft' },
  { mesh: `${id}_jaw_a`, body: 'jaw_a' }, { mesh: `${id}_jaw_b`, body: 'jaw_b' },
];
const scene = {
  version: 1, units: 'm', frame: 'D',
  provenance: { git_commit: 'stub', sources: { 'web/twin/tools/make_stub_scene.mjs': 'stub' }, built: '2026-10-31T00:00:00Z', stub: true },
  bench: { z: BENCH_Z, material: 'bench_white', extent: [[-0.40, -0.30], [0.30, 0.36]] },
  meshes: meshList, cameras, ports,
  scope: { port: 'C', pivot: ports[2].pivot, default: { dir_D: scopeDir, insertion: 0.016, roll: 0.0 }, mesh_parts: ['scope_shaft', 'scope_tip_face'], emitter_mesh: 'scope_light' },
  instruments: [
    { id: 'L', port: 'L', kind: 'fenestrated', spec: { shaft_radius: 0.0025, housing_length: 0.007, jaw_length: 0.018, jaw_max: 1.187 }, parts: parts('L'), hinge_in_S: [0, 0, 0] },
    { id: 'R', port: 'R', kind: 'maryland', spec: { shaft_radius: 0.0025, housing_length: 0.007, jaw_length: 0.018, jaw_max: 1.187 }, parts: parts('R'), hinge_in_S: [0, 0, 0] },
  ],
  boards: [],
};
writeFileSync(join(OUT, 'scene.json'), JSON.stringify(scene, null, 1));
console.log(`stub scene -> ${OUT}: ${meshList.length} meshes, ${meshList.reduce((s, m) => s + m.triangle_count, 0)} triangles`);
