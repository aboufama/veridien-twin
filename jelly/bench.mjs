#!/usr/bin/env node
// Judge runner for scripts/judge_jelly.py (docs/JELLY.md "judge interface"): runs one reference case on the CPU solver
// (the WebGPU solver runs the same algorithm; its parity with this one is judged separately) at >= 2 resolutions.
//   node web/twin/jelly/bench.mjs --case 1_uniaxial --spec spec.json --out result.json [--levels 0,1] [--quick]
// Output: {case, solver, runs: [{level, mesh, h_s, tets, seconds, result: {...}}]} with result in the judge's schema.
import { readFileSync, writeFileSync } from 'node:fs';
import { JellyCPU } from './cpu.js';
import { gridBlock, cutPlane } from './mesh.js';

const MM = 1e-3;
const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, arr) => (v.startsWith('--') ? [...a, [v.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]] : a), []));
const caseName = args.case;
const spec = args.spec ? JSON.parse(readFileSync(args.spec, 'utf8')) : {};
const mat = spec.material ?? {};
const quick = !!args.quick;

function params(extra = {}) {
  const p = { substeps: 1, gravity: [0, 0, 0], floor_z: null, mu_s: 0, mu_k: 0 };
  for (const k of ['E', 'nu', 'rho', 'eta', 'Gc', 'W0']) if (mat[k] != null) p[k] = Number(mat[k]);
  return { ...p, ...extra };
}
const g0 = () => -(mat.g ?? 9.81);
const sel = (j, f) => { const out = []; for (let i = 0; i < j.N; i++) if (f(j.X0[3 * i], j.X0[3 * i + 1], j.X0[3 * i + 2])) out.push(i); return out; };
const near = (a, b) => Math.abs(a - b) < 1e-9;

/** Advance the solver by `dur` seconds in substeps h (one solver step per substep), calling each(t) after every step. */
function run(j, dur, h, cols, each) {
  const n = Math.round(dur / h);
  for (let k = 0; k < n; k++) { j.step(h, typeof cols === 'function' ? cols() : cols); if (each) each(); }
}

const LEVELS = {
  '1_uniaxial': [{ n: [8, 2, 2], h: 1e-4 }, { n: [16, 4, 4], h: 5e-5 }],
  '2_overhang': [{ n: [10, 8, 4], h: 1e-4 }, { n: [20, 16, 6], h: 5e-5 }],
  '3_indent': [{ n: [10, 8, 6], h: 1e-4 }, { n: [16, 12, 10], h: 5e-5 }],
  '4_cantilever_release': [{ n: [10, 8, 4], h: 1e-4 }, { n: [20, 16, 6], h: 5e-5 }],
  '5_fracture': [{ n: [30, 10, 2], h: 1e-4 }, { n: [45, 15, 3], h: 5e-5 }],
};

const CASE = {
  '1_uniaxial'(lv) {
    const [L, W, H] = spec.spec?.geometry_m ?? [0.020, 0.005, 0.005];
    const j = new JellyCPU({ params: params({ fracture: false }), mesh: gridBlock({ size: [L, W, H], n: lv.n }) });
    const x0 = sel(j, (x) => near(x, 0)), xL = sel(j, (x) => near(x, L));
    j.prescribe(x0, 1); j.prescribe(xL, 1); j.prescribe(sel(j, (x, y) => near(y, 0)), 2); j.prescribe(sel(j, (x, y, z) => near(z, 0)), 4);
    const prY = j.probe([L / 2, W, H]);
    const rows = [{ stretch: 1, P11: 0, lateral_stretch: 1 }];
    const Tr = 0.05, Th = quick ? 0.05 : 0.1, Ta = 0.03;
    for (let k = 1; k <= 10; k++) {
      j.setPrescribedVelocity(xL, [0.1 * L / Tr, 0, 0], 1);
      run(j, Tr, lv.h, []);
      j.setPrescribedVelocity(xL, [0, 0, 0], 1);
      run(j, Th - Ta, lv.h, []);
      let F = 0, c = 0;
      run(j, Ta, lv.h, [], () => { for (const i of xL) F -= j.react[3 * i]; c++; });
      const stretch = (j.x[3 * xL[0]] - j.x[3 * x0[0]]) / L;
      rows.push({ stretch, P11: F / c / (W * H), lateral_stretch: j.probeAt(prY)[1] / W });
    }
    return { rows };
  },

  '2_overhang'(lv) {
    const [L, W, H] = spec.spec?.geometry_m ?? [0.040, 0.030, 0.012];
    const j = new JellyCPU({ params: params({ gravity: [0, 0, g0()] }), mesh: gridBlock({ size: [L, W, H], n: lv.n }) });
    j.prescribe(sel(j, (x, y, z) => near(x, 0) && near(y, W / 2) && near(z, 0)), 1);
    j.prescribe(sel(j, (x, y) => near(y, W / 2)), 2);
    const floor = { id: 'floor', kind: 'convex', center: [-0.5, 0, -0.5], radius: 3, mu_s: 0, mu_k: 0, T: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] },
      planes: [0, 0, 1, 0, 1, 0, 0, 0.025, -1, 0, 0, 1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 0, -1, 1] };
    const pr = { mid: j.probe([L, W / 2, H / 2]), bot: j.probe([L, W / 2, 0]), top: j.probe([L, W / 2, H]), back: j.probe([0, W / 2, 0]) };
    const dur = quick ? 0.15 : 0.3, Ta = 0.05;
    run(j, dur - Ta, lv.h, [floor]);
    const acc = { mid: 0, bot: 0, top: 0, back: 0 }; let c = 0;
    run(j, Ta, lv.h, [floor], () => { acc.mid += j.probeAt(pr.mid)[2] - H / 2; acc.bot += j.probeAt(pr.bot)[2]; acc.top += j.probeAt(pr.top)[2] - H; acc.back += j.probeAt(pr.back)[2]; c++; });
    return { tip_deflection_m: -acc.mid / c, tip_deflection_bottom_m: -acc.bot / c, tip_deflection_top_m: -acc.top / c, lift_back_m: acc.back / c, kinetic_J: j.energy().kinetic };
  },

  '3_indent'(lv) {
    // quarter model: x in [0, 20] mm, y in [0, 15] mm, symmetry planes x = 20 mm (u_x = 0) and y = 15 mm (u_y = 0)
    const [Lf, Wf, H] = spec.spec?.geometry_m ?? [0.040, 0.030, 0.012];
    const L = Lf / 2, W = Wf / 2, R = 0.0025;
    const j = new JellyCPU({ params: params({ gravity: [0, 0, g0()], floor_z: 0 }), mesh: gridBlock({ size: [L, W, H], n: lv.n }) });
    j.prescribe(sel(j, (x) => near(x, L)), 1); j.prescribe(sel(j, (x, y) => near(y, W)), 2);
    let d = 0;
    const cyl = () => [{ id: 'cyl', kind: 'capsule', a: [0, -0.1, 0], b: [0, 0.1, 0], r: R, mu_s: 0, mu_k: 0, T: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [L, 0, H - d + R] } }];
    run(j, 0.05, lv.h, cyl);
    const rows = [];
    const Tr = 0.02, Th = quick ? 0.03 : 0.04, Ta = 0.015;
    for (let k = 1; k <= 12; k++) {
      const d0 = d, d1 = 0.25e-3 * k, n = Math.round(Tr / lv.h); let s = 0;
      run(j, Tr, lv.h, () => { s++; d = d0 + (d1 - d0) * s / n; return cyl(); });
      d = d1;
      run(j, Th - Ta, lv.h, cyl);
      let F = 0, c = 0;
      run(j, Ta, lv.h, cyl, () => { F -= j.colliderForce.get('cyl')[2]; c++; });
      rows.push({ depth_m: d1, force_N: 4 * F / c });
    }
    return { rows };
  },

  '4_cantilever_release'(lv) {
    const [L, W, H] = spec.spec?.geometry_m ?? [0.040, 0.030, 0.012];
    const j = new JellyCPU({ params: params({ gravity: [0, 0, g0()] }), mesh: gridBlock({ size: [L, W, H], n: lv.n }) });
    j.prescribe(sel(j, (x) => near(x, 0)), 7);
    const pr = j.probe([L, W / 2, H / 2]);
    const t_s = [], uz_m = [];
    const dur = quick ? 0.3 : 1.0, every = Math.max(1, Math.round(1e-3 / lv.h));
    let k = 0;
    t_s.push(0); uz_m.push(0);
    run(j, dur, lv.h, [], () => { if (++k % every === 0) { t_s.push(k * lv.h); uz_m.push(j.probeAt(pr)[2] - H / 2); } });
    const q = Math.floor(uz_m.length * 3 / 4), tail = uz_m.slice(q);
    return { trace: { t_s, uz_m }, static_sag_m: -tail.reduce((a, b) => a + b, 0) / tail.length };
  },

  '5_fracture'(lv) {
    const [L, W, H] = spec.spec?.geometry_m ?? [0.030, 0.010, 0.002];
    const vel = 0.06;                                  // grip separation rate, m/s (strain rate 2 /s)
    const sent = (() => {
      const g = cutPlane(gridBlock({ size: [L, W, H], n: lv.n }), 0, L / 2, (p) => p[1] < 0.003 - 1e-9);
      const j = new JellyCPU({ params: params(), mesh: g });
      const a = sel(j, (x) => near(x, 0)), b = sel(j, (x) => near(x, L));
      j.prescribe(a, 3, [-vel / 2, 0, 0]); j.prescribe(b, 3, [vel / 2, 0, 0]);
      const F_N = [0], delta_m = [0]; let first = null, k = 0, peak = 0;
      const every = Math.max(1, Math.round(1e-3 / lv.h));
      const maxT = quick ? 0.5 : 1.0;
      for (let s = 0; s < Math.round(maxT / lv.h); s++) {
        j.step(lv.h, []);
        if (first === null && j.failed > 0) first = vel * (s + 1) * lv.h;
        if (++k % every === 0) {
          let F = 0; for (const i of b) F -= j.react[3 * i];
          F_N.push(F); delta_m.push(vel * (s + 1) * lv.h); peak = Math.max(peak, F);
          if (peak > 0 && F < 0.2 * peak && first !== null) break;
        }
      }
      return { F_N, delta_m, first_failure_delta_m: first, failed_tets: j.failed };
    })();
    const no_notch = (() => {
      const j = new JellyCPU({ params: params(), mesh: gridBlock({ size: [L, W, H], n: [Math.max(6, lv.n[0] / 3) | 0, 2, 1] }) });
      const a = sel(j, (x) => near(x, 0)), b = sel(j, (x) => near(x, L));
      j.prescribe(a, 1, [-vel / 2, 0, 0]); j.prescribe(b, 1, [vel / 2, 0, 0]);
      j.prescribe(sel(j, (x, y) => near(y, 0)), 2); j.prescribe(sel(j, (x, y, z) => near(z, 0)), 4);
      const F_N = [0], stretch = [1]; let peak = 0;
      const every = Math.max(1, Math.round(1e-3 / lv.h));
      for (let s = 0; s < Math.round(1.5 / lv.h); s++) {
        j.step(lv.h, []);
        if ((s + 1) % every === 0) {
          let F = 0; for (const i of b) F -= j.react[3 * i];
          F_N.push(F); stretch.push(1 + vel * (s + 1) * lv.h / L); peak = Math.max(peak, F);
          if (peak > 0 && F < 0.2 * peak) break;
        }
      }
      return { F_N, stretch, failed_tets: j.failed, l_e_m: j.l_e[0] };
    })();
    return { sent, no_notch };
  },
};

const levels = (LEVELS[caseName] ?? []).map((lv, i) => ({ ...lv, level: i }))
  .filter((lv) => !args.levels || String(args.levels).split(',').map(Number).includes(lv.level));
if (!CASE[caseName]) { console.error(`unknown case ${caseName}`); process.exit(2); }
const runs = [];
for (const lv of levels) {
  const t0 = Date.now();
  const result = CASE[caseName](lv);
  const tets = 6 * lv.n[0] * lv.n[1] * lv.n[2];
  runs.push({ level: lv.level, mesh: lv.n, h_s: lv.h, tets, seconds: (Date.now() - t0) / 1000, result });
  console.error(`${caseName} level ${lv.level} mesh ${lv.n} h ${lv.h} tets ${tets}: ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}
const out = { case: caseName, solver: 'web/twin/jelly cpu.js: small-step XPBD, stable Neo-Hookean (Macklin 2021), coloured GS, 1 iteration per substep', quick, runs };
if (args.out) writeFileSync(args.out, JSON.stringify(out)); else console.log(JSON.stringify(out));
