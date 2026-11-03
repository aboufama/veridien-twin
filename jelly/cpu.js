// CPU reference of the jelly solver (docs/JELLY.md). Plain JS, f64, deterministic; node and browser.
// Small-step XPBD (Macklin et al. 2019): n substeps per step, one Gauss-Seidel pass per substep over a vertex-disjoint
// tet colouring (the WebGPU solver runs the same colours in the same order, one dispatch per colour).
// Per substep:
//   1. predict      v += h g; xp = x; x += h v   (prescribed axes: x += h v_bc)
//   2. elasticity   per tet, the stable Neo-Hookean pair (C_D, C_H) of Macklin & Mueller 2021 solved together (2x2)
//   3. contact      floor plane and rigid colliders (convex pieces, capsules, spheres): push-out + Coulomb friction on
//                   the displacement relative to the collider's own surface motion
//   4. velocity     v = (x - xp) / h
//      viscosity    Kelvin-Voigt sigma_v = 2 eta D: implicit projection of v over the 6 strain-rate components per tet
//   5. fracture     crack band: H = max Psi while lambda_max > 1; d(H); d = 1 -> tet removed, its mass with it
import { gridBlock, colourTets, faceNeighbours, vertexTets, placeBlock, TET_FACES } from './mesh.js';
import { resolveParams, inv3, det3, damageOf, maxEigSym3 } from './material.js';
import { poseFromT, lerpPose, toLocal, rotWorld } from './pose.js';

const SQ2 = Math.SQRT1_2;
/** Largest damage short of failure (the f32 GPU solver uses the same cap: 1 - 1e-9 rounds to 1 in f32). */
export const DMAX = 1 - 1e-6;
const VQ = new Uint32Array(4), VDS = new Float64Array(9), VG = new Float64Array(12), VGR = new Float64Array(12);
const PV_GT = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
const PV_IDS = new Uint32Array(256), PV_G = new Float64Array(768), PV_DS = new Float64Array(9);
const FF = new Float64Array(9), FC = new Float64Array(9), GD = new Float64Array(12), GH = new Float64Array(12);

export class JellyCPU {
  /** opts: {params, mesh: gridBlock() result | {size, n, origin}, pose: {yaw, translate}} */
  constructor(opts = {}) {
    this.P = resolveParams(opts.params);
    const g = opts.mesh?.tets ? opts.mesh : gridBlock(opts.mesh ?? { size: [0.04, 0.03, 0.015], n: [16, 12, 6] });
    const X = opts.pose ? placeBlock(g.X, opts.pose) : g.X;
    const N = X.length / 3;
    const col = colourTets(g.tets, g.l_e, N);
    this.grid = g;
    this.N = N; this.T = col.tets.length / 4;
    this.tets = col.tets; this.l_e = col.l_e; this.colourStart = col.colourStart; this.tetOrder = col.order;
    this.nbr = faceNeighbours(this.tets);
    this.vt = vertexTets(this.tets, N);
    this.X0 = Float64Array.from(X);
    this.DmInv = new Float64Array(9 * this.T); this.V0 = new Float64Array(this.T);
    for (let t = 0; t < this.T; t++) {
      const q = this.tets.subarray(4 * t, 4 * t + 4);
      const Dm = this.ds(X, q);
      this.V0[t] = det3(Dm) / 6;
      this.DmInv.set(inv3(Dm), 9 * t);
    }
    this.vcol = colourPatches(this.tets, this.vt, N);   // vertex order such that same-colour patches share no vertex
    this.kinMask = new Uint8Array(N);          // bit a: axis a prescribed
    this.vBC = new Float64Array(3 * N);        // prescribed velocity on those axes
    this.reset();
  }

  ds(x, q) {
    const a = 3 * q[0];
    const m = new Array(9);
    for (let k = 0; k < 3; k++) { const b = 3 * q[k + 1]; for (let r = 0; r < 3; r++) m[3 * r + k] = x[b + r] - x[a + r]; }
    return m;
  }

  reset() {
    const { N, T } = this;
    this.x = Float64Array.from(this.X0); this.xp = Float64Array.from(this.X0); this.v = new Float64Array(3 * N);
    this.alive = new Uint8Array(T).fill(1); this.H = new Float64Array(T); this.dmg = new Float64Array(T);
    this.react = new Float64Array(3 * N);
    this.colliderPrev = new Map(); this.colliderForce = new Map();
    this._stamp = 0; this._stampOf = new Uint32Array(N); this._slot = new Uint32Array(N);
    this.time = 0; this.topoVersion = 0; this.version = 0; this.failed = 0; this.lastContacts = 0; this.lastMs = 0;
    this.failedLog = [];
    this.recomputeMass();
  }

  /** Vertex masses from alive tets (rho V0 / 4 each); per-axis inverse masses (0 on prescribed axes). */
  recomputeMass() {
    const { N, T, P } = this;
    this.mass = new Float64Array(N);
    for (let t = 0; t < T; t++) if (this.alive[t]) for (let l = 0; l < 4; l++) this.mass[this.tets[4 * t + l]] += P.rho * this.V0[t] / 4;
    this.Vv0 = new Float64Array(N);            // rest volume of each vertex patch (alive tets)
    for (let t = 0; t < T; t++) if (this.alive[t]) for (let l = 0; l < 4; l++) this.Vv0[this.tets[4 * t + l]] += this.V0[t];
    this.invm = new Float64Array(3 * N); this.active = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      this.active[i] = this.mass[i] > 0 ? 1 : 0;
      for (let a = 0; a < 3; a++) this.invm[3 * i + a] = this.mass[i] > 0 && !(this.kinMask[i] >> a & 1) ? 1 / this.mass[i] : 0;
    }
  }

  /** Prescribe axes of vertices: idx list, mask (bit per axis), velocity [3] (m/s). */
  prescribe(idx, mask, vel = [0, 0, 0]) {
    for (const i of idx) { this.kinMask[i] |= mask; for (let a = 0; a < 3; a++) if (mask >> a & 1) this.vBC[3 * i + a] = vel[a]; }
    this.recomputeMass();
  }
  setPrescribedVelocity(idx, vel, mask = 7) { for (const i of idx) for (let a = 0; a < 3; a++) if (mask >> a & 1) this.vBC[3 * i + a] = vel[a]; }

  /** Mark tets failed (an initial notch): list of tet indices in this solver's order. */
  failTets(list) {
    for (const t of list) if (this.alive[t]) { this.alive[t] = 0; this.dmg[t] = 1; this.failed++; }
    this.topoVersion++; this.recomputeMass();
  }

  // ---------------------------------------------------------------- step
  /** step(dt, colliders): colliders = [{id, kind: 'convex'|'capsule'|'sphere', T (col-major 4x4 | {R,t}), ...}]. */
  step(dt, colliders = []) {
    const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
    const n = this.P.substeps, h = dt / n;
    const cols = colliders.map((c) => {
      const now = poseFromT(c.T);
      const prev = this.colliderPrev.get(c.id) ?? now;
      return { c, now, prev };
    });
    this.react.fill(0);
    const force = new Map(); for (const { c } of cols) force.set(c.id, [0, 0, 0]);
    let contacts = 0;
    for (let s = 1; s <= n; s++) {
      const poses = cols.map(({ c, now, prev }) => ({ c, P1: lerpPose(prev, now, s / n), P0: lerpPose(prev, now, (s - 1) / n), f: force.get(c.id) }));
      this.predict(h);
      this.solveElastic(h);
      contacts = this.solveContacts(h, poses);
      this.updateVelocity(h);
      if (this.P.viscosity && this.P.eta > 0) this.solveViscous(h);
      if (this.P.fracture) this.updateDamage();
      this.time += h;
    }
    for (const { c, now } of cols) this.colliderPrev.set(c.id, now);
    for (const [k, f] of force) this.colliderForce.set(k, f.map((v) => v / n));
    for (let i = 0; i < this.react.length; i++) this.react[i] /= n;
    this.lastContacts = contacts; this.version++;
    this.lastMs = (typeof performance !== 'undefined' ? performance : Date).now() - t0;
  }

  predict(h) {
    const { N, x, xp, v, invm, kinMask, vBC, active } = this;
    const g = this.P.gravity;
    for (let i = 0; i < N; i++) {
      if (!active[i]) continue;
      for (let a = 0; a < 3; a++) {
        const k = 3 * i + a;
        xp[k] = x[k];
        if (kinMask[i] >> a & 1) { v[k] = vBC[k]; x[k] += h * vBC[k]; }
        else if (invm[k] > 0) { v[k] += h * g[a]; x[k] += h * v[k]; }
      }
    }
  }

  /** The 2x2 coupled projection of (C_D, C_H) on tet t; returns nothing, moves x. */
  projectTet(t, h) {
    const { x, invm, tets, DmInv, P } = this;
    const q0 = tets[4 * t], q1 = tets[4 * t + 1], q2 = tets[4 * t + 2], q3 = tets[4 * t + 3];
    const gdeg = 1 - this.dmg[t];
    const V0 = this.V0[t];
    const M = DmInv.subarray(9 * t, 9 * t + 9);
    // Ds columns
    const d1x = x[3 * q1] - x[3 * q0], d1y = x[3 * q1 + 1] - x[3 * q0 + 1], d1z = x[3 * q1 + 2] - x[3 * q0 + 2];
    const d2x = x[3 * q2] - x[3 * q0], d2y = x[3 * q2 + 1] - x[3 * q0 + 1], d2z = x[3 * q2 + 2] - x[3 * q0 + 2];
    const d3x = x[3 * q3] - x[3 * q0], d3y = x[3 * q3 + 1] - x[3 * q0 + 1], d3z = x[3 * q3 + 2] - x[3 * q0 + 2];
    const cd = tetConstraints(d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z, M, P.alphaRest, GD, GH);
    if (cd === null) return;
    const CD = cd[0], CH = cd[1], gD = GD, gH = GH;
    const qs = VQ; qs[0] = q0; qs[1] = q1; qs[2] = q2; qs[3] = q3;
    let wDD = 0, wDH = 0, wHH = 0;
    for (let l = 0; l < 4; l++) for (let a = 0; a < 3; a++) {
      const w = invm[3 * qs[l] + a], gd = gD[3 * l + a], gh = gH[3 * l + a];
      wDD += w * gd * gd; wDH += w * gd * gh; wHH += w * gh * gh;
    }
    const aD = 1 / (gdeg * P.mu * V0 * h * h), aH = 1 / (gdeg * P.lamTet * V0 * h * h);
    const A = wDD + aD, B = wDH, Cc = wHH + aH;
    const det = A * Cc - B * B;
    if (!(Math.abs(det) > 0)) return;
    const lD = (-CD * Cc + CH * B) / det;
    const lH = (-CH * A + CD * B) / det;
    for (let l = 0; l < 4; l++) for (let a = 0; a < 3; a++) {
      const k = 3 * qs[l] + a, gl = gD[3 * l + a] * lD + gH[3 * l + a] * lH;
      if (invm[k] > 0) x[k] += invm[k] * gl;
      else this.react[k] += gl / (h * h);          // force the gel exerts on a prescribed axis
    }
  }

  solveElastic(h) {
    const { T, alive } = this;
    for (let t = 0; t < T; t++) if (alive[t]) this.projectTet(t, h);
    if (this.P.lamNode > 0) { const o = this.vcol.order; for (let k = 0; k < o.length; k++) this.projectVertex(o[k], h); }
  }

  /** Nodal volume constraint of vertex v: C = J_v - 1, J_v = sum V_t / sum V_t0 over the alive tets of its patch,
   * energy w_v lamNode / 2 C^2 with w_v = sum V_t0 / 4 (so sum_v w_v = the body's volume). */
  projectVertex(v, h) {
    const { x, invm, tets, vt, alive, P } = this;
    const V0p = this.Vv0[v];
    if (!(V0p > 0) || !this.active[v]) return;
    const ids = PV_IDS, gr = PV_G; let n = 0, Vp = 0;
    const stamp = ++this._stamp;
    for (let k = vt.start[v]; k < vt.start[v + 1]; k++) {
      const t = vt.list[k] >> 2;
      if (!alive[t]) continue;
      const q0 = tets[4 * t], a0 = 3 * q0;
      const m = PV_DS;
      for (let c = 0; c < 3; c++) { const b = 3 * tets[4 * t + c + 1]; for (let r = 0; r < 3; r++) m[3 * r + c] = x[b + r] - x[a0 + r]; }
      // cof(Ds) columns = dV/dx_{1..3} * 6
      const c0 = [m[4] * m[8] - m[5] * m[7], m[5] * m[6] - m[3] * m[8], m[3] * m[7] - m[4] * m[6]];
      const c1 = [m[2] * m[7] - m[1] * m[8], m[0] * m[8] - m[2] * m[6], m[1] * m[6] - m[0] * m[7]];
      const c2 = [m[1] * m[5] - m[2] * m[4], m[2] * m[3] - m[0] * m[5], m[0] * m[4] - m[1] * m[3]];
      Vp += (m[0] * c0[0] + m[1] * c0[1] + m[2] * c0[2]) / 6;
      // c0, c1, c2 are the rows of cof(Ds); column j of cof(Ds) / 6 = dV/d(x_{j+1})
      const g = PV_GT;
      g[1][0] = c0[0] / 6; g[1][1] = c1[0] / 6; g[1][2] = c2[0] / 6;
      g[2][0] = c0[1] / 6; g[2][1] = c1[1] / 6; g[2][2] = c2[1] / 6;
      g[3][0] = c0[2] / 6; g[3][1] = c1[2] / 6; g[3][2] = c2[2] / 6;
      for (let a = 0; a < 3; a++) g[0][a] = -(g[1][a] + g[2][a] + g[3][a]);
      for (let l = 0; l < 4; l++) {
        const u = tets[4 * t + l];
        let slot;
        if (this._stampOf[u] !== stamp) { this._stampOf[u] = stamp; this._slot[u] = slot = n++; ids[slot] = u; gr[3 * slot] = 0; gr[3 * slot + 1] = 0; gr[3 * slot + 2] = 0; }
        else slot = this._slot[u];
        gr[3 * slot] += g[l][0]; gr[3 * slot + 1] += g[l][1]; gr[3 * slot + 2] += g[l][2];
      }
    }
    const C = Vp / V0p - 1;
    let w = 0;
    for (let s = 0; s < n; s++) for (let a = 0; a < 3; a++) { gr[3 * s + a] /= V0p; w += invm[3 * ids[s] + a] * gr[3 * s + a] ** 2; }
    const alpha = 1 / (P.lamNode * (V0p / 4) * h * h);
    const dl = -C / (w + alpha);
    for (let s = 0; s < n; s++) for (let a = 0; a < 3; a++) {
      const k = 3 * ids[s] + a, gl = gr[3 * s + a] * dl;
      if (invm[k] > 0) x[k] += invm[k] * gl; else this.react[k] += gl / (h * h);
    }
  }

  /** Kelvin-Voigt (velocity level, after the velocity update; positions are not moved): one GS pass over the tets. */
  solveViscous(h) {
    const { T, alive } = this;
    for (let t = 0; t < T; t++) if (alive[t]) this.viscousTet(t, h);
  }

  /** Kelvin-Voigt on tet t: minimise 1/2 |v - v*|_M^2 + h vol eta |D(v)|^2 (sigma_v = 2 eta D), one GS pass over the
   * 6 orthonormal strain-rate components (D_xx, D_yy, D_zz, sqrt2 D_xy, sqrt2 D_yz, sqrt2 D_xz). Gradients in the current
   * configuration. (Applying it to positions inside the elastic pass was tried and is unstable with friction: ADR in JELLY.md.) */
  viscousTet(t, h) {
    const { tets, x, v, invm, P } = this;
    const q0 = tets[4 * t], q1 = tets[4 * t + 1], q2 = tets[4 * t + 2], q3 = tets[4 * t + 3];
    const qs = VQ; qs[0] = q0; qs[1] = q1; qs[2] = q2; qs[3] = q3;
    const a0 = 3 * q0;
    const Ds = VDS;
    for (let k = 0; k < 3; k++) { const b = 3 * qs[k + 1]; for (let r = 0; r < 3; r++) Ds[3 * r + k] = x[b + r] - x[a0 + r]; }
    const m = Ds;
    const c00 = m[4] * m[8] - m[5] * m[7], c01 = m[5] * m[6] - m[3] * m[8], c02 = m[3] * m[7] - m[4] * m[6];
    const dd = m[0] * c00 + m[1] * c01 + m[2] * c02;
    const vol = Math.abs(dd) / 6;
    if (!(vol > 1e-18)) return;
    // rows of Ds^-1 = grad of the shape fns of x1..x3 (current configuration)
    const g = VG, inv = 1 / dd;
    g[3] = c00 * inv; g[4] = (m[2] * m[7] - m[1] * m[8]) * inv; g[5] = (m[1] * m[5] - m[2] * m[4]) * inv;
    g[6] = c01 * inv; g[7] = (m[0] * m[8] - m[2] * m[6]) * inv; g[8] = (m[2] * m[3] - m[0] * m[5]) * inv;
    g[9] = c02 * inv; g[10] = (m[1] * m[6] - m[0] * m[7]) * inv; g[11] = (m[0] * m[4] - m[1] * m[3]) * inv;
    for (let a = 0; a < 3; a++) g[a] = -(g[3 + a] + g[6 + a] + g[9 + a]);
    const alpha = 1 / (2 * h * vol * P.eta * (1 - this.dmg[t]));
    const grad = VGR;
    for (let comp = 0; comp < 6; comp++) {
      grad.fill(0);
      if (comp < 3) for (let l = 0; l < 4; l++) grad[3 * l + comp] = g[3 * l + comp];
      else {
        const a = comp === 4 ? 1 : 0, b = comp === 3 ? 1 : 2;
        for (let l = 0; l < 4; l++) { grad[3 * l + a] = SQ2 * g[3 * l + b]; grad[3 * l + b] = SQ2 * g[3 * l + a]; }
      }
      let C = 0, w = 0;
      for (let l = 0; l < 4; l++) for (let a = 0; a < 3; a++) {
        const kk = 3 * qs[l] + a, gr = grad[3 * l + a];
        C += gr * v[kk]; w += invm[kk] * gr * gr;
      }
      const dl = -C / (w + alpha);
      for (let l = 0; l < 4; l++) for (let a = 0; a < 3; a++) {
        const kk = 3 * qs[l] + a, gr = grad[3 * l + a];
        if (invm[kk] > 0) v[kk] += invm[kk] * gr * dl; else this.react[kk] += gr * dl / h;
      }
    }
  }

  solveContacts(h, poses) {
    const { N, x, xp, invm, active, kinMask, mass } = this;
    const fz = this.P.floor_z;
    let count = 0;
    const pl = [0, 0, 0], pp = [0, 0, 0], nw = [0, 0, 0], u = [0, 0, 0];
    for (let i = 0; i < N; i++) {
      if (!active[i] || kinMask[i] === 7) continue;
      const k = 3 * i, km = kinMask[i];
      if (fz !== null && fz !== undefined && x[k + 2] < fz && !(km & 4)) {
        const delta = fz - x[k + 2];
        x[k + 2] = fz;
        this.friction(k, delta, 0, 0, 1, 0, 0, 0, this.P.mu_s, this.P.mu_k, km);
        count++;
      }
      for (const { c, P1, P0, f } of poses) {
        toLocal(P1, x[k], x[k + 1], x[k + 2], pl);
        const hit = contactLocal(c, pl);
        if (!hit) continue;
        // hit = [depth, nlx, nly, nlz] (local normal, out of the collider)
        rotWorld(P1, hit[1], hit[2], hit[3], nw);
        const bx = x[k], by = x[k + 1], bz = x[k + 2];
        if (!(km & 1)) x[k] += hit[0] * nw[0];
        if (!(km & 2)) x[k + 1] += hit[0] * nw[1];
        if (!(km & 4)) x[k + 2] += hit[0] * nw[2];
        // collider surface motion at xp over this substep: P1 (P0^-1 xp) - xp
        toLocal(P0, xp[k], xp[k + 1], xp[k + 2], pp);
        const R = P1.R;
        u[0] = R[0] * pp[0] + R[1] * pp[1] + R[2] * pp[2] + P1.t[0] - xp[k];
        u[1] = R[3] * pp[0] + R[4] * pp[1] + R[5] * pp[2] + P1.t[1] - xp[k + 1];
        u[2] = R[6] * pp[0] + R[7] * pp[1] + R[8] * pp[2] + P1.t[2] - xp[k + 2];
        this.friction(k, hit[0], nw[0], nw[1], nw[2], u[0], u[1], u[2], c.mu_s ?? this.P.mu_s, c.mu_k ?? this.P.mu_k, km);
        const m = mass[i] / (h * h);
        f[0] += m * (x[k] - bx); f[1] += m * (x[k + 1] - by); f[2] += m * (x[k + 2] - bz);   // force on the gel by the collider
        count++;
      }
    }
    return count;
  }

  /** Coulomb friction on the displacement this substep relative to the surface motion u (normal n, depth delta). */
  friction(k, delta, nx, ny, nz, ux, uy, uz, mus, muk, km = 0) {
    const { x, xp } = this;
    const dx = x[k] - xp[k] - ux, dy = x[k + 1] - xp[k + 1] - uy, dz = x[k + 2] - xp[k + 2] - uz;
    const dn = dx * nx + dy * ny + dz * nz;
    const tx = dx - dn * nx, ty = dy - dn * ny, tz = dz - dn * nz;
    const tl = Math.hypot(tx, ty, tz);
    if (tl <= 0) return;
    const s = tl < mus * delta ? 1 : Math.min(muk * delta / tl, 1);
    if (!(km & 1)) x[k] -= s * tx;
    if (!(km & 2)) x[k + 1] -= s * ty;
    if (!(km & 4)) x[k + 2] -= s * tz;
  }

  updateVelocity(h) {
    const { N, x, xp, v, active } = this;
    for (let i = 0; i < 3 * N; i++) if (active[(i / 3) | 0]) v[i] = (x[i] - xp[i]) / h;
  }

  /** Per-tet F (row-major) at the current positions (out: optional 9-array to fill). */
  F(t, x = this.x, out = null) {
    const tq = this.tets, M = this.DmInv, o = 9 * t, a0 = 3 * tq[4 * t];
    const Ds = VDS;
    for (let k = 0; k < 3; k++) { const b = 3 * tq[4 * t + k + 1]; for (let r = 0; r < 3; r++) Ds[3 * r + k] = x[b + r] - x[a0 + r]; }
    const F = out ?? new Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) F[3 * i + j] = Ds[3 * i] * M[o + j] + Ds[3 * i + 1] * M[o + 3 + j] + Ds[3 * i + 2] * M[o + 6 + j];
    return F;
  }
  updateDamage() {
    const { T, alive, P } = this;
    let changed = false;
    for (let t = 0; t < T; t++) {
      if (!alive[t]) continue;
      const F = this.F(t, this.x, FF);
      const C = rcg(F, FC);
      if (maxEigSym3(C) <= 1) continue;
      const e = psiF(F, P);
      if (e > this.H[t]) {
        this.H[t] = e;
        const d = damageOf(e, this.l_e[t], P);
        this.dmg[t] = Math.min(d, DMAX);
        if (d >= 1) { alive[t] = 0; this.dmg[t] = 1; this.failed++; changed = true; this.failedLog.push({ t, H: e, time: this.time }); }
      }
    }
    if (changed) { this.topoVersion++; this.recomputeMass(); }
  }

  // ---------------------------------------------------------------- read-outs
  energy() {
    const { T, N, alive, P, V0, mass, v, x } = this;
    let el = 0, ke = 0, pe = 0;
    for (let t = 0; t < T; t++) if (alive[t]) el += (1 - this.dmg[t]) * V0[t] * psiTet(this.F(t), P);
    if (P.lamNode > 0) for (let v = 0; v < N; v++) { const J = this.patchJ(v); if (J !== null) el += 0.5 * P.lamNode * (this.Vv0[v] / 4) * (J - 1) ** 2; }
    for (let i = 0; i < N; i++) {
      ke += 0.5 * mass[i] * (v[3 * i] ** 2 + v[3 * i + 1] ** 2 + v[3 * i + 2] ** 2);
      pe -= mass[i] * (P.gravity[0] * x[3 * i] + P.gravity[1] * x[3 * i + 1] + P.gravity[2] * x[3 * i + 2]);
    }
    return { elastic: el, kinetic: ke, gravity: pe, total: el + ke + pe };
  }
  patchJ(v) {
    if (!(this.Vv0[v] > 0)) return null;
    let V = 0;
    for (let k = this.vt.start[v]; k < this.vt.start[v + 1]; k++) { const t = this.vt.list[k] >> 2; if (this.alive[t]) V += this.V0[t] * det3(this.F(t)); }
    return V / this.Vv0[v];
  }
  momentum() {
    const p = [0, 0, 0], L = [0, 0, 0];
    for (let i = 0; i < this.N; i++) {
      const m = this.mass[i], x = this.x.subarray(3 * i, 3 * i + 3), v = this.v.subarray(3 * i, 3 * i + 3);
      for (let a = 0; a < 3; a++) p[a] += m * v[a];
      L[0] += m * (x[1] * v[2] - x[2] * v[1]); L[1] += m * (x[2] * v[0] - x[0] * v[2]); L[2] += m * (x[0] * v[1] - x[1] * v[0]);
    }
    return { p, L };
  }
  /** Barycentric probe of a REST point: {tet, w[4]}; then probeAt(pr) gives its current position. */
  probe(X) {
    let best = null, bestMin = -Infinity;
    for (let t = 0; t < this.T; t++) {
      const q = this.tets.subarray(4 * t, 4 * t + 4), M = this.DmInv.subarray(9 * t, 9 * t + 9);
      const d = [X[0] - this.X0[3 * q[0]], X[1] - this.X0[3 * q[0] + 1], X[2] - this.X0[3 * q[0] + 2]];
      const b = [0, 1, 2].map((r) => M[3 * r] * d[0] + M[3 * r + 1] * d[1] + M[3 * r + 2] * d[2]);
      const w = [1 - b[0] - b[1] - b[2], b[0], b[1], b[2]];
      const mn = Math.min(...w);
      if (mn > bestMin) { bestMin = mn; best = { tet: t, w, q: Array.from(q) }; }
      if (mn >= -1e-12) break;
    }
    return best;
  }
  probeAt(pr) {
    const out = [0, 0, 0];
    for (let l = 0; l < 4; l++) for (let a = 0; a < 3; a++) out[a] += pr.w[l] * this.x[3 * pr.q[l] + a];
    return out;
  }
  totalMass() { let m = 0; for (let i = 0; i < this.N; i++) m += this.mass[i]; return m; }
  stats() {
    let maxS = 1, alive = 0;
    for (let t = 0; t < this.T; t++) if (this.alive[t]) { alive++; maxS = Math.max(maxS, Math.sqrt(maxEigSym3(rcg(this.F(t))))); }
    return { tets: alive, failed: this.failed, energy: this.energy(), maxStretch: maxS, contacts: this.lastContacts, ms: this.lastMs, time: this.time };
  }

  /** Boundary surface of the alive tets: {positions, normals (area-weighted, per vertex), indices, count, version}. */
  surface() {
    const { T, N, alive, nbr, tets, x } = this;
    const idx = [];
    for (let t = 0; t < T; t++) {
      if (!alive[t]) continue;
      for (let f = 0; f < 4; f++) {
        const o = nbr[4 * t + f];
        if (o >= 0 && alive[o]) continue;
        const fv = TET_FACES[f];
        idx.push(tets[4 * t + fv[0]], tets[4 * t + fv[1]], tets[4 * t + fv[2]]);
      }
    }
    const positions = new Float32Array(3 * N), normals = new Float32Array(3 * N);
    for (let i = 0; i < 3 * N; i++) positions[i] = x[i];
    const nacc = new Float64Array(3 * N);
    for (let f = 0; f < idx.length; f += 3) {
      const a = 3 * idx[f], b = 3 * idx[f + 1], c = 3 * idx[f + 2];
      const e1 = [x[b] - x[a], x[b + 1] - x[a + 1], x[b + 2] - x[a + 2]], e2 = [x[c] - x[a], x[c + 1] - x[a + 1], x[c + 2] - x[a + 2]];
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      for (const p of [a, b, c]) for (let r = 0; r < 3; r++) nacc[p + r] += n[r];
    }
    for (let i = 0; i < N; i++) {
      const l = Math.hypot(nacc[3 * i], nacc[3 * i + 1], nacc[3 * i + 2]) || 1;
      for (let r = 0; r < 3; r++) normals[3 * i + r] = nacc[3 * i + r] / l;
    }
    return { positions, normals, indices: Uint32Array.from(idx), count: idx.length, version: this.topoVersion, frame: this.version };
  }
}

// ---------------------------------------------------------------- shared helpers

const CDH = [0, 0];
/** The stable Neo-Hookean constraint pair of one tet (docs/JELLY.md energy): C_D = sqrt(I_C), C_H = J - alphaRest, and
 * their gradients wrt the 4 vertices (gD, gH: 12 each, vertex-major). d1..d3 = x1-x0, x2-x0, x3-x0; M = Dm^-1 (row-major,
 * at offset M.byteOffset). Returns [C_D, C_H] (a shared array) or null when F = 0. */
export function tetConstraints(d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z, M, alphaRest, gD, gH) {
  // F = Ds M (row-major)
    const F = [
      d1x * M[0] + d2x * M[3] + d3x * M[6], d1x * M[1] + d2x * M[4] + d3x * M[7], d1x * M[2] + d2x * M[5] + d3x * M[8],
      d1y * M[0] + d2y * M[3] + d3y * M[6], d1y * M[1] + d2y * M[4] + d3y * M[7], d1y * M[2] + d2y * M[5] + d3y * M[8],
      d1z * M[0] + d2z * M[3] + d3z * M[6], d1z * M[1] + d2z * M[4] + d3z * M[7], d1z * M[2] + d2z * M[5] + d3z * M[8]];
    let Ic = 0; for (let i = 0; i < 9; i++) Ic += F[i] * F[i];
    const CD = Math.sqrt(Ic);
    if (CD < 1e-12) return null;
    // cofactor (row-major)
    const cof = [
      F[4] * F[8] - F[5] * F[7], F[5] * F[6] - F[3] * F[8], F[3] * F[7] - F[4] * F[6],
      F[2] * F[7] - F[1] * F[8], F[0] * F[8] - F[2] * F[6], F[1] * F[6] - F[0] * F[7],
      F[1] * F[5] - F[2] * F[4], F[2] * F[3] - F[0] * F[5], F[0] * F[4] - F[1] * F[3]];
    const J = F[0] * cof[0] + F[1] * cof[1] + F[2] * cof[2];
    const CH = J - alphaRest;
    // gradients wrt x1..x3: G = (dC/dF) M^T ; columns are the vertex gradients
    for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) {
      let sD = 0, sH = 0;
      for (let j = 0; j < 3; j++) { sD += F[3 * r + j] * M[3 * k + j]; sH += cof[3 * r + j] * M[3 * k + j]; }
      gD[3 * (k + 1) + r] = sD / CD; gH[3 * (k + 1) + r] = sH;
    }
    for (let r = 0; r < 3; r++) { gD[r] = -(gD[3 + r] + gD[6 + r] + gD[9 + r]); gH[r] = -(gH[3 + r] + gH[6 + r] + gH[9 + r]); }
    CDH[0] = CD; CDH[1] = CH;
    return CDH;
}


export function rcg(F, out = null) {
  const C = out ?? new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = F[i] * F[j] + F[3 + i] * F[3 + j] + F[6 + i] * F[6 + j];
  return C;
}
/** Per-tet part of the discrete energy density: mu/2 (I_C - 3) - mu (J - 1) + lamTet/2 (J - 1)^2. */
export function psiTet(F, P) {
  let Ic = 0; for (let i = 0; i < 9; i++) Ic += F[i] * F[i];
  const J = det3(F);
  return 0.5 * P.mu * (Ic - 3) - P.mu * (J - 1) + 0.5 * P.lamTet * (J - 1) * (J - 1);
}

/** Vertex patches coloured so that two patches of one colour share no vertex (2-ring colouring); returns
 * {order: vertices in colour order, colourStart}. */
export function colourPatches(tets, vt, N) {
  const nb = new Array(N);
  for (let v = 0; v < N; v++) {
    const s = new Set();
    for (let k = vt.start[v]; k < vt.start[v + 1]; k++) { const t = vt.list[k] >> 2; for (let l = 0; l < 4; l++) s.add(tets[4 * t + l]); }
    nb[v] = s;
  }
  const colour = new Int32Array(N).fill(-1); let C = 0;
  for (let v = 0; v < N; v++) {
    const used = new Set();
    for (const u of nb[v]) for (const w of nb[u]) if (colour[w] >= 0) used.add(colour[w]);
    let c = 0; while (used.has(c)) c++;
    colour[v] = c; C = Math.max(C, c + 1);
  }
  const colourStart = new Uint32Array(C + 1);
  for (let v = 0; v < N; v++) colourStart[colour[v] + 1]++;
  for (let c = 0; c < C; c++) colourStart[c + 1] += colourStart[c];
  const fill = colourStart.slice(0, C), order = new Uint32Array(N);
  for (let v = 0; v < N; v++) order[fill[colour[v]]++] = v;
  return { order, colourStart, colour };
}

export function psiF(F, P) {
  let Ic = 0; for (let i = 0; i < 9; i++) Ic += F[i] * F[i];
  const J = det3(F);
  return 0.5 * P.mu * (Ic - 3) - P.mu * (J - 1) + 0.5 * P.lamp * (J - 1) * (J - 1);
}

/** Point (collider-local) vs collider: null or [depth > 0, nx, ny, nz] (local outward normal). */
export function contactLocal(c, p) {
  if (c.kind === 'convex') {
    const cx = p[0] - c.center[0], cy = p[1] - c.center[1], cz = p[2] - c.center[2];
    if (cx * cx + cy * cy + cz * cz > c.radius * c.radius) return null;
    const pl = c.planes;
    let best = -Infinity, bi = -1;
    for (let k = 0; k < pl.length; k += 4) {
      const s = pl[k] * p[0] + pl[k + 1] * p[1] + pl[k + 2] * p[2] - pl[k + 3];
      if (s >= 0) return null;
      if (s > best) { best = s; bi = k; }
    }
    return [-best, pl[bi], pl[bi + 1], pl[bi + 2]];
  }
  if (c.kind === 'capsule') {
    const a = c.a, b = c.b;
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    const L2 = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2;
    let u = L2 > 0 ? (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / L2 : 0;
    u = Math.max(0, Math.min(1, u));
    const d = [ap[0] - u * ab[0], ap[1] - u * ab[1], ap[2] - u * ab[2]];
    const l = Math.hypot(d[0], d[1], d[2]);
    if (l >= c.r || l < 1e-15) return null;
    return [c.r - l, d[0] / l, d[1] / l, d[2] / l];
  }
  if (c.kind === 'sphere') {
    const d = [p[0] - c.c[0], p[1] - c.c[1], p[2] - c.c[2]];
    const l = Math.hypot(d[0], d[1], d[2]);
    if (l >= c.r || l < 1e-15) return null;
    return [c.r - l, d[0] / l, d[1] / l, d[2] / l];
  }
  if (c.kind === 'halfspace') {          // local plane n.p <= d is solid
    const s = c.n[0] * p[0] + c.n[1] * p[1] + c.n[2] * p[2] - c.d;
    if (s >= 0) return null;
    return [-s, c.n[0], c.n[1], c.n[2]];
  }
  return null;
}
