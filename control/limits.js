// Joint limits of the twin's tools, against the scene's own geometry (docs/WEB_TWIN.md §5: "insertion is clamped so
// that the jaws never pass through the bench or the shell").
//
//   * port cone: the shaft stays within cone_half_deg of the port axis (scene.ports[].cone_half_deg);
//   * insertion in [min, max] (spec.min_insertion / spec.max_insertion, defaults 0 and 0.33 m as twin.instrument);
//   * jaw in [0, spec.jaw_max];
//   * no shaft, hinge or jaw-tip point below the bench plane (plus the part's radius and a margin);
//   * no shaft or jaw segment through the static shell: rays in a BVH of the static meshes (shell, hatch, lid, lens
//     fronts; the port membrane and cap, which the shaft passes through by design, are left out).
// When a requested state breaks a geometric limit the insertion is pulled back (bisection) until it holds, so a tool
// swung into the wall withdraws instead of passing through it. Pure ES module (node + browser).

import { buildBVH, packTriangles, traverse } from '../gpu/bvh.js';
import { DEG, port, spec, forward, scopeForward, add, scale, coneClamp, scopeMinInsertion } from './kinematics.js';

// the slit silicone (and any cap) is passed by design; dynamic bodies (tools, trocars, scope, board) are not static
const PASS_THROUGH_GROUPS = new Set(['membrane', 'port_cap', 'instrument_part', 'scope', 'board', 'trocar']);

export const LIMIT_DEFAULTS = {
  marginShell: 0.0010,        // m kept between any tool point and the shell
  marginBench: 0.0005,        // m above the bench
  skipNearPivot: 0.003,       // m: hits closer than this to the pivot are the port's own bore
  minInsertion: 0.005,        // m, hinge past the membrane (spec.min_insertion overrides)
  nearPort: 0.025,            // m: a conflict with the hinge closer than this to the pivot closes the jaws (the bore
                              //    closes them) instead of withdrawing the tool
  scopeMinInsertion: 0.004,
  scopeMaxInsertion: 0.20,
  bisectSteps: 18,
};

/** Merge every static collision mesh into one BVH. meshes: {name: {P, I}} as decoded from scene meshes/*.bin. */
export function collisionBVH(scene, meshes) {
  const parts = scene.meshes.filter((m) => !m.dynamic && !PASS_THROUGH_GROUPS.has(m.group) && meshes[m.name]);
  let nv = 0, ni = 0;
  for (const m of parts) { nv += meshes[m.name].P.length; ni += meshes[m.name].I.length; }
  if (!ni) return null;
  const P = new Float32Array(nv), I = new Uint32Array(ni);
  let vo = 0, io = 0;
  for (const m of parts) {
    const { P: p, I: idx } = meshes[m.name];
    P.set(p, vo);
    const base = vo / 3;
    for (let k = 0; k < idx.length; k++) I[io + k] = idx[k] + base;
    vo += p.length; io += idx.length;
  }
  const bvh = buildBVH(P, I, { leafSize: 4 });
  const T = packTriangles(P, I, bvh.order);
  return { nodes: bvh.nodes, T, triCount: I.length / 3, names: parts.map((m) => m.name) };
}

export class Limits {
  constructor(scene, meshes = null, opts = {}) {
    this.scene = scene;
    this.o = { ...LIMIT_DEFAULTS, ...opts };
    this.bvh = meshes ? collisionBVH(scene, meshes) : null;
    this.benchZ = scene.bench?.z ?? 0;
    this.ports = {};
    for (const p of scene.ports) this.ports[p.id] = port(scene, p.id);
    this.specs = {};
    for (const ins of scene.instruments) this.specs[ins.id] = spec(ins);
  }

  /** First hit distance along a ray (Infinity on a miss). */
  hit(o, d, tmin = 0, tmax = Infinity) {
    if (!this.bvh) return Infinity;
    return traverse(this.bvh.nodes, this.bvh.T, o, d, { tmin, tmax, anyHit: false }).t;
  }

  /** Clamp the shaft direction into the port cone (kinematics.coneClamp). */
  static cone(q, cone) { return coneClamp(q, cone); }

  /** Is every tool point of instrument `id` at state q clear of the bench and the shell? */
  instrumentClear(id, q, portRec, sp) {
    const o = this.o;
    const f = forward(portRec.pivot, portRec.R, sp, q);
    const zmin = this.benchZ + o.marginBench;
    if (f.hinge[2] < zmin + sp.shaft_radius) return 'bench';
    for (const t of f.jaw_tips) if (t[2] < zmin + (sp.jaw_radius ?? 0.0012)) return 'bench';
    if (this.bvh) {
      const tAxis = this.hit(portRec.pivot, f.d, o.skipNearPivot, q.insertion + o.marginShell + sp.shaft_radius);
      if (tAxis < Infinity) return 'shell';
      for (const a of f.jaw_dirs) {
        if (this.hit(f.hinge, a, 0, sp.jaw_length + o.marginShell + (sp.jaw_radius ?? 0.0012)) < Infinity) return 'shell';
      }
    }
    return null;
  }

  clampInstrument(id, q0) {
    const ins = this.scene.instruments.find((i) => i.id === id);
    const pr = this.ports[ins.port], sp = this.specs[id];
    const limited = [];
    let q = coneClamp(q0, pr.cone);
    if (q._cone) limited.push('cone');
    delete q._cone;
    const jmax = sp.jaw_max;
    if (q.jaw < 0) { q.jaw = 0; limited.push('jaw'); } else if (q.jaw > jmax) { q.jaw = jmax; limited.push('jaw'); }
    const lo = sp.min_insertion ?? this.o.minInsertion, hi = sp.max_insertion ?? 0.33;
    if (q.insertion < lo) { q.insertion = lo; limited.push('insertion_min'); }
    if (q.insertion > hi) { q.insertion = hi; limited.push('insertion_max'); }
    const clearAt = (qq, s, jaw) => !this.instrumentClear(id, { ...qq, insertion: s, jaw }, pr, sp);
    const why = this.instrumentClear(id, q, pr, sp);
    if (!why) return { q, limited };
    limited.push(why);
    const floor = Math.min(q.insertion, Math.max(lo, this.o.nearPort));
    if (!clearAt(q, floor, 0)) {
      // even withdrawn to the floor with the jaws closed the shaft meets the shell: the direction itself is out of
      // reach (e.g. a tilt the port's cone table allows but the cannula and shaft cannot take inside this dome).
      // Swing back toward the port axis, keeping the azimuth, to the largest clear tilt.
      const back = this._tiltBack(q, (qq) => clearAt(qq, floor, 0));
      if (back) { q = back; limited.push('tilt'); } else { q.insertion = floor; q.jaw = 0; limited.push('stuck'); return { q, limited }; }
      if (!this.instrumentClear(id, q, pr, sp)) return { q, limited };
    }
    if (clearAt(q, floor, q.jaw)) {
      // withdraw: largest clear insertion in [floor, request]
      let a = floor, b = q.insertion;
      for (let k = 0; k < this.o.bisectSteps; k++) { const m = 0.5 * (a + b); if (clearAt(q, m, q.jaw)) a = m; else b = m; }
      q.insertion = a;
    } else {
      // the conflict is at the floor: close the jaws to the largest clear opening there
      q.insertion = floor;
      let a = 0, b = q.jaw;
      for (let k = 0; k < this.o.bisectSteps; k++) { const m = 0.5 * (a + b); if (clearAt(q, floor, m)) a = m; else b = m; }
      q.jaw = a;
      limited.push('jaw_bore');
    }
    return { q, limited };
  }

  /** Scale a state's tilt from the port axis (same azimuth) down to the largest value for which ok() holds. */
  _tiltBack(q, ok) {
    const cp = Math.cos(q.pitch);
    const dP = [cp * Math.sin(q.yaw), -Math.sin(q.pitch), cp * Math.cos(q.yaw)];
    const tilt = Math.acos(Math.max(-1, Math.min(1, dP[2])));
    const l = Math.hypot(dP[0], dP[1]) || 1;
    const at = (f) => {
      const t = tilt * f, s = Math.sin(t);
      const d = [dP[0] / l * s, dP[1] / l * s, Math.cos(t)];
      return { ...q, yaw: Math.atan2(d[0], d[2]), pitch: -Math.asin(Math.max(-1, Math.min(1, d[1]))) };
    };
    if (!ok(at(0))) return null;
    let a = 0, b = 1;
    for (let k = 0; k < this.o.bisectSteps; k++) { const m = 0.5 * (a + b); if (ok(at(m))) a = m; else b = m; }
    return at(a);
  }

  scopeClear(q) {
    const o = this.o;
    const s = scopeForward(this.scene, q);
    const r = 0.005;
    if (s.lens[2] < this.benchZ + o.marginBench + r) return 'bench';
    if (this.bvh && this.hit(s.pivot, s.d, o.skipNearPivot, q.insertion + o.marginShell + r) < Infinity) return 'shell';
    return null;
  }

  clampScope(q0) {
    const pr = this.ports[this.scene.scope?.port ?? 'C'];
    const limited = [];
    let q = coneClamp(q0, pr.cone);
    if (q._cone) limited.push('cone');
    delete q._cone;
    const lo = Math.max(this.o.scopeMinInsertion, scopeMinInsertion(this.scene)), hi = Math.max(this.o.scopeMaxInsertion, lo + 0.05);
    if (q.insertion < lo) { q.insertion = lo; limited.push('insertion_min'); }
    if (q.insertion > hi) { q.insertion = hi; limited.push('insertion_max'); }
    const why = this.scopeClear(q);
    if (!why) return { q, limited };
    limited.push(why);
    if (this.scopeClear({ ...q, insertion: lo })) {
      const back = this._tiltBack(q, (qq) => !this.scopeClear({ ...qq, insertion: lo }));
      if (!back) { q.insertion = lo; limited.push('stuck'); return { q, limited }; }
      q = back; limited.push('tilt');
      if (!this.scopeClear(q)) return { q, limited };
    }
    let a = lo, b = q.insertion;
    for (let k = 0; k < this.o.bisectSteps; k++) { const m = 0.5 * (a + b); if (!this.scopeClear({ ...q, insertion: m })) a = m; else b = m; }
    q.insertion = a;
    return { q, limited };
  }

  /** Clamp a whole §5 state. Returns {state, limited: {L: [...], R: [...], scope: [...]}}. */
  clampState(state) {
    const limited = {};
    const instruments = state.instruments.map((q) => {
      const r = this.clampInstrument(q.id, { ...q });
      limited[q.id] = r.limited;
      return r.q;
    });
    let scope = state.scope;
    if (scope && this.scene.scope) {
      const r = this.clampScope({ ...scope });
      limited.scope = r.limited;
      scope = r.q;
    }
    return { state: { ...state, instruments, scope }, limited };
  }

  /** Points of an instrument for external checks: [pivot..hinge samples, jaw tips] (tests use it). */
  samplePoints(id, q, n = 24) {
    const ins = this.scene.instruments.find((i) => i.id === id);
    const pr = this.ports[ins.port], sp = this.specs[id];
    const f = forward(pr.pivot, pr.R, sp, q);
    const pts = [];
    for (let k = 0; k <= n; k++) pts.push(add(pr.pivot, scale(f.d, (q.insertion * k) / n)));
    return { pivot: pr.pivot, hinge: f.hinge, jaw_tips: f.jaw_tips, shaft: pts, d: f.d };
  }
}

export { DEG };
