// Scripted autoplay, seeded and deterministic (same scene + seed -> the same state at every t, in node and in the
// page). A script is planned once into per-body tracks of minimum-jerk moves; stateAt(t) is a pure function of t.
//
//   task_set           : reaches across the workspace, grasp open/close cycles, the two instruments crossing and
//                        occluding each other, a 10 s still hold (tip jitter, CLAUDE.md targets), scope re-aims.
//                        Tip speeds are surgical (min-jerk mean 8-50 mm/s, so peaks stay <= 100 mm/s); yaw/pitch
//                        stay inside 85 % of each port cone; every target is checked against the clamp.
//   charuco_rig        : the ChArUco board held still ~1 s at random poses in the cameras' shared view, then moved
//                        (the input `veridien multicam calibrate` + `triangulate` expect: still views, >= 2 cameras).
//   charuco_intrinsics : one camera; the board swept over its whole field, image corners included, at varied tilt.
//
// Minimum-jerk position profile s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 (Flash & Hogan 1985), the standard model of
// a skilled point-to-point reach. Hinge targets are Cartesian (straight lines in D); joints follow by aim_at.

import { Rng } from './rng.js';
import {
  DEG, port, spec, forward, aimAt, rotX, rotY, add, sub, scale, norm, normalize, dot, cross, defaultState, truthFromState,
  workspaceCentre, scopeDefault, fromRows, frameFromZ, matmul, rotZ, quatFromR, rFromQuat, coneClamp, ensureLenses,
  scopeMinInsertion,
} from './kinematics.js';

export const SCRIPTS = ['task_set', 'charuco_rig', 'charuco_intrinsics'];

const mj = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * u * (10 + u * (-15 + 6 * u)));
/** Peak speed of a min-jerk move = 1.875 x mean speed. */
export const MJ_PEAK = 1.875;

// ------------------------------------------------------------------ tracks

class Track {
  constructor(v0) { this.segs = []; this.v = v0; this.t = 0; }
  /** Move from the current value to v over [t0, t0 + dur]; holds before. */
  move(t0, dur, v, kind = 'move') { this.segs.push({ t0, t1: t0 + dur, a: this.v, b: v, kind }); this.v = v; this.t = t0 + dur; return this.t; }
  hold(t0, dur, kind = 'hold') { this.segs.push({ t0, t1: t0 + dur, a: this.v, b: this.v, kind }); this.t = t0 + dur; return this.t; }
  at(t, lerp) {
    const s = this.segs;
    if (!s.length) return this.v;
    if (t <= s[0].t0) return s[0].a;
    // binary search the last segment starting at or before t
    let lo = 0, hi = s.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (s[m].t0 <= t) lo = m; else hi = m - 1; }
    const g = s[lo];
    if (t >= g.t1) return g.b;
    return lerp(g.a, g.b, mj((t - g.t0) / (g.t1 - g.t0)));
  }
  kindAt(t) {
    for (const g of this.segs) if (t >= g.t0 && t < g.t1) return g.kind;
    return 'idle';
  }
}

const lerpTool = (a, b, w) => ({ hinge: a.hinge.map((x, i) => x + (b.hinge[i] - x) * w), roll: a.roll + (b.roll - a.roll) * w, jaw: a.jaw + (b.jaw - a.jaw) * w });
const lerpJoint = (a, b, w) => ({ yaw: a.yaw + (b.yaw - a.yaw) * w, pitch: a.pitch + (b.pitch - a.pitch) * w, roll: a.roll + (b.roll - a.roll) * w, insertion: a.insertion + (b.insertion - a.insertion) * w });
function lerpBoard(a, b, w) {
  if (!a.T || !b.T) return w < 1 ? a : b;
  const qa = quatFromR(a.R); let qb = quatFromR(b.R);
  if (qa.reduce((s, x, i) => s + x * qb[i], 0) < 0) qb = qb.map((x) => -x);
  const c = Math.min(1, qa.reduce((s, x, i) => s + x * qb[i], 0));
  let q;
  if (c > 0.9995) { q = qa.map((x, i) => x + (qb[i] - x) * w); const n = Math.hypot(...q); q = q.map((x) => x / n); }
  else { const th = Math.acos(c), s = Math.sin(th); q = qa.map((x, i) => (Math.sin((1 - w) * th) * x + Math.sin(w * th) * qb[i]) / s); }
  const R = rFromQuat(q);
  return { R, p: a.p.map((x, i) => x + (b.p[i] - x) * w), T: true };
}

/** Min-jerk duration for a distance at a mean speed (s), never shorter than tmin. */
const durFor = (dist, vMean, tmin = 0.35) => Math.max(tmin, dist / vMean);

// ------------------------------------------------------------------ planner helpers

class Planner {
  constructor(scene, seed, { limits = null } = {}) {
    this.scene = scene;
    this.rng = new Rng(seed >>> 0);
    this.limits = limits;
    this.base = defaultState(scene);
    this.tools = {};
    for (const ins of scene.instruments) {
      const p = port(scene, ins.port), sp = spec(ins);
      const q = this.base.instruments.find((x) => x.id === ins.id);
      const f = forward(p.pivot, p.R, sp, q);
      this.tools[ins.id] = { ins, p, sp, track: new Track({ hinge: f.hinge, roll: q.roll, jaw: q.jaw }) };
    }
    this.scope = new Track(scopeDefault(scene));
    this.scopeDefault = scopeDefault(scene);
    this.boards = {};
    for (const b of scene.boards || []) this.boards[b.name] = { rec: b, track: new Track({ T: false }) };
    this.centre = workspaceCentre(scene);
    this.labels = [];
  }

  label(t0, t1, name) { this.labels.push({ t0, t1, name }); }

  /** A seeded start state: each grasper at a random clear target with its own roll and opening, the scope swung
   * up to 0.12 rad from its default (inside its cone and clear by the clamp). */
  randomStart() {
    for (const tool of Object.values(this.tools)) {
      const h = this.target(tool, { jaw: 0.9, radius: 0.05 });
      tool.track.v = { hinge: [...h], roll: this.rng.uniform(-1.2, 1.2), jaw: this.rng.uniform(0.1, 0.8) };
    }
    const sd = this.scopeDefault;
    let q = { ...sd, yaw: sd.yaw + this.rng.uniform(-0.12, 0.12), pitch: sd.pitch + this.rng.uniform(-0.12, 0.12), roll: sd.roll + this.rng.uniform(-0.5, 0.5) };
    if (this.limits) q = this.limits.clampScope(q).q;
    this.scope.v = q;
  }

  /** Joint state of one tool from its Cartesian track value. */
  joints(tool, v) {
    const q = aimAt(tool.p.pivot, tool.p.R, v.hinge, v.roll, v.jaw);
    return { id: tool.ins.id, ...q };
  }

  /** Is a hinge target reachable and clear (cone with 15 % to spare, clamp passes it unchanged with jaws open)? */
  okTarget(tool, hinge, jawCheck) {
    const q = aimAt(tool.p.pivot, tool.p.R, hinge, 0, jawCheck);
    const c = coneClamp(q, tool.p.cone * 0.85);
    if (c._cone) return false;
    if (q.insertion < (tool.sp.min_insertion ?? 0) + 0.003 || q.insertion > (tool.sp.max_insertion ?? 0.33) * 0.95) return false;
    if (hinge[2] < (this.scene.bench?.z ?? 0) + 0.008) return false;
    if (this.limits) {
      const r = this.limits.clampInstrument(tool.ins.id, { ...q });
      if (r.limited.length) return false;
    }
    return true;
  }

  /** A random hinge target inside the tool's reachable cone (85 %), clear by the clamp, seen by >= minCams cameras
   * (relaxed to fewer when none is), preferring points near the shared workspace (radius around the reach centre).
   * side: '+' / '-' keeps the target on that side (D y) of the midline between the working ports. */
  target(tool, { side = null, jaw = 0.9, minCams = 2, radius = 0.05 } = {}) {
    const ys = this.scene.ports.filter((p) => p.role !== 'scope').map((p) => p.pivot[1]);
    const yMid = ys.reduce((a, b) => a + b, 0) / Math.max(1, ys.length);
    const centre = this.reachCentre();
    for (const need of [minCams, Math.min(minCams, 1), 0]) {
      for (let k = 0; k < 300; k++) {
        const r = Math.sqrt(this.rng.u()) * tool.p.cone * 0.85, az = this.rng.uniform(0, 2 * Math.PI);
        const dP = [Math.sin(r) * Math.cos(az), Math.sin(r) * Math.sin(az), Math.cos(r)];
        const d = [0, 1, 2].map((i) => tool.p.R[i][0] * dP[0] + tool.p.R[i][1] * dP[1] + tool.p.R[i][2] * dP[2]);
        const lo = Math.max(0.05, (tool.sp.min_insertion ?? 0) + 0.003);
        const h = add(tool.p.pivot, scale(d, this.rng.uniform(lo, Math.max(lo + 0.01, Math.min(lo + 0.16, (tool.sp.max_insertion ?? 0.33) * 0.9)))));
        if (radius && norm(sub(h, centre)) > radius * 1.8) continue;
        if (side === '+' && h[1] < yMid) continue;
        if (side === '-' && h[1] > yMid) continue;
        if (!this.okTarget(tool, h, jaw)) continue;
        if (need > 0 && this.camerasSeeing(h) < need) continue;
        return h;
      }
    }
    return [...tool.track.v.hinge];                    // nothing found: stay
  }

  /** The centre of the region both working tools reach and the cameras see (computed once by sampling). */
  reachCentre() {
    if (this._rc) return this._rc;
    const rng = new Rng(12345);
    const pts = [];
    for (const tool of Object.values(this.tools)) {
      for (let k = 0; k < 600; k++) {
        const r = Math.sqrt(rng.u()) * tool.p.cone * 0.85, az = rng.uniform(0, 2 * Math.PI);
        const dP = [Math.sin(r) * Math.cos(az), Math.sin(r) * Math.sin(az), Math.cos(r)];
        const d = [0, 1, 2].map((i) => tool.p.R[i][0] * dP[0] + tool.p.R[i][1] * dP[1] + tool.p.R[i][2] * dP[2]);
        const lo = Math.max(0.05, (tool.sp.min_insertion ?? 0) + 0.003);
        const h = add(tool.p.pivot, scale(d, rng.uniform(lo, lo + 0.16)));
        if (h[2] < (this.scene.bench?.z ?? 0) + 0.01) continue;
        if (this.camerasSeeing(h) < 2) continue;
        if (this.limits && this.limits.clampInstrument(tool.ins.id, { ...aimAt(tool.p.pivot, tool.p.R, h, 0, 0.3) }).limited.length) continue;
        pts.push(h);
      }
    }
    this._rc = pts.length ? scale(pts.reduce(add, [0, 0, 0]), 1 / pts.length) : this.centre;
    return this._rc;
  }

  camerasSeeing(pD) {
    let n = 0;
    for (const cam of ensureLenses(this.scene)) {
      const X = [0, 1, 2].map((i) => cam.T_C_D[i] * pD[0] + cam.T_C_D[4 + i] * pD[1] + cam.T_C_D[8 + i] * pD[2] + cam.T_C_D[12 + i]);
      if (X[2] <= 0.005) continue;
      const th = Math.atan2(Math.hypot(X[0], X[1]), X[2]);
      if (th > (cam.lens.thetaMax ?? 1.2) * 0.95) continue;
      const fx = cam.lens.fx, cx = cam.lens.cx, cy = cam.lens.cy, fy = cam.lens.fy;
      const k = cam.lens.k, t2 = th * th;
      const thd = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
      const r = Math.hypot(X[0], X[1]) || 1e-12;
      const u = fx * (thd * X[0] / r) + cx, v = fy * (thd * X[1] / r) + cy;
      if (u > 40 && v > 40 && u < cam.size[0] - 40 && v < cam.size[1] - 40) n++;
    }
    return n;
  }

  /** Move a tool's hinge to h at a random surgical mean speed. Returns the end time. */
  reach(id, t0, h, { roll = null, jaw = null, vMin = 0.008, vMax = 0.050, kind = 'reach' } = {}) {
    const tool = this.tools[id];
    const cur = tool.track.v;
    const dist = norm(sub(h, cur.hinge));
    const v = this.rng.uniform(vMin, vMax);
    const dur = durFor(dist, v);
    return tool.track.move(t0, dur, { hinge: h, roll: roll ?? cur.roll, jaw: jaw ?? cur.jaw }, kind);
  }

  jaw(id, t0, jaw, dur = null, kind = 'jaw') {
    const tool = this.tools[id];
    const cur = tool.track.v;
    const d = dur ?? Math.max(0.25, Math.abs(jaw - cur.jaw) / 1.6);   // ~1.6 rad/s opening rate
    return tool.track.move(t0, d, { ...cur, jaw }, kind);
  }

  holdAll(t0, dur, kind = 'hold') {
    for (const t of Object.values(this.tools)) { if (t.track.t < t0) t.track.hold(t.track.t, t0 - t.track.t); t.track.hold(t0, dur, kind); }
    if (this.scope.t < t0) this.scope.hold(this.scope.t, t0 - this.scope.t);
    this.scope.hold(t0, dur, kind);
    return t0 + dur;
  }

  /** Park the instruments along their port axes, withdrawn to their minimum (jaws just past the cannula toe) and
   * closed, and the scope at its minimum, so a board can move freely. */
  park(t0) {
    let t1 = t0;
    for (const tool of Object.values(this.tools)) {
      const q = { yaw: 0, pitch: 0, roll: 0, insertion: (tool.sp.min_insertion ?? 0.03) + 0.0005, jaw: 0 };
      const f = forward(tool.p.pivot, tool.p.R, tool.sp, q);
      t1 = Math.max(t1, tool.track.move(t0, durFor(norm(sub(f.hinge, tool.track.v.hinge)), 0.05, 0.8), { hinge: f.hinge, roll: 0, jaw: 0 }, 'park'));
    }
    const sd = this.scopeDefault;
    t1 = Math.max(t1, this.scope.move(t0, 0.8, { ...sd, insertion: Math.max(scopeMinInsertion(this.scene) + 0.0005, Math.min(sd.insertion, 0.012)) }, 'park'));
    this.parked = { until: t1 };
    return t1;
  }

  build(duration, name, seed) {
    const T = duration;
    const tools = this.tools, scope = this.scope, boards = this.boards;
    const scene = this.scene;
    const labels = this.labels;
    const self = this;
    return {
      name, seed, duration: T, labels,
      stateAt(t) {
        const instruments = Object.values(tools).map((tool) => self.joints(tool, tool.track.at(t, lerpTool)));
        const sc = scope.at(t, lerpJoint);
        const bs = (scene.boards || []).map((b) => {
          const v = boards[b.name].track.at(t, lerpBoard);
          if (!v.T) return { name: b.name, T_D_B: null, visible: false };
          const R = v.R, p = v.p;
          return { name: b.name, visible: true, T_D_B: [[R[0][0], R[0][1], R[0][2], p[0]], [R[1][0], R[1][1], R[1][2], p[1]], [R[2][0], R[2][1], R[2][2], p[2]], [0, 0, 0, 1]] };
        });
        return { t, instruments, scope: { ...sc }, boards: bs };
      },
      phaseAt(t) {
        for (const l of labels) if (t >= l.t0 && t < l.t1) return l.name;
        return 'idle';
      },
    };
  }
}

// ------------------------------------------------------------------ task_set

function taskSet(scene, seed, opts) {
  const P = new Planner(scene, seed, opts);
  if (opts.varyStart) P.randomStart();
  // phase sizes (defaults: the full set; the dataset uses a compact ~30 s set with every phase)
  const nReach = opts.reaches ?? 6, nGrasp = opts.graspCycles ?? 3, nCross = opts.crossings ?? 2;
  const stillS = opts.stillHold ?? 10.0, nReaim = opts.reaims ?? 3;
  const ids = Object.keys(P.tools);
  const [A, B] = ids;
  let t = 0.5;
  // 1. reaches across the workspace, the two hands alternating and overlapping
  const t1s = t;
  for (let k = 0; k < nReach; k++) {
    for (const id of ids) {
      const tool = P.tools[id];
      const h = P.target(tool, { jaw: 0.9, radius: 0.05 });
      const start = Math.max(tool.track.t, t + (id === B ? P.rng.uniform(0.1, 0.6) : 0));
      if (tool.track.t < start) tool.track.hold(tool.track.t, start - tool.track.t);
      P.reach(id, start, h, { roll: P.rng.uniform(-1.2, 1.2), jaw: P.rng.uniform(0.2, 0.8) });
    }
    t = Math.max(...ids.map((id) => P.tools[id].track.t));
  }
  P.label(t1s, t, 'reach');
  // 2. grasp cycles: open wide, close fully (a grasp), hold, reopen; three per instrument
  const t2s = t;
  for (const id of ids) {
    const tool = P.tools[id];
    let tt = Math.max(t, tool.track.t);
    const h = P.target(tool, { jaw: tool.sp.jaw_max, radius: 0.035 });
    tt = P.reach(id, tt, h, { jaw: 0.3 });
    for (let c = 0; c < nGrasp; c++) {
      tt = P.jaw(id, tt, tool.sp.jaw_max * P.rng.uniform(0.75, 1.0), null, 'grasp_open');
      tt = tool.track.hold(tt, P.rng.uniform(0.2, 0.5));
      tt = P.jaw(id, tt, 0, null, 'grasp_close');
      tt = tool.track.hold(tt, P.rng.uniform(0.3, 0.8), 'grasp_hold');
    }
    tt = P.jaw(id, tt, 0.4);
  }
  t = Math.max(...ids.map((id) => P.tools[id].track.t));
  P.label(t2s, t, 'grasp');
  // 3. crossing: each tip to the other side, so the shafts cross and occlude each other in the camera views
  const t3s = t;
  if (B) {
    for (let k = 0; k < nCross; k++) {
      const portY = (id) => P.tools[id].p.pivot[1];
      const yMid = (portY(A) + portY(B)) / 2;
      const sideOf = (id) => (portY(id) > yMid ? '-' : '+');            // the side away from its own port
      const hA = [...P.target(P.tools[A], { side: sideOf(A), jaw: 0.6, radius: 0.04 })];
      const hB = [...P.target(P.tools[B], { side: sideOf(B), jaw: 0.6, radius: 0.04 })];
      // pass at different heights so one shaft goes over the other
      hA[2] += 0.008; hB[2] -= 0.008;
      const ok = (tool, h) => P.okTarget(tool, h, 0.6);
      const ta = ok(P.tools[A], hA) ? P.reach(A, t, hA, { vMin: 0.015, vMax: 0.035, kind: 'cross' }) : t;
      const tb = ok(P.tools[B], hB) ? P.reach(B, t + 0.2, hB, { vMin: 0.015, vMax: 0.035, kind: 'cross' }) : t;
      t = Math.max(ta, tb);
      for (const id of ids) P.tools[id].track.hold(P.tools[id].track.t, t + 0.6 - P.tools[id].track.t);
      t += 0.6;
      // back to own side
      const backA = P.target(P.tools[A], { side: sideOf(A) === '+' ? '-' : '+', radius: 0.04 });
      const backB = P.target(P.tools[B], { side: sideOf(B) === '+' ? '-' : '+', radius: 0.04 });
      t = Math.max(P.reach(A, t, backA, { kind: 'uncross' }), P.reach(B, t + 0.15, backB, { kind: 'uncross' }));
    }
  }
  for (const id of ids) { const tr = P.tools[id].track; if (tr.t < t) tr.hold(tr.t, t - tr.t); }
  P.label(t3s, t, 'cross');
  // 4. a 10 s still hold of everything (the stationary-instrument jitter statistic)
  const t4s = t;
  t = P.holdAll(t, stillS, 'still');
  P.label(t4s, t, 'still_hold');
  // 5. scope re-aims: 3 new directions inside its cone, ~20 deg/s, and a roll of the 30 deg optic
  const t5s = t;
  const sp = port(scene, scene.scope?.port ?? 'C');
  const sd = P.scopeDefault;
  for (let k = 0; k < nReaim; k++) {
    const q = coneClamp({ yaw: sd.yaw + P.rng.uniform(-0.25, 0.25), pitch: sd.pitch + P.rng.uniform(-0.25, 0.25), roll: sd.roll + P.rng.uniform(-0.6, 0.6), insertion: sd.insertion + P.rng.uniform(-0.004, 0.012) }, sp.cone * 0.85);
    delete q._cone;
    let qq = q;
    if (opts.limits) { const r = opts.limits.clampScope({ ...q }); qq = r.q; }
    const cur = P.scope.v;
    const ang = Math.hypot(qq.yaw - cur.yaw, qq.pitch - cur.pitch, (qq.roll - cur.roll) * 0.5);
    t = P.scope.move(t, Math.max(0.8, ang / (20 * DEG)), qq, 'scope_reaim');
    t = P.scope.hold(t, 1.0);
    // instruments keep working while the view moves
    for (const id of ids) {
      const tool = P.tools[id];
      if (tool.track.t < t - 1.5) tool.track.hold(tool.track.t, t - 1.5 - tool.track.t);
      P.reach(id, tool.track.t, P.target(tool, { radius: 0.04 }));
    }
  }
  t = P.scope.move(Math.max(t, P.scope.t), 1.2, { ...sd }, 'scope_reaim');
  t = Math.max(t, ...ids.map((id) => P.tools[id].track.t));
  P.label(t5s, t, 'scope_reaim');
  for (const id of ids) { const tr = P.tools[id].track; if (tr.t < t + 0.5) tr.hold(tr.t, t + 0.5 - tr.t); }
  return P.build(t + 0.5, 'task_set', seed);
}

// ------------------------------------------------------------------ ChArUco board scripts

function boardRec(scene) {
  const b = (scene.boards || [])[0];
  if (!b) return null;
  // corner object points: scene.boards[].object_points as data, or ChArUco B (7 x 5 squares of 12 mm)
  const sq = b.square_m ?? 0.012, nx = b.squares?.[0] ?? b.squares_x ?? 7, ny = b.squares?.[1] ?? b.squares_y ?? 5;
  const corners = [];
  for (let j = 1; j < ny; j++) for (let i = 1; i < nx; i++) corners.push([i * sq, j * sq, 0]);
  const size = b.size_m ?? [nx * sq, ny * sq];
  const mg = b.margin_m ?? 0;
  const outline = [[-mg, -mg, 0], [size[0] + mg, -mg, 0], [size[0] + mg, size[1] + mg, 0], [-mg, size[1] + mg, 0]];
  const centre = [(1 * sq + (nx - 1) * sq) / 2, (1 * sq + (ny - 1) * sq) / 2, 0];
  return { name: b.name, corners, outline, centre, size, margin: mg, thickness: b.thickness_m ?? 0.003 };
}

/** Board pose with its corner centre at c, its printed face (-z_B) toward `towards`, spun by `spin` about its normal. */
function boardPose(rb, c, towards, spin) {
  const zB = normalize(sub(c, towards));             // z into the board, away from the viewer
  let R = frameFromZ(zB, [1, 0, 0]);
  R = matmul(R, rotZ(spin));
  const p = sub(c, [0, 1, 2].map((i) => R[i][0] * rb.centre[0] + R[i][1] * rb.centre[1] + R[i][2] * rb.centre[2]));
  return { R, p, T: true };
}
const boardPoint = (pose, x) => add(pose.p, [0, 1, 2].map((i) => pose.R[i][0] * x[0] + pose.R[i][1] * x[1] + pose.R[i][2] * x[2]));

/** The parked tools' shaft segments (pivot -> jaw tips) with a radius, and the scope's, for the board check. */
function toolSegments(P, t) {
  const st = P.stateAtPlan ? P.stateAtPlan(t) : null;
  const segs = [];
  for (const tool of Object.values(P.tools)) {
    const v = st ? st.instruments.find((q) => q.id === tool.ins.id) : null;
    const q = v ?? aimAt(tool.p.pivot, tool.p.R, tool.track.v.hinge, 0, 0);
    const f = forward(tool.p.pivot, tool.p.R, tool.sp, q);
    segs.push({ a: tool.p.pivot, b: add(f.hinge, scale(f.d, tool.sp.jaw_length)), r: 0.0045 });     // cannula 4.25 mm
  }
  if (P.scene.scope) {
    const sp = port(P.scene, P.scene.scope.port ?? 'C');
    const q = P.scope.v;
    const R = matmul(matmul(matmul(sp.R, rotY(q.yaw)), rotX(q.pitch)), rotZ(q.roll));
    const d = [R[0][2], R[1][2], R[2][2]];
    segs.push({ a: sp.pivot, b: add(sp.pivot, scale(d, q.insertion)), r: 0.0065 });
  }
  return segs;
}

function boardHitsTools(P, rb, pose, margin = 0.004) {
  const inv = (p) => { const d = sub(p, pose.p); return [0, 1, 2].map((j) => pose.R[0][j] * d[0] + pose.R[1][j] * d[1] + pose.R[2][j] * d[2]); };
  const [W, H] = rb.size, m = rb.margin ?? 0.006;
  for (const sg of toolSegments(P)) {
    const L = norm(sub(sg.b, sg.a)), n = Math.max(2, Math.ceil(L / 0.002));
    for (let k = 0; k <= n; k++) {
      const x = inv(add(sg.a, scale(sub(sg.b, sg.a), k / n)));
      const out = Math.max(-m - x[0], x[0] - W - m, -m - x[1], x[1] - H - m, 0);
      const zgap = x[2] < 0 ? -x[2] : Math.max(0, x[2] - (rb.thickness ?? 0.003));
      if (Math.hypot(out, zgap) < sg.r + margin) return true;
    }
  }
  return false;
}

function boardClear(P, rb, pose) {
  const bz = (P.scene.bench?.z ?? 0) + 0.004;
  const pts = rb.outline.map((x) => boardPoint(pose, x));
  if (pts.some((q) => q[2] < bz)) return false;
  if (boardHitsTools(P, rb, pose)) return false;
  if (P.limits?.bvh) {
    // every outline edge, and the line from the workspace centre to each outline corner, clear of the shell
    for (let k = 0; k < 4; k++) {
      const a = pts[k], b = pts[(k + 1) % 4];
      const d = sub(b, a), L = norm(d);
      if (P.limits.hit(a, scale(d, 1 / L), 0, L) < Infinity) return false;
      const e = sub(a, P.centre), Le = norm(e);
      if (P.limits.hit(P.centre, scale(e, 1 / Le), 0, Le + 0.004) < Infinity) return false;
    }
  }
  return true;
}

function cornersSeen(cam, pose, rb, margin = 8) {
  let n = 0;
  const normalB = [-pose.R[0][2], -pose.R[1][2], -pose.R[2][2]];          // printed face normal (toward viewers)
  for (const x of rb.corners) {
    const pD = boardPoint(pose, x);
    const ray = normalize(sub(cam.centre, pD));
    if (dot(ray, normalB) < Math.cos(70 * DEG)) continue;               // grazing views fail detection
    const X = [0, 1, 2].map((i) => cam.T_C_D[i] * pD[0] + cam.T_C_D[4 + i] * pD[1] + cam.T_C_D[8 + i] * pD[2] + cam.T_C_D[12 + i]);
    if (X[2] <= 0.005) continue;
    const th = Math.atan2(Math.hypot(X[0], X[1]), X[2]);
    if (th > (cam.lens.thetaMax ?? 1.2)) continue;
    const k = cam.lens.k, t2 = th * th;
    const thd = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
    const r = Math.hypot(X[0], X[1]) || 1e-12;
    const u = cam.lens.fx * (thd * X[0] / r) + cam.lens.cx, v = cam.lens.fy * (thd * X[1] / r) + cam.lens.cy;
    if (u > margin && v > margin && u < cam.size[0] - margin && v < cam.size[1] - margin) n++;
  }
  return n;
}

function charucoRig(scene, seed, opts) {
  const P = new Planner(scene, seed, opts);
  const rb = boardRec(scene);
  if (!rb) throw new Error('charuco_rig: the scene has no board');
  const bt = P.boards[rb.name].track;
  const cams = ensureLenses(scene);
  const C = cams.map((c) => c.centre);
  const camMean = scale(C.reduce(add, [0, 0, 0]), 1 / C.length);
  const aim = scale(cams.map((c) => add(c.centre, scale([c.T_D_C[8], c.T_D_C[9], c.T_D_C[10]], 0.10))).reduce(add, [0, 0, 0]), 1 / cams.length);
  let t = P.park(0.2);
  const nPoses = opts.poses ?? 24;
  let placed = 0, tries = 0, prev = null;
  while (placed < nPoses && tries < nPoses * 300) {
    tries++;
    const c = add(aim, [P.rng.uniform(-0.035, 0.035), P.rng.uniform(-0.035, 0.035), P.rng.uniform(-0.03, 0.03)]);
    const towards = add(camMean, [P.rng.normal(0, 0.04), P.rng.normal(0, 0.04), P.rng.normal(0, 0.04)]);
    const pose = boardPose(rb, c, towards, P.rng.uniform(0, 2 * Math.PI));
    if (!boardClear(P, rb, pose)) continue;
    const seen = cams.map((cam) => cornersSeen(cam, pose, rb));
    if (seen.filter((n) => n >= 8).length < 2) continue;
    // move (min-jerk, ~40 mm/s) then hold still ~1 s
    if (!prev) { bt.move(t, 0.0001, pose, 'appear'); t += 0.0001; }
    else {
      const dist = norm(sub(pose.p, prev.p));
      t = bt.move(t, Math.max(0.8, dist / 0.04), pose, 'board_move');
    }
    const hold = P.rng.uniform(0.95, 1.25);
    P.label(t, t + hold, 'board_still');
    t = bt.hold(t, hold, 'board_still');
    prev = pose;
    placed++;
  }
  for (const tool of Object.values(P.tools)) tool.track.hold(tool.track.t, Math.max(0, t - tool.track.t));
  P.scope.hold(P.scope.t, Math.max(0, t - P.scope.t));
  const s = P.build(t + 0.3, 'charuco_rig', seed);
  s.boardPoses = placed;
  return s;
}

function charucoIntrinsics(scene, seed, opts) {
  const P = new Planner(scene, seed, opts);
  const rb = boardRec(scene);
  if (!rb) throw new Error('charuco_intrinsics: the scene has no board');
  const bt = P.boards[rb.name].track;
  const camName = opts.camera ?? scene.cameras[0].name;
  const cam = ensureLenses(scene).find((c) => c.name === camName);
  if (!cam) throw new Error(`charuco_intrinsics: no camera ${camName}`);
  let t = P.park(0.2);
  // pixel targets: a 5 x 3 grid including the four image corners, then random fill
  const [W, H] = cam.size;
  const grid = [];
  for (const fy of [0.12, 0.5, 0.88]) for (const fx of [0.1, 0.3, 0.5, 0.7, 0.9]) grid.push([fx * W, fy * H]);
  for (const c of [[0.04, 0.06], [0.96, 0.06], [0.04, 0.94], [0.96, 0.94]]) grid.push([c[0] * W, c[1] * H]);
  const nPoses = opts.poses ?? 36;
  let placed = 0, prev = null, gi = 0, tries = 0;
  const unproject = (uv) => {
    const L = cam.lens;
    const yd = (uv[1] - L.cy) / L.fy, xd = (uv[0] - L.cx) / L.fx;
    const thd = Math.hypot(xd, yd);
    let th = thd;
    for (let i = 0; i < 30; i++) {
      const t2 = th * th, k = L.k;
      const f = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - thd;
      const d = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
      th -= f / d;
    }
    const s = thd > 1e-12 ? Math.sin(th) / thd : 0;
    const rC = [xd * s, yd * s, Math.cos(th)];
    const T = cam.T_D_C;
    return normalize([T[0] * rC[0] + T[4] * rC[1] + T[8] * rC[2], T[1] * rC[0] + T[5] * rC[1] + T[9] * rC[2], T[2] * rC[0] + T[6] * rC[1] + T[10] * rC[2]]);
  };
  while (placed < nPoses && tries < nPoses * 400) {
    tries++;
    const uv = gi < grid.length ? grid[gi] : [P.rng.uniform(0.05, 0.95) * W, P.rng.uniform(0.05, 0.95) * H];
    const ray = unproject(uv);
    const dist = P.rng.uniform(0.045, 0.11);
    const c = add(cam.centre, scale(ray, dist));
    // face the camera, tilted up to 50 deg about a random in-plane axis
    const tilt = P.rng.uniform(0, 50 * DEG), az = P.rng.uniform(0, 2 * Math.PI);
    const toCam = normalize(sub(cam.centre, c));
    const R0 = frameFromZ(toCam, [1, 0, 0]);
    const ax = add(scale(col0(R0), Math.cos(az)), scale(col1(R0), Math.sin(az)));
    const towards = add(c, rotateAbout(toCam, ax, tilt));
    const pose = boardPose(rb, c, towards, P.rng.uniform(0, 2 * Math.PI));
    if (!boardClear(P, rb, pose)) { if (tries % 60 === 0) gi++; continue; }
    if (cornersSeen(cam, pose, rb, 4) < 10) { if (tries % 60 === 0) gi++; continue; }
    if (!prev) { bt.move(t, 0.0001, pose, 'appear'); t += 0.0001; }
    else t = bt.move(t, Math.max(0.8, norm(sub(pose.p, prev.p)) / 0.04), pose, 'board_move');
    const hold = P.rng.uniform(0.6, 0.9);
    P.label(t, t + hold, 'board_still');
    t = bt.hold(t, hold, 'board_still');
    prev = pose; placed++; gi++;
  }
  for (const tool of Object.values(P.tools)) tool.track.hold(tool.track.t, Math.max(0, t - tool.track.t));
  P.scope.hold(P.scope.t, Math.max(0, t - P.scope.t));
  const s = P.build(t + 0.3, 'charuco_intrinsics', seed);
  s.camera = camName;
  s.boardPoses = placed;
  return s;
}
const col0 = (R) => [R[0][0], R[1][0], R[2][0]];
const col1 = (R) => [R[0][1], R[1][1], R[2][1]];
function rotateAbout(v, axis, a) {        // Rodrigues
  const k = normalize(axis), c = Math.cos(a), s = Math.sin(a);
  return add(add(scale(v, c), scale(cross(k, v), s)), scale(k, dot(k, v) * (1 - c)));
}

// ------------------------------------------------------------------ entry

/** makeScript(name, scene, seed, {limits, camera, poses}) -> {name, seed, duration, stateAt(t), phaseAt(t), labels}. */
export function makeScript(name, scene, seed = 0, opts = {}) {
  if (name === 'task_set') return taskSet(scene, seed, opts);
  if (name === 'charuco_rig') return charucoRig(scene, seed, opts);
  if (name === 'charuco_intrinsics') return charucoIntrinsics(scene, seed, opts);
  throw new Error(`unknown script ${name}; known: ${SCRIPTS.join(', ')}`);
}

/** Tip speed statistics of a script (mm/s), sampled at dt, for the tests and the report. */
export function tipSpeeds(script, scene, dt = 0.01) {
  const out = {};
  let prev = null;
  for (let t = 0; t <= script.duration; t += dt) {
    const tr = truthFromState(scene, script.stateAt(t));
    if (prev) for (const ins of tr.instruments) {
      const p = prev.instruments.find((x) => x.id === ins.id);
      const v = norm(sub(ins.hinge_D, p.hinge_D)) / dt * 1e3;
      (out[ins.id] ??= []).push(v);
    }
    prev = tr;
  }
  return out;
}

export { fromRows };
