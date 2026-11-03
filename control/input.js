// Mouse + keyboard control of the twin's three tools, tuned to feel like a laparoscope: small, scaled, clutched.
//
//   1 / 2 / 3      select the fenestrated grasper (port L), the Maryland (port R), the scope (port C)
//   drag           swing the selected tool about its pivot. The button is the clutch: release, reposition, drag again.
//                  The tip follows the drag in the overview's screen plane at 0.08 mm per px (about 3:1 from the
//                  hand on a desk mouse), 0.02 mm per px with Shift (fine)
//   arrows         swing at 12 mm/s of tip speed (Shift 3 mm/s)
//   wheel, W / S   insert / withdraw (wheel 0.4 mm per notch, keys 15 mm/s; Shift x 1/4)
//   Q / E          roll the shaft (60 deg/s, Shift 15 deg/s); on the scope this turns the 30 deg optic
//   Space (hold)   close the jaws (grasp); release reopens them to the set opening
//   J / K (hold)   open / close the set opening; jaws move at <= 2.5 rad/s (a smooth first-order ramp)
//   right-drag or Alt-drag   orbit the overview; Alt-wheel zoom; F reset view; C cutaway on/off
//
// Increments are SI (m, rad). The controller only proposes a state; control/limits.js clamps it.

import { port, spec, forward, inverseDirection, add, sub, scale, dot, normalize, scopeForward } from './kinematics.js';

export const GAINS = {
  dragTip: 0.00008,          // m of tip motion per px of drag
  fine: 0.25,
  arrowTip: 0.012,           // m/s
  wheel: 0.000004,           // m per wheel delta unit (Chrome: ~100 units per notch -> 0.4 mm)
  keyInsert: 0.015,          // m/s
  roll: 60 * Math.PI / 180,  // rad/s
  jawSet: 0.8,               // rad/s change of the set opening (J/K)
  jawRate: 2.5,              // rad/s max jaw speed
  jawTau: 0.06,              // s first-order jaw response
  keyTau: 0.07,              // s key velocity smoothing
  scopeLever: 0.05,          // m: scope drags act as if on a 5 cm lever (scope insertion is short)
  orbit: 0.006,              // rad per px
};

export const LEGEND = [
  ['1 / 2 / 3', 'fenestrated (L) / Maryland (R) / scope'],
  ['drag', 'swing about the pivot (button = clutch)'],
  ['arrows', 'swing'],
  ['wheel, W / S', 'insert / withdraw'],
  ['Q / E', 'roll'],
  ['Space', 'hold to grasp'],
  ['J / K', 'open / close jaws'],
  ['Shift', 'fine'],
  ['right or Alt drag', 'orbit view'],
  ['Alt wheel', 'zoom'],
  ['F / C', 'reset view / cutaway'],
];

const TOOLS = ['L', 'R', 'scope'];

export class Controls extends EventTarget {
  /**
   * @param {object} o
   * @param {HTMLElement} o.surface   element that receives drags and the wheel (the overview)
   * @param {object} o.scene
   * @param {() => {right: number[], up: number[]}} o.screenBasis   overview screen axes in D (unit vectors)
   * @param {(dAz: number, dEl: number, dZoom: number) => void} o.orbit
   */
  constructor({ surface, scene, screenBasis, orbit, keyTarget = globalThis }) {
    super();
    this.surface = surface;
    this.scene = scene;
    this.screenBasis = screenBasis;
    this.orbit = orbit ?? (() => {});
    this.selected = 'L';
    this.keys = new Set();
    this.shift = false;
    this.drag = { dx: 0, dy: 0 };
    this.orbitAcc = { dx: 0, dy: 0, zoom: 0 };
    this.wheel = 0;
    this.vel = { swing: [0, 0], ins: 0, roll: 0 };
    this.jawSet = {};            // set opening per instrument (rad)
    this.enabled = true;
    this.activity = 0;           // performance.now() of the last user input (autoplay yields to the user)
    this._bind(keyTarget);
  }

  select(tool) {
    if (!TOOLS.includes(tool) && !this.scene.instruments.some((i) => i.id === tool)) return;
    this.selected = tool;
    this.dispatchEvent(new CustomEvent('select', { detail: { tool } }));
  }

  _bind(keyTarget) {
    const typing = (e) => /^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName ?? '');
    const handled = new Set(['KeyW', 'KeyS', 'KeyQ', 'KeyE', 'KeyJ', 'KeyK', 'Space', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown']);
    keyTarget.addEventListener('keydown', (e) => {
      if (typing(e) || e.metaKey || e.ctrlKey) return;
      this.shift = e.shiftKey;
      if (handled.has(e.code)) { e.preventDefault(); this.keys.add(e.code); this._touch(); }
      if (e.repeat) return;
      if (e.code === 'Digit1') this.select('L');
      else if (e.code === 'Digit2') this.select('R');
      else if (e.code === 'Digit3') this.select('scope');
      else if (e.code === 'KeyF') this.dispatchEvent(new CustomEvent('toggle', { detail: { what: 'reset-view' } }));
      else if (e.code === 'KeyC') this.dispatchEvent(new CustomEvent('toggle', { detail: { what: 'cutaway' } }));
    });
    keyTarget.addEventListener('keyup', (e) => { this.keys.delete(e.code); this.shift = e.shiftKey; });
    globalThis.addEventListener?.('blur', () => { this.keys.clear(); this.shift = false; });
    const s = this.surface;
    if (!s) return;
    s.addEventListener('contextmenu', (e) => e.preventDefault());
    let mode = null, lx = 0, ly = 0, pid = null;
    s.addEventListener('pointerdown', (e) => {
      s.focus?.({ preventScroll: true });
      mode = e.button === 2 || e.altKey ? 'orbit' : e.button === 0 ? 'swing' : null;
      if (!mode) return;
      pid = e.pointerId; lx = e.clientX; ly = e.clientY;
      try { s.setPointerCapture(pid); } catch { /* capture refused: plain events still work */ }
      e.preventDefault();
    });
    s.addEventListener('pointermove', (e) => {
      if (!mode || e.pointerId !== pid) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      this.shift = e.shiftKey;
      if (mode === 'orbit') { this.orbitAcc.dx += dx; this.orbitAcc.dy += dy; } else { this.drag.dx += dx; this.drag.dy += dy; this._touch(); }
    });
    const end = (e) => { if (e.pointerId === pid) { mode = null; pid = null; } };
    s.addEventListener('pointerup', end);
    s.addEventListener('pointercancel', end);
    s.addEventListener('wheel', (e) => {
      e.preventDefault();
      const k = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1;
      if (e.altKey) this.orbitAcc.zoom += e.deltaY * k;
      else { this.wheel += e.deltaY * k; this.shift = e.shiftKey; this._touch(); }
    }, { passive: false });
  }

  _touch() { this.activity = globalThis.performance?.now?.() ?? Date.now(); }

  /** Apply the input gathered since the last call to `state` (not clamped). Returns a new state. */
  step(state, dt) {
    const G = GAINS;
    const fine = this.shift ? G.fine : 1;
    const k = this.keys;
    const a = 1 - Math.exp(-dt / G.keyTau);
    const ax = (k.has('ArrowRight') ? 1 : 0) - (k.has('ArrowLeft') ? 1 : 0);
    const ay = (k.has('ArrowUp') ? 1 : 0) - (k.has('ArrowDown') ? 1 : 0);
    this.vel.swing[0] += (ax * G.arrowTip * fine - this.vel.swing[0]) * a;
    this.vel.swing[1] += (ay * G.arrowTip * fine - this.vel.swing[1]) * a;
    const ins = (k.has('KeyW') ? 1 : 0) - (k.has('KeyS') ? 1 : 0);
    this.vel.ins += (ins * G.keyInsert * fine - this.vel.ins) * a;
    const roll = (k.has('KeyE') ? 1 : 0) - (k.has('KeyQ') ? 1 : 0);
    this.vel.roll += (roll * G.roll * fine - this.vel.roll) * a;
    // a released key stops its motion within a few time constants (no endless exponential creep)
    const snap = (v, held, scale) => (!held && Math.abs(v) < 0.02 * scale ? 0 : v);
    this.vel.swing[0] = snap(this.vel.swing[0], ax !== 0, G.arrowTip);
    this.vel.swing[1] = snap(this.vel.swing[1], ay !== 0, G.arrowTip);
    this.vel.ins = snap(this.vel.ins, ins !== 0, G.keyInsert);
    this.vel.roll = snap(this.vel.roll, roll !== 0, G.roll);
    // screen-plane tip displacement from the drag and the arrows (D frame)
    const { right, up } = this.screenBasis();
    const sx = this.drag.dx * G.dragTip * fine + this.vel.swing[0] * dt;
    const sy = -this.drag.dy * G.dragTip * fine + this.vel.swing[1] * dt;
    const disp = add(scale(right, sx), scale(up, sy));
    const dIns = -this.wheel * G.wheel * fine + this.vel.ins * dt;
    const dRoll = this.vel.roll * dt;
    this.drag.dx = this.drag.dy = 0; this.wheel = 0;
    if (this.orbitAcc.dx || this.orbitAcc.dy || this.orbitAcc.zoom) {
      this.orbit(-this.orbitAcc.dx * G.orbit, this.orbitAcc.dy * G.orbit, this.orbitAcc.zoom);
      this.orbitAcc = { dx: 0, dy: 0, zoom: 0 };
    }
    const out = { ...state, instruments: state.instruments.map((q) => ({ ...q })), scope: state.scope ? { ...state.scope } : state.scope };
    const sel = this.selected;
    if (sel === 'scope' && out.scope) {
      const s = scopeForward(this.scene, out.scope);
      const perp = sub(disp, scale(s.d, dot(disp, s.d)));
      const d2 = normalize(add(s.d, scale(perp, 1 / G.scopeLever)));
      const p = port(this.scene, this.scene.scope.port ?? 'C');
      const [yaw, pitch] = inverseDirection(p.R, d2);
      Object.assign(out.scope, { yaw, pitch, roll: out.scope.roll + dRoll, insertion: out.scope.insertion + dIns });
    } else {
      const q = out.instruments.find((x) => x.id === sel);
      if (q) {
        const insRec = this.scene.instruments.find((i) => i.id === sel);
        const p = port(this.scene, insRec.port);
        const f = forward(p.pivot, p.R, spec(insRec), q);
        if (sx || sy) {
          // move the tip (hinge) in the screen plane, perpendicular to the shaft; the pivot stays fixed
          const perp = sub(disp, scale(f.d, dot(disp, f.d)));
          const h2 = add(f.hinge, perp);
          const [yaw, pitch] = inverseDirection(p.R, sub(h2, p.pivot));
          q.yaw = yaw; q.pitch = pitch;
        }
        q.insertion += dIns;
        q.roll += dRoll;
      }
    }
    // jaws: every grasper relaxes toward its target (Space closes the selected one)
    for (const q of out.instruments) {
      this.jawSet[q.id] ??= q.jaw;                 // the set opening starts at the state's own opening
      const insRec = this.scene.instruments.find((i) => i.id === q.id);
      const jmax = spec(insRec).jaw_max;
      if (q.id === sel) {
        const js = (k.has('KeyJ') ? 1 : 0) - (k.has('KeyK') ? 1 : 0);
        this.jawSet[q.id] = Math.max(0, Math.min(jmax, this.jawSet[q.id] + js * G.jawSet * fine * dt));
      }
      const target = q.id === sel && k.has('Space') ? 0 : this.jawSet[q.id];
      const want = (target - q.jaw) * (1 - Math.exp(-dt / G.jawTau));
      const lim = G.jawRate * dt;
      q.jaw += Math.max(-lim, Math.min(lim, want));
    }
    out.t = state.t;
    return out;
  }

  /** Keep the set openings in step with an externally driven state (autoplay -> manual hand-over). */
  adopt(state) {
    for (const q of state.instruments) this.jawSet[q.id] = q.jaw;
  }
}
