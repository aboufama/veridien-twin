// The god's-eye view's orbit camera with damping (docs/WEB_TWIN.md §19): left-drag orbits about the target, right- or
// Shift-drag pans, the wheel zooms, F resets to V1_surgeon. Pure (node tests): input adds to a pending delta, and
// update(dt) applies the fraction 1 - exp(-dt / tau) of it, so motion eases out over ~3 tau after the hand stops.
// Frame D, metres, radians. Spherical coordinates as godview.yaml: eye = target + d (cos el cos az, cos el sin az, sin el).

const DEG = Math.PI / 180;

export const ORBIT_GAINS = Object.freeze({
  orbit: 0.0055,         // rad per px of drag
  zoom: 0.0011,          // ln(distance) per wheel unit (Chrome: ~100 per notch -> 11 %)
  tau: 0.075,            // s, damping time constant
  elMin: -4 * DEG, elMax: 89 * DEG,
  dMin: 0.12, dMax: 2.6,
  panLimit: 0.6,         // m: the target stays within this of the home target
});

export class OrbitCamera {
  /** home: {target, az, el, d, vfov_deg} (radians). */
  constructor(home) {
    this.home = structuredClone(home);
    this.reset(true);
  }

  static fromViewpoint(gv, name) {
    const v = gv.viewpoints[name];
    return new OrbitCamera({ target: [...gv.centre], az: v.az_deg * DEG, el: v.el_deg * DEG, d: v.d, vfov_deg: gv.vfov_deg });
  }

  /** Reset to home: animated (eases there) unless `now`. */
  reset(now = false) {
    const h = this.home;
    if (now || !this.target) {
      this.target = [...h.target]; this.az = h.az; this.el = h.el; this.d = h.d; this.vfov_deg = h.vfov_deg;
      this.pend = { az: 0, el: 0, lnd: 0, pan: [0, 0, 0] };
      return;
    }
    // shortest way round in azimuth
    let da = h.az - this.az;
    da = Math.atan2(Math.sin(da), Math.cos(da));
    this.pend = { az: da, el: h.el - this.el, lnd: Math.log(h.d / this.d), pan: [h.target[0] - this.target[0], h.target[1] - this.target[1], h.target[2] - this.target[2]] };
  }

  /** Drag by (dx, dy) CSS px: right drag turns the view to the right around the target, up drag raises the eye. */
  orbitBy(dx, dy) {
    this.pend.az -= dx * ORBIT_GAINS.orbit;
    this.pend.el += dy * ORBIT_GAINS.orbit;
  }

  /** Pan by (dx, dy) CSS px of a view `heightPx` tall: the point under the cursor stays under it. */
  panBy(dx, dy, heightPx) {
    const { right, up } = this.basis();
    const m = (2 * this.d * Math.tan((this.vfov_deg * DEG) / 2)) / Math.max(1, heightPx);
    for (let k = 0; k < 3; k++) this.pend.pan[k] += (-dx * right[k] + dy * up[k]) * m;
  }

  zoomBy(wheel) { this.pend.lnd += wheel * ORBIT_GAINS.zoom; }

  /** Apply the damped share of the pending motion. Returns true while the view still moves. */
  update(dt) {
    const p = this.pend, G = ORBIT_GAINS;
    const a = dt <= 0 ? 0 : 1 - Math.exp(-dt / G.tau);
    const step = (x) => (Math.abs(x) < 1e-6 ? x : x * a);
    const dAz = step(p.az), dEl = step(p.el), dLn = step(p.lnd);
    const dPan = p.pan.map(step);
    this.az += dAz; p.az -= dAz;
    const el = Math.max(G.elMin, Math.min(G.elMax, this.el + dEl));
    p.el = (p.el - dEl) * (el === this.el + dEl ? 1 : 0);    // a clamp ends the pending motion
    this.el = el;
    const d = Math.max(G.dMin, Math.min(G.dMax, this.d * Math.exp(dLn)));
    p.lnd = d === this.d * Math.exp(dLn) ? p.lnd - dLn : 0;
    this.d = d;
    for (let k = 0; k < 3; k++) { this.target[k] += dPan[k]; p.pan[k] -= dPan[k]; }
    const h = this.home.target;
    const off = [this.target[0] - h[0], this.target[1] - h[1], this.target[2] - h[2]];
    const lo = Math.hypot(...off);
    if (lo > G.panLimit) { for (let k = 0; k < 3; k++) this.target[k] = h[k] + off[k] * (G.panLimit / lo); p.pan = [0, 0, 0]; }
    return this.moving();
  }

  moving() {
    const p = this.pend;
    return Math.abs(p.az) > 1e-5 || Math.abs(p.el) > 1e-5 || Math.abs(p.lnd) > 1e-5 || Math.hypot(...p.pan) > 1e-6;
  }

  eye() {
    const c = Math.cos(this.el);
    return [this.target[0] + this.d * c * Math.cos(this.az), this.target[1] + this.d * c * Math.sin(this.az), this.target[2] + this.d * Math.sin(this.el)];
  }

  /** Screen axes in D: right and up (unit vectors), and the view direction. */
  basis() {
    const e = this.eye();
    const f = norm3([this.target[0] - e[0], this.target[1] - e[1], this.target[2] - e[2]]);
    let r = cross3(f, [0, 0, 1]);
    if (Math.hypot(...r) < 1e-9) r = [Math.sin(this.az), -Math.cos(this.az), 0];
    r = norm3(r);
    const u = cross3(r, f);
    return { right: r, up: u, forward: f, eye: e };
  }

  /** The GodView camera block. */
  camera() { return { eye: this.eye(), target: [...this.target], up: [0, 0, 1], vfov_deg: this.vfov_deg }; }

  /** Project a D point to CSS px of a w x h view (pinhole, vfov), or null behind the eye. */
  project(p, w, h) {
    const { right, up, forward, eye } = this.basis();
    const d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
    const z = d[0] * forward[0] + d[1] * forward[1] + d[2] * forward[2];
    if (z <= 1e-4) return null;
    const f = (h / 2) / Math.tan((this.vfov_deg * DEG) / 2);
    const x = d[0] * right[0] + d[1] * right[1] + d[2] * right[2];
    const y = d[0] * up[0] + d[1] * up[1] + d[2] * up[2];
    return [w / 2 + (f * x) / z, h / 2 - (f * y) / z];
  }
}

function cross3(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm3(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
