// The live tip filter of the in-browser tracker: the JS twin of tipnet.py's OneEuro and TipFilter (ADR-141): One-Euro
// (min_cutoff 0.4 Hz, beta 500 /(m/s), d_cutoff 1 Hz), a one-frame spike hold with a motion-scaled gate, and a short
// hold-last on dropouts. Never extrapolates. Vectors are 3-arrays in metres, times in seconds (capture time).

const sub = (a, b) => a.map((x, i) => x - b[i]);
const nrm = (a) => Math.sqrt(a.reduce((s, x) => s + x * x, 0));

export class OneEuro {
  constructor(min_cutoff = 1.0, beta = 50.0, d_cutoff = 1.0) {
    this.mc = min_cutoff; this.beta = beta; this.dc = d_cutoff;
    this.x = null; this.dx = null; this.t = null;
  }

  static alpha(fc, dt) {
    const tau = 1.0 / (2 * Math.PI * fc);
    return 1.0 / (1.0 + tau / dt);
  }

  update(z, t) {
    if (z == null) return this.x === null ? null : this.x.slice();
    z = z.slice();
    if (this.x === null) {
      this.x = z.slice(); this.dx = z.map(() => 0); this.t = t;
      return z.slice();
    }
    const dt = Math.max(t - this.t, 1e-4);
    this.t = t;
    const dz = z.map((v, i) => (v - this.x[i]) / dt);
    const ad = OneEuro.alpha(this.dc, dt);
    this.dx = dz.map((v, i) => ad * v + (1 - ad) * this.dx[i]);
    const fc = this.mc + this.beta * nrm(this.dx);
    const a = OneEuro.alpha(fc, dt);
    this.x = z.map((v, i) => a * v + (1 - a) * this.x[i]);
    return this.x.slice();
  }
}

export const TIP_FILTER_DEFAULTS = Object.freeze({ min_cutoff: 0.4, beta: 500.0, d_cutoff: 1.0, gate_m: 0.004, gate_travel: 3.0,
  gate_accel: 4.0, hold_frames: 3, max_held: 2, gate_cutoff: 10.0 });

export class TipFilter {
  constructor(opts = {}) {
    const o = { ...TIP_FILTER_DEFAULTS, ...opts };
    this.euro = new OneEuro(o.min_cutoff, o.beta, o.d_cutoff);
    this.gate_cutoff = o.gate_cutoff;
    this.v = null;
    this.z_last = null;                 // [z, t] of the last accepted measurement
    this.gate0 = o.gate_m; this.gate_travel = o.gate_travel; this.gate_accel = o.gate_accel;
    this.gate = o.gate_m; this.hold_frames = o.hold_frames; this.max_held = o.max_held;
    this.pending = null;                // a held-back measurement [z, t]
    this.missed = 0;
    this.n_held = 0;
    this.held = false;                  // the last output was not updated by this call's measurement
  }

  _pred(t) {
    const e = this.euro;
    if (e.x === null) return null;
    const dt = Math.max(t - e.t, 0.0);
    return e.x.map((x, i) => x + (this.v !== null ? this.v[i] : 0.0) * dt);
  }

  _accept(z, t, v = null) {
    if (v !== null) this.v = v;
    else if (this.z_last !== null) {
      const dt = Math.max(t - this.z_last[1], 1e-4);
      const a = OneEuro.alpha(this.gate_cutoff, dt);
      const vz = z.map((x, i) => (x - this.z_last[0][i]) / dt);
      this.v = this.v === null ? vz : vz.map((x, i) => a * x + (1 - a) * this.v[i]);
    }
    this.z_last = [z, t];
    return this.euro.update(z, t);
  }

  update(z, t) {
    const e = this.euro;
    if (z == null) {
      this.missed += 1;
      this.held = true;
      if (this.missed > this.hold_frames) { this.reset(); return null; }
      return e.x === null ? null : e.x.slice();
    }
    z = z.slice();
    this.missed = 0;
    const pred = this._pred(t);
    if (e.x !== null) {
      const dt = Math.max(t - e.t, 0.0);
      const speed = this.v !== null ? nrm(this.v) : 0.0;
      this.gate = this.gate0 + this.gate_travel * speed * dt + 0.5 * this.gate_accel * dt * dt;
    }
    if (pred === null || nrm(sub(z, pred)) <= this.gate) {
      this.pending = null; this.n_held = 0; this.held = false;
      return this._accept(z, t);
    }
    let confirmed = false;
    if (this.pending !== null) {
      const [zp, tp] = this.pending;
      const v = zp.map((x, i) => (x - e.x[i]) / Math.max(tp - e.t, 1e-4));
      const ahead = zp.map((x, i) => x + v[i] * (t - tp));
      confirmed = Math.min(nrm(sub(z, zp)), nrm(sub(z, ahead))) <= this.gate;
    }
    if (confirmed || this.n_held >= this.max_held) {
      this.euro = new OneEuro(e.mc, e.beta, e.dc);
      let v = null;
      if (confirmed) {
        const [zp, tp] = this.pending;
        this.euro.update(zp, tp);
        v = z.map((x, i) => (x - zp[i]) / Math.max(t - tp, 1e-4));
      } else {
        this.v = null; this.z_last = null;
      }
      this.pending = null; this.n_held = 0; this.held = false;
      return this._accept(z, t, v);
    }
    this.pending = [z, t]; this.held = true;
    this.n_held += 1;
    return e.x.slice();
  }

  reset() {
    this.euro = new OneEuro(this.euro.mc, this.euro.beta, this.euro.dc);
    this.pending = null; this.missed = 0; this.n_held = 0; this.held = false;
    this.v = null; this.z_last = null; this.gate = this.gate0;
  }
}
