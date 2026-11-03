// Seeded, platform-independent random numbers for the scripted sessions (same seed -> same states, in node and in
// every browser). mulberry32 for uniforms; Box-Muller for normals.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Rng {
  constructor(seed) { this.u = mulberry32(seed); this._spare = null; }
  uniform(lo = 0, hi = 1) { return lo + (hi - lo) * this.u(); }
  int(lo, hi) { return lo + Math.floor(this.u() * (hi - lo + 1)); }
  normal(mu = 0, sigma = 1) {
    if (this._spare != null) { const s = this._spare; this._spare = null; return mu + sigma * s; }
    let u = 0, v = 0;
    while (u <= 1e-300) u = this.u();
    v = this.u();
    const r = Math.sqrt(-2 * Math.log(u));
    this._spare = r * Math.sin(2 * Math.PI * v);
    return mu + sigma * r * Math.cos(2 * Math.PI * v);
  }
  pick(arr) { return arr[Math.floor(this.u() * arr.length)]; }
}
