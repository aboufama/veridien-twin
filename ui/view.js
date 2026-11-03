// The overview's orbit camera (a pinhole in OpenCV convention, posed in D) and its projection, shared by the stub
// renderer, the 3-D overlays and the mouse mapping (screen axes in D).
import { add, sub, scale, normalize, cross, dot } from '../control/kinematics.js';

export class OrbitView {
  constructor(target, { az = Math.PI * 1.0, el = 32 * Math.PI / 180, dist = 0.62, fovy_deg = 38 } = {}) {
    this.home = { target: [...target], az, el, dist, fovy_deg };
    this.reset();
    this.cutaway = true;
    this.width = 1; this.height = 1;
  }
  reset() { Object.assign(this, structuredClone(this.home)); }
  orbit(dAz, dEl, dZoom = 0) {
    this.az += dAz;
    this.el = Math.max(-10 * Math.PI / 180, Math.min(88 * Math.PI / 180, this.el + dEl));
    this.dist = Math.max(0.12, Math.min(2.0, this.dist * Math.exp(dZoom * 0.0015)));
  }
  eye() {
    const c = Math.cos(this.el);
    return add(this.target, scale([c * Math.cos(this.az), c * Math.sin(this.az), Math.sin(this.el)], this.dist));
  }
  /** Row-major T_D_V of the OpenCV-convention view camera (x right, y down, z forward). */
  basis() {
    const e = this.eye();
    const z = normalize(sub(this.target, e));
    let x = cross(z, [0, 0, 1]);
    if (Math.hypot(...x) < 1e-9) x = [0, -1, 0];
    x = normalize(x);
    const y = cross(z, x);
    return { e, x, y, z };
  }
  T_D_V() {
    const { e, x, y, z } = this.basis();
    return [[x[0], y[0], z[0], e[0]], [x[1], y[1], z[1], e[1]], [x[2], y[2], z[2], e[2]], [0, 0, 0, 1]];
  }
  /** Screen axes in D: right and up (unit), for the mouse mapping. */
  screenBasis() { const { x, y } = this.basis(); return { right: x, up: scale(y, -1) }; }
  /** The renderer's view block (WEB_TWIN.md §8 renderOverview(view, opts)). */
  view() {
    const { e } = this.basis();
    let n = [e[0] - this.target[0], e[1] - this.target[1], 0];
    const ln = Math.hypot(n[0], n[1]);
    n = ln > 1e-9 ? [n[0] / ln, n[1] / ln, 0] : [1, 0, 0];
    return {
      T_D_V: this.T_D_V(), fovy_deg: this.fovy_deg, width: this.width, height: this.height,
      eye: e, target: [...this.target], up: [0, 0, 1],
      cutaway: this.cutaway ? { normal: n, offset: dot(n, this.target) - 0.005 } : null,
    };
  }
  /** Project a D point to overview canvas pixels (CSS px of a w x h canvas), or null behind the eye. */
  project(p, w = this.width, h = this.height) {
    const { e, x, y, z } = this.basis();
    const d = sub(p, e);
    const zc = dot(d, z);
    if (zc <= 1e-4) return null;
    const f = (h / 2) / Math.tan((this.fovy_deg * Math.PI / 180) / 2);
    return [w / 2 + f * dot(d, x) / zc, h / 2 + f * dot(d, y) / zc];
  }
}
