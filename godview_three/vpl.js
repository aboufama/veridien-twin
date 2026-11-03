// Indirect light of the scope inside the dome: a reflective shadow map (Dachsbacher & Stamminger 2005). The scope's
// view is rendered at G x G (positions, normals, diffuse albedo) with the cutaway applied; each texel becomes a virtual
// point light carrying the flux the outlet sends through it, I(theta) d(omega), times the surface albedo, re-emitted
// diffusely. Fragments gather the VPLs (no visibility; distances clamped) and scale by a cavity multi-bounce factor
// 1 / (1 - rho (1 - V_env)), V_env being the environment visibility (the dome is close to an integrating sphere).

import * as THREE from 'three/webgpu';
import { Fn, vec3, vec4, float, uniform, uniformArray, dot, positionWorld, normalWorld, Loop, max } from 'three/tsl';

export class ScopeVPL {
  constructor(renderer, { grid = 16, fovDeg = 116, dmin = 1.5 }) {
    this.r = renderer;
    this.G = grid; this.n = grid * grid;
    this.cam = new THREE.PerspectiveCamera(fovDeg, 1, 0.2, 100);
    this.cam.layers.set(2);
    this.rtP = new THREE.RenderTarget(grid, grid, { type: THREE.FloatType, samples: 0, depthBuffer: true });
    this.rtN = new THREE.RenderTarget(grid, grid, { type: THREE.FloatType, samples: 0, depthBuffer: true });
    this.matP = new THREE.MeshBasicNodeMaterial(); this.matP.colorNode = vec4(positionWorld, 1); this.matP.side = THREE.DoubleSide;
    this.matN = new THREE.MeshBasicNodeMaterial(); this.matN.side = THREE.DoubleSide;
    this.albedo = uniform(0.8);
    this.matN.colorNode = vec4(normalWorld, 1);
    const z = () => new THREE.Vector4(0, 0, 0, 0);
    this.pos = uniformArray(Array.from({ length: this.n }, z), 'vec4');
    this.nrm = uniformArray(Array.from({ length: this.n }, z), 'vec4');
    this.flux = uniformArray(Array.from({ length: this.n }, z), 'vec4');
    this.dmin2 = float(dmin * dmin);
    this.busy = false;
    this.version = 0;
  }

  /** TSL node: first-bounce irradiance (scene units: W/m^2-equivalent after the S^2 scaling of the intensities). */
  node() {
    const N = this.n, pos = this.pos, nrm = this.nrm, flux = this.flux, dmin2 = this.dmin2;
    return Fn(() => {
      const n = normalWorld.normalize();
      const p = positionWorld;
      const E = vec3(0).toVar();
      Loop(N, ({ i }) => {
        const vp = pos.element(i), vn = nrm.element(i), f = flux.element(i);
        const d = vp.xyz.sub(p);
        const r2 = max(dot(d, d), dmin2);
        const l = d.div(dot(d, d).max(1e-8).sqrt());
        const cx = max(dot(n, l), 0), cv = max(dot(vn.xyz, l.negate()), 0);
        E.addAssign(f.xyz.mul(cx.mul(cv).div(r2)));
      });
      return E.div(Math.PI);
    })();
  }

  /** Refresh from the scope light's current pose (async readback; the previous VPLs stay until the new ones land). */
  async update(scene, spot, intensityFn, albedoFn) {
    if (this.busy) return false;
    this.busy = true;
    try {
      const cam = this.cam;
      cam.position.copy(spot.position);
      cam.up.set(0, 0, 1);
      if (Math.abs(spot.target.position.clone().sub(spot.position).normalize().z) > 0.95) cam.up.set(1, 0, 0);
      cam.lookAt(spot.target.position);
      cam.updateMatrixWorld(); cam.updateProjectionMatrix();
      const r = this.r, prev = r.getRenderTarget(), ov = scene.overrideMaterial, bg = scene.background;
      scene.background = null;
      const pass = (mat, rt) => {
        const restore = this.swap ? this.swap(mat) : null;
        if (!restore) scene.overrideMaterial = mat;
        r.setRenderTarget(rt); r.setClearColor(0x000000, 0); r.clear(); r.render(scene, cam);
        if (restore) restore(); else scene.overrideMaterial = ov;
      };
      pass(this.matP, this.rtP);
      pass(this.matN, this.rtN);
      scene.overrideMaterial = ov; scene.background = bg; r.setRenderTarget(prev);
      const [P, Nn] = await Promise.all([r.readRenderTargetPixelsAsync(this.rtP, 0, 0, this.G, this.G),
        r.readRenderTargetPixelsAsync(this.rtN, 0, 0, this.G, this.G)]);
      const G = this.G, fov = cam.fov * Math.PI / 180, t = Math.tan(fov / 2);
      const axis = spot.target.position.clone().sub(spot.position).normalize();
      const o = spot.position;
      for (let k = 0; k < this.n; k++) {
        const a = P[4 * k + 3];
        const pv = this.pos.array[k], nv = this.nrm.array[k], fv = this.flux.array[k];
        if (!(a > 0.5)) { pv.set(0, 0, 0, 0); nv.set(0, 0, 0, 0); fv.set(0, 0, 0, 0); continue; }
        const x = P[4 * k], y = P[4 * k + 1], zz = P[4 * k + 2];
        const dx = x - o.x, dy = y - o.y, dz = zz - o.z, dl = Math.hypot(dx, dy, dz) || 1;
        const cosA = (dx * axis.x + dy * axis.y + dz * axis.z) / dl;          // angle from the emission axis
        // solid angle of a pinhole texel: (2t/G)^2 cos^3(angle from the camera axis); camera axis = emission axis
        const dOmega = (2 * t / G) * (2 * t / G) * Math.pow(Math.max(cosA, 0), 3);
        const I = intensityFn(Math.acos(Math.min(1, Math.max(-1, cosA))));    // [r,g,b] radiant intensity, scene units
        const rho = albedoFn(k, x, y, zz);
        pv.set(x, y, zz, 1);
        let nx = Nn[4 * k], ny = Nn[4 * k + 1], nz = Nn[4 * k + 2];
        // face the light (double-sided cut surfaces)
        if (nx * -dx + ny * -dy + nz * -dz < 0) { nx = -nx; ny = -ny; nz = -nz; }
        nv.set(nx, ny, nz, 0);
        fv.set(I[0] * dOmega * rho[0], I[1] * dOmega * rho[1], I[2] * dOmega * rho[2], 0);
      }
      this.version++;
      return true;
    } finally {
      this.busy = false;
    }
  }
}
