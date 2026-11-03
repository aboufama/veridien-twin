// Environment visibility for the god's-eye view: N orthographic depth maps of the rig (one per direction on a Fibonacci
// sphere, tiled in one depth atlas), rendered with the cutaway applied, so a fragment knows which of N directions of
// the studio environment it actually sees. The cosine- and radiance-weighted visible fraction multiplies three's
// image-based light (material.aoNode), which handles what screen-space AO cannot: the inside of the dome sees the
// studio only through the cut, and the bench near the dome sees the dome instead of the sky.
// The bench itself is not an occluder: below the horizon the environment already carries the bench's radiance.

import * as THREE from 'three/webgpu';
import { Fn, vec2, vec3, float, int, uniform, uniformArray, texture, dot, positionWorld, normalWorld, Loop, max, select, frontFacing } from 'three/tsl';

export const OCCLUDER_LAYER = 2;

export class SkyVis {
  constructor(renderer, { centre, radius, n = 64, res = 512, envUp = 1, envDown = 1 }) {
    this.r = renderer;
    this.n = n; this.res = res;
    this.T = Math.ceil(Math.sqrt(n));
    this.C = centre.slice();                                  // scene units
    this.Rb = radius;                                         // half extent of each ortho view
    this.R2 = radius * 3;                                      // camera distance from the centre
    this.near = this.R2 - radius * 1.8;
    this.far = this.R2 + radius * 1.8;
    const ga = Math.PI * (3 - Math.sqrt(5));
    this.dirs = []; this.rights = []; this.ups = []; this.Lw = [];
    for (let i = 0; i < n; i++) {
      const z = 1 - 2 * (i + 0.5) / n, rr = Math.sqrt(1 - z * z), ph = i * ga;
      const d = new THREE.Vector3(rr * Math.cos(ph), rr * Math.sin(ph), z);
      const hint = Math.abs(d.z) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
      const right = new THREE.Vector3().crossVectors(hint, d).normalize();   // camera looks along -d
      const up = new THREE.Vector3().crossVectors(d, right).normalize();
      this.dirs.push(d); this.rights.push(right); this.ups.push(up);
      this.Lw.push(d.z >= 0 ? envUp : envDown);
    }
    const W = this.T * res;
    this.depth = new THREE.DepthTexture(W, W);
    this.depth.type = THREE.FloatType;
    this.depth.compareFunction = THREE.LessEqualCompare;
    this.depth.minFilter = this.depth.magFilter = THREE.LinearFilter;
    this.rt = new THREE.RenderTarget(W, W, { depthBuffer: true, depthTexture: this.depth, type: THREE.UnsignedByteType, samples: 0 });
    this.cams = this.dirs.map((d, i) => {
      const c = new THREE.OrthographicCamera(-radius, radius, radius, -radius, this.near, this.far);
      c.position.set(this.C[0] + d.x * this.R2, this.C[1] + d.y * this.R2, this.C[2] + d.z * this.R2);
      c.up.copy(this.ups[i]);
      c.lookAt(this.C[0], this.C[1], this.C[2]);
      c.updateMatrixWorld();
      c.layers.set(OCCLUDER_LAYER);
      return c;
    });
    this.override = new THREE.MeshBasicNodeMaterial({ colorWrite: false });
    this.override.side = THREE.DoubleSide;
    this.flipV = uniform(1);
    this.bias = uniform(0.004);                               // normalised depth units (x (far-near) = 0.15 cm at Rb = 40)
    this.normalOffset = uniform(0.25);                        // scene units along the normal (2.5 mm)
    this.dirty = true;
    this.next = 0;
    this.taps = uniform(new THREE.Vector2(4.0, 1.6));          // tap offset (texels), rotated by 90 deg per tap
  }

  /** TSL node: weighted visible fraction of the environment for the current fragment (1 = unoccluded). */
  node() {
    const N = this.n, T = this.T;
    const dirs = uniformArray(this.dirs, 'vec3'), rights = uniformArray(this.rights, 'vec3'), ups = uniformArray(this.ups, 'vec3');
    const Lw = uniformArray(this.Lw, 'float');
    const C = uniform(new THREE.Vector3(...this.C));
    const Rb = float(this.Rb), R2 = float(this.R2), near = float(this.near), far = float(this.far);
    const depth = this.depth, flipV = this.flipV, bias = this.bias, off = this.normalOffset;
    return Fn(() => {
      const nrm = normalWorld.normalize().mul(select(frontFacing, float(1), float(-1)));   // back faces: their own side
      const q = positionWorld.add(nrm.mul(off)).sub(C);
      const num = float(0).toVar(), den = float(0).toVar();
      Loop(N, ({ i }) => {
        const d = dirs.element(i);
        const w = max(dot(nrm, d), 0).mul(Lw.element(i));
        const u = dot(q, rights.element(i)).div(Rb.mul(2)).add(0.5).clamp(0.001, 0.999);
        const v0 = dot(q, ups.element(i)).div(Rb.mul(2)).add(0.5).clamp(0.001, 0.999);
        const v = flipV.greaterThan(0.5).select(v0.oneMinus(), v0);
        const tx = float(i.mod(int(T))), ty = float(i.div(int(T)));
        const uv = vec2(tx.add(u), ty.add(v)).div(T);
        const z = R2.sub(dot(q, d)).sub(near).div(far.sub(near)).sub(bias);
        // 4 rotated taps (bilinear compare each): a ~2 r-texel footprint softens each direction's hard shadow edge
        const px = this.taps.mul(1 / (T * this.res));
        const vis = texture(depth, uv.add(vec2(px.x, px.y))).compare(z)
          .add(texture(depth, uv.add(vec2(px.y.negate(), px.x))).compare(z))
          .add(texture(depth, uv.add(vec2(px.x.negate(), px.y.negate()))).compare(z))
          .add(texture(depth, uv.add(vec2(px.y, px.x.negate()))).compare(z)).mul(0.25);
        num.addAssign(w.mul(vis));
        den.addAssign(w);
      });
      return num.div(max(den, 1e-6));
    })();
  }

  /** Render `count` tiles (all when count is omitted); returns true when a full pass completed. */
  update(scene, count = this.n) {
    const r = this.r;
    const prevRT = r.getRenderTarget(), prevAuto = r.autoClear, prevOverride = scene.overrideMaterial;
    const prevBg = scene.background, prevEnv = scene.environment;
    // swap(m) puts m (and its world-space cutaway-masked twin on the cut materials) on every occluder; the
    // ClippingGroup's view-space planes are computed for the main camera and do not hold for these cameras
    const restore = this.swap ? this.swap(this.override) : null;
    if (!restore) scene.overrideMaterial = this.override;
    scene.background = null;
    r.setRenderTarget(this.rt);
    r.autoClear = false;
    let done = false;
    for (let k = 0; k < count; k++) {
      const i = this.next;
      if (i === 0) r.clear(true, true, false);
      const tx = i % this.T, ty = Math.floor(i / this.T);
      this.rt.viewport.set(tx * this.res, ty * this.res, this.res, this.res);
      this.rt.scissor.set(tx * this.res, ty * this.res, this.res, this.res);
      this.rt.scissorTest = true;
      r.render(scene, this.cams[i]);
      this.next = (i + 1) % this.n;
      if (this.next === 0) { done = true; break; }
    }
    this.rt.scissorTest = false;
    this.rt.viewport.set(0, 0, this.rt.width, this.rt.height);
    this.rt.scissor.set(0, 0, this.rt.width, this.rt.height);
    r.setRenderTarget(prevRT);
    r.autoClear = prevAuto;
    scene.overrideMaterial = prevOverride;
    if (restore) restore();
    scene.background = prevBg;
    scene.environment = prevEnv;
    if (done) this.dirty = false;
    return done;
  }
}

/** Multi-bounce correction of a visibility term (Jimenez et al. 2016, GTAO): occluders are lit too. */
export function multiBounce(vis, albedo) {
  const a = 2.0404 * albedo - 0.3324, b = -4.7951 * albedo + 0.6417, c = 2.7552 * albedo + 0.6903;
  return max(vis, vis.mul(a).add(b).mul(vis).add(c).mul(vis));
}
