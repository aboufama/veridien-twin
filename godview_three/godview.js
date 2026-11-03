// God's-eye view renderer, candidate 1: three.js r180 WebGPURenderer (docs/WEB_TWIN.md section 19, configs/twin/godview.yaml).
//
//   const g = await GodView.create({ canvas, scene, physics, godview, device? });
//   g.setPoses(poses); g.setCamera({ eye, target, up, vfov_deg }); g.setCutaway(true); g.render();
//   const rgb = await g.captureLinear(1280, 800);   // scene-referred linear radiance, w*h*3, top row first
//
// Light transport (all of it from godview.yaml + physics.yaml; nothing is tuned by eye):
//   * materials: physics.yaml section 3 mapped onto three's physical model (GGX, height-correlated Smith, alpha = r^2:
//     the same lobe as the contract BSDF; Lambert diffuse with rho_d; conductors with RGB f0; the coated cannula as a
//     clear-coat over steel);
//   * environment: godview.lights.environment (constant radiance) as a PMREM; below the horizon the bench, which is a
//     2 x 2 m diffuse plane, so the lower hemisphere carries the bench's own radiance (rho_b/pi x its irradiance);
//   * key: a directional light (irradiance from godview.yaml) with a 4096^2 shadow map over the rig;
//   * scope: a spot light at the light outlet with the tabulated angular profile p(theta) cos(theta) (an IES texture),
//     intensity L0 x area x gain (radiant intensity, W/sr per channel), and soft (area-light) shadows;
//   * cutaway: a clipping plane on the PLA meshes (shadows clipped too, as the reference removes that geometry);
//     back faces seen through the cut are shaded as a cap with the plane's normal.
// Units: three's punctual-light falloff clamps 1/d^2 below d = 0.1 scene units, so the scene is built in
// centimetres (S = 100 units per metre); radiance is unaffected, intensities are scaled by S^2.

import * as THREE from 'three/webgpu';
import { SkyVis, multiBounce, OCCLUDER_LAYER } from './skyvis.js';
import { ScopeVPL } from './vpl.js';
import {
  Fn, vec2, vec3, vec4, float, uniform, uniformArray, texture, frontFacing, cameraViewMatrix, normalView,
  positionWorld, normalWorld, select, Loop, smoothstep, output, int, dot, max, min, clamp, sqrt, mix, exp, abs, If, vec4 as v4,
} from 'three/tsl';

export const S = 100;                                     // scene units per metre
const DEG = Math.PI / 180;
const unwrap = (x) => (x && typeof x === 'object' && !Array.isArray(x) && 'value' in x ? x.value : x);
const lum = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

// ------------------------------------------------------------------ scene loading (scene.json + meshes/*.bin)

function decodeMesh(meta, buf) {
  const lay = meta.layout, nv = meta.vertex_count, nt = meta.triangle_count;
  const vec = (c) => (c === nv * 3 && nv * 3 !== nv ? nv : c);
  const idx = (c) => (c === nt ? nt * 3 : c);
  const [po, pc] = lay.positions, [io, ic] = lay.indices;
  const P = new Float32Array(buf.slice(po, po + vec(pc) * 12));
  const N = lay.normals ? new Float32Array(buf.slice(lay.normals[0], lay.normals[0] + vec(lay.normals[1]) * 12)) : null;
  const I = new Uint32Array(buf.slice(io, io + idx(ic) * 4));
  let UV = null;
  if (lay.uvs) { const [uo, uc] = lay.uvs; UV = new Float32Array(buf.slice(uo, uo + (uc === nv * 2 && nv * 2 !== nv ? nv : uc) * 8)); }
  return { P, N, I, UV };
}

export async function loadSceneBundle(base) {
  const root = base.endsWith('/') ? base : base + '/';
  const scene = await (await fetch(root + 'scene.json')).json();
  const meshes = {};
  await Promise.all(scene.meshes.map(async (m) => { meshes[m.name] = decodeMesh(m, await (await fetch(root + m.file)).arrayBuffer()); }));
  return { scene, meshes, base: root };
}

// ------------------------------------------------------------------ materials (physics.yaml section 3)

function rgb(v, fallback = [0.5, 0.5, 0.5]) {
  const x = unwrap(v);
  if (Array.isArray(x)) return x.map(Number);
  if (typeof x === 'number') return [x, x, x];
  return fallback;
}

/** physics.yaml profile p(theta) -> IES texture of p(theta) cos(theta), u = theta / 180 deg (three's IESSpotLightNode). */
function profileTexture(physics) {
  const prof = physics?.light?.profile || {};
  const th = unwrap(prof.theta_deg) || [0, 25, 30, 35, 40, 45, 50, 55, 60];
  const p = unwrap(prof.p) || [1, 1, 0.933, 0.75, 0.5, 0.25, 0.067, 0, 0];
  const W = 720;
  const data = new Float32Array(W * 4);
  for (let i = 0; i < W; i++) {
    const t = (i + 0.5) / W * 180;
    let v = 0;
    if (t <= th[th.length - 1]) {
      let k = 0;
      while (k < th.length - 2 && t > th[k + 1]) k++;
      const w = (t - th[k]) / (th[k + 1] - th[k]);
      v = p[k] + (p[k + 1] - p[k]) * Math.min(1, Math.max(0, w));
    }
    data[4 * i] = data[4 * i + 1] = data[4 * i + 2] = v * Math.max(0, Math.cos(t * DEG));
    data[4 * i + 3] = 1;
  }
  const tex = new THREE.DataTexture(data, W, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return { tex, thetaMax: th[th.length - 1] };
}

// ------------------------------------------------------------------ soft shadow filters

/** Poisson-disk PCF with a radius in shadow-map texels (key light: small; scope: the outlet's penumbra). */
function poissonFilter(radiusTexels, taps = 16) {
  const pts = [];
  const ga = Math.PI * (3 - Math.sqrt(5));                 // golden-angle spiral: even, deterministic
  for (let i = 0; i < taps; i++) { const r = Math.sqrt((i + 0.5) / taps); pts.push([r * Math.cos(i * ga), r * Math.sin(i * ga)]); }
  const offs = uniformArray(pts.map(([x, y]) => new THREE.Vector2(x, y)), 'vec2');
  const rad = uniform(radiusTexels);
  const fn = Fn(({ depthTexture, shadowCoord, shadow }) => {
    const mapSize = vec2(shadow.mapSize.x, shadow.mapSize.y);
    const texel = vec2(1).div(mapSize);
    const sum = float(0).toVar();
    Loop(taps, ({ i }) => {
      const uv = shadowCoord.xy.add(offs.element(i).mul(texel).mul(rad));
      sum.addAssign(texture(depthTexture, uv).compare(shadowCoord.z));
    });
    return sum.div(taps);
  });
  fn.radius = rad;
  return fn;
}

let HALF = null;
function halfToFloat(u16) {
  if (!HALF) {
    HALF = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      HALF[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
  }
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = HALF[u16[i]];
  return out;
}

// ------------------------------------------------------------------ the renderer

export class GodView {
  static async create(opts) {
    const g = new GodView();
    await g._init(opts);
    return g;
  }

  async _init(opts) {
    const { canvas, scene, physics, godview, device, meshes, base, log } = opts;
    const t0 = performance.now();
    this.log = log || (() => {});
    this.canvas = canvas || document.createElement('canvas');
    this.physics = physics || {};
    this.gv = godview || {};
    let bundle = scene && scene.scene && scene.meshes ? scene : null;
    if (!bundle && meshes) bundle = { scene, meshes, base };
    if (!bundle) bundle = await loadSceneBundle(base || scene?.__base || '../assets/');
    this.bundle = bundle;
    this.sceneJson = bundle.scene;

    const params = { canvas: this.canvas, antialias: true, trackTimestamp: true, powerPreference: 'high-performance' };
    if (device) params.device = device;
    const r = new THREE.WebGPURenderer(params);
    await r.init();
    this.backendName = r.backend?.isWebGPUBackend ? 'webgpu' : 'webgl2';
    r.setPixelRatio(1);
    r.toneMapping = THREE.NeutralToneMapping;                          // Khronos PBR Neutral
    r.toneMappingExposure = this.exposure();
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap;
    this.r = r;

    this.scene = new THREE.Scene();
    this.dbg = opts.debug || {};
    // rays that miss the rig see the constant studio environment (the reference's background)
    const envc = rgb(this.gv.lights?.environment?.radiance, [1, 1, 1]);
    this.scene.background = new THREE.Color(envc[0], envc[1], envc[2]);
    this.camera = new THREE.PerspectiveCamera(Number(this.gv.vfov_deg ?? 35), 16 / 10, 1, 500);
    this.camera.up.set(0, 0, 1);
    this.centre = (this.gv.centre || [-0.05, 0.027, 0.10]).map(Number);
    this.cutOn = this.gv.cutaway?.default !== false && this.gv.cutaway?.default !== 'off';
    this._envBelow();
    const sc = this.centre.map((x) => x * S);
    this.sky = new SkyVis(r, { centre: sc, radius: 36, n: 64, res: 512, envUp: lum(this.envUp), envDown: lum(this.envBelowL) });
    this.skyNode = this.sky.node();
    this.vpl = new ScopeVPL(r, { grid: Number(opts.vplGrid ?? 16) });
    this.vplNode = this.vpl.node();
    // world-space cutaway mask for the auxiliary passes (sky tiles, VPL map): exact per fragment
    this.cutC = uniform(new THREE.Vector3(...sc));
    this.cutN = uniform(new THREE.Vector3(0, 0, 1));
    this.cutOff = uniform(1e6);
    this.backFaceGain = uniform(Number(opts.backFaceGain ?? 0.18));
    this.cavW = uniform(Number(opts.cavW ?? 0));               // weight of the cavity multi-bounce factor on the VPL light
    this.mbInside = uniform(Number(opts.mbInside ?? 0));     // weight of plain visibility (vs multi-bounce) inside
    this.cutMask = dot(positionWorld.sub(this.cutC), this.cutN).lessThanEqual(this.cutOff);
    this._masked = new Map();
    this.sky.swap = this.vpl.swap = (plain) => this._swap(plain);
    this.cutMats = new Set(this.gv.cutaway?.materials || ['pla_matte_white']);
    this._insideNode = this._domeInside();
    if (opts.flipV != null) this.sky.flipV.value = Number(opts.flipV);
    this._buildMaterials();
    this._buildMeshes();
    await this._buildLights();
    this._times = [];
    this.frames = 0;
    // first frame compiles every pipeline (counted in the load time, not in the frame times)
    this.setCamera(this.viewpoint(Object.keys(this.gv.viewpoints || { V1: {} })[0]));
    this.setCutaway(this.cutOn);
    await r.compileAsync(this.scene, this.camera);
    this.loadMs = performance.now() - t0;
  }

  // ---------------------------------------------------------------- materials
  _buildMaterials() {
    const mats = this.physics.materials || {};
    this.cavityAlbedo = uniform(lum(rgb(mats.pla_matte_white?.rho_d, [0.829, 0.808, 0.746])));
    const env = rgb(this.gv.lights?.environment?.radiance, [1, 1, 1]);
    this.cap = { normal: uniform(new THREE.Vector3(0, 0, 1)) };
    const capNormalView = cameraViewMatrix.mul(vec4(this.cap.normal, 0)).xyz.normalize();
    const cutMats = new Set(this.gv.cutaway?.materials || ['pla_matte_white']);
    this.cutMats = cutMats;
    this.mat = {};
    const make = (name) => {
      const m = mats[name] || {};
      const model = unwrap(m.model) || 'dielectric';
      let mat;
      if (model === 'conductor') {
        mat = new THREE.MeshStandardNodeMaterial({ metalness: 1, roughness: Number(unwrap(m.roughness) ?? 0.3) });
        mat.color.setRGB(...rgb(m.f0, [0.56, 0.57, 0.58]));
      } else if (model === 'coated_conductor') {
        mat = new THREE.MeshPhysicalNodeMaterial({ metalness: 1, roughness: Number(unwrap(m.roughness) ?? 0.3),
          clearcoat: 1, clearcoatRoughness: Number(unwrap(m.coat_roughness) ?? 0.08) });
        const tau = rgb(m.coat_transmittance, [1, 1, 1]);
        const f0 = rgb(m.f0, [0.56, 0.57, 0.58]);
        mat.color.setRGB(f0[0] * tau[0] * tau[0], f0[1] * tau[1] * tau[1], f0[2] * tau[2] * tau[2]);
      } else if (model === 'emitter') {
        mat = new THREE.MeshBasicNodeMaterial();
        const L0 = rgb(this.physics.materials?.emitter?.L0, [10000, 9200, 8800]);
        this.emitterL0 = L0;
        mat.colorNode = uniform(new THREE.Color(L0[0], L0[1], L0[2]).multiplyScalar(this.scopeGain()));
        this.emitterColor = mat.colorNode;
      } else {
        let rho = m.rho_d ? rgb(m.rho_d) : null;
        if (!rho && m.rho_d_paper) rho = rgb(m.rho_d_paper);
        if (!rho && m.albedo) rho = rgb(m.albedo);
        rho = rho || [0.5, 0.5, 0.5];
        const rough = Number(unwrap(m.roughness ?? m.roughness_paper) ?? 0.5);
        mat = new THREE.MeshPhysicalNodeMaterial({ metalness: 0, roughness: rough, ior: Number(unwrap(m.ior) ?? 1.5), specularIntensity: 1 });
        mat.color.setRGB(rho[0], rho[1], rho[2]);
      }
      if (model !== 'emitter' && !this.dbg.noao && !(this.dbg.noaoTools && model !== 'dielectric')) {
        const alb = model === 'conductor' || model === 'coated_conductor' ? lum(rgb(m.f0, [0.56, 0.57, 0.58]))
          : lum(rgb(m.rho_d ?? m.rho_d_paper ?? m.albedo, [0.5, 0.5, 0.5]));
        // multi-bounce (Jimenez 2016) assumes the occluders are lit like the receiver: true outside, not inside the
        // dome, whose walls are lit only through the cut; back faces (inside the hollow wall) skip the sky maps
        const ao = mix(multiBounce(this.skyNode, alb), this.skyNode, this._insideNode.mul(this.mbInside));
        mat.aoNode = ao;
        if (model === 'dielectric' && !this.dbg.novpl) {
          // scope light re-emitted by the dome's inside (first bounce from the VPLs, cavity multi-bounce factor)
          const rho = rgb(m.rho_d ?? m.rho_d_paper ?? m.albedo, [0.5, 0.5, 0.5]);
          const cav = mix(float(1), float(1).div(float(1).sub(this.cavityAlbedo.mul(float(1).sub(this.skyNode))).max(0.2)), this.cavW);
          const ind = vec3(rho[0], rho[1], rho[2]).mul(this.vplNode).mul(cav).mul(this._insideNode).div(Math.PI);
          mat.emissiveNode = ind;
        }
      }
      if (cutMats.has(name)) {
        mat.side = THREE.DoubleSide;
        mat.shadowSide = THREE.DoubleSide;
        // the cutaway, in world space and per fragment: main pass (mask), shadow passes (alpha-tested cast colour)
        mat.maskNode = this.cutMask;
        mat.alphaTest = 0.5;
        mat.castShadowNode = vec4(0, 0, 0, select(this.cutMask, float(1), float(0)));
        // one-sided BSDF (as the reference): a solid's back faces, seen only through the cut, are black
        // back faces (the inside of the hollow wall, seen only through the cut) get light only through the slot
        if (!this.dbg.cap) mat.outputNode = select(frontFacing, output, vec4(output.rgb.mul(this.backFaceGain), 1));
        if (this.dbg.cap) mat.normalNode = select(frontFacing, normalView, capNormalView);
      }
      mat.name = name;
      return mat;
    };
    this.getMat = (name) => (this.mat[name] ??= make(name));
  }

  /** Put `plain` on every occluder mesh (its masked twin on the cut materials); returns the restore function. */
  _swap(plain) {
    let masked = this._masked.get(plain);
    if (!masked) { masked = plain.clone(); masked.maskNode = this.cutMask; this._masked.set(plain, masked); }
    const saved = [];
    this.scene.traverse((o) => {
      if (!o.isMesh || o.name === 'bench') return;
      saved.push([o, o.material]);
      o.material = this.cutMats.has(o.material.name) ? masked : plain;
    });
    const clipWas = this.clip.enabled;
    this.clip.enabled = false;
    return () => { for (const [o, m] of saved) o.material = m; this.clip.enabled = clipWas; };
  }

  /** 1 inside the dome's cavity, 0 outside: a superellipsoid fitted to the shell's inner extent (PLA bboxes less the
   * 10 mm wall). Gates the VPL light, which carries no visibility, so it cannot leak through the wall. */
  _domeInside() {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const m of this.sceneJson.meshes) {
      if (!this.cutMats.has(m.material) || !m.bbox) continue;
      for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], m.bbox[0][k]); hi[k] = Math.max(hi[k], m.bbox[1][k]); }
    }
    const bz = Number(this.sceneJson.bench?.z ?? 0);
    const wall = 0.010;
    const c = [(lo[0] + hi[0]) / 2 * S, (lo[1] + hi[1]) / 2 * S, bz * S];
    const a = [((hi[0] - lo[0]) / 2 - wall) * S, ((hi[1] - lo[1]) / 2 - wall) * S, (hi[2] - bz - wall) * S];
    this.domeBox = { c, a };
    const C = vec3(...c), A = vec3(...a);
    return Fn(() => {
      const q = positionWorld.sub(C).div(A).abs();
      const r = q.x.pow(3).add(q.y.pow(3)).add(q.z.pow(3)).pow(1 / 3);
      return float(1).sub(smoothstep(0.98, 1.06, r));
    })();
  }

  // ---------------------------------------------------------------- meshes
  _buildMeshes() {
    const sj = this.sceneJson, M = this.bundle.meshes;
    this.bodies = {};
    this.clip = new THREE.ClippingGroup();
    this.clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 1e6);
    this.clip.clippingPlanes = [this.clipPlane];
    this.clip.clipShadows = false;
    this.clip.enabled = false;                                // superseded by the world-space mask (exact in every pass)
    this.scene.add(this.clip);
    this.staticGroup = new THREE.Group();
    this.scene.add(this.staticGroup);
    for (const m of sj.meshes) {
      const d = M[m.name];
      if (!d) continue;
      const geo = new THREE.BufferGeometry();
      const P = new Float32Array(d.P.length);
      for (let i = 0; i < P.length; i++) P[i] = d.P[i] * S;
      geo.setAttribute('position', new THREE.BufferAttribute(P, 3));
      if (d.N) geo.setAttribute('normal', new THREE.BufferAttribute(d.N, 3));
      if (d.UV) geo.setAttribute('uv', new THREE.BufferAttribute(d.UV, 2));
      geo.setIndex(new THREE.BufferAttribute(d.I, 1));
      if (!d.N) geo.computeVertexNormals();
      geo.computeBoundingSphere();
      const mesh = new THREE.Mesh(geo, this.getMat(m.material));
      mesh.name = m.name;
      const emitter = m.material === 'emitter';
      mesh.castShadow = !emitter;
      mesh.receiveShadow = !emitter;
      if (!m.dynamic) mesh.layers.enable(OCCLUDER_LAYER);       // instruments are too thin to occlude the studio
      if (m.dynamic) {
        const b = m.body || m.name;
        let g = this.bodies[b];
        if (!g) {
          g = new THREE.Group(); g.matrixAutoUpdate = false; g.name = b;
          g.visible = !b.startsWith('board/');
          this.bodies[b] = g; this.scene.add(g);
        }
        g.add(mesh);
        if (emitter) this.emitterMesh = { mesh, P: d.P, N: d.N };
      } else if (this.cutMats.has(m.material)) {
        this.clip.add(mesh);
      } else {
        this.staticGroup.add(mesh);
      }
    }
    // the bench: a bigger plane than the camera sims need (godview.bench.extent), same material and z
    const bz = Number(sj.bench?.z ?? 0) * S;
    const ex = this.gv.bench?.extent || sj.bench?.extent || [[-1, -1], [1, 1]];
    const bw = (ex[1][0] - ex[0][0]) * S, bh = (ex[1][1] - ex[0][1]) * S;
    const bgeo = new THREE.PlaneGeometry(bw, bh, 1, 1);
    const bench = new THREE.Mesh(bgeo, this.getMat(sj.bench?.material || 'bench_white'));
    bench.position.set((ex[0][0] + ex[1][0]) / 2 * S, (ex[0][1] + ex[1][1]) / 2 * S, bz);
    bench.receiveShadow = true; bench.castShadow = false; bench.name = 'bench';
    this.staticGroup.add(bench);
    this.benchZ = bz;
    // the floor inside the dome as a sky occluder only (layer OCCLUDER_LAYER, never drawn in the view): from inside
    // the cavity, directions below the horizon see this shaded floor, not the open bench the environment carries
    if (this.domeBox && !this.dbg.nofloorocc) {
      const { c, a } = this.domeBox, pts = [];
      for (let k = 0; k < 96; k++) {
        const t = k / 96 * 2 * Math.PI, ct = Math.cos(t), st = Math.sin(t);
        const r = 1 / Math.cbrt(Math.abs(ct) ** 3 + Math.abs(st) ** 3);
        pts.push(new THREE.Vector2(c[0] + a[0] * 1.02 * r * ct, c[1] + a[1] * 1.02 * r * st));
      }
      const fg = new THREE.ShapeGeometry(new THREE.Shape(pts));
      const floor = new THREE.Mesh(fg, this.getMat(sj.bench?.material || 'bench_white'));
      floor.position.z = bz + 0.05;
      floor.layers.set(OCCLUDER_LAYER);
      floor.name = 'floor_occluder';
      floor.castShadow = false; floor.receiveShadow = false;
      this.staticGroup.add(floor);
    }
  }

  // ---------------------------------------------------------------- lights
  /** godview.display.exposure; until the reference agent sets it, the same rule computed here: key-lit exterior PLA
   * (normal toward the key: L = rho/pi (pi L_env + E_key)) maps to 0.8 before the tone curve. */
  exposure() {
    const e = Number(unwrap(this.gv.display?.exposure));
    if (Number.isFinite(e) && e > 0) return e;
    const rho = lum(rgb(this.physics.materials?.pla_matte_white?.rho_d, [0.829, 0.808, 0.746]));
    const env = lum(rgb(this.gv.lights?.environment?.radiance, [1, 1, 1])), ek = lum(rgb(this.gv.lights?.key?.irradiance, [2, 2, 2]));
    return 0.8 / (rho / Math.PI * (Math.PI * env + ek));
  }

  scopeGain() { return Number(unwrap(this.gv.lights?.scope?.gain) ?? 1.0) || 1.0; }

  /** Environment above the horizon; below it the bench's radiance (rho_b/pi x (pi L_env + E_key sin el_key)). */
  _envBelow() {
    const L = this.gv.lights || {};
    const env = rgb(L.environment?.radiance, [1, 1, 1]);
    const key = L.key || {};
    const Ek = rgb(key.irradiance, [2, 2, 2]);
    const el = Number(key.from_el_deg ?? 65) * DEG;
    const rb = rgb(this.physics.materials?.bench_white?.rho_d, [0.86, 0.86, 0.84]);
    this.envUp = env;
    this.envBelowL = [0, 1, 2].map((i) => rb[i] / Math.PI * (Math.PI * env[i] + Ek[i] * Math.max(0, Math.sin(el))));
  }

  async _buildLights() {
    const L = this.gv.lights || {};
    const env = rgb(L.environment?.radiance, [1, 1, 1]);
    const key = L.key || {};
    const Ek = rgb(key.irradiance, [2, 2, 2]);
    const az = Number(key.from_az_deg ?? 150) * DEG, el = Number(key.from_el_deg ?? 65) * DEG;
    this.keyDir = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
    const c = this.centre.map((x) => x * S);

    // key: directional, shadow map over the rig (4096^2 over 0.8 m: 0.2 mm texels)
    const kl = new THREE.DirectionalLight(new THREE.Color(Ek[0], Ek[1], Ek[2]), 1.0);
    kl.position.set(c[0] + this.keyDir[0] * 150, c[1] + this.keyDir[1] * 150, c[2] + this.keyDir[2] * 150);
    kl.target.position.set(c[0], c[1], c[2]);
    kl.castShadow = !this.dbg.nokeyshadow;
    kl.shadow.mapSize.set(4096, 4096);
    const half = 40;
    Object.assign(kl.shadow.camera, { left: -half, right: half, top: half, bottom: -half, near: 50, far: 260 });
    kl.shadow.camera.updateProjectionMatrix();
    kl.shadow.bias = -0.0001;
    kl.shadow.normalBias = 0.03;                              // 0.3 mm along the normal
    this.keyFilter = poissonFilter(1.5, 12);
    kl.shadow.filterNode = this.keyFilter;
    this.scene.add(kl); this.scene.add(kl.target);
    this.keyLight = kl;

    // environment: constant radiance above the horizon, the bench's radiance below it
    const Lb = this.envBelowL;
    const envScene = new THREE.Scene();
    const sph = new THREE.SphereGeometry(100, 64, 32);
    const top = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide });
    const posW = positionWorld;
    top.colorNode = select(posW.z.greaterThanEqual(0), vec3(env[0], env[1], env[2]), vec3(Lb[0], Lb[1], Lb[2]));
    const s = new THREE.Mesh(sph, top);
    envScene.add(s);
    // PMREM's cube faces assume y-up; our up is z, and positionWorld in the env scene is in the cube camera's frame.
    // The mapping is world-direction based, so the split at z = 0 lands on the D-frame horizon.
    const pm = new THREE.PMREMGenerator(this.r);
    const rt = await pm.fromSceneAsync(envScene, 0, 0.1, 1000);
    this.scene.environment = rt.texture;
    this.scene.environmentIntensity = 1.0;
    pm.dispose();

    // scope: spot at the light outlet; radiant intensity I(theta) = L0 p(theta) A cos(theta) x gain
    const { tex, thetaMax } = profileTexture(this.physics);
    const sp = new THREE.IESSpotLight(0xffffff, 1.0, 0, Math.min(89, thetaMax + 2) * DEG, 0, 2);
    sp.iesMap = tex;
    sp.castShadow = true;
    sp.shadow.mapSize.set(2048, 2048);
    sp.shadow.camera.near = 0.3; sp.shadow.camera.far = 80;
    sp.shadow.bias = -0.0002;
    sp.shadow.normalBias = 0.02;
    sp.shadow.focus = 1;
    this.scopeFilter = poissonFilter(6.0, 16);
    sp.shadow.filterNode = this.scopeFilter;
    this.scene.add(sp); this.scene.add(sp.target);
    this.scopeLight = sp;
    this._setScopeIntensity();
    this._emitterLocal = this._emitterFrame();
  }

  /** Radiant intensity of the outlet at angle th (rad) from its axis, [r,g,b] in scene units (x S^2). */
  _scopeI(th) {
    const prof = this.physics?.light?.profile || {};
    const tt = unwrap(prof.theta_deg) || [0, 25, 30, 35, 40, 45, 50, 55, 60];
    const pp = unwrap(prof.p) || [1, 1, 0.933, 0.75, 0.5, 0.25, 0.067, 0, 0];
    const t = th / DEG;
    let v = 0;
    if (t <= tt[tt.length - 1]) {
      let k = 0;
      while (k < tt.length - 2 && t > tt[k + 1]) k++;
      v = pp[k] + (pp[k + 1] - pp[k]) * Math.min(1, Math.max(0, (t - tt[k]) / (tt[k + 1] - tt[k])));
    }
    const c = this.scopeLight.color;
    const f = v * Math.max(0, Math.cos(th));
    return [c.r * f, c.g * f, c.b * f];
  }

  async _updateVPL() {
    if (!this.scopeLight || !this.scopePose) return;
    const rb = rgb(this.physics.materials?.bench_white?.rho_d, [0.86, 0.86, 0.84]);
    const rp = rgb(this.physics.materials?.pla_matte_white?.rho_d, [0.829, 0.808, 0.746]);
    const bz = this.benchZ;
    const clear = new THREE.Color(); this.r.getClearColor(clear); const ca = this.r.getClearAlpha();
    await this.vpl.update(this.scene, this.scopeLight, (th) => this._scopeI(th), (k, x, y, z) => (Math.abs(z - bz) < 0.2 ? rb : rp));
    this.r.setClearColor(clear, ca);
  }

  _setScopeIntensity() {
    const sj = this.sceneJson;
    const L0 = rgb(this.physics.materials?.emitter?.L0, [10091.9, 9193.2, 8830.7]);
    const A = Number(sj.scope?.emitter_area_physics_m2 ?? unwrap(this.physics.scope?.light_outlet?.area_on_face) ?? 4.793e-5);
    const k = A * this.scopeGain() * S * S;                   // W/sr per channel, x S^2 for the centimetre falloff
    this.scopeLight.color.setRGB(L0[0] * k, L0[1] * k, L0[2] * k);
    this.scopeLight.intensity = 1.0;
    if (this.emitterColor) this.emitterColor.value.setRGB(L0[0], L0[1], L0[2]).multiplyScalar(this.scopeGain());
  }

  /** The outlet centroid and emission axis in the scope body frame (metres). */
  _emitterFrame() {
    const sj = this.sceneJson;
    const ax = sj.scope?.emission_axis_K || [0, 0.5, 0.8660254];
    let cx = 0, cy = 0, cz = 0, n = 0;
    if (this.emitterMesh) {
      const P = this.emitterMesh.P;
      for (let i = 0; i < P.length; i += 3) { cx += P[i]; cy += P[i + 1]; cz += P[i + 2]; n++; }
      cx /= n; cy /= n; cz /= n;
    }
    return { c: [cx, cy, cz], a: ax };
  }

  // ---------------------------------------------------------------- API
  setPoses(poses) {
    const B = poses?.bodies || poses || {};
    for (const [name, g] of Object.entries(this.bodies)) {
      if (!(name in B)) continue;
      const T = B[name];
      if (!T) { g.visible = false; continue; }
      g.visible = true;
      g.matrix.fromArray(Array.from(T));
      g.matrix.elements[12] *= S; g.matrix.elements[13] *= S; g.matrix.elements[14] *= S;
      g.matrixWorldNeedsUpdate = true;
    }
    this.vplDirty = true;                                     // the sky maps hold static geometry only
    const T = B.scope;
    if (T && this.scopeLight) {
      const e = this._emitterLocal;
      const tp = (p) => [T[0] * p[0] + T[4] * p[1] + T[8] * p[2] + T[12], T[1] * p[0] + T[5] * p[1] + T[9] * p[2] + T[13], T[2] * p[0] + T[6] * p[1] + T[10] * p[2] + T[14]];
      const td = (v) => [T[0] * v[0] + T[4] * v[1] + T[8] * v[2], T[1] * v[0] + T[5] * v[1] + T[9] * v[2], T[2] * v[0] + T[6] * v[1] + T[10] * v[2]];
      const p = tp(e.c), a = td(e.a);
      // start the light a hair in front of the outlet so the scope's own tip face does not shadow it
      const o = [p[0] + a[0] * 2e-4, p[1] + a[1] * 2e-4, p[2] + a[2] * 2e-4];
      this.scopeLight.position.set(o[0] * S, o[1] * S, o[2] * S);
      this.scopeLight.target.position.set((o[0] + a[0] * 0.1) * S, (o[1] + a[1] * 0.1) * S, (o[2] + a[2] * 0.1) * S);
      this.scopeLight.target.updateMatrixWorld();
      this.scopePose = { p: o, a };
    }
  }

  viewpoint(name) {
    const v = this.gv.viewpoints?.[name] || { az_deg: 200, el_deg: 40, d: 0.75 };
    const az = v.az_deg * DEG, el = v.el_deg * DEG, c = this.centre;
    return { eye: [c[0] + v.d * Math.cos(el) * Math.cos(az), c[1] + v.d * Math.cos(el) * Math.sin(az), c[2] + v.d * Math.sin(el)],
      target: c.slice(), up: [0, 0, 1], vfov_deg: Number(this.gv.vfov_deg ?? 35) };
  }

  setCamera({ eye, target, up, vfov_deg }) {
    const cam = this.camera;
    cam.position.set(eye[0] * S, eye[1] * S, eye[2] * S);
    cam.up.set(...(up || [0, 0, 1]));
    const t = target || this.centre;
    cam.lookAt(t[0] * S, t[1] * S, t[2] * S);
    if (vfov_deg) cam.fov = vfov_deg;
    const dist = Math.hypot(eye[0] - t[0], eye[1] - t[1], eye[2] - t[2]) * S;
    cam.near = Math.max(0.5, dist * 0.2); cam.far = dist + 400;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
    this.eye = eye.slice();
    this._updateCut();
  }

  setCutaway(on) { this.cutOn = !!on; this._updateCut(); }

  _updateCut() {
    if (!this.eye) return;
    const c = this.centre, e = this.eye;
    const n = [e[0] - c[0], e[1] - c[1], e[2] - c[2]];
    const l = Math.hypot(...n) || 1;
    n[0] /= l; n[1] /= l; n[2] /= l;
    const off = Number(this.gv.cutaway?.offset ?? 0.02);
    // keep dot(p - c, n) <= off  <=>  -n.p + (n.c + off) >= 0  (three keeps the positive side)
    this.clipPlane.normal.set(-n[0], -n[1], -n[2]);
    this.clipPlane.constant = this.cutOn ? (n[0] * c[0] + n[1] * c[1] + n[2] * c[2] + off) * S : 1e6;
    this.clip.enabled = false;
    this.cap.normal.value.set(n[0], n[1], n[2]);
    if (this.cutN) { this.cutN.value.set(n[0], n[1], n[2]); this.cutOff.value = this.cutOn ? off * S : 1e6; }
    const key = this.cutOn ? n.map((x) => x.toFixed(4)).join(',') : 'off';
    if (key !== this._cutKey) { this._cutKey = key; if (this.sky) this.sky.dirty = true; this.vplDirty = true; }
  }

  _fit(w, h) {
    const cv = this.canvas;
    if (cv.width !== w || cv.height !== h) this.r.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() {
    const cv = this.canvas;
    const w = cv.clientWidth || cv.width, h = cv.clientHeight || cv.height;
    if (w && h) this._fit(Math.round(w * (this.pixelRatio || 1)), Math.round(h * (this.pixelRatio || 1)));
    if (this.sky.dirty) this.sky.update(this.scene, this.skyTilesPerFrame ?? 8);
    if (this.vplDirty && !this.vpl.busy) { this.vplDirty = false; this._updateVPL(); }
    this.r.render(this.scene, this.camera);
    this.frames++;
    if (!this._pending) {
      this._pending = true;
      this.r.resolveTimestampsAsync(THREE.TimestampQuery.RENDER).then((ms) => {
        this._pending = false;
        if (typeof ms === 'number' && ms > 0) { this._times.push(ms); if (this._times.length > 240) this._times.shift(); }
      }).catch(() => { this._pending = false; });
    }
  }

  /** Scene-referred linear radiance of the current view, w*h*3 floats, top row first (2x2 supersampled + MSAA). */
  async captureLinear(width, height, { ss = 2 } = {}) {
    const W = width * ss, H = height * ss;
    // rgba16float: rgba32float cannot be multisampled in WebGPU; half precision is 5e-4 relative, far below the judge
    const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, samples: 4, depthBuffer: true,
      colorSpace: THREE.LinearSRGBColorSpace });
    const cam = this.camera;
    const aspect = cam.aspect;
    cam.aspect = width / height; cam.updateProjectionMatrix();
    const tm = this.r.toneMapping;
    this.r.toneMapping = THREE.NoToneMapping;
    if (this.sky.dirty || this.sky.next !== 0) { this.sky.next = 0; this.sky.update(this.scene); }
    while (this.vpl.busy) await new Promise((res) => setTimeout(res, 5));
    this.vplDirty = false;
    await this._updateVPL();
    this.r.setRenderTarget(rt);
    this.r.render(this.scene, cam);
    this.r.setRenderTarget(null);
    this.r.toneMapping = tm;
    const raw = await this.r.readRenderTargetPixelsAsync(rt, 0, 0, W, H);
    rt.dispose();
    const px = raw instanceof Uint16Array ? halfToFloat(raw) : raw;
    cam.aspect = aspect; cam.updateProjectionMatrix();
    const out = new Float32Array(width * height * 3);
    const flip = this.backendName !== 'webgpu';              // WebGL reads bottom row first
    const inv = 1 / (ss * ss);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let r0 = 0, g0 = 0, b0 = 0;
        for (let j = 0; j < ss; j++) {
          const sy = flip ? H - 1 - (y * ss + j) : y * ss + j;
          for (let i = 0; i < ss; i++) {
            const k = (sy * W + x * ss + i) * 4;
            r0 += px[k]; g0 += px[k + 1]; b0 += px[k + 2];
          }
        }
        const o = (y * width + x) * 3;
        out[o] = r0 * inv; out[o + 1] = g0 * inv; out[o + 2] = b0 * inv;
      }
    }
    return out;
  }

  timings() {
    const t = this._times.slice().sort((a, b) => a - b);
    return { gpu_ms_last: this._times[this._times.length - 1] ?? null, gpu_ms_median: t.length ? t[t.length >> 1] : null,
      frames: this.frames, load_ms: this.loadMs, backend: this.backendName };
  }

  destroy() {
    this.scene.traverse((o) => { o.geometry?.dispose?.(); });
    for (const m of Object.values(this.mat)) m.dispose();
    this.r.dispose();
  }
}
