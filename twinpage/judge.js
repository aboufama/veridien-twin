// The god's-eye judge page for any GodView (docs/WEB_TWIN.md §19): twinpage/judge.html?god=three|pt|<dir>|placeholder.
//   window.godJudge = { ready, info(), async capture(viewpointName, stateObject) -> base64 of the Float32Array (w*h*3,
//   top row first) from captureLinear at godview.yaml image size, at that viewpoint's eye, cutaway on, all lights,
//   async timings(frames) -> display timings over `frames` renders at the canvas size }.
// The state is a judge state (src/veridien/webtwin/judges.py mitsuba_states), clamped by control/limits.js and posed
// by control/kinematics.js, as the reference does. ?spp=<n>&mode=traced|raster pass through to captureLinear (a
// renderer that takes no options ignores them).
import { loadScene } from '../gpu/scene_gpu.js';
import * as K from '../control/kinematics.js';
import { Limits } from '../control/limits.js';
import { normaliseGodview, viewpointCamera } from './config.js';
import { godOrder, godModuleUrl, rendererClass } from './loader.js';

const Q = new URLSearchParams(location.search);
const ROOT = new URL('../', import.meta.url);
const say = (t) => { document.getElementById('s').textContent = t; };

function b64(f32) {
  const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}

async function boot() {
  const t0 = performance.now();
  const bundle = await loadScene(new URL(Q.get('scene') || 'assets/', ROOT).href);
  const physics = await (await fetch('/veridien-twin/configs/twin/physics.json', { cache: 'no-store' })).json().catch(() => ({}));
  const raw = await fetch('/veridien-twin/configs/twin/godview.json', { cache: 'no-store' }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const gv = normaliseGodview(raw);
  const [W, H] = gv.image || [1280, 800];
  const canvas = document.getElementById('c');
  canvas.width = W; canvas.height = H;
  const failed = [];
  let god = null, dir = null;
  for (const d of godOrder(Q.get('god'))) {
    try {
      const mod = await import(new URL(godModuleUrl(d), ROOT).href);
      const G = rendererClass(mod, ['GodView']);
      god = await G.create({ canvas, scene: bundle, physics, godview: gv, log: (...a) => console.log(`[god ${d}]`, ...a) });
      dir = d;
      break;
    } catch (e) { failed.push({ dir: d, error: String(e?.message || e) }); console.warn(d, e); }
  }
  if (!god) throw new Error(`no GodView: ${JSON.stringify(failed)}`);
  const loadMs = performance.now() - t0;
  say(`${dir} ready in ${loadMs.toFixed(0)} ms`);
  const api = window.godJudge;
  api.god = god;
  api.info = () => ({ god: dir, failed, load_ms: loadMs, capture_opts: capOpts, image: [W, H], godview_source: raw ? 'configs/twin/godview.yaml' : 'built-in copy', viewpoints: Object.keys(gv.viewpoints) });
  const limits = new Limits(bundle.scene, bundle.meshes);
  const capOpts = {};
  if (Q.get('spp')) capOpts.spp = Number(Q.get('spp'));
  if (Q.get('mode')) capOpts.mode = Q.get('mode');
  api.capture = async (viewpoint, state) => {
    const st = limits.clampState(state || K.defaultState(bundle.scene)).state;
    god.setCamera(viewpointCamera(gv, viewpoint));
    god.setCutaway(true);
    god.setPoses(K.posesFromState(bundle.scene, st));
    const lin = await god.captureLinear(W, H, { ...capOpts });
    if (!(lin instanceof Float32Array) || lin.length !== W * H * 3) throw new Error(`captureLinear gave ${lin?.constructor?.name} of ${lin?.length}, want ${W * H * 3}`);
    god.render();
    return b64(lin);
  };
  api.timings = async (frames = 60, viewpoint = 'V1_surgeon') => {
    god.setCamera(viewpointCamera(gv, viewpoint));
    god.setCutaway(true);
    god.setPoses(K.posesFromState(bundle.scene, K.defaultState(bundle.scene)));
    for (let i = 0; i < frames; i++) { god.render(); await new Promise((r) => requestAnimationFrame(r)); }
    return god.timings();
  };
  return api.info();
}

window.godJudge = { ready: null };
window.godJudge.ready = boot().catch((e) => { say(`failed: ${e?.message || e}`); throw e; });
