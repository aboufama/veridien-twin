// The camera column's renderer in a module worker (docs/WEB_TWIN.md §19, ADR-137). Setting up a camera-simulation
// renderer costs seconds of CPU (scene preparation, BVH builds: lite_pt ~7 s on a pod), which on the page's main
// thread froze the god's-eye view for that long. Here it runs beside the page: the worker loads the scene, creates the
// first renderer of the given order that starts (§18.1 API), draws tiles straight into OffscreenCanvases the page
// transferred, and renders the model's bundles (CAM1-CAM3 at native size, read back to the page for the JPEG stage).
//
// Messages in: {id, type, ...}; replies: {id, ok, out | error}. Unsolicited: {type: 'lost', message}.
//   init   {order, base, physics}            -> {dir, failed, cameras, kind, loadMs}
//   canvas {key, canvas: OffscreenCanvas}    -> {}
//   tile   {name, size, poses|null, seed, frame, targets: [{key, w, h}]}  -> {ms}
//   bundle {poses, cams: [{name, size}], pts, seed, frame}  -> {cams: [{name, vis, rb}]}
//   timings {}                               -> renderer timings
import { loadScene } from '../gpu/scene_gpu.js';
import { camsModuleUrl, rendererClass } from './loader.js';

const ROOT = new URL('../', import.meta.url);
let r = null, dir = null, presenter = null, fmt = null;
const canvases = new Map(), ctxs = new Map(), viewSize = {};

async function init({ order, base, physics }) {
  const t0 = performance.now();
  const bundle = await loadScene(base);
  const failed = [];
  for (const d of order) {
    try {
      const mod = await import(new URL(camsModuleUrl(d), ROOT).href);
      const R = rendererClass(mod, d === 'trace' ? ['TwinRenderer'] : d === 'stub' ? ['StubRenderer'] : ['LiteRenderer']);
      if (!R) throw new Error(`${d} exports no renderer`);
      r = await R.create({ scene: bundle, physics, preset: 'live', log: (...a) => console.log(`[cams ${d}]`, ...a) });
      dir = d;
      break;
    } catch (e) { failed.push({ dir: d, error: String(e?.message || e) }); console.warn(`[cams worker] ${d} unavailable:`, e); }
  }
  if (!r) return { dir: null, failed, cameras: [], kind: null, loadMs: performance.now() - t0 };
  r.device?.lost?.then?.((i) => self.postMessage({ type: 'lost', message: i?.message || 'lost' }));
  r.device?.addEventListener?.('uncapturederror', (e) => self.postMessage({ type: 'gpuerror', message: String(e.error?.message || e.error || 'error').split('\n')[0] }));
  fmt = navigator.gpu.getPreferredCanvasFormat();
  if (typeof r.present !== 'function' && r.device) {
    try { const { Presenter } = await import('../ui/present.js'); presenter = new Presenter(r.device); } catch (e) { console.warn('[cams worker] presenter', e); }
  }
  const cameras = Array.isArray(r.cameras) ? [...r.cameras] : Object.keys(r.cameras || {});
  const kind = r.readbackKinds?.includes?.('yuv420') || r.isLite ? 'yuv420' : 'rgb8';
  return { dir, failed, cameras, kind, loadMs: performance.now() - t0 };
}

function setSize(name, size) {
  if (!r.setViewSize) return;
  const cur = viewSize[name];
  if (!cur || cur[0] !== size[0] || cur[1] !== size[1]) { r.setViewSize(name, size); viewSize[name] = [...size]; }
}

async function renderAndSense(name, seed, frame) {
  const lt = r.renderCamera(name, { await: false, wait: false, seed });
  if (lt?.then) await lt;
  const tex = r.sensor(name, { seed: (frame * 7919) >>> 0, frame });
  return tex?.then ? await tex : tex;
}

function ctxFor(key) {
  let ctx = ctxs.get(key);
  if (!ctx) {
    ctx = canvases.get(key).getContext('webgpu');
    if (r.device) ctx.configure({ device: r.device, format: fmt, alphaMode: 'opaque' });
    ctxs.set(key, ctx);
  }
  return ctx;
}

async function tile({ name, size, poses, seed, frame, targets }) {
  const t0 = performance.now();
  if (poses) r.setPoses(poses);
  setSize(name, size);
  const tex = await renderAndSense(name, seed, frame);
  for (const { key, w, h } of targets || []) {
    const c = canvases.get(key);
    if (!c) continue;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    if (typeof r.present === 'function') r.present(name, ctxFor(key), { fit: 'contain' });
    else if (tex && presenter) presenter.present(tex, c);
  }
  await r.device?.queue?.onSubmittedWorkDone?.();        // the reply's time includes the GPU work (the page's duty cap)
  return { ms: performance.now() - t0 };
}

async function bundle({ poses, cams, pts, seed, frame }) {
  r.setPoses(poses);
  const out = [];
  const pending = [];
  for (const { name, size } of cams) {
    setSize(name, size);
    const vis = r.visible ? Array.from(await r.visible(name, pts)) : pts.map(() => true);
    await renderAndSense(name, seed, frame);
    const kind = r.readbackKinds?.includes?.('yuv420') || r.isLite ? 'yuv420' : 'rgb8';
    const item = { name, vis, rb: null };
    out.push(item);
    pending.push(r.readback(name, kind).then((rb) => { item.rb = rb; }));   // lands while the next camera renders
  }
  await Promise.all(pending);
  return { cams: out };
}

self.onmessage = async (ev) => {
  const m = ev.data || {};
  try {
    let out;
    if (m.type === 'init') out = await init(m);
    else if (m.type === 'canvas') { canvases.set(m.key, m.canvas); ctxs.delete(m.key); out = {}; }
    else if (!r) throw new Error('no camera renderer');
    else if (m.type === 'tile') out = await tile(m);
    else if (m.type === 'bundle') out = await bundle(m);
    else if (m.type === 'timings') out = r.timings?.() ?? null;
    else throw new Error(`unknown message ${m.type}`);
    self.postMessage({ id: m.id, ok: true, out });
  } catch (e) {
    self.postMessage({ id: m.id, ok: false, error: String(e?.message || e) });
  }
};
