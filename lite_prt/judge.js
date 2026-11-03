// Judge page for the lite PRT renderer (owner: lite_prt racer). window.twinJudge with the §8 contract API, as
// judges.py capture_twin / headless.TwinPage expect, plus: perf(), gen() (PRT training data from the path tracer's
// converged node solver, ?prt=0) and info(). Query: ?prt=0 (node solver instead of the PRT asset), ?ls=N light samples.
import { LitePRTRenderer, scopeFeatures } from './renderer.js';
import * as K from '../control/kinematics.js';

const log = (s) => { const el = document.getElementById('log'); if (el) el.textContent += '\n' + s; console.log('[lite-prt]', s); };
const params = new URLSearchParams(location.search);
const sceneDir = params.get('scene') || '../assets/';
let resolveReady, rejectReady;
const ready = new Promise((a, b) => { resolveReady = a; rejectReady = b; });
const J = { ready, kind: 'lite_prt', renderer: null, scene: null };
window.twinJudge = J;

const b64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };

(async () => {
  try {
    const t0 = performance.now();
    const scene = await (await fetch(sceneDir + 'scene.json')).json();
    const physics = await (await fetch(params.get('physics') || './assets/physics.json')).json();
    const opts = { scene: sceneDir, physics, layerLines: false, log, prt: params.get('prt') === '0' ? false : (params.get('prtdir') || undefined) };
    if (params.has('ls')) opts.liteLightSamples = Number(params.get('ls'));
    const r = await LitePRTRenderer.create(opts);
    J.renderer = r; J.scene = scene;
    J.loadMs = performance.now() - t0;
    log(`ready in ${J.loadMs.toFixed(0)} ms, prt ${r.prtMode}`);
    resolveReady(true);
  } catch (e) { log('ERROR ' + (e.stack || e)); rejectReady(e); }
})();

J.poses = (state) => K.posesFromState(J.scene, state);
J.defaultState = () => K.defaultState(J.scene);
J.timings = () => J.renderer.timings();
J.info = () => ({ stats: J.renderer.stats, prt: J.renderer.prtMode, loadMs: J.loadMs, cameras: Object.keys(J.renderer.cameras), adapter: J.renderer.adapterInfo });

J.render = async (state, o = {}) => {
  const r = J.renderer;
  const st = { ...K.defaultState(J.scene), ...state };
  const p0 = K.posesFromState(J.scene, st);
  const p1 = o.state1 ? K.posesFromState(J.scene, { ...st, ...o.state1 }) : null;
  r.setPoses(p0, p1);
  const t0 = performance.now();
  if (r.converge && o.converge !== false) await r.converge(o.iters ?? 256);
  const out = { _meta: { converge_ms: performance.now() - t0 } };
  for (const cam of o.cameras || Object.keys(r.cameras)) {
    await r.renderCamera(cam, { seed: o.seed ?? 0, exposure: o.exposure });
    const rec = { width: r.cameras[cam].size[0], height: r.cameras[cam].size[1] };
    if (o.linear !== false) rec.linear = await r.readback(cam, 'linear');
    if (o.rgb8) { r.sensor(cam, {}); rec.rgb8 = await r.readback(cam, 'rgb8'); }
    out[cam] = rec;
  }
  await r.idle();
  out._meta.timings = r.timings();
  return out;
};

/** render() with the arrays base64-encoded (fast through Playwright). */
J.renderB64 = async (state, o = {}) => {
  const out = await J.render(state, o);
  const enc = { _meta: out._meta };
  for (const [k, v] of Object.entries(out)) {
    if (k === '_meta') continue;
    enc[k] = { width: v.width, height: v.height };
    if (v.linear) enc[k].linear = b64(new Uint8Array(v.linear.buffer));
    if (v.rgb8) enc[k].rgb8 = b64(v.rgb8);
  }
  return enc;
};

/** GPU ms per camera frame: n renders of `cams` at the state, scope still (live path: no converge). */
J.perf = async (state, cams = ['SCOPE'], n = 30, o = {}) => {
  const r = J.renderer;
  const st = { ...K.defaultState(J.scene), ...state };
  const res = {};
  for (const cam of cams) {
    const ms = [], wall = [];
    for (let k = 0; k < n; k++) {
      const s2 = { ...st, instruments: st.instruments.map((q, i) => ({ ...q, yaw: q.yaw + (o.move ? 0.02 * Math.sin(k + i) : 0) })) };
      r.frame = (r.frame || 0) + 1;
      const fno = r.frame;
      r.setPoses(K.posesFromState(J.scene, s2));
      const t0 = performance.now();
      await r.renderCamera(cam, { seed: 0 });
      if (o.sensor !== false) r.sensor(cam, {});
      await r.idle();
      wall.push(performance.now() - t0);
      ms.push(r.timings().frames.find((f) => f.frame === fno) || {});
      await new Promise((res) => setTimeout(res, 20));
    }
    res[cam] = { wall, frames: ms };
  }
  return res;
};

/**
 * PRT training data: for each scope state (tools at `base`), converge the path tracer's node solver (warm-started
 * from the previous pose) and return E_free (slot 0 rgb) of every active node as base64 float32, with the features.
 * Needs ?prt=0.
 */
J.gen = async (scopes, o = {}) => {
  const r = J.renderer;
  if (r.prtMode) throw new Error('gen needs ?prt=0 (the node solver)');
  const base = { ...K.defaultState(J.scene), ...(o.base || {}) };
  const out = [];
  const act = r.activeList;
  for (const sc of scopes) {
    const st = { ...base, scope: { ...base.scope, ...sc } };
    const P = K.posesFromState(J.scene, st);
    r.setPoses(P);
    await r.converge(o.iters ?? 64);
    const sum = await r.nodeSummary();
    const s = sum.state, E = new Float32Array(act.length * 3);
    for (let j = 0; j < act.length; j++) { const i = act[j]; E[3 * j] = s[16 * i]; E[3 * j + 1] = s[16 * i + 1]; E[3 * j + 2] = s[16 * i + 2]; }
    out.push({ scope: sc, feat: scopeFeatures(J.scene, r.S, P.bodies.scope), E: b64(new Uint8Array(E.buffer)) });
  }
  return out;
};

J.activeB64 = () => b64(new Uint8Array(J.renderer.activeList.buffer.slice(0)));
J.nodeInfo = () => ({ staticVerts: J.renderer.S.staticVerts, nActive: J.renderer.activeCount });
