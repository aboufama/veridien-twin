// The god's-eye view page of the godview_pt candidate: one orbitable view, cutaway on (C), default state.
import { loadScene } from '../gpu/scene_gpu.js';
import * as K from '../control/kinematics.js';
import { GodView } from './godview.js';

const canvas = document.getElementById('view');
const line = (t) => { document.getElementById('line').textContent = t || ''; };
async function getJSON(urls) {
  for (const u of urls) { try { const r = await fetch(u, { cache: 'no-store' }); if (r.ok) return await r.json(); } catch { /* next */ } }
  throw new Error(`none of ${urls.join(', ')} loaded`);
}
const DEG = Math.PI / 180;

async function main() {
  const [bundle, physics, godview] = await Promise.all([loadScene('../assets/'), getJSON(['/veridien-twin/configs/twin/physics.json', './physics.json']), getJSON(['/veridien-twin/configs/twin/godview.json', './godview.json'])]);
  const fit = () => { const d = Math.min(devicePixelRatio || 1, 2); canvas.width = Math.round(innerWidth * d); canvas.height = Math.round(innerHeight * d); };
  fit();
  const g = await GodView.create({ canvas, scene: bundle, physics, godview });
  g.setPoses(K.posesFromState(bundle.scene, K.defaultState(bundle.scene)));     // scene.default_state (already inside the limits)
  const vps = Object.keys(godview.viewpoints);
  let orbit = { ...godview.viewpoints[vps[0]] }, target = [...godview.centre], cut = true;
  const apply = () => {
    const az = orbit.az_deg * DEG, el = orbit.el_deg * DEG;
    const eye = [target[0] + orbit.d * Math.cos(el) * Math.cos(az), target[1] + orbit.d * Math.cos(el) * Math.sin(az), target[2] + orbit.d * Math.sin(el)];
    g.setCamera({ eye, target, up: [0, 0, 1], vfov_deg: godview.vfov_deg });
  };
  apply();
  let drag = null;
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {
      const az = orbit.az_deg * DEG, s = orbit.d * 0.0015;
      const right = [-Math.sin(az), Math.cos(az), 0];
      target = [target[0] - right[0] * dx * s, target[1] - right[1] * dx * s, target[2] + dy * s];
    } else {
      orbit.az_deg -= dx * 0.3;
      orbit.el_deg = Math.max(-5, Math.min(89, orbit.el_deg + dy * 0.3));
    }
    apply();
  });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); orbit.d = Math.max(0.15, Math.min(3, orbit.d * Math.exp(e.deltaY * 0.001))); apply(); }, { passive: false });
  addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') { orbit = { ...godview.viewpoints[vps[0]] }; target = [...godview.centre]; apply(); }
    if (e.key === 'c' || e.key === 'C') { cut = !cut; g.setCutaway(cut); }
    const i = Number(e.key) - 1;
    if (i >= 0 && i < vps.length) { orbit = { ...godview.viewpoints[vps[i]] }; target = [...godview.centre]; apply(); }
  });
  addEventListener('resize', () => { fit(); g._changed(); });
  const loop = () => { g.render(); if (g.lostInfo || g.lastError) line(g.lostInfo || g.lastError); requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
  window.__god = g;
}
main().catch((e) => { line(`error: ${e.message}`); console.error(e); });
