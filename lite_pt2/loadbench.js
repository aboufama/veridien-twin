// Load bench of lite_pt2 (the side-column start the twin page sees): page start -> create() -> the four tiles
// rendering round-robin (one camera per rAF, as the column does) for `seconds`, while the deferred gather table
// builds. Records, from the browser itself: create() wall time, first frame, the gather table's arrival, and every
// main-thread long task (PerformanceObserver 'longtask', > 50 ms) from page start. ?renderer=lite_pt (or any
// <dir>) runs the same bench on another renderer. Driven by tools/load_bench.py through TwinPage (window.twinJudge).
import { loadScene } from '../gpu/scene_gpu.js';
import * as K from '../control/kinematics.js';
import { Limits } from '../control/limits.js';

const T0 = performance.now();
const LT = [];
try { new PerformanceObserver((l) => { for (const e of l.getEntries()) LT.push([Math.round(e.startTime), Math.round(e.duration)]); }).observe({ type: 'longtask', buffered: true }); } catch { /* none */ }
const Q = new URLSearchParams(location.search);
const NAME = (Q.get('renderer') || 'lite_pt2').replace(/[^a-z0-9_]/gi, '');
const SECONDS = Number(Q.get('seconds') || 12);
const out = { renderer: NAME, seconds: SECONDS };
const msg = (t) => { document.getElementById('msg').textContent = t; };
const raf = () => new Promise((r) => requestAnimationFrame(r));

async function run() {
  const bundle = await loadScene('../assets/');
  let physics = {};
  try { const r = await fetch('/veridien-twin/configs/twin/physics.json'); if (r.ok) physics = await r.json(); } catch { /* static */ }
  out.scene_ms = performance.now() - T0;
  // the page's own boot work (Limits, the first poses) before the renderer starts, as twinpage/main.js does it
  const tl = performance.now();
  const lim = new Limits(bundle.scene, bundle.meshes);
  const st = lim.clampState({ ...K.defaultState(bundle.scene) }).state;
  const poses = K.posesFromState(bundle.scene, st);
  out.marks = { scene_loaded: Math.round(tl - T0), limits_ms: Math.round(performance.now() - tl) };
  await new Promise((res) => setTimeout(res, 50));
  const mod = await import(`../${NAME}/renderer.js`);
  const R = mod.LitePTRenderer || mod.default;
  const tc = performance.now();
  // a rAF ticker during create(): frames the page could not paint show up as gaps
  let ticking = true, lastTick = performance.now(), maxGap = 0;
  (async () => { while (ticking) { await raf(); const t = performance.now(); maxGap = Math.max(maxGap, t - lastTick); lastTick = t; } })();
  const r = await R.create({ scene: bundle, physics, preset: 'live', log: (...a) => console.log(`[${NAME}]`, ...a) });
  out.create_ms = performance.now() - tc;
  out.marks.create_start = Math.round(tc - T0); out.marks.create_end = Math.round(performance.now() - T0);
  out.raf_max_gap_during_create_ms = maxGap;
  r.setPoses(poses);
  const cams = ['SCOPE', 'CAM1', 'CAM2', 'CAM3'].filter((c) => r.cameras.includes(c));
  let first = null, frames = 0, gapMax = 0, last = performance.now();
  const tEnd = performance.now() + SECONDS * 1000;
  r.gatherReady?.then(() => { out.gather_ready_ms = performance.now() - T0; });
  for (let i = 0; performance.now() < tEnd; i++) {
    const c = cams[i % cams.length];
    await r.renderCamera(c);
    r.sensor(c);
    r.present(c, document.getElementById(c));
    if (first === null) { out.marks.first_submit = Math.round(performance.now() - T0); await r.device.queue.onSubmittedWorkDone(); first = performance.now() - T0; out.first_frame_ms = first; }
    if (i < 4) out.marks[`frame${i}_${c}`] = Math.round(performance.now() - T0);
    frames++;
    await raf();
    const t = performance.now(); gapMax = Math.max(gapMax, t - last); last = t;
  }
  ticking = false;
  await r.device.queue.onSubmittedWorkDone();
  out.frames = frames; out.fps = frames / SECONDS; out.raf_max_gap_after_create_ms = gapMax;
  out.long_tasks = LT.slice(); out.long_task_max_ms = LT.length ? Math.max(...LT.map((x) => x[1])) : 0;
  out.long_tasks_over_100 = LT.filter((x) => x[1] > 100).length;
  out.timings = r.timings?.().litePT ?? null;
  msg(JSON.stringify({ create_ms: out.create_ms, first_frame_ms: out.first_frame_ms, long_task_max_ms: out.long_task_max_ms }));
  return out;
}
const ready = run().catch((e) => { out.error = String(e?.stack || e); msg(out.error); return out; });
window.twinJudge = { ready: ready.then(() => true), result: () => ready };
