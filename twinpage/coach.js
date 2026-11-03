// Coach panel: live feedback from the tracked instrument tips, plus an optional AI coach (Claude).
// Reads window.twinPage.rig.fed ({frame_id, t, tips: {L: tip_D, R: tip_D}}, dome frame, metres), which the page fills from the
// camera server or the in-browser tracker alike. Shows raw measurements with units and plain observations, never a grade or a
// skill score (CLAUDE.md: no unvalidated grades, not clinical decision support). Live rules run in the page; the AI coach is
// opt-in: the user's own Anthropic API key, kept in this browser's localStorage and sent only to api.anthropic.com.
// ?coach=0 hides the panel.

const SDK_URL = 'https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.135.0/+esm';
const MODEL = 'claude-opus-5-5';
const WINDOW_S = 10;           // rolling window for the live numbers
const IDLE_MM_S = 5;           // tip speed below this counts as idle
const FAST_MM_S = 120;         // guide: faster than this is flagged
const CLOSE_MM = 8;            // guide: tips closer than this are flagged
const LOST_S = 3;              // guide: out of view longer than this is flagged
const KEY_STORE = 'veridien.coach.key';

const Q = new URLSearchParams(location.search);

/** Per-tool rolling history of [t_s, x, y, z] in mm, plus session totals. */
class ToolTrack {
  constructor() { this.h = []; this.path = 0; this.lastSeen = null; this.lostSince = null; this.first = null; }
  push(t, p) {
    const last = this.h[this.h.length - 1];
    if (last && t <= last[0]) return;
    if (last && t - last[0] < 1) this.path += Math.hypot(p[0] - last[1], p[1] - last[2], p[2] - last[3]);
    this.h.push([t, p[0], p[1], p[2]]);
    while (this.h.length && t - this.h[0][0] > WINDOW_S) this.h.shift();
    this.lastSeen = t; this.lostSince = null; if (this.first == null) this.first = t;
  }
  miss(t) { if (this.lostSince == null) this.lostSince = t; }
  /** Rolling-window numbers: speed median/peak (mm/s), idle fraction, movement units (speed peaks above 20 mm/s), path (mm). */
  stats() {
    const h = this.h, sp = [];
    for (let i = 1; i < h.length; i++) {
      const dt = h[i][0] - h[i - 1][0];
      if (dt > 0 && dt < 0.5) sp.push(Math.hypot(h[i][1] - h[i - 1][1], h[i][2] - h[i - 1][2], h[i][3] - h[i - 1][3]) / dt);
    }
    if (sp.length < 3) return null;
    const sm = sp.map((_, i) => (sp[Math.max(0, i - 1)] + sp[i] + sp[Math.min(sp.length - 1, i + 1)]) / 3);
    let units = 0;
    for (let i = 1; i < sm.length - 1; i++) if (sm[i] > 20 && sm[i] >= sm[i - 1] && sm[i] > sm[i + 1]) units++;
    const sorted = [...sm].sort((a, b) => a - b);
    let win = 0;
    for (let i = 1; i < h.length; i++) win += Math.hypot(h[i][1] - h[i - 1][1], h[i][2] - h[i - 1][2], h[i][3] - h[i - 1][3]);
    const span = h[h.length - 1][0] - h[0][0];
    return { speed_med: sorted[sorted.length >> 1], speed_peak: sorted[Math.floor(sorted.length * 0.95)],
             idle: sm.filter((v) => v < IDLE_MM_S).length / sm.length, units, units_per_10s: span > 0 ? units * 10 / span : 0,
             path_window: win, span };
  }
}

const tracks = { L: new ToolTrack(), R: new ToolTrack() };
let lastFrame = null, t0 = null, lastTips = {};

function sample() {
  const fed = window.twinPage?.rig?.fed;
  if (!fed || fed.frame_id === lastFrame) return;
  lastFrame = fed.frame_id;
  const t = Number.isFinite(fed.t) ? (fed.t > 1e11 ? fed.t / 1e3 : fed.t) : performance.now() / 1e3;   // feed t: s (epoch); ms if huge
  if (t0 == null) t0 = t;
  for (const id of ['L', 'R']) {
    const p = fed.tips?.[id];
    if (Array.isArray(p) && p.every(Number.isFinite)) {
      const mm = Math.max(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2])) < 2 ? p.map((v) => v * 1e3) : p;
      tracks[id].push(t, mm); lastTips[id] = mm;
    } else { tracks[id].miss(t); delete lastTips[id]; }
  }
  traceStep();
}

/** The live picture: numbers per tool and plain-language observations (guides, not grades). */
function snapshot() {
  const now = tracks.L.lastSeen ?? tracks.R.lastSeen ?? 0;
  const out = { elapsed_s: t0 == null ? 0 : Math.max(tracks.L.lastSeen ?? 0, tracks.R.lastSeen ?? 0) - t0, tools: {}, notes: [] };
  const name = { L: 'Left', R: 'Right' };
  for (const id of ['L', 'R']) {
    const tr = tracks[id], s = tr.stats();
    const lost = tr.lostSince != null ? Math.max(0, now - tr.lostSince) : 0;
    out.tools[id] = { visible: tr.lostSince == null && tr.lastSeen != null, out_of_view_s: +lost.toFixed(1),
                      path_total_mm: Math.round(tr.path), ...(s ? { speed_median_mm_s: +s.speed_med.toFixed(1), speed_p95_mm_s: +s.speed_peak.toFixed(1),
                      idle_pct: Math.round(s.idle * 100), movement_units_per_10s: +s.units_per_10s.toFixed(1) } : {}) };
    if (tr.lastSeen != null && lost > LOST_S) out.notes.push({ k: 'lost', t: `${name[id]} tool out of the cameras' view for ${lost.toFixed(0)} s: bring it back into view.` });
    if (!s) continue;
    if (s.speed_peak > FAST_MM_S) out.notes.push({ k: 'fast', t: `${name[id]} tool peaking at ${s.speed_peak.toFixed(0)} mm/s: slow down for control near tissue.` });
    if (s.units_per_10s > 8) out.notes.push({ k: 'jerky', t: `${name[id]} tool: ${s.units_per_10s.toFixed(0)} movement units per 10 s: aim for fewer, smoother motions.` });
  }
  const sL = tracks.L.stats(), sR = tracks.R.stats();
  if (sL && sR) {
    const [idle, busy, a] = sL.idle > sR.idle ? [sL, sR, 'Left'] : [sR, sL, 'Right'];
    if (idle.idle > 0.8 && busy.idle < 0.4) out.notes.push({ k: 'bimanual', t: `${a} hand idle ${Math.round(idle.idle * 100)} % of the last ${WINDOW_S} s while the other works: use both hands.` });
  }
  if (lastTips.L && lastTips.R) {
    const d = Math.hypot(lastTips.L[0] - lastTips.R[0], lastTips.L[1] - lastTips.R[1], lastTips.L[2] - lastTips.R[2]);
    out.tip_separation_mm = Math.round(d);
    if (d < CLOSE_MM) out.notes.push({ k: 'close', t: `Tips ${d.toFixed(0)} mm apart: watch for instrument clashes.` });
  }
  if (!out.notes.length && (sL || sR)) out.notes.push({ k: 'ok', t: 'Steady: no flags in the last 10 s.' });
  return out;
}

// ---------------------------------------------------------------- trace task
// Draw a shape on the floor with one tool tip (no jaw state needed). Start puts the shape's centre at the tip's current
// position (top-down: dome frame x, y; z up), then every tracked point is scored against the target outline.
const SHAPES = { circle: { label: 'Circle, 20 mm radius', r: 20 }, square: { label: 'Square, 40 mm side', a: 20 } };
const trace = { on: false, shape: 'circle', tool: null, c: null, pts: [], bins: new Set(), t0: null, zs: [] };
function shapeDist(shape, x, y) {
  if (shape === 'circle') return Math.abs(Math.hypot(x, y) - SHAPES.circle.r);
  const a = SHAPES.square.a, dx = Math.abs(x) - a, dy = Math.abs(y) - a;
  return Math.abs(Math.min(Math.max(dx, dy), 0) + Math.hypot(Math.max(dx, 0), Math.max(dy, 0)));
}
function traceStart(shape) {
  const id = ['L', 'R'].sort((a, b) => (tracks[b].stats()?.speed_med ?? -1) - (tracks[a].stats()?.speed_med ?? -1)).find((k) => lastTips[k]);
  if (!id) return 'No tool tip is tracked yet.';
  Object.assign(trace, { on: true, shape, tool: id, c: lastTips[id].slice(), pts: [], bins: new Set(), t0: tracks[id].lastSeen, zs: [] });
  return null;
}
function traceStep() {
  if (!trace.on) return;
  const p = lastTips[trace.tool];
  if (!p) return;
  const x = p[0] - trace.c[0], y = p[1] - trace.c[1], last = trace.pts[trace.pts.length - 1];
  if (last && Math.hypot(x - last[0], y - last[1]) < 0.3) return;
  const d = shapeDist(trace.shape, x, y);
  trace.pts.push([x, y, d]); trace.zs.push(p[2] - trace.c[2]);
  if (d < 5) trace.bins.add(Math.floor(((Math.atan2(y, x) + Math.PI) / (2 * Math.PI)) * 36) % 36);
}
function traceStats() {
  if (!trace.c || trace.pts.length < 5) return null;
  const ds = trace.pts.map((q) => q[2]).sort((a, b) => a - b), zs = [...trace.zs].sort((a, b) => a - b);
  let path = 0;
  for (let i = 1; i < trace.pts.length; i++) path += Math.hypot(trace.pts[i][0] - trace.pts[i - 1][0], trace.pts[i][1] - trace.pts[i - 1][1]);
  return { shape: SHAPES[trace.shape].label, tool: trace.tool === 'L' ? 'left' : 'right', points: trace.pts.length,
           deviation_mean_mm: +(ds.reduce((a, b) => a + b, 0) / ds.length).toFixed(1), deviation_p95_mm: +ds[Math.floor(ds.length * 0.95)].toFixed(1),
           outline_covered_pct: Math.round(trace.bins.size / 36 * 100), path_mm: Math.round(path),
           height_range_mm: +(zs[Math.floor(zs.length * 0.95)] - zs[Math.floor(zs.length * 0.05)]).toFixed(1),
           elapsed_s: +((tracks[trace.tool].lastSeen ?? trace.t0) - trace.t0).toFixed(1) };
}
function drawTrace() {
  const cv = document.getElementById('c-trace');
  if (!cv) return;
  const g = cv.getContext('2d'), W = cv.width, k = W / 70, o = W / 2;   // 70 mm across
  g.clearRect(0, 0, W, W);
  g.strokeStyle = 'rgba(29,35,33,.18)'; g.lineWidth = 1; g.strokeRect(0.5, 0.5, W - 1, W - 1);
  g.strokeStyle = '#3b8a5e'; g.lineWidth = 2; g.setLineDash([4, 3]); g.beginPath();
  if (trace.shape === 'circle') g.arc(o, o, SHAPES.circle.r * k, 0, 2 * Math.PI);
  else g.rect(o - SHAPES.square.a * k, o - SHAPES.square.a * k, 2 * SHAPES.square.a * k, 2 * SHAPES.square.a * k);
  g.stroke(); g.setLineDash([]);
  if (!trace.pts.length) return;
  g.lineWidth = 2;
  for (let i = 1; i < trace.pts.length; i++) {
    const [x0, y0] = trace.pts[i - 1], [x1, y1, d] = trace.pts[i];
    g.strokeStyle = d < 2 ? '#1f6f45' : d < 5 ? '#c08a1a' : '#b8432f';
    g.beginPath(); g.moveTo(o + x0 * k, o - y0 * k); g.lineTo(o + x1 * k, o - y1 * k); g.stroke();
  }
  const [x, y] = trace.pts[trace.pts.length - 1];
  g.fillStyle = '#1d2321'; g.beginPath(); g.arc(o + x * k, o - y * k, 3.5, 0, 2 * Math.PI); g.fill();
}

// ---------------------------------------------------------------- UI
const css = `
#coach{position:fixed;left:12px;top:12px;z-index:50;width:min(300px,calc(100vw - 24px));font:12.5px/1.45 system-ui,-apple-system,sans-serif;color:#1d2321;
  background:rgba(255,255,255,.94);border:1px solid rgba(0,0,0,.12);border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.08);padding:10px 12px}
#coach h3{margin:0 0 6px;font-size:13px;font-weight:600;display:flex;justify-content:space-between;align-items:center}
#coach h3 button{font:inherit;font-size:11px;border:0;background:none;color:#6b7470;cursor:pointer}
#coach table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;margin-bottom:6px}
#coach td{padding:1px 0;color:#4a524e}#coach td.v{text-align:right;color:#1d2321}
#coach .notes div{padding:3px 0;border-top:1px solid rgba(0,0,0,.06)}
#coach .ai{margin-top:8px;padding-top:6px;border-top:1px solid rgba(0,0,0,.12)}
#coach .ai p{margin:4px 0;white-space:pre-wrap;color:#1d2321}
#coach input{width:100%;box-sizing:border-box;font:inherit;font-size:11.5px;padding:4px 6px;border:1px solid rgba(0,0,0,.2);border-radius:6px}
#coach .row{display:flex;gap:6px;align-items:center;margin-top:5px}
#coach select{font:inherit;font-size:12px;padding:2px 4px;border:1px solid rgba(0,0,0,.25);border-radius:6px;background:#fff}
#coach .row button{font:inherit;font-size:12px;padding:3px 9px;border:1px solid rgba(0,0,0,.25);border-radius:6px;background:#fff;cursor:pointer}
#coach .muted{color:#7a837f;font-size:11px}
#coach.min .body{display:none}`;

function mount() {
  if (Q.get('coach') === '0' || document.getElementById('coach')) return;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
  const el = document.createElement('div'); el.id = 'coach';
  el.innerHTML = `<h3>Coach <button id="coach-min">hide</button></h3><div class="body">
    <table><tr><td></td><td class="v">Left</td><td class="v">Right</td></tr>
      <tr><td>speed (median)</td><td class="v" id="c-sL">–</td><td class="v" id="c-sR">–</td></tr>
      <tr><td>speed (p95)</td><td class="v" id="c-pL">–</td><td class="v" id="c-pR">–</td></tr>
      <tr><td>idle</td><td class="v" id="c-iL">–</td><td class="v" id="c-iR">–</td></tr>
      <tr><td>path</td><td class="v" id="c-dL">–</td><td class="v" id="c-dR">–</td></tr></table>
    <div class="muted" id="c-sep"></div>
    <div class="notes" id="c-notes"><div class="muted">Waiting for tracked tools…</div></div>
    <div class="ai"><b>Trace a shape</b> <span class="muted">(tip on the floor)</span>
      <div class="row"><select id="c-shape"><option value="circle">Circle, 20 mm radius</option><option value="square">Square, 40 mm side</option></select>
        <button id="c-trace-go">Start</button></div>
      <canvas id="c-trace" width="200" height="200" style="display:block;margin:6px auto 2px;width:200px;height:200px"></canvas>
      <div class="muted" id="c-trace-txt">Hold the tip at the shape's centre, press Start, then trace the outline.</div></div>
    <div class="ai"><b>Live cue</b> <span class="muted">(Jev)</span>
      <label class="muted" style="float:right"><input id="c-voice" type="checkbox" style="width:auto"> speak cues</label>
      <p id="c-jev" class="muted">Waiting for movement…</p></div>
    <div class="ai"><b>AI coach</b> <span class="muted">(Claude)</span>
      <div id="c-keyrow" class="row"><input id="c-key" type="password" placeholder="Anthropic API key (stays in this browser)"></div>
      <div class="row"><button id="c-ask">Ask now</button><label class="muted"><input id="c-auto" type="checkbox" style="width:auto"> every 20 s</label></div>
      <p id="c-ai" class="muted">Gives a short tip from the live numbers above.</p></div></div>`;
  document.body.appendChild(el);
  el.querySelector('#coach-min').onclick = (e) => { el.classList.toggle('min'); e.target.textContent = el.classList.contains('min') ? 'show' : 'hide'; };
  const key = el.querySelector('#c-key');
  try { key.value = localStorage.getItem(KEY_STORE) || ''; } catch { /* storage blocked */ }
  key.onchange = () => { try { localStorage.setItem(KEY_STORE, key.value.trim()); } catch { /* storage blocked */ } };
  el.querySelector('#c-ask').onclick = () => askAI();
  const go = el.querySelector('#c-trace-go');
  go.onclick = () => {
    if (trace.on) { trace.on = false; go.textContent = 'Start'; return; }
    const err = traceStart(el.querySelector('#c-shape').value);
    if (err) { el.querySelector('#c-trace-txt').textContent = err; return; }
    go.textContent = 'Stop';
  };
  el.querySelector('#c-shape').onchange = (e) => { if (!trace.on) { trace.shape = e.target.value; drawTrace(); } };
  setInterval(() => { if (el.querySelector('#c-auto').checked) askAI(); }, 20000);
}

const fmt = (v, u) => (Number.isFinite(v) ? `${v} ${u}` : '–');
function render() {
  const s = snapshot(), $ = (id) => document.getElementById(id);
  if (!$('coach')) return s;
  for (const id of ['L', 'R']) {
    const t = s.tools[id];
    $(`c-s${id}`).textContent = t.visible === false && t.out_of_view_s > 0 ? 'not seen' : fmt(t.speed_median_mm_s, 'mm/s');
    $(`c-p${id}`).textContent = fmt(t.speed_p95_mm_s, 'mm/s');
    $(`c-i${id}`).textContent = fmt(t.idle_pct, '%');
    $(`c-d${id}`).textContent = t.path_total_mm ? `${(t.path_total_mm / 10).toFixed(0)} cm` : '–';
  }
  $('c-sep').textContent = Number.isFinite(s.tip_separation_mm) ? `Tip separation ${s.tip_separation_mm} mm · ${s.elapsed_s.toFixed(0)} s tracked` : '';
  if (s.notes.length) $('c-notes').innerHTML = s.notes.map((n) => `<div>${n.t}</div>`).join('');
  if (performance.now() < jevDown && s.notes[0] && s.notes[0].k !== 'ok') say(s.notes[0].t.split(':')[0]);
  const ts = traceStats();
  if (ts) $('c-trace-txt').textContent = `${ts.tool} tool · off the outline ${ts.deviation_mean_mm} mm mean, ${ts.deviation_p95_mm} mm p95 · ` +
    `${ts.outline_covered_pct} % of the outline traced · height range ${ts.height_range_mm} mm · ${ts.elapsed_s} s`;
  drawTrace();
  return s;
}

// ---------------------------------------------------------------- voice
// The browser's own speech synthesis: no download, no service, near-zero cost. A cue is spoken only when it changes,
// at most every 6 s, and never over itself.
let lastSpoken = '', lastSpokenAt = 0;
function say(text) {
  if (!document.getElementById('c-voice')?.checked || !('speechSynthesis' in window)) return;
  const now = performance.now();
  if (text === lastSpoken || now - lastSpokenAt < 6000 || speechSynthesis.speaking) return;
  const u = new SpeechSynthesisUtterance(text.replace(/\s*\(.*?\)\s*$/, ''));
  u.rate = 1.05; u.pitch = 1; u.volume = 1;
  speechSynthesis.speak(u);
  lastSpoken = text; lastSpokenAt = now;
}

// ---------------------------------------------------------------- Jev live cues
// TypeSafe Jev (System One) picks one cue from a fixed list with calibrated probabilities in ~0.2 s, so it can run live.
// Its API has no CORS, so the page asks the local relay (scripts/jev_relay.py, holds the key); ?jev=<url> moves it.
const JEV_URL = (Q.get('jev') || 'http://127.0.0.1:8899').replace(/\/$/, '') + '/jev';
const CUES = {
  slow_down: 'Slow down: the tip is moving fast for fine control.',
  smoother: 'Smoother: fewer, more continuous movements.',
  stay_on_line: 'Stay on the outline: you are drifting off the shape.',
  keep_on_floor: 'Keep light contact with the floor: the tip is lifting.',
  use_both_hands: 'Use both hands: one tool is idle.',
  find_tool: 'Bring the tool back into the cameras\' view.',
  keep_going: 'Good: keep the same pace and control.',
};
let jevBusy = false, jevDown = 0;
async function jevCue() {
  const out = document.getElementById('c-jev');
  if (!out || jevBusy || performance.now() < jevDown) return;
  const s = snapshot(), ts = traceStats();
  if (!Object.values(s.tools).some((t) => t.speed_median_mm_s != null)) return;
  jevBusy = true;
  const t0 = performance.now();
  try {
    const r = await fetch(JEV_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
      model: 'jev-latest',
      state: { task: ts ? `tracing a ${ts.shape.toLowerCase()} on the floor with the ${ts.tool} grasper tip` : 'free practice with two graspers',
               tools: s.tools, tip_separation_mm: s.tip_separation_mm, trace: ts },
      questions: { cue: { type: 'choice', instructions: 'Pick the single most useful coaching cue for this trainee right now, based only on `state` (objective measurements).', criteria: CUES } },
    }) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json(), a = j.answers?.cue;
    if (!a?.choice) throw new Error('no answer');
    out.className = '';
    out.textContent = `${CUES[a.choice] || a.choice} (${Math.round((a.probabilities?.[a.choice] ?? a.confidence ?? 0) * 100)} %, ${Math.round(performance.now() - t0)} ms)`;
    say(CUES[a.choice] || a.choice);
  } catch (e) {
    out.className = 'muted';
    out.textContent = 'Jev relay not running: python scripts/jev_relay.py';
    jevDown = performance.now() + 10000;
  } finally { jevBusy = false; }
}

// ---------------------------------------------------------------- AI coach
const SYSTEM = `You are a coach for laparoscopic skills practice on a box trainer (Veridien). You receive live, objective
measurements of two grasper tips tracked by cameras: speed (mm/s), idle share, movement units per 10 s (speed peaks:
more means jerkier), path length, time out of the cameras' view, and tip separation. Give ONE short, specific, encouraging
coaching tip (max 2 sentences) that refers to the actual numbers. Rules: never give a score, grade or skill rating; never
claim learning outcomes or surgical competence; no clinical or patient advice; if the numbers look fine, say what to keep
doing; if there is too little data, say so briefly. If trace_task is present, the trainee is tracing a shape on the floor
with one tool tip: deviation is the distance off the target outline (mm), outline_covered_pct how much of it they have
traced, height_range_mm how much the tip lifted off the floor; coach that task first.`;

let client = null, busy = false;
async function askAI() {
  const out = document.getElementById('c-ai'), key = document.getElementById('c-key')?.value.trim();
  if (!out || busy) return;
  if (!key) { out.textContent = 'Paste an Anthropic API key above to turn on the AI coach.'; return; }
  busy = true; out.className = ''; out.textContent = '…';
  try {
    if (!client || client.apiKey !== key) {
      const { default: Anthropic } = await import(SDK_URL);
      client = new Anthropic({ apiKey: key, dangerouslyAllowBrowser: true });
    }
    const s = snapshot();
    const stream = client.beta.messages.stream({
      model: MODEL, max_tokens: 2000, output_config: { effort: 'low' },
      betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default',
      system: SYSTEM,
      messages: [{ role: 'user', content: `Live measurements (JSON):\n${JSON.stringify({ tools: s.tools, tip_separation_mm: s.tip_separation_mm, elapsed_s: Math.round(s.elapsed_s), live_flags: s.notes.map((n) => n.t), trace_task: traceStats() })}` }],
    });
    let text = '';
    stream.on('text', (d) => { text += d; out.textContent = text; });
    const msg = await stream.finalMessage();
    if (msg.stop_reason === 'refusal') out.textContent = 'The coach could not answer this one.';
    else if (!text) out.textContent = msg.content.filter((b) => b.type === 'text').map((b) => b.text).join('') || '(no tip)';
  } catch (e) {
    out.className = 'muted';
    out.textContent = e?.status === 401 ? 'That API key was rejected.' : `AI coach unavailable: ${e?.message || e}`;
  } finally { busy = false; }
}

// ---------------------------------------------------------------- loop
function start() {
  mount();
  setInterval(sample, 30);          // tips arrive at up to ~30 Hz; frame ids dedupe
  setInterval(render, 500);
  if (Q.get('jev') !== '0') setInterval(jevCue, 2000);
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
export { snapshot, ToolTrack };
