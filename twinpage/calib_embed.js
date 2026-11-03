// The real cameras and their calibration, inside the twin page: the one tab that runs everything (the user: "integrate
// this page into the main digital clone view subtly in a calibration section ... just to run off that main page").
//
// One iframe of the tracker's camera page (http://localhost:8833/?live=1&embed=1) opens the three cameras, feeds the
// tracker, and holds the calibration workspace (camera calibration, truth jig, recording). Collapsed, the dock is a
// bar (status, Record, Calibrate...) over a thin live strip of the three cameras (the page is told it is collapsed and
// shows only its camera tiles). The videos must stay visibly on screen: a near-invisible muted video is treated as
// off-screen and stops rendering, and the captured frames go black (2026-11-01).
// Expanded, the same iframe fills a panel over the 3-D view, so capture never restarts. It always stays mounted and on
// screen: Chrome throttles cross-origin frames that are hidden or scrolled out of view, which would stall capture.
//
// Only one page may hold the cameras: the tracker gives the stream to the page that started most recently and tells
// an older one (a separate camera tab) to retire, which releases the cameras. Two pages capturing at once fought over
// USB bandwidth and CPU (30 -> 12 frame sets/s).
//
// That is the 'server' source (twinpage/tracker_source.js). With the 'browser' source (no server: the hosted site, or
// ?tracker=browser) there is no iframe and nothing reaches for localhost: the dock is a bar (status, Start cameras,
// Record, Calibrate...) over this page's own tracker (twinpage/browser_source.js). Record downloads a ZIP in the
// server recorder's layout; Calibrate... shows which camera is which (swap them, or match them by their views) and
// says that camera calibration and the truth jig need the camera server. When the page asks for the server (the
// local twin server's default, ?tracker=server, ?rig=<url>) the camera page mounts at once, exactly as before; otherwise
// the dock waits for main.js to announce the source it picked ('twin-source': auto's probe takes <= 1.5 s).

import { browserNote } from './browser_core.js';
import { trackerModeFromQuery } from './tracker_source.js';

const BASE = new URLSearchParams(location.search).get('cams') || 'http://localhost:8833';
const SRC = BASE + '/?live=1&embed=1';

const css = `
  #calib-dock { position: fixed; right: 0; bottom: 0; width: var(--col, 420px); z-index: 20;
    background: var(--paper, #f3f3ef); border-top: 1px solid var(--rule, #dcdcd6); border-left: 1px solid var(--rule, #dcdcd6); }
  body.collapsed #calib-dock { width: 360px; border-radius: 10px 0 0 0; }
  #calib-dock .bar { display: flex; align-items: center; gap: 10px; padding: 9px 14px; font-size: 13px; color: var(--grey, #6b6c68); }
  #calib-dock .bar b { font-size: 14px; }
  #calib-dock #calib-note { font-variant-numeric: tabular-nums; }
  #calib-dock .bar b { color: var(--ink, #111211); font-weight: 600; }
  #calib-dock .bar button { margin-left: auto; border: 1px solid var(--rule, #dcdcd6); background: var(--card, #fbfbf8);
    color: var(--ink, #111211); border-radius: 7px; padding: 5px 14px; font: inherit; cursor: pointer; }
  #calib-dock .bar button:hover { border-color: var(--grey2, #a9aaa4); }
  #calib-dock .bar #calib-rec { margin-left: auto; }
  #calib-dock .bar #calib-rec + #calib-toggle { margin-left: 0; }
  #calib-dock .bar #calib-rec.on { background: #b3261e; border-color: #b3261e; color: #fff; }
  #calib-frame-wrap { position: relative; height: 64px; overflow: hidden; pointer-events: none; }
  #calib-frame { border: 0; width: 1200px; height: 760px; transform-origin: 0 0; background: transparent; }
  #calib-dock.open { left: 0; top: 0; width: auto; right: 0; z-index: 30; display: flex; flex-direction: column;
    border-left: 0; }                     /* the whole window: no second set of camera views beside the workspace */
  body.collapsed #calib-dock.open { right: 0; border-radius: 0; }
  #calib-dock.open #calib-frame-wrap { flex: 1; height: auto; pointer-events: auto; }
  #calib-dock.open #calib-frame { width: 100%; height: 100%; transform: none !important; }
  /* the browser source: a bar, and Calibrate... opens a panel (no camera page to embed) */
  #calib-dock.local .bar button { margin-left: 0; }
  #calib-dock.local .bar #calib-note { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #calib-dock.local .bar #calib-start[hidden] { display: none; }
  #calib-dock.local .bar #calib-start { background: var(--ink, #111211); border-color: var(--ink, #111211); color: var(--paper, #f3f3ef); }
  #calib-panel { display: none; }
  #calib-dock.open #calib-panel { display: block; flex: 1; overflow-y: auto; padding: 8px 22px 28px; }
  #calib-panel .cp { max-width: 820px; }
  #calib-panel h2 { font-size: 15px; font-weight: 600; margin: 10px 0 4px; }
  #calib-panel h3 { font-size: 13px; font-weight: 600; margin: 22px 0 4px; }
  #calib-panel p { margin: 4px 0 10px; color: var(--grey, #6b6c68); max-width: 68ch; }
  #calib-panel code { font: 12px var(--mono, ui-monospace, monospace); background: rgba(17, 18, 17, 0.05); padding: 1px 5px; border-radius: 4px; color: var(--ink, #111211); }
  #calib-panel .cp-row { display: grid; grid-template-columns: 76px 224px 224px 1fr; gap: 12px; align-items: center; padding: 8px 0;
    border-top: 1px solid var(--rule, #dcdcd6); }
  #calib-panel .cp-row canvas { width: 224px; height: 126px; border-radius: 6px; background: var(--frame, #0c0d0c); display: block; }
  #calib-panel .cp-cap { font-size: 11.5px; color: var(--grey, #6b6c68); margin-top: 3px; }
  #calib-panel .cp-name { font-weight: 600; }
  #calib-panel .cp-swap { display: flex; flex-direction: column; gap: 6px; align-items: flex-start; }
  #calib-panel button { border: 1px solid var(--rule, #dcdcd6); background: var(--card, #fbfbf8); color: var(--ink, #111211);
    border-radius: 7px; padding: 4px 12px; font: inherit; cursor: pointer; }
  #calib-panel button:hover { border-color: var(--grey2, #a9aaa4); }
  #calib-panel button:disabled { color: var(--grey2, #a9aaa4); cursor: default; border-color: var(--rule, #dcdcd6); }
  #calib-panel .cp-actions { display: flex; gap: 12px; align-items: center; margin-top: 10px; color: var(--grey, #6b6c68); }
  #calib-panel label { display: flex; gap: 8px; align-items: center; margin-top: 14px; }
  @media (max-width: 760px) { #calib-panel .cp-row { grid-template-columns: 1fr 1fr; } #calib-panel .cp-row canvas { width: 100%; height: auto; aspect-ratio: 16 / 9; } }
`;
const FLIP_KEY = 'veridien.flip';                // the camera page's setting: on unless "0"
const flipOn = () => { try { return localStorage.getItem(FLIP_KEY) !== '0'; } catch { return true; } };

function mount() {
  if (document.getElementById('calib-dock')) return;
  mountStyle();
  const dock = document.createElement('section');
  dock.id = 'calib-dock';
  dock.setAttribute('aria-label', 'Cameras and calibration');
  dock.innerHTML = `
    <div class="bar"><b>Cameras</b><span id="calib-note">starting…</span>
      <button type="button" id="calib-rec" title="Record all three cameras and every predicted tip">Record</button>
      <button type="button" id="calib-toggle" aria-expanded="false">Calibrate…</button></div>
    <div id="calib-frame-wrap"><iframe id="calib-frame" title="Tracking cameras" allow="camera; autoplay; local-network-access" src="${SRC}"></iframe></div>`;
  document.body.appendChild(dock);
  const frame = dock.querySelector('#calib-frame'), wrap = dock.querySelector('#calib-frame-wrap');
  const btn = dock.querySelector('#calib-toggle');
  const fit = () => {                     // collapsed: scale the camera row of the embed page into the strip
    if (dock.classList.contains('open')) return;
    frame.style.transform = `scale(${wrap.clientWidth / 1200})`;
  };
  const tell = () => {                    // the camera page shows only its tiles while collapsed
    try { frame.contentWindow.postMessage({ veridienDock: dock.classList.contains('open') ? 'open' : 'collapsed' }, '*'); } catch { /* not loaded */ }
  };
  frame.addEventListener('load', tell);
  setInterval(tell, 3000);
  const set = (open) => {
    dock.classList.toggle('open', open);
    setTimeout(tell, 0);
    btn.textContent = open ? 'Close' : 'Calibrate…';
    btn.setAttribute('aria-expanded', String(open));
    fit();
  };
  btn.addEventListener('click', () => set(!dock.classList.contains('open')));
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && dock.classList.contains('open')) set(false); });
  new ResizeObserver(fit).observe(wrap);
  fit();
  // Record: the tracker keeps every frame set's three JPEGs (15/s) and each predicted tip (tracks.jsonl) in
  // data/multicam/<date>_session_NN; a second click stops. Auto-stop after 5 minutes (disk).
  const rec = dock.querySelector('#calib-rec');
  let recT0 = 0, recTimer = null, recStop = null;
  const post = (body) => fetch(BASE + '/api/record', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                       body: JSON.stringify(body) }).then((r) => r.json());
  const stopRec = async () => {
    clearInterval(recTimer); clearTimeout(recStop); recTimer = recStop = null;
    rec.classList.remove('on');
    try {
      const j = await post({ stop: true });
      rec.textContent = 'Record';
      note.textContent = j.stopped ? `saved ${j.frames} frame sets: ${j.stopped.split('/').pop()}` : 'not recording';
    } catch { rec.textContent = 'Record'; note.textContent = 'stop failed: tracker offline'; }
  };
  rec.addEventListener('click', async () => {
    if (recTimer) return stopRec();
    try {
      const j = await post({ tag: 'session' });
      if (!j.recording) throw new Error(j.error || 'no session');
      recT0 = performance.now();
      rec.classList.add('on');
      const tick = () => { rec.textContent = `Stop ${Math.floor((performance.now() - recT0) / 1000)} s`; };
      tick(); recTimer = setInterval(tick, 250);
      recStop = setTimeout(stopRec, 5 * 60 * 1000);
    } catch (e) { note.textContent = 'record failed: ' + e.message; }
  });
  // live status from the tracker: frame sets per second, so a stall is visible at a glance
  const note = dock.querySelector('#calib-note');
  setInterval(async () => {
    if (document.hidden) return;
    try {
      const t = await (await fetch(BASE + '/api/timing', { cache: 'no-store' })).json();
      if (recTimer) return;              // the Record button's own status stays visible while recording
      note.textContent = t.arrival_hz ? `${t.arrival_hz.toFixed(0)} frame sets/s` : 'waiting for cameras';
    } catch { note.textContent = 'tracker offline'; }
  }, 2000);
}

// Display white balance for the twin's own camera images (real_cams.js tiles and the enlarged view): the tracker
// publishes per-camera gains (/api/wb, from the white dome walls); here they become SVG colour matrices that the GPU
// applies to the images. Frames are never re-encoded, so it costs the tracker nothing.
function mountWB(src) {
  const local = src?.kind === 'browser';
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', '0'); svg.setAttribute('height', '0'); svg.style.position = 'absolute';
  svg.setAttribute('aria-hidden', 'true');
  const mats = {};
  for (const cam of ['CAM1', 'CAM2', 'CAM3']) {
    const f = document.createElementNS(NS, 'filter');
    f.id = 'vwb-' + cam; f.setAttribute('color-interpolation-filters', 'sRGB');
    const m = document.createElementNS(NS, 'feColorMatrix');
    m.setAttribute('type', 'matrix'); m.setAttribute('values', '1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0');
    f.appendChild(m); svg.appendChild(f); mats[cam] = m;
  }
  document.body.appendChild(svg);
  const style = document.createElement('style');
  style.textContent = ['CAM1', 'CAM2', 'CAM3'].map((c) =>
    `.tile.real[data-view="${c}"] .frame > img, .tile.real[data-view="${c}"] .frame > canvas.img, #real-large[data-cam="${c}"] > img, #real-large[data-cam="${c}"] > canvas.img, .cp-row[data-cam="${c}"] canvas.cp-live { filter: url(#vwb-${c}); }`).join('\n');
  document.head.appendChild(style);
  // the cameras look back at the person: their images (and the overlays drawn on them) are shown turned 180 deg
  const flip = document.createElement('style');
  flip.textContent = `body:not(.cam-noflip) .tile.real .frame > img, body:not(.cam-noflip) .tile.real .frame > canvas, body:not(.cam-noflip) #real-large > img, body:not(.cam-noflip) #real-large > canvas { transform: rotate(180deg); }`;
  document.head.appendChild(flip);
  if (local) document.body.classList.toggle('cam-noflip', !flipOn());
  const large = () => {                    // the enlarged view: which camera it shows, from its label ("Camera 2 · real")
    const big = document.getElementById('real-large');
    const lab = big && big.querySelector('.real-large-label');
    const m = lab && lab.textContent.match(/(\d)/);
    if (big && m) big.dataset.cam = 'CAM' + m[1];
  };
  setInterval(async () => {
    if (document.hidden) return;
    large();
    try {
      const g = local ? src.wb() : await (await fetch(BASE + '/api/wb', { cache: 'no-store' })).json();
      for (const [cam, v] of Object.entries(g)) {
        if (mats[cam] && v.length === 3) mats[cam].setAttribute('values', `${v[0]} 0 0 0 0 0 ${v[1]} 0 0 0 0 0 ${v[2]} 0 0 0 0 0 1 0`);
      }
    } catch { /* tracker offline: keep the last gains */ }
  }, 1000);
}

function mountStyle() {
  if (document.getElementById('calib-dock-css')) return;
  const style = document.createElement('style');
  style.id = 'calib-dock-css';
  style.textContent = css;
  document.head.appendChild(style);
}

const fmtMB = (b) => `${(b / 1e6).toFixed(b < 1e8 ? 1 : 0)} MB`;
const camNo = (cam) => cam.replace(/^CAM/, '');

function mountBrowser(src) {
  if (document.getElementById('calib-dock')) return;
  mountStyle();
  const dock = document.createElement('section');
  dock.id = 'calib-dock'; dock.className = 'local';
  dock.setAttribute('aria-label', 'Cameras and calibration');
  dock.innerHTML = `
    <div class="bar"><b>Cameras</b><span id="calib-note">starting…</span>
      <button type="button" id="calib-start" hidden title="Open the three tracking cameras in this page">Start cameras</button>
      <button type="button" id="calib-rec" title="Record all three cameras and every predicted tip to a ZIP file">Record</button>
      <button type="button" id="calib-toggle" aria-expanded="false">Calibrate…</button></div>
    <div id="calib-panel"><div class="cp">
      <h2>Cameras on this page</h2>
      <p>No server is running: this page opens the three tracking cameras and runs the tip tracker itself. <span id="cp-tracker"></span></p>
      <h3>Which camera is which</h3>
      <p>Each row is one camera place in the dome: what that camera shows now, and its reference view. If a row does not match, swap it with the one that does. Your choice is kept in this browser.</p>
      <div id="cp-rows"></div>
      <div class="cp-actions"><button type="button" id="cp-suggest">Match them by their views</button><span id="cp-match"></span></div>
      <label><input type="checkbox" id="cp-flip"> Show the camera images turned 180° (the cameras face you)</label>
      <h3>Camera calibration and the truth jig</h3>
      <p>Camera calibration, the truth jig and guided data collection need the camera server on this computer. From the Veridien repository run <code>veridien multicam preview</code>, then open this page with <code>?tracker=server</code>.</p>
    </div></div>`;
  document.body.appendChild(dock);
  const $ = (s) => dock.querySelector(s);
  const note = $('#calib-note'), startBtn = $('#calib-start'), rec = $('#calib-rec'), btn = $('#calib-toggle');
  const set = (open) => {
    dock.classList.toggle('open', open);
    btn.textContent = open ? 'Close' : 'Calibrate…';
    btn.setAttribute('aria-expanded', String(open));
    if (open) rows();
  };
  btn.addEventListener('click', () => set(!dock.classList.contains('open')));
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && dock.classList.contains('open')) set(false); });
  startBtn.addEventListener('click', () => { src.startCameras(); paint(); });

  // Record: a ZIP of <date>_browser_<time>/{frames.jsonl, tracks.jsonl, CAMk/NNNNN.jpg, meta.json}, downloaded on stop
  let recTimer = null;
  const stopRec = (why = null) => {
    clearInterval(recTimer); recTimer = null;
    rec.classList.remove('on'); rec.textContent = 'Record';
    const j = src.record.active() ? src.record.stop(why) : null;
    if (j?.stopped) note.textContent = `saved ${j.frames} frame sets (${fmtMB(j.bytes)}): ${j.stopped}`;
  };
  rec.addEventListener('click', () => {
    if (recTimer) return stopRec();
    const j = src.record.start('session');
    if (j.error) { note.textContent = `record: ${j.error}`; return; }
    rec.classList.add('on');
    const tick = () => {
      const i = src.record.info();
      if (!i) { stopRec(); return; }
      rec.textContent = `Stop ${Math.floor(i.s)} s · ${fmtMB(i.bytes)}`;
    };
    tick(); recTimer = setInterval(tick, 250);
  });
  src.on((what) => {
    if (what === 'record' && recTimer && !src.record.active()) { clearInterval(recTimer); recTimer = null; rec.classList.remove('on'); rec.textContent = 'Record'; note.textContent = 'recording saved (size cap or 5 min)'; }
    if (what === 'names' || what === 'status') { paint(); if (dock.classList.contains('open')) rows(); }
  });

  const paint = () => {
    startBtn.hidden = !!src.replay || !['idle', 'error', 'moved'].includes(src.status);
    if (!recTimer) note.textContent = browserNote(src);
    const w = src.workerInfo, t = src.timing?.();
    $('#cp-tracker').textContent = !w ? 'The tracker is loading.'
      : w.tracker === 'stub' ? `The tracker module did not load, so the views show without tips${w.note ? ` (${w.note})` : ''}.`
      : `Tracker: ${w.model ?? 'TipNet'}${(t?.ep ?? w.ep) ? ` on ${t?.ep ?? w.ep}` : ''}.`;
  };
  setInterval(() => { if (!document.hidden) paint(); }, 1000);
  paint();

  // which camera is which: live view | reference view | swap buttons
  const flip = $('#cp-flip');
  flip.checked = flipOn();
  flip.addEventListener('change', () => {
    try { localStorage.setItem(FLIP_KEY, flip.checked ? '1' : '0'); } catch { /* private window */ }
    document.body.classList.toggle('cam-noflip', !flip.checked);
  });
  $('#cp-suggest').disabled = !!src.replay;
  $('#cp-suggest').addEventListener('click', () => { $('#cp-match').textContent = 'looking…'; src.suggestNames(); });
  const cams = (src.calib?.cameras || []).map((c) => c.name).filter(Boolean);
  const drawRef = (c, cam) => {
    const r = src.refView(cam);
    if (!r) return;
    const im = new ImageData(r.w, r.h);
    for (let i = 0; i < r.gray.length; i++) { im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = r.gray[i]; im.data[4 * i + 3] = 255; }
    const tmp = new OffscreenCanvas(r.w, r.h); tmp.getContext('2d').putImageData(im, 0, 0);
    c.width = 448; c.height = 252;
    const g = c.getContext('2d');
    if (flipOn()) { g.translate(c.width, c.height); g.rotate(Math.PI); }
    g.drawImage(tmp, 0, 0, c.width, c.height);
  };
  const rows = () => {
    const box = $('#cp-rows');
    const names = src.names();
    box.innerHTML = '';
    for (const cam of cams) {
      const dev = names.find((n) => n.name === cam);
      const row = document.createElement('div');
      row.className = 'cp-row'; row.dataset.cam = cam;
      row.innerHTML = `<div><div class="cp-name">Camera ${camNo(cam)}</div><div class="cp-cap">${dev ? dev.tile : src.replay ? 'replay' : 'no camera'}</div></div>
        <div><canvas class="cp-live" width="448" height="252"></canvas><div class="cp-cap">now</div></div>
        <div><canvas class="cp-ref" width="448" height="252"></canvas><div class="cp-cap">reference view</div></div>
        <div class="cp-swap"></div>`;
      const sw = row.querySelector('.cp-swap');
      for (const other of cams.filter((c) => c !== cam)) {
        const b = document.createElement('button');
        b.type = 'button'; b.textContent = `Swap with camera ${camNo(other)}`;
        b.disabled = !names.length;
        b.addEventListener('click', () => src.swap(cam, other));
        sw.appendChild(b);
      }
      drawRef(row.querySelector('.cp-ref'), cam);
      box.appendChild(row);
    }
    $('#cp-match').textContent = src.replay ? 'a replay keeps the names it was recorded with' : src.matchNote ?? (src.namesFrom === 'saved' ? 'names remembered from last time' : src.namesFrom === 'order' ? 'names in the order the browser lists the cameras' : '');
  };
  setInterval(() => {                     // the live column, twice a second while the panel is open
    if (!dock.classList.contains('open') || document.hidden) return;
    for (const c of dock.querySelectorAll('.cp-row')) {
      const got = src.frames.get(c.dataset.cam);
      const cv = c.querySelector('.cp-live'), g = cv.getContext('2d');
      g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = '#0c0d0c'; g.fillRect(0, 0, cv.width, cv.height);
      if (!got?.bitmap) continue;
      if (flipOn()) { g.translate(cv.width, cv.height); g.rotate(Math.PI); }
      g.drawImage(got.bitmap, 0, 0, cv.width, cv.height);
    }
  }, 500);
}

/** The source main.js picked (twinpage/tracker_source.js), once it has. */
const sourceReady = () => new Promise((res) => {
  if (window.__twinSource) res(window.__twinSource);
  else window.addEventListener('twin-source', (e) => res(e.detail), { once: true });
});

const start = async () => {
  const siteDefault = document.querySelector('meta[name="veridien-tracker"]')?.content || null;
  if (trackerModeFromQuery(new URLSearchParams(location.search), siteDefault) === 'server') { mount(); mountWB(null); return; }
  const src = await sourceReady();
  if (src.kind === 'browser') mountBrowser(src); else mount();
  mountWB(src);
};
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
