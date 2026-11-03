// The in-page tracker's worker (module worker, started by twinpage/browser_source.js): the live path off the page's
// thread. It holds the newest frame of each camera (MediaStreamTrackProcessor readables, or ImageBitmaps posted by the
// page for the <video> fallback and for replays), takes a frame set once every camera has a new frame (or after 60 ms,
// as web/cameras/capture.js does), reads each 1280x720 frame's pixels once, and runs the tracker
// (web/twin/tracker/tracker.js: TipNet in onnxruntime-web + decode + RCM solve + TipFilter + SteadyTips) on them.
// Frames that arrive while a set is being tracked replace each other: the tracker only ever sees the newest.
// Per set it posts back the /api/tips payload (browser_core.finalizePayload), small ImageBitmaps for the camera views,
// display white-balance gains, and, while recording, the set's JPEGs (q 0.85, the camera page's encoder settings).
// With the WebGPU net (the default when its start-up self-test passes; ADR-150) the frames go straight to the GPU
// (tracker.trackFrames): no full-frame pixel read on the CPU. The first set of each kind of frame (VideoFrame / ImageBitmap,
// and the frame's pixel format) is also read through the canvas and its GPU resize compared with the JS one; a kind
// that differs keeps the canvas pixels + JS preprocess (the net stays on WebGPU), and the status says why.
//
// Tracker contract (web/twin/tracker/tracker.js; a stub stands in when it does not load):
//   new Tracker({rig: rig.json object, net}) with net = await createTipNet({model: url, ep}) from tracker/net.js
//     (or the module's own createTracker(opts) when it has one)
//   track(images {CAMk: ImageData, RGBA, full resolution}, t_s, {frameId}) -> Promise<{payload, instruments, timing}>
//   optional: framesOk(frames) / trackFrames(frames {CAMk: VideoFrame | ImageBitmap}, t_s, {frameId}) (the GPU path)
//     payload: the /api/tips shape; instruments: the rig-frame dict the server's tracks.jsonl keeps
//   reset() on a camera re-assignment or a replay loop; model for the status line

import { finalizePayload, wbGains, DisplayWB, areaGray, crc32, CAM_NAMES } from './browser_core.js';

let cfg = null, tracker = null, trackerKind = 'none', trackerNote = null, rig = null;
let slotNames = [];                         // camera name of each readable / slot
const latest = [], fresh = [];              // MSTP: the newest VideoFrame per slot
let pending = null;                         // posted sets (fallback, replay): only the newest waits
let busy = false, stopped = false, frameId = 0, rec = null, sigWanted = false;
let streamGen = 0;                          // bumped by 'streams' / 'streams-stop': older pumps and loops end
const canvases = {}, wb = new DisplayWB();
let wbCanvas = null;
const frameKinds = {};                      // frame kind -> 'gpu' | 'cpu' (the first set's canvas cross-check), + why
const stat = { sets: 0, t0: Date.now() / 1000, hz: 0, track_ms: [], total_ms: [] };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
/** One trip through the event loop (no timer clamp): messages already queued for this worker are handled first. */
const yieldTask = () => new Promise((r) => { const ch = new MessageChannel(); ch.port1.onmessage = () => { ch.port1.close(); r(); }; ch.port2.postMessage(0); });
const say = (m) => postMessage(m);

class StubTracker {
  constructor(r) { this.rig = r; this.model = 'none (tracker not loaded)'; this.ep = null; }
  track(images) {
    return { type: 'predictions', model: this.model, instruments: [], keypoints_px: {},
      image_size: Object.fromEntries(Object.entries(images).map(([k, im]) => [k, [im.width, im.height]])) };
  }
}

async function loadTracker() {
  try {
    rig = await (await fetch(cfg.rigUrl, { cache: 'no-cache' })).json();
  } catch (e) { say({ type: 'error', msg: `rig.json did not load: ${e.message || e}` }); }
  if (cfg.trackerUrl && cfg.tracker !== 'stub') {
    try {
      const mod = await import(cfg.trackerUrl);
      let t;
      if (typeof mod.createTracker === 'function') t = await mod.createTracker({ rig, modelUrl: cfg.modelUrl, ep: cfg.ep || null });
      else {
        if (typeof mod.Tracker !== 'function') throw new Error('tracker.js exports no Tracker');
        const nm = await import(new URL('./net.js', cfg.trackerUrl).href);
        const net = await nm.createTipNet({ model: cfg.modelUrl, ep: cfg.ep || null, graphCapture: !!cfg.graphCapture });
        t = new mod.Tracker({ rig, net });
        t.ep = net.ep; t.backend = net.backend ?? null; t.selftest = net.selftest ?? null;
      }
      tracker = t; trackerKind = 'tracker';
      return;
    } catch (e) { trackerNote = `the in-page tracker did not load (${e.message || e}); camera views only`; console.warn('[tracker worker]', trackerNote); }
  }
  tracker = new StubTracker(rig); trackerKind = 'stub';
}

function canvasFor(name, w, h) {
  let c = canvases[name];
  if (!c || c.width !== w || c.height !== h) c = canvases[name] = new OffscreenCanvas(w, h);
  return c;
}

const kindOf = (fr) => `${fr?.constructor?.name ?? 'frame'}${fr?.format ? ':' + fr.format : ''}`;

/** The canvas pixels of a frame (RGBA ImageData), drawn into the camera's canvas. */
function pixels(n, fr, w, h) {
  const c = canvasFor(n, w, h), g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(fr, 0, 0);
  return { c, im: g.getImageData(0, 0, w, h) };
}

/** One frame set {CAMk: VideoFrame | ImageBitmap}; closes every frame. -> whether a 'result' was posted. */
async function processSet(byName, { t_s, source = 'live', replay = null, capMs = {}, detail = false }) {
  const t0 = performance.now();
  const names = CAM_NAMES.filter((n) => byName[n]);
  const images = {}, views = {}, size = {}, grays = sigWanted ? {} : null;
  const nowS = Date.now() / 1000;
  const doWb = wb.due(nowS);
  // this set's recording, held: a 'record off' that arrives while its JPEGs encode must not lose it (or throw)
  const recSet = rec && t_s - rec.last >= 1 / rec.hz - 0.004 ? rec : null, recNow = !!recSet;
  const jpegs = [];
  if (!names.length) { for (const fr of Object.values(byName)) fr?.close?.(); return false; }
  const frames = Object.fromEntries(names.map((n) => [n, byName[n]]));
  // the GPU path: the frames themselves go to the GPU; the canvas only when something needs CPU pixels
  let gpu = cfg.gpuFrames !== false && typeof tracker?.trackFrames === 'function' && tracker.framesOk?.(frames);
  const kind = kindOf(frames[names[0]]);
  let check = null;
  if (gpu && !frameKinds[kind]) check = kind;                       // first set of this kind: cross-check it
  else if (gpu && frameKinds[kind].mode === 'cpu') gpu = false;
  let raw = null, trackMs = 0, t1 = performance.now();
  const id = frameId++;
  try {
    for (const n of names) {
      const fr = byName[n];
      const w = fr.displayWidth ?? fr.width, h = fr.displayHeight ?? fr.height;
      let c = null;
      if (!gpu || check || recNow || grays) {
        const p = pixels(n, fr, w, h); c = p.c;
        if (!gpu || check || grays) images[n] = p.im;
      }
      size[n] = [w, h];
      const vw = Math.min(w, cfg.viewW), vh = Math.round(vw * h / w);
      views[n] = await createImageBitmap(c ?? fr, { resizeWidth: vw, resizeHeight: vh, resizeQuality: 'medium' });
      if (doWb) {
        wbCanvas ??= new OffscreenCanvas(160, 90);
        const wg = wbCanvas.getContext('2d', { willReadFrequently: true });
        wg.drawImage(c ?? fr, 0, 0, 160, 90);
        wb.update(n, wbGains(wg.getImageData(0, 0, 160, 90).data), nowS);
      }
      if (grays && w % 160 === 0 && h === w * 9 / 16) grays[n] = areaGray(images[n].data, w, h, w / 160);
      if (recNow) jpegs.push(c.convertToBlob({ type: 'image/jpeg', quality: recSet.quality }).then(async (b) => [n, new Uint8Array(await b.arrayBuffer())]));
    }
    if (check) {
      try {
        const r = await tracker.net.gpu.checkFrames(names.map((n) => frames[n]), names.map((n) => images[n]));
        frameKinds[kind] = r.ok ? { mode: 'gpu' } : { mode: 'cpu', why: `GPU copy of ${kind} frames differs from the canvas pixels (${names.filter((_, i) => !r.slots[i]).join(', ')}); JS preprocess for them` };
      } catch (e) { frameKinds[kind] = { mode: 'cpu', why: `frame check failed (${e.message || e}); JS preprocess` }; }
      if (frameKinds[kind].why) { console.warn('[tracker worker]', frameKinds[kind].why); trackerNote = frameKinds[kind].why; }
      if (frameKinds[kind].mode === 'cpu') gpu = false;
    }
    t1 = performance.now();
    try {
      const r = gpu ? await tracker.trackFrames(frames, t_s, { frameId: id }) : await tracker.track(images, t_s, { frameId: id });
      raw = r?.payload ? { ...r.payload, instruments_rig: r.instruments ?? {} } : r;
      if (detail && r?.res) detail = { res: Object.fromEntries(Object.entries(r.res).map(([p, pt]) => [p, pt ? pt.tip : null])), steady: r.steady ?? null, path: gpu ? 'gpu' : 'cpu',
        cands: Object.fromEntries(Object.entries(r.cands ?? {}).map(([n, d]) => [n, Object.fromEntries(Object.entries(d).map(([ch, c]) => [ch, c.length]))])), net: tracker.net?.gpu?.timing ? { ...tracker.net.gpu.timing } : null };
    } catch (e) { say({ type: 'error', msg: `tracker: ${e.message || e}` }); }
    trackMs = performance.now() - t1;
  } finally { for (const fr of Object.values(byName)) fr?.close?.(); }
  const encoded = Promise.all(jpegs);
  const payload = finalizePayload(raw, { frame_id: id, t: t_s, source, replay, now: Date.now() / 1000, model: tracker?.model ?? null,
    image_size: size });
  stat.sets++; stat.track_ms.push(trackMs); if (stat.track_ms.length > 60) stat.track_ms.shift();
  const totalMs = performance.now() - t0;
  stat.total_ms.push(totalMs); if (stat.total_ms.length > 60) stat.total_ms.shift();
  const now = Date.now() / 1000;
  if (now - stat.t0 >= 1) { stat.hz = stat.sets / (now - stat.t0); stat.sets = 0; stat.t0 = now; }
  const med = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : null);
  say({ type: 'result', payload, views, size, wb: wb.gains(), detail: detail || undefined,
    timing: { hz: stat.hz, track_ms: med(stat.track_ms), total_ms: med(stat.total_ms), cameras: names.length, tracker: trackerKind,
      ep: tracker?.ep ?? null, frames: gpu ? 'gpu' : 'cpu', model: tracker?.model ?? null, note: trackerNote, stages_ms: tracker?.timing ?? null,
      set_ms: { track: trackMs, total: totalMs } } }, Object.values(views));
  if (grays && Object.keys(grays).length) { sigWanted = false; say({ type: 'sig', grays }, Object.values(grays).map((g) => g.buffer)); }
  if (recNow) {
    recSet.last = t_s;
    const files = (await encoded).map(([n, bytes]) => ({ cam: n, bytes, crc: crc32(bytes) }));
    const index = recSet.index++;
    // t_ms: the capture time on this worker's performance clock (the server keeps the page's; only differences matter)
    say({ type: 'rec', index, wallMs: t_s * 1000, tMs: Math.round((t_s * 1000 - performance.timeOrigin) * 10) / 10,
      cameras: names, capMs, ms: Math.round(trackMs), payload, files }, files.map((f) => f.bytes.buffer));
  }
  return true;
}

// ------------------------------------------------------------------ MSTP: the newest frame per camera, in this worker

const readers = [];

async function pump(readable, k, gen) {
  const reader = readable.getReader();
  readers.push(reader);
  try {
    while (!stopped && gen === streamGen) {
      const { value, done } = await reader.read();
      if (done) return;
      if (gen !== streamGen) { value.close(); return; }
      if (latest[k]) latest[k].close();
      latest[k] = value; fresh[k] = true;
    }
  } catch { /* the track ended or the reader was cancelled */ }
}

/** Drop the cameras' readables and their newest frames (the page stopped its cameras, or the worker stops). */
function endStreams() {
  streamGen++;
  for (const r of readers.splice(0)) r.cancel().catch(() => {});
  latest.forEach((f) => f?.close()); latest.length = 0; fresh.length = 0;
}

async function streamLoop(gen) {
  while (!stopped && gen === streamGen) {
    const m = slotNames.length, t_wait = Date.now();
    while (!stopped && gen === streamGen && Date.now() - t_wait < 60 && fresh.filter(Boolean).length < m) await sleep(2);
    if (stopped || gen !== streamGen) return;
    if (latest.filter(Boolean).length < 2) { await sleep(20); continue; }
    const t_s = Date.now() / 1000;
    const byName = {};
    latest.forEach((f, k) => { if (f && slotNames[k]) { byName[slotNames[k]] = f; latest[k] = null; } });
    fresh.fill(false);
    busy = true;
    try { await processSet(byName, { t_s, source: 'live' }); } catch (e) { say({ type: 'error', msg: String(e.message || e) }); } finally { busy = false; }
  }
}

// ------------------------------------------------------------------ posted sets (the <video> fallback, replays)

async function drain() {
  if (busy) return;
  busy = true;
  try {
    while (pending && !stopped) {
      await yieldTask();                          // let queued messages in first: a newer set replaces this one
      if (!pending || stopped) break;
      const s = pending; pending = null;
      let posted = false;
      try { posted = await processSet(s.byName, s); } catch (e) { say({ type: 'error', msg: String(e.message || e) }); }
      // the page waits for one answer per posted set (browser_source.js inflight): a set that gave no result says so
      if (!posted) say({ type: 'skip' });
    }
  } finally { busy = false; }
}

// ------------------------------------------------------------------ a synthetic source (browser checks: no camera opened)

async function synthLoop(hz) {
  const W = 1280, H = 720;
  const cs = Object.fromEntries(CAM_NAMES.map((n) => [n, new OffscreenCanvas(W, H)]));
  const t0 = Date.now() / 1000;
  while (!stopped) {
    const t = Date.now() / 1000, byName = {};
    CAM_NAMES.forEach((n, k) => {
      const g = cs[n].getContext('2d');
      const grd = g.createRadialGradient(W / 2, H / 2, 40, W / 2, H / 2, W * 0.7);
      grd.addColorStop(0, `rgb(${236 - 8 * k}, ${232 - 4 * k}, 222)`); grd.addColorStop(1, 'rgb(70, 66, 60)');
      g.fillStyle = grd; g.fillRect(0, 0, W, H);
      const a = (t - t0) * 0.7 + k * 2.1;
      for (const [s, col] of [[1, '#1b1c1b'], [-1, '#262725']]) {
        const x0 = W / 2 + s * 520, y0 = 80, x1 = W / 2 + s * 120 + 140 * Math.cos(a), y1 = H / 2 + 90 * Math.sin(a * 1.3);
        g.strokeStyle = col; g.lineWidth = 26; g.lineCap = 'round'; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      }
      g.fillStyle = 'rgba(17,18,17,0.65)'; g.font = '600 34px ui-sans-serif, system-ui, sans-serif';
      g.fillText(`${n} synthetic`, 36, H - 40);
      byName[n] = cs[n].transferToImageBitmap ? cs[n].transferToImageBitmap() : null;
    });
    if (pending) for (const b of Object.values(pending.byName)) b?.close?.();
    pending = { byName, t_s: t, source: 'replay', replay: 'synthetic' };
    drain();
    await sleep(1000 / hz);
  }
}

onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    cfg = { viewW: 640, ...m };
    slotNames = m.slots ?? [];
    await loadTracker();
    say({ type: 'ready', tracker: trackerKind, note: trackerNote, ep: tracker?.ep ?? null, model: tracker?.model ?? null,
      backend: tracker?.backend ?? null, selftest: tracker?.selftest ?? null });
  } else if (m.type === 'streams') {             // (again after a restart of the cameras: one loop, fresh state)
    endStreams();
    const gen = streamGen;
    slotNames = m.slots;
    tracker?.reset?.();
    m.readables.forEach((rs, k) => pump(rs, k, gen));
    streamLoop(gen);
  } else if (m.type === 'streams-stop') {
    endStreams();
  } else if (m.type === 'set') {
    // a set that replaces a waiting one is answered as skipped: the page counts one answer per posted set
    if (pending) { for (const b of Object.values(pending.byName)) b?.close?.(); say({ type: 'skip' }); }
    pending = m;
    drain();
  } else if (m.type === 'synthetic') {
    synthLoop(m.hz ?? 15);
  } else if (m.type === 'names') {
    slotNames = m.slots;
    tracker?.reset?.();
  } else if (m.type === 'record') {
    rec = m.on ? { hz: m.hz ?? 10, quality: m.quality ?? 0.85, last: -1e9, index: 0 } : null;
  } else if (m.type === 'signatures') {
    sigWanted = true;
  } else if (m.type === 'stop') {
    stopped = true;
    endStreams();
    if (pending) for (const b of Object.values(pending.byName)) b?.close?.();
    pending = null;
    // free the net (WebGPU: session, buffers, device) once no set is in flight; terminate() would also free it
    for (let i = 0; i < 100 && busy; i++) await sleep(10);
    try { tracker?.net?.release?.(); } catch { /* */ }
  }
};
