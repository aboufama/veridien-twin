// The in-page tracker source (twinpage/tracker_source.js 'browser' mode): no server. This page opens the dome's three
// tracking cameras ("Global Shutter Camera", 1280x720 at 30 fps, which over USB 2 means MJPEG), hands their tracks to
// tracker_worker.js (MediaStreamTrackProcessor; a <video> + requestVideoFrameCallback fallback for browsers without it)
// and feeds the page from the worker's results: the tips payload into a LocalFeed (rig_link.js consumers), camera views
// into a FramesRing (real_cams.js), display white balance (calib_embed.js), and recordings as a downloaded ZIP in the
// server recorder's layout (data/multicam/<session>/{frames.jsonl, tracks.jsonl, CAMk/NNNNN.jpg}).
//
// Which camera is which: browser device ids are per site, so the hosted page cannot reuse the server's names. A name
// chosen here is remembered per device id (localStorage); without one the cameras are suggested from their views
// (browser_core.matchViews against rig.json's reference views) and otherwise taken in the order the browser lists them.
// The user checks and swaps them under Calibrate... (calib_embed.js).
//
//   ?replay=<session url>   replay a recorded session (frames.jsonl + CAMk/NNNNN.jpg, served with CORS); no camera opens
//   ?replay=synthetic       a drawn test pattern; no camera opens
//   ?ep=wasm|webgpu         onnxruntime-web's backend for TipNet. Default: WebGPU end to end when its start-up self-test
//                           is bit-identical with the live Python chain, else WASM (ADR-150); wasm forces WASM
//   ?trackerModule=<url>    the tracker module (default ../tracker/tracker.js); ?trackerModule=stub: views only
//   ?model=<url>            the ONNX model (default ../tracker/tipnet.onnx)
//   ?recHz=10 ?recMB=250    recording rate (sets/s) and size cap (held in this tab until the download: 250 MB keeps a
//                           16 GB Mac with a live tracker comfortable; ~25 s of three 720p cameras at 10 sets/s, q 0.85)

import { LocalFeed, FramesRing, LOCAL_BASE, calibrationFromRig } from './tracker_source.js';
import { ZipStore, assignNames, swapNames, refSignatures, viewSignature, matchViews, recordFrameLine, recordTrackLine,
  sessionName, CAM_NAMES, b64bytes } from './browser_core.js';

const TRACKING = /global shutter/i;
const NAMES_KEY = 'veridien.twin.camNames';
const PAGE_ID = 'twin-' + Math.random().toString(36).slice(2, 8);
const pad5 = (i) => String(i).padStart(5, '0');
const tileId = (deviceId) => 'cam-' + String(deviceId).replace(/[^a-z0-9]/gi, '').slice(0, 6).toLowerCase();
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k) ?? 'null'); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private window */ } },
};

export class BrowserSource {
  constructor({ q = new URLSearchParams(location.search) } = {}) {
    this.kind = 'browser'; this.base = LOCAL_BASE; this.q = q;
    this.feed = new LocalFeed(LOCAL_BASE); this.frames = new FramesRing(8);
    this.rig = null; this.calib = null; this.worker = null; this.workerInfo = null;
    this.status = 'idle'; this.error = null; this.devices = []; this.slots = []; this.tracks = []; this.videos = [];
    this.gains = {}; this.timingInfo = null; this.namesFrom = null; this.listeners = new Set(); this.outstanding = 0;
    this.replay = q.get('replay') || null;
    const u = (p) => new URL(p, import.meta.url).href;
    this.urls = {
      rig: q.get('rigJson') || u('../tracker/rig.json'),
      model: q.get('model') || u('../tracker/tipnet.onnx'),
      tracker: q.get('trackerModule') === 'stub' ? null : (q.get('trackerModule') || u('../tracker/tracker.js')),
      worker: u('./tracker_worker.js'),
    };
    this.recCfg = { hz: Number(q.get('recHz')) || 10, capBytes: (Number(q.get('recMB')) || 250) * 1e6, maxS: 5 * 60, quality: 0.85 };
    this.rec = null;
    this.bc = typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel('veridien-twin-cameras') : null;
    this.bc?.addEventListener('message', (e) => { if (e.data?.hello && e.data.hello !== PAGE_ID && this.tracks.length) this._moved(); });
    // leaving the page for good: the worker drops its frames and stops (a bfcache'd page keeps it for its return)
    addEventListener('pagehide', (e) => { if (!e.persisted) this.close(); });
  }

  /** Stop everything: cameras, recording (downloaded), the worker. */
  close() {
    if (this.rec) this.record.stop('page closed');
    this.stopCameras(false);
    this.worker?.postMessage({ type: 'stop' });
    this.worker?.terminate();
    this.worker = null;
  }

  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  _emit(what) { for (const fn of this.listeners) try { fn(what, this); } catch (e) { console.warn('[twin] source listener', e); } }
  _set(status, error = null) { this.status = status; this.error = error; this._emit('status'); }

  async init() {
    try { this.rig = await (await fetch(this.urls.rig, { cache: 'no-cache' })).json(); } catch (e) { console.warn('[twin] rig.json', e); }
    this.calib = calibrationFromRig(this.rig);
    this.worker = new Worker(this.urls.worker, { type: 'module', name: 'veridien-tracker' });
    this.worker.onmessage = (e) => this._onWorker(e.data);
    this.worker.onerror = (e) => { console.error('[twin] tracker worker', e.message || e); this._set('error', `tracker worker: ${e.message || 'failed'}`); };
    const ready = new Promise((res) => { this._ready = res; });
    this.worker.postMessage({ type: 'init', rigUrl: this.urls.rig, modelUrl: this.urls.model, trackerUrl: this.urls.tracker,
      tracker: this.urls.tracker ? 'auto' : 'stub', ep: this.q.get('ep') || null, viewW: Number(this.q.get('viewW')) || 640, slots: [] });
    // the page boots on; the cameras wait for the tracker to load (the model and onnxruntime-web download once)
    this.readyP = ready;
    if (this.replay) this.readyP.then(() => (this.replay === 'synthetic' ? this.startSynthetic() : this.startReplay(this.replay)));
    else this._autoStart();
    return this;
  }

  /** Open the cameras by themselves only when the site already has camera permission (else a click starts them). */
  async _autoStart() {
    try {
      const p = await navigator.permissions?.query({ name: 'camera' });
      if (p?.state === 'granted' && this.q.get('autostart') !== '0') this.startCameras();
    } catch { /* Safari / Firefox: wait for the click */ }
  }

  calibration() { return Promise.resolve(this.calib); }
  wb() { return this.gains; }
  timing() { return this.timingInfo; }
  info() {
    return { mode: 'browser', status: this.status, error: this.error, replay: this.replay, tracker: this.workerInfo, timing: this.timingInfo,
      devices: this.devices.map((d, i) => ({ tile: tileId(d.deviceId), label: d.label, name: this.slots[i] ?? null })), namesFrom: this.namesFrom,
      frames: this.frames.cams(), recording: this.rec ? { sets: this.rec.sets, bytes: this.rec.zip.bytes } : null, feed: this.feed.info() };
  }

  _onWorker(m) {
    if (m.type === 'ready') { this.workerInfo = m; this._ready?.(); this._emit('ready'); return; }
    if (m.type === 'error') { console.warn('[twin] tracker:', m.msg); this.lastError = m.msg; return; }
    if (m.type === 'sig') { this._onSignatures(m.grays); return; }
    if (m.type === 'rec') { this._onRec(m); return; }
    // every posted set gets exactly one answer: a result, or a skip (no named camera, or a newer set replaced it)
    if (m.type === 'skip') { this._answered(); return; }
    if (m.type !== 'result') return;
    this._answered();
    this.gains = m.wb || this.gains;
    this.timingInfo = m.timing;
    this.frames.put(m.payload.frame_id, m.views, m.size);
    this.feed.push(m.payload);
    if (!this._named && this.status === 'running') { this._named = true; this._suggestIfUnnamed(); }
  }

  /** Posted sets not answered yet (live fallback: at most 1; replay: at most 2, so one loads while one is tracked). */
  get inflight() { return this.outstanding > 0; }
  _answered() { this.outstanding = Math.max(0, this.outstanding - 1); }
  _post(msg) { this.outstanding++; this.worker.postMessage(msg, Object.values(msg.byName)); }

  // ---------------------------------------------------------------- cameras

  async startCameras() {
    if (this.replay || this.status === 'starting' || this.status === 'running') return;
    await this.readyP;
    this._set('starting');
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('this browser cannot open cameras (getUserMedia)');
      let list = await navigator.mediaDevices.enumerateDevices();
      if (!list.some((d) => d.kind === 'videoinput' && d.label)) {          // labels need permission once
        const s = await navigator.mediaDevices.getUserMedia({ video: true });
        s.getTracks().forEach((t) => t.stop());
        list = await navigator.mediaDevices.enumerateDevices();
      }
      const devices = list.filter((d) => d.kind === 'videoinput' && (this.q.get('allCams') === '1' || TRACKING.test(d.label))).slice(0, 3);
      if (devices.length < 2) throw new Error(`found ${devices.length} tracking camera(s) ("Global Shutter Camera"); the 3-D tips need 2 or more`);
      const half = this.q.get('res') === '360';
      const want = (d) => ({ deviceId: { exact: d.deviceId }, width: { ideal: half ? 640 : 1280 }, height: { ideal: half ? 360 : 720 } });
      const streams = await Promise.all(devices.map(async (d) => {
        // >= 25 fps at 720p is only possible in MJPEG over USB 2: this keeps the browser off 5 fps YUY2
        try { return await navigator.mediaDevices.getUserMedia({ video: { ...want(d), frameRate: { min: 25, ideal: 30 } } }); }
        catch { return navigator.mediaDevices.getUserMedia({ video: { ...want(d), frameRate: { ideal: 30 } } }); }
      }));
      this.devices = devices;
      this.tracks = streams.map((s) => s.getVideoTracks()[0]);
      const saved = store.get(NAMES_KEY) || {};
      this.slots = assignNames(devices.map((d) => d.deviceId), saved);
      this.namesFrom = devices.every((d) => CAM_NAMES.includes(saved[d.deviceId])) ? 'saved' : 'order';
      this._named = this.namesFrom === 'saved';
      this.bc?.postMessage({ hello: PAGE_ID, at: Date.now() });
      if (typeof MediaStreamTrackProcessor !== 'undefined' && this.q.get('mstp') !== '0') {
        const readables = this.tracks.map((t) => new MediaStreamTrackProcessor({ track: t }).readable);
        this.worker.postMessage({ type: 'streams', readables, slots: this.slots }, readables);
        this.capture = 'MediaStreamTrackProcessor';
      } else {
        this._videoLoop(streams);
        this.capture = 'video';
      }
      this.tracks.forEach((t) => t.addEventListener('ended', () => { if (this.status === 'running') this._set('error', 'a camera was unplugged'); }));
      this._set('running');
    } catch (e) {
      this.stopCameras(false);
      this._set('error', e?.name === 'NotAllowedError' ? 'camera permission was not given' : (e?.message || String(e)));
    }
  }

  /** The fallback without MediaStreamTrackProcessor: videos (kept on screen, tiny) and requestVideoFrameCallback. */
  _videoLoop(streams) {
    const box = this.videoBox = document.createElement('div');
    box.style.cssText = 'position:fixed;left:0;bottom:0;width:48px;height:27px;overflow:hidden;pointer-events:none;z-index:1;opacity:1';
    box.setAttribute('aria-hidden', 'true');
    document.body.appendChild(box);
    this.videos = streams.map((s) => {
      const v = document.createElement('video');
      v.muted = true; v.playsInline = true; v.autoplay = true; v.srcObject = s; v.style.cssText = 'width:16px;height:9px';
      box.appendChild(v); v.play().catch(() => {});
      v.newFrames = 0;
      const tick = () => { v.newFrames++; if (this.videos.includes(v)) v.requestVideoFrameCallback?.(tick); };
      if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(tick); else v.addEventListener('timeupdate', tick);
      return v;
    });
    const videos = this.videos;
    let sentAt = 0;
    const loop = async () => {
      if (this.videos !== videos || !videos.length || !this.worker) return;           // stopped (or restarted)
      try {
        const t0 = performance.now();
        while (performance.now() - t0 < 60 && videos.some((v) => !v.newFrames)) await new Promise((r) => setTimeout(r, 4));
        // a set unanswered for 3 s (the worker lost it) no longer holds the loop
        if (this.inflight && performance.now() - sentAt > 3000) this.outstanding = 0;
        if (!this.inflight && videos.filter((v) => v.videoWidth).length >= 2) {
          videos.forEach((v) => { v.newFrames = 0; });
          const t_s = Date.now() / 1000, byName = {};
          try {
            await Promise.all(videos.map(async (v, k) => { if (v.videoWidth && this.slots[k]) byName[this.slots[k]] = await createImageBitmap(v); }));
          } catch (e) { Object.values(byName).forEach((b) => b.close?.()); throw e; }
          if (Object.keys(byName).length && this.videos === videos) {
            sentAt = performance.now();
            this._post({ type: 'set', byName, t_s, source: 'live' });
          } else Object.values(byName).forEach((b) => b.close?.());
        }
      } catch (e) { console.warn('[twin] camera frame', e?.message || e); }
      setTimeout(loop, 4);
    };
    loop();
  }

  stopCameras(emit = true) {
    if (this.capture === 'MediaStreamTrackProcessor' && this.tracks.length) this.worker?.postMessage({ type: 'streams-stop' });
    this.tracks.forEach((t) => t.stop());
    this.tracks = []; this.videos = []; this.videoBox?.remove(); this.videoBox = null; this.outstanding = 0;
    this.frames.clear();
    if (emit) this._set('idle');
  }

  _moved() {
    if (this.rec) this.record.stop();
    this.stopCameras(false);
    this._set('moved', 'the cameras moved to a newer tab of this page');
  }

  // ---------------------------------------------------------------- replay (no camera)

  startSynthetic() { this._set('replay'); this.namesFrom = 'synthetic'; this.worker.postMessage({ type: 'synthetic', hz: Number(this.q.get('replayHz')) || 15 }); }

  async startReplay(url) {
    const base = String(url).replace(/\/+$/, '');
    const name = this.replayName = base.split('/').pop();
    this._set('replay');
    this.namesFrom = 'replay';
    let lines;
    try {
      const txt = await (await fetch(`${base}/frames.jsonl`, { cache: 'no-cache' })).text();
      lines = txt.split('\n').filter(Boolean).map((l) => JSON.parse(l));
    } catch (e) { this._set('error', `replay: ${base}/frames.jsonl did not load (${e.message || e})`); return; }
    const from = Number(this.q.get('replayFrom')) || 0, to = Number(this.q.get('replayTo')) || lines.length;
    lines = lines.filter((l) => l.index >= from && l.index < to);
    if (!lines.length) { this._set('error', 'replay: no frame sets'); return; }
    const speed = Number(this.q.get('replaySpeed')) || 1;
    const load = async (l) => {
      const byName = {};
      await Promise.all((l.cameras || CAM_NAMES).map(async (cam) => {
        const r = await fetch(`${base}/${cam}/${pad5(l.index)}.jpg`);
        if (r.ok) byName[cam] = await createImageBitmap(await r.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
      }));
      return byName;
    };
    // played in real time: a set's t is its recorded capture time moved to now (the filters see the recorded spacing);
    // sets the tracker cannot keep up with are skipped, as the live path drops them (only the newest is tracked)
    const closeAll = (b) => Object.values(b).forEach((x) => x.close?.());
    for (;;) {
      const t0 = lines[0].t_ms, wall0 = performance.now(), unix0 = Date.now() / 1000;
      for (let i = 0; i < lines.length; i++) {
        if (this.status !== 'replay') return;
        const due = wall0 + (lines[i].t_ms - t0) / speed;
        if (performance.now() - due > 150 && i + 1 < lines.length) continue;            // late: the next one
        // one set tracked and one waiting at most: a slower tracker skips sets (as live does) instead of queueing them
        // in the worker (a queue there grew without bound and starved its other messages: record, signatures)
        if (this.outstanding >= 2 && i + 1 < lines.length) {
          const tw = performance.now();
          while (this.outstanding >= 2 && this.status === 'replay' && performance.now() - tw < 3000) await new Promise((r) => setTimeout(r, 5));
          if (this.outstanding >= 2) this.outstanding = 1;                // an answer lost: do not stall the replay
          if (performance.now() - due > 150) continue;
        }
        const byName = await load(lines[i]);
        const wait = due - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        if (this.status !== 'replay') { closeAll(byName); return; }
        const t_s = unix0 + (lines[i].t_ms - t0) / 1000 / speed;
        this._post({ type: 'set', byName, t_s, source: 'replay', replay: name });
      }
      if (this.q.get('replayLoop') === '0') break;
      await new Promise((r) => setTimeout(r, 1500));                 // a gap the filters see as a restart
      this.worker.postMessage({ type: 'names', slots: [] });         // tracker.reset()
    }
    this._set('idle');
  }

  // ---------------------------------------------------------------- which camera is which

  /** The cameras' names now: [{tile, label, name}] (the order the browser lists them). */
  names() { return this.devices.map((d, i) => ({ tile: tileId(d.deviceId), deviceId: d.deviceId, label: d.label, name: this.slots[i] ?? null })); }

  setSlots(slots, from = 'manual') {
    this.slots = [...slots];
    this.namesFrom = from;
    if (from === 'manual') {
      const saved = store.get(NAMES_KEY) || {};
      this.devices.forEach((d, i) => { if (this.slots[i]) saved[d.deviceId] = this.slots[i]; });
      store.set(NAMES_KEY, saved);
    }
    this.frames.clear();
    this.worker.postMessage({ type: 'names', slots: this.slots });
    this._emit('names');
  }

  /** Swap two cameras' names (CAM names), remembered for these devices. */
  swap(a, b) { if (a !== b) this.setSlots(swapNames(this.slots, a, b)); }

  /** The reference view of a camera (rig.json view_refs, newest): {w, h, gray: Uint8Array} or null. */
  refView(cam) {
    const list = this.rig?.view_refs?.[cam];
    const r = list?.[list.length - 1];
    return r ? { w: r.w, h: r.h, gray: b64bytes(r.gray_u8_b64), source: r.source } : null;
  }

  /** Ask the worker for the next set's 160x90 grays; names follow when the views match the references clearly. */
  suggestNames() { this._suggest = { manual: true }; this.worker.postMessage({ type: 'signatures' }); }
  _suggestIfUnnamed() { this._suggest = { manual: false }; this.worker.postMessage({ type: 'signatures' }); }

  _onSignatures(grays) {
    const refs = this._refs ??= refSignatures(this.rig?.view_refs, { minStd: this.rig?.view_match?.min_std ?? 4 });
    const sigs = {};
    for (const [cam, g] of Object.entries(grays)) {
      const i = this.slots.indexOf(cam);
      if (i >= 0) sigs[String(i)] = viewSignature(g, 160, 90);
    }
    const vm = this.rig?.view_match || {};
    const r = matchViews(sigs, refs, { minNcc: vm.min_ncc ?? 0.4, minMargin: vm.min_margin ?? 0.12 });
    const got = Object.entries(r.names);
    const manual = !!this._suggest?.manual;
    this.lastMatch = { ...r, at: Date.now() };
    if (got.length === this.devices.length && got.length) {
      const slots = this.devices.map((_, i) => r.names[String(i)]);
      const changed = slots.some((n, i) => n !== this.slots[i]);
      if (changed) this.setSlots(slots, manual ? 'manual' : 'views');
      else if (manual) this.setSlots(slots, 'manual');
      else this.namesFrom = 'views';
      this.matchNote = changed ? 'names set from the camera views' : 'the camera views agree with these names';
    } else {
      this.matchNote = `the views named ${got.length} of ${this.devices.length} cameras clearly; check them by eye and swap`;
    }
    this._emit('names');
  }

  // ---------------------------------------------------------------- recording: a ZIP downloaded by this page

  get record() {
    const self = this;
    return {
      active: () => !!self.rec,
      start(tag = 'session') {
        if (self.rec) return { recording: true };
        if (!(self.status === 'running' || self.status === 'replay')) return { error: 'no cameras running' };
        const t0 = Date.now();
        self.rec = { zip: new ZipStore({ date: new Date(t0) }), name: sessionName(t0, tag === 'session' ? 'browser' : tag), t0, sets: 0,
          frames: [], tracks: [], cameras: new Set(), stopped: false };
        self.rec.timer = setTimeout(() => self.record.stop('5 min'), self.recCfg.maxS * 1000);
        self.worker.postMessage({ type: 'record', on: true, hz: self.recCfg.hz, quality: self.recCfg.quality });
        self._emit('record');
        return { recording: self.rec.name };
      },
      stop(why = null) {
        const r = self.rec;
        if (!r) return { stopped: null };
        self.rec = null;
        clearTimeout(r.timer);
        self.worker.postMessage({ type: 'record', on: false });
        const tiles = Object.fromEntries(self.devices.map((d, i) => [tileId(d.deviceId), self.slots[i]]));
        r.zip.addText(`${r.name}/frames.jsonl`, r.frames.join('\n') + (r.frames.length ? '\n' : ''));
        r.zip.addText(`${r.name}/tracks.jsonl`, r.tracks.join('\n') + (r.tracks.length ? '\n' : ''));
        r.zip.addText(`${r.name}/meta.json`, JSON.stringify({ recorded_by: 'twin page (browser tracker)', session: r.name, started: new Date(r.t0).toISOString(),
          sets: r.sets, hz: self.recCfg.hz, jpeg_quality: self.recCfg.quality, tiles, names_from: self.namesFrom, tracker: self.workerInfo,
          timing: self.timingInfo, replay: self.replay, user_agent: navigator.userAgent, stopped_by: why }, null, 1) + '\n');
        const blob = r.zip.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `${r.name}.zip`; document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 120000);
        self._emit('record');
        return { stopped: `${r.name}.zip`, frames: r.sets, bytes: blob.size, why };
      },
      info: () => (self.rec ? { name: self.rec.name, sets: self.rec.sets, bytes: self.rec.zip.bytes, s: (Date.now() - self.rec.t0) / 1000, capBytes: self.recCfg.capBytes } : null),
    };
  }

  _onRec(m) {
    const r = this.rec;
    if (!r) return;
    const idx = r.sets;
    for (const f of m.files) r.zip.add(`${r.name}/${f.cam}/${pad5(idx)}.jpg`, f.bytes, f.crc);
    const tiles = Object.fromEntries(this.devices.map((d, i) => [tileId(d.deviceId), this.slots[i]]));
    r.frames.push(recordFrameLine({ index: idx, wallMs: m.wallMs, tMs: m.tMs, cameras: m.files.map((f) => f.cam), tiles, capMs: m.capMs }));
    r.tracks.push(recordTrackLine({ index: idx, wallMs: m.wallMs, tMs: m.tMs, cameras: m.files.map((f) => f.cam), ms: m.ms, payload: m.payload }));
    r.sets++;
    if (r.zip.bytes >= this.recCfg.capBytes) this.record.stop(`${Math.round(this.recCfg.capBytes / 1e6)} MB cap`);
  }
}
