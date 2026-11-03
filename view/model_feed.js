// The model socket path of the one-view page (docs/WEB_TWIN.md §7, §14, §18.1): CAM1-CAM3 render offscreen at 20 fps,
// whole bundles only, through readback -> camera JPEG (ui/feeds.js workers) -> model/socket.js, and ONLY while a model is
// attached (the server's status says so). Latest-only: a tick that finds the previous bundle still in flight is
// skipped and counted. Model-vs-truth markers (M) draw from the newest score.
import * as K from '../control/kinematics.js';
import { TwinSocket } from '../model/socket.js';
import { Feeds } from '../ui/feeds.js';
import { qualityTables } from '../ui/jpeg_encoder.js';

const TRACKING = ['CAM1', 'CAM2', 'CAM3'];
const PERIOD_MS = 50;

export class ModelFeed extends EventTarget {
  constructor(V) {
    super();
    this.V = V;
    this.attached = false;
    this.everAttached = false;
    this.socketState = 'offline';
    this.busy = false;
    this.frameId = 0;
    this.stats = { bundles: 0, skipped: 0, ms: [], sent: 0, times: [] };
    this.scores = [];
    this.posesDirty = false;
    this.feeds = null;
    const scene = V.scene;
    this.socket = new TwinSocket({ hello: () => ({ calibration: this.calibration(), scene_ref: { renderer: `webgpu-${V.kind}`, page: 'view' } }) });
    this.socket.addEventListener('state', () => { this.socketState = this.socket.state; this.dispatchEvent(new Event('change')); });
    this.socket.addEventListener('status', (e) => this.onStatus(e.detail));
    this.socket.addEventListener('score', (e) => { this.scores.push(e.detail); if (this.scores.length > 60) this.scores.shift(); });
    this.cams = TRACKING.filter((n) => V.cams.has(n) && scene.cameras.some((c) => c.name === n));
    this.timer = null;
  }

  calibration() {
    const s = this.V.scene;
    return {
      cameras: s.cameras.map((c) => ({ name: c.name, size: c.size, lens: c.lens, T_D_C: c.T_D_C })),
      ports: s.ports, instruments: s.instruments.map((i) => ({ id: i.id, port: i.port, kind: i.kind, spec: i.spec })), source: 'sim',
    };
  }

  onStatus(st) {
    const on = !!(st?.model?.connected || st?.recording);
    if (on === this.attached) return;
    this.attached = on;
    if (on) {
      this.everAttached = true;
      if (!this.feeds) {
        this.feeds = new Feeds(this.V.scene.cameras.filter((c) => this.cams.includes(c.name)), {});
        const q = Number(this.V.physics?.jpeg?.quality?.value ?? this.V.physics?.jpeg?.quality) || 91;
        this.feeds.setTables({ ...qualityTables(q), quality: q, source: `IJG q${q}` });
      }
      this.timer = setInterval(() => this.tick(), PERIOD_MS);
    } else {
      clearInterval(this.timer); this.timer = null;
    }
    this.dispatchEvent(new Event('change'));
  }

  tick() {
    if (!this.attached || !this.cams.length) return;
    if (this.busy) { this.stats.skipped++; return; }                          // latest only: the last bundle is in flight
    this.busy = true;
    const t0 = performance.now();
    this.bundle().then(() => { this.stats.ms.push(performance.now() - t0); if (this.stats.ms.length > 400) this.stats.ms.shift(); },
      (e) => console.warn('[view] bundle', e)).finally(() => { this.busy = false; });
  }

  async bundle() {
    const V = this.V, r = V.r, scene = V.scene;
    const frameId = ++this.frameId;
    const state = V.state;
    const truth = K.truthFromState(scene, state);
    const kps = K.keypointList(truth), pts = kps.map((x) => x.p);
    r.setPoses(K.posesFromState(scene, state));
    this.posesDirty = true;                                       // the display sets its own poses again
    // issue every camera's render, sensor and readback first, then wait: the three readbacks overlap on the GPU queue
    // and each encode starts in its worker as soon as its readback lands
    const per = [];
    for (const name of this.cams) {
      const cam = scene.cameras.find((c) => c.name === name);
      const [w, h] = cam.size;
      const visP = Promise.resolve(r.visible ? r.visible(name, pts) : pts.map(() => true));
      const out = r.renderCamera(name, V.kind === 'trace' ? { seed: 1009 * (TRACKING.indexOf(name) + 2) } : { size: [w, h] });
      if (out?.then) await out;
      r.sensor(name, { seed: (frameId * 7919) >>> 0, frame: frameId });
      const enc = r.readback(name, 'rgb8').then((rb) => this.feeds.encode(name, rb, w, h, { draw: false, channels: Math.round(rb.length / (w * h)) || 3 }));
      per.push({ name, w, h, enc, visP });
    }
    for (const p of per) p.vis = await p.visP;
    const encs = await Promise.all(per.map((p) => p.enc));
    const t = V.t;
    const visMap = (vis) => { const o = {}; kps.forEach((k, i) => { (o[k.id] ??= {})[k.kp] = !!vis[i]; }); return o; };
    const truthMsg = {
      frame_id: frameId, t_ref: t, ref: { instruments: truth.instruments },
      cameras: Object.fromEntries(per.map((p, i) => [p.name, {
        t_capture: t, index: frameId, exposure_s: 0.010,
        instruments: truth.instruments.map((ins) => {
          const px = truth.px[p.name][ins.id], inside = truth.in_image[p.name][ins.id], un = visMap(p.vis)[ins.id] ?? {};
          return { ...ins, px, in_image: inside, unoccluded: un, visible: Object.fromEntries(Object.entries(un).map(([k, b]) => [k, !!(b && inside[k])])) };
        }),
      }])),
      state,
    };
    const header = { type: 'frames', frame_id: frameId, source: 'sim',
      cameras: per.map((p) => ({ name: p.name, t_capture: t, bytes: 0, width: p.w, height: p.h, index: frameId })) };
    this.lastVis = Object.fromEntries(per.map((p) => [p.name, visMap(p.vis)]));
    this.lastTruth = truth;
    if (this.socket.state === 'open' && this.socket.sendFrame(truthMsg, header, encs.map((e) => e.jpeg))) this.stats.sent++;
    this.stats.bundles++;
    this.stats.times.push(performance.now());
    if (this.stats.times.length > 400) this.stats.times.shift();
  }

  /** Model-vs-truth markers over a tracking-camera view (M): truth rings, prediction crosses, from the newest score. */
  async drawMarkers(name, slot) {
    if (!TRACKING.includes(name)) return;
    const { drawMarkers } = await import('../ui/markers.js');
    const cam = K.ensureLenses(this.V.scene).find((c) => c.name === name);
    const last = this.scores.at(-1);
    const frame = { cam, truth: K.truthFromState(this.V.scene, this.V.state), visible: this.lastVis?.[name], prediction: last?.prediction ?? null };
    drawMarkers(slot.ov, frame, { x: 0, y: 0, w: slot.rect.w, h: slot.rect.h });
  }

  report() {
    const ms = [...this.stats.ms].sort((a, b) => a - b);
    const now = performance.now(), recent = this.stats.times.filter((t) => now - t <= 3000);
    return { attached: this.attached, socket: this.socketState, bundles: this.stats.bundles, sent: this.stats.sent, skipped: this.stats.skipped,
      bundles_per_s_3s: recent.length / 3, bundle_ms_p50: ms.length ? ms[ms.length >> 1] : null,
      bundle_ms_p95: ms.length ? ms[Math.min(ms.length - 1, Math.floor(0.95 * ms.length))] : null };
  }
}
