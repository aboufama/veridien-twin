// The model socket on the twin page (docs/WEB_TWIN.md §7, §14, §19): the page says hello on /ws/twin (served by
// `veridien webtwin serve`; offline with retries otherwise). Only while a model is attached (the server's status says
// so), the three tracking cameras render offscreen at their native size at 20 fps, each image goes through the camera
// JPEG stage (ui/feeds.js workers, physics.yaml tables) and the truth + §7.2 bundle are sent (model/socket.js,
// latest-only). The server scores every prediction; the newest predicted tools become the god view's ghosts (key M).
//
// Simplified against app.js's calendar (control/gpu_sched.js): one bundle at a time, the state at issue time for the
// whole exposure (no motion blur), truth at that instant. The judges and the dataset generator keep using
// index.html?judge=1, whose bundle path is the reference.
import * as K from '../control/kinematics.js';
import { TwinSocket } from '../model/socket.js';
import { Feeds } from '../ui/feeds.js';
import { qualityTables } from '../ui/jpeg_encoder.js';

const BUNDLE_HZ = 20;

function jpegTables(physics) {
  const j = physics?.jpeg;
  const val = (x) => (x && typeof x === 'object' && !Array.isArray(x) && 'value' in x ? x.value : x);
  const q = Number(val(j?.quality)) || 91;
  const std = qualityTables(q);
  const flat = (t) => (t == null ? null : Uint8Array.from(Array.isArray(t[0]) ? t.flat() : t));
  const qt = j?.quant_tables;
  const luma = flat(val(qt?.luma) ?? val(qt)?.luma), chroma = flat(val(qt?.chroma) ?? val(qt)?.chroma);
  return { qY: luma?.length === 64 ? luma : std.qY, qC: chroma?.length === 64 ? chroma : std.qC, quality: q, source: `IJG q${q}` };
}

export class ModelLink {
  constructor(page) {
    this.page = page;
    this.busy = false;
    this.frameId = 0;
    this.sent = 0;
    this.status = null;
    this.everOpen = false;
    this.scores = [];
    const scene = page.scene;
    this.tracking = scene.cameras.map((c) => c.name).filter((n) => page.camNames?.has(n));
    try {
      this.socket = new TwinSocket({ hello: () => ({ calibration: this.calibration(), scene_ref: { renderer: page.camsDir, page: 'twin.html' } }) });
    } catch (e) { this.socket = null; console.warn('[twin] socket', e); return; }
    this.socket.addEventListener('state', () => { if (this.socket.state === 'open') this.everOpen = true; });
    this.socket.addEventListener('status', (e) => { this.status = e.detail; });
    this.socket.addEventListener('score', (e) => this.onScore(e.detail));
    this.loop();
  }

  calibration() {
    const s = this.page.scene;
    return { cameras: s.cameras.map((c) => ({ name: c.name, size: c.size, lens: c.lens, T_D_C: c.T_D_C })),
      ports: s.ports, instruments: s.instruments.map((i) => ({ id: i.id, port: i.port, kind: i.kind, spec: i.spec })), source: 'sim' };
  }

  attached() { return !!this.status?.model?.connected; }

  problem() {
    if (this.everOpen && this.socket?.state !== 'open') return 'The model socket is down (the server stopped?)';
    if (this.err) return `Model frames: ${this.err}`;
    return null;
  }

  onScore(s) {
    this.scores.push(s);
    if (this.scores.length > 30) this.scores.shift();
    const pred = s?.prediction;
    if (!pred?.instruments?.length) return;
    // ghosts: the predicted shaft (pivot -> hinge) and jaws (hinge -> tip) as lines; a renderer with setGhosts gets
    // body poses rebuilt from the predicted directions
    const page = this.page, lines = [];
    const bodies = {};
    for (const p of pred.instruments) {
      const ins = page.scene.instruments.find((i) => i.id === p.id);
      if (!ins || !p.hinge_D) continue;
      const port = K.port(page.scene, ins.port);
      lines.push([port.pivot, p.hinge_D, 4]);
      if (p.tip_D) lines.push([p.hinge_D, p.tip_D, 3]);
      const dir = K.normalize(K.sub(p.hinge_D, port.pivot));
      const [yaw, pitch] = K.inverseDirection(port.R, dir);
      const q = page.state.instruments.find((x) => x.id === p.id);
      const insertion = K.norm(K.sub(p.hinge_D, port.pivot));
      const P = K.posesFromState(page.scene, { ...page.state, instruments: [{ id: p.id, yaw, pitch, roll: q?.roll ?? 0, insertion, jaw: p.jaw ?? q?.jaw ?? 0 }] });
      for (const [k, T] of Object.entries(P.bodies)) if (k.startsWith(`${p.id}/`) && !k.endsWith('cannula')) bodies[k] = T;
    }
    page.ghosts = { lines, bodies, frame_id: s.frame_id };
    if (page.ghostsOn) page.godDirty = true;
  }

  /** 20 Hz: the next bundle starts 50 ms after the previous one started, or at once when that one ran late. */
  async loop() {
    const t0 = performance.now();
    await this.tick();
    setTimeout(() => this.loop(), Math.max(0, 1000 / BUNDLE_HZ - (performance.now() - t0)));
  }

  async tick() {
    const page = this.page, r = page.cams;
    if (!r || this.busy || !this.attached() || this.socket?.state !== 'open' || document.hidden || page.tileBusy) return;
    this.busy = true;
    try {
      if (!this.feeds) {
        this.feeds = new Feeds(page.scene.cameras, {});
        this.feeds.setTables(jpegTables(page.physics));
      }
      const id = ++this.frameId;
      const t = performance.now() / 1000;
      const state = page.state;
      const P = K.posesFromState(page.scene, state);
      if (!r.isWorkerHost) r.setPoses(P);
      page.camsPosesV = -1;                                   // the tiles set their own poses again
      const truth = K.truthFromState(page.scene, state);
      const kps = K.keypointList(truth), pts = kps.map((x) => x.p);
      const per = [];
      if (r.isWorkerHost) {
        // the renderer lives in the camera worker (twinpage/cams_worker.js): one round trip renders, senses and reads
        // back all three; the JPEG stage starts here as before
        const res = await r.bundle({ poses: P, cams: this.tracking.map((name) => ({ name, size: page.scene.cameras.find((c) => c.name === name).size })), pts, seed: 11, frame: id });
        for (const { name, vis, rb } of res.cams) {
          const cam = page.scene.cameras.find((c) => c.name === name);
          const enc = rb?.y ? this.feeds.encodeYUV(name, rb, { draw: false })
            : this.feeds.encode(name, rb, cam.size[0], cam.size[1], { draw: false, channels: rb.length === cam.size[0] * cam.size[1] * 4 ? 4 : 3 });
          per.push({ name, cam, vis, enc });
        }
      }
      for (const name of r.isWorkerHost ? [] : this.tracking) {
        const cam = page.scene.cameras.find((c) => c.name === name);
        if (r.setViewSize) { const cur = page.camSize?.[name]; if (!cur || cur[0] !== cam.size[0] || cur[1] !== cam.size[1]) { r.setViewSize(name, cam.size); (page.camSize ??= {})[name] = [...cam.size]; } }
        const vis = r.visible ? await r.visible(name, pts) : pts.map(() => true);
        const lt = r.renderCamera(name, { await: false, wait: false, seed: 11 });
        if (lt?.then) await lt;
        r.sensor(name, { seed: (id * 7919) >>> 0, frame: id });
        const kind = r.readbackKinds?.includes?.('yuv420') || r.isLite ? 'yuv420' : 'rgb8';
        // the readback lands while the next camera renders; its JPEG starts the moment it lands
        const enc = r.readback(name, kind).then((rb) => (rb?.y ? this.feeds.encodeYUV(name, rb, { draw: false })
          : this.feeds.encode(name, rb, cam.size[0], cam.size[1], { draw: false, channels: rb.length === cam.size[0] * cam.size[1] * 4 ? 4 : 3 })));
        per.push({ name, cam, vis, enc });
      }
      const encs = await Promise.all(per.map((p) => p.enc));
      const visMap = (vis) => { const out = {}; kps.forEach((k, i) => { (out[k.id] ??= {})[k.kp] = !!vis[i]; }); return out; };
      const camTruth = (name, vis) => truth.instruments.map((ins) => {
        const px = truth.px[name][ins.id], inside = truth.in_image[name][ins.id];
        const un = Object.fromEntries(Object.entries(vis[ins.id] ?? {}).map(([k, b]) => [k, px[k] == null ? null : b]));
        return { ...ins, px, in_image: inside, unoccluded: un, visible: Object.fromEntries(Object.entries(un).map(([k, b]) => [k, b == null ? null : !!(b && inside[k])])) };
      });
      const truthMsg = { frame_id: id, t_ref: t, ref: { instruments: truth.instruments },
        cameras: Object.fromEntries(per.map((p) => [p.name, { t_capture: t, index: id, exposure_s: 0.01, instruments: camTruth(p.name, visMap(p.vis)) }])), state };
      const header = { type: 'frames', frame_id: id, source: 'sim',
        cameras: per.map((p) => ({ name: p.name, t_capture: t, bytes: 0, width: p.cam.size[0], height: p.cam.size[1], index: id })) };
      if (this.socket.sendFrame(truthMsg, header, encs.map((e) => e.jpeg))) this.sent++;
      this.err = null;
    } catch (e) {
      this.err = String(e?.message || e);
      if (!this.errLogged) { console.error('[twin] model frames', e); this.errLogged = true; }
    } finally { this.busy = false; }
  }
}
