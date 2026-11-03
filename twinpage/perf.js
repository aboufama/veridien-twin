// Startup profile harness of the twin page (?perf=1). Loaded by twinpage/main.js only when ?perf is in the URL; it
// changes nothing the page draws or tracks: it reads the page's performance marks ('twin:*' from main.js, 'gv:*' from
// godview_pt3), resource timing, long tasks, and watches the tracker source for its first results.
//
//   twin.html?perf=1&tracker=browser&replay=synthetic&autostart=0      a run with no camera and no localhost
//   window.twinPerf.report()          -> { marks, resources, totals, longtasks, god, tracker, env }  (ms from navigation)
//   window.twinPerf.done              a Promise of that report once the milestones are in (or ?perfTimeout ms, 60 s)
//   window.twinPerf.sceneHash()       -> sha-256 of the god view's prepared scene (geometry, BVH, materials): byte proof
//   window.twinPerf.tracedHash(w, h, spp)  -> sha-256 of the path-traced god view (captureLinear 'traced', seed 0, spp
//                                     samples, default 320x180x4: ~1 s of GPU): pixel proof, independent of the bake's timing
//   window.twinPerf.rasterHash(w, h)  the raster view after a FULL bake (tens of seconds of GPU: avoid on the 16 GB Air)
//   window.twinPerf.proof()           starts sceneHash + tracedHash; poll window.twinPerf.proofResult
//
// The report prints once to the console as one line 'TWIN_PERF {json}' so a driver can grep it.
// Workers keep their own resource timelines: the tracker worker's fetches (ORT, its wasm, tipnet.onnx, rig.json) are not
// in `resources`; their total shows up as tracker.readyMs (twin:tracker:ready) and in the network panel.

const MILESTONES = ['twin:god:firstFrameGpu', 'twin:tracker:firstTip'];
const marks = () => Object.fromEntries(performance.getEntriesByType('mark').filter((m) => /^(twin|gv):/.test(m.name)).map((m) => [m.name, +m.startTime.toFixed(1)]));
const mk = (n) => { if (!performance.getEntriesByName(n, 'mark').length) performance.mark(n); };

const longtasks = [];
let installed = false;
let resolveDone;
const done = new Promise((r) => { resolveDone = r; });

export function install() {
  if (installed) return window.twinPerf;
  installed = true;
  try {
    new PerformanceObserver((l) => { for (const e of l.getEntries()) longtasks.push([+e.startTime.toFixed(0), +e.duration.toFixed(0)]); })
      .observe({ type: 'longtask', buffered: true });
  } catch { /* no longtask timing */ }
  watchSource();
  const timeout = Number(new URLSearchParams(location.search).get('perfTimeout')) || 60000;
  const t = setInterval(() => {
    const m = marks();
    if (MILESTONES.every((k) => k in m) || performance.now() > timeout) {
      clearInterval(t);
      // one more second for late marks (sim tile, real views), then report
      setTimeout(() => { const r = report(); console.log('TWIN_PERF ' + JSON.stringify(r)); resolveDone(r); }, 1000);
    }
  }, 100);
  window.twinPerf = { report, done, marks, sceneHash, rasterHash, tracedHash, proof, proofResult: null };
  return window.twinPerf;
}

/** The tracker source's milestones: worker ready (ORT session built), first views (a frame set back), first tip. */
function watchSource() {
  const tick = () => {
    const src = window.twinPage?.source;
    if (!src) { setTimeout(tick, 10); return; }
    if (src.kind !== 'browser') { mk('twin:tracker:server'); return; }
    if (src.workerInfo) mk('twin:tracker:ready');
    src.on?.((what) => { if (what === 'ready') mk('twin:tracker:ready'); });
    src.frames?.onPut?.((k) => { if (k != null) mk('twin:tracker:firstViews'); });
    const feed = src.feed, push = feed.push.bind(feed);
    feed.push = (j) => {
      const ok = push(j);
      if (j && Array.isArray(j.instruments)) {
        mk('twin:tracker:firstResult');
        if (j.instruments.some((q) => Array.isArray(q.tip_D ?? q.tip ?? q.tip_mm ?? null) || q.ok || q.seen)) mk('twin:tracker:firstTip');
      }
      return ok;
    };
  };
  tick();
}

export function report() {
  const nav = performance.getEntriesByType('navigation')[0];
  const res = performance.getEntriesByType('resource').map((e) => ({
    name: e.name.replace(location.origin, ''), type: e.initiatorType, start: +e.startTime.toFixed(0), end: +e.responseEnd.toFixed(0),
    dur: +e.duration.toFixed(0), transfer: e.transferSize, encoded: e.encodedBodySize, decoded: e.decodedBodySize,
    cached: e.transferSize === 0 && e.decodedBodySize > 0, revalidated: e.transferSize > 0 && e.transferSize < 1000 && e.decodedBodySize > 1000,
    status: e.responseStatus,
  })).sort((a, b) => a.start - b.start);
  const sum = (k, f = () => true) => res.filter(f).reduce((a, r) => a + (r[k] || 0), 0);
  const p = window.twinPage || {};
  const g = p.god || {};
  const src = p.source || {};
  return {
    url: location.href, at: new Date().toISOString(), ua: navigator.userAgent,
    nav: nav ? { responseEnd: +nav.responseEnd.toFixed(0), domInteractive: +nav.domInteractive.toFixed(0), dcl: +nav.domContentLoadedEventEnd.toFixed(0), load: +nav.loadEventEnd.toFixed(0), transfer: nav.transferSize } : null,
    marks: marks(),
    totals: { requests: res.length, transferMB: +(sum('transfer') / 1e6).toFixed(2), decodedMB: +(sum('decoded') / 1e6).toFixed(2),
      fromCache: res.filter((r) => r.cached).length, revalidated: res.filter((r) => r.revalidated).length,
      lastResponseEnd: res.reduce((a, r) => Math.max(a, r.end), 0),
      meshes: { n: res.filter((r) => /\/meshes\//.test(r.name)).length, transferMB: +(sum('transfer', (r) => /\/meshes\//.test(r.name)) / 1e6).toFixed(2),
        firstStart: Math.min(...res.filter((r) => /\/meshes\//.test(r.name)).map((r) => r.start)), lastEnd: Math.max(0, ...res.filter((r) => /\/meshes\//.test(r.name)).map((r) => r.end)) } },
    longtasks: { n: longtasks.length, totalMs: longtasks.reduce((a, t) => a + t[1], 0), top: [...longtasks].sort((a, b) => b[1] - a[1]).slice(0, 8) },
    god: { dir: p.godDir, cached: g.cached, prepareMs: g.prepareMs && +g.prepareMs.toFixed(0), loadMs: g.loadMs && +g.loadMs.toFixed(0), pageLoadMs: p.godLoadMs && +p.godLoadMs.toFixed(0),
      tris: g.S?.triCount, adapter: g.adapterInfo, failed: p.godFailed },
    tracker: { kind: src.kind, status: src.status, error: src.error, worker: src.workerInfo ? { ...src.workerInfo, type: undefined } : null, timing: src.timingInfo ?? null },
    sim: { tileFrames: p.tileFrames, camsMode: p.camsMode },
    resources: res,
  };
}

async function sha(bufs) {
  const parts = bufs.map((b) => (b instanceof ArrayBuffer ? new Uint8Array(b) : new Uint8Array(b.buffer, b.byteOffset, b.byteLength)));
  const n = parts.reduce((a, b) => a + b.length, 0), all = new Uint8Array(n);
  let o = 0; for (const b of parts) { all.set(b, o); o += b.length; }
  const h = await crypto.subtle.digest('SHA-256', all);
  return [...new Uint8Array(h)].map((x) => x.toString(16).padStart(2, '0')).join('');
}

/** The god view's prepared scene, every array and the JSON-able tables (stats and timings excluded). */
export async function sceneHash() {
  const S = window.twinPage?.god?.S;
  if (!S) return null;
  const meta = JSON.stringify({ nodeCount: S.nodeCount, triCount: S.triCount, vertCount: S.vertCount, staticVerts: S.staticVerts, staticTris: S.staticTris,
    blas: S.blas, meshTable: S.meshTable, materials: S.materials, materialIndex: S.materialIndex,
    emitter: { ...S.emitter } });
  const arrays = { nodes: S.nodes, tris: S.tris, verts: S.verts, adjStart: S.adjacency?.start, adjList: S.adjacency?.list };
  const per = {};
  for (const [k, a] of Object.entries(arrays)) per[k] = a ? (await sha([a])).slice(0, 16) : null;
  return { all: await sha([new TextEncoder().encode(meta), ...Object.values(arrays).filter(Boolean)]), meta: (await sha([new TextEncoder().encode(meta)])).slice(0, 16), ...per };
}

/** A raster capture of the god view at the current camera (full bake first): its sha-256 and mean. */
export async function rasterHash(w = 640, h = 360) {
  const g = window.twinPage?.god;
  if (!g?.captureLinear) return null;
  const px = await g.captureLinear(w, h, { mode: 'raster' });
  let m = 0; for (let i = 0; i < px.length; i++) m += px[i];
  return { sha: await sha([px]), mean: m / px.length, w, h };
}

export async function tracedHash(w = 320, h = 180, spp = 4) {
  const g = window.twinPage?.god;
  if (!g?.captureLinear) return null;
  const px = await g.captureLinear(w, h, { mode: 'traced', spp, seed: 0 });
  let m = 0; for (let i = 0; i < px.length; i++) m += px[i];
  return { sha: await sha([px]), mean: m / px.length, w, h, spp };
}

/** Both proofs, in the background (a driver polls twinPerf.proofResult; a capture can outlast a 45 s tool call). */
export function proof() {
  const out = window.twinPerf.proofResult = { started: performance.now() };
  (async () => {
    out.scene = await sceneHash();
    out.traced = await tracedHash();
    out.ms = +(performance.now() - out.started).toFixed(0);
  })().catch((e) => { out.error = String(e?.message || e); });
  return out;
}
