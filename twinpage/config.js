// The god's-eye view's configuration (configs/twin/godview.yaml, docs/WEB_TWIN.md §19) and its viewpoint formula.
// Pure ES module (node tests). The page fetches /configs/twin/godview.json (the server parses the YAML); a static
// server cannot, so DEFAULT_GODVIEW mirrors godview.yaml as of 2026-11-01 (gain 0.365, exposure 1.0) and is used only then (the status
// line says so).

export const DEFAULT_GODVIEW = Object.freeze({
  centre: [-0.05, 0.027, 0.10],
  vfov_deg: 35.0,
  image: [1280, 800],
  viewpoints: {
    V1_surgeon: { az_deg: 200.0, el_deg: 40.0, d: 0.75 },
    V2_left: { az_deg: 110.0, el_deg: 30.0, d: 0.75 },
    V3_top: { az_deg: 180.0, el_deg: 80.0, d: 0.80 },
    V4_far: { az_deg: 330.0, el_deg: 35.0, d: 0.75 },
  },
  cutaway: { materials: ['pla_matte_white'], offset: 0.02, default: 'on', zoom: { open: 0.85, closed: 1.30, closed_offset: 0.27 } },
  bench: { extent: [[-1.0, -1.0], [1.0, 1.0]] },
  lights: {
    environment: { type: 'constant', radiance: [1.0, 1.0, 1.0] },
    key: { type: 'directional', irradiance: [2.0, 2.0, 2.0], from_az_deg: 150.0, from_el_deg: 65.0 },
    scope: { type: 'physics_emitter', gain: 0.365 },
  },
  display: { tone_map: 'khronos_pbr_neutral', exposure: 1.0 },
  states: ['default', 'crossing'],
  materials: 'physics',
});

export const DEFAULT_VIEWPOINT = 'V1_surgeon';
const DEG = Math.PI / 180;

/** YAML `on`/`off` may arrive as true/false (YAML 1.1) or as strings. */
export function cutawayDefault(gv) {
  const d = gv?.cutaway?.default;
  return !(d === false || d === 'off' || d === 0);
}

/** eye = centre + d (cos el cos az, cos el sin az, sin el); az from +x toward +y (godview.yaml). */
export function viewpointEye(gv, name) {
  const v = gv.viewpoints?.[name];
  if (!v) throw new Error(`no viewpoint ${name}`);
  const c = gv.centre, az = v.az_deg * DEG, el = v.el_deg * DEG;
  return [c[0] + v.d * Math.cos(el) * Math.cos(az), c[1] + v.d * Math.cos(el) * Math.sin(az), c[2] + v.d * Math.sin(el)];
}

/** The camera block every GodView takes: {eye, target, up, vfov_deg} for a named viewpoint. */
export function viewpointCamera(gv, name) {
  return { eye: viewpointEye(gv, name), target: [...gv.centre], up: [0, 0, 1], vfov_deg: gv.vfov_deg };
}

/** The cutaway plane for an eye: n = unit(eye - centre); a shell fragment p is removed when dot(p - centre, n) > offset. */
export function cutawayPlane(gv, eye) {
  const c = gv.centre;
  const d = [eye[0] - c[0], eye[1] - c[1], eye[2] - c[2]];
  const l = Math.hypot(d[0], d[1], d[2]) || 1;
  const n = [d[0] / l, d[1] / l, d[2] / l];
  return { n, offset: Number(gv.cutaway?.offset ?? 0.02), centre: [...c], materials: [...(gv.cutaway?.materials || ['pla_matte_white'])] };
}

// ---------------------------------------------------------------- the cutaway follows the zoom (godview.yaml cutaway.zoom)
/** {open, closed} orbit distances (m), the open offset (cutaway.offset) and the closed one (past the dome's reach) */
export function zoomCutParams(gv) {
  const z = gv?.cutaway?.zoom || {};
  const open = Number(z.open ?? 0.85);
  const closed = Math.max(Number(z.closed ?? 1.30), open + 1e-6);
  return { open, closed, offOpen: Number(gv?.cutaway?.offset ?? 0.02), offClosed: Number(z.closed_offset ?? 0.27) };
}

/** The 'view' plane's offset at orbit distance d: cutaway.offset for d <= open, closed_offset (nothing cut) for
 *  d >= closed, smoothstep in distance between. */
export function zoomCutOffset(gv, d) {
  const p = zoomCutParams(gv);
  const t = Math.min(1, Math.max(0, (d - p.open) / (p.closed - p.open)));
  return p.offOpen + t * t * (3 - 2 * t) * (p.offClosed - p.offOpen);
}

/** The fixed section (precomputed caps at one plane, so it cannot slide) shown at distance d: it closes past `closed`
 *  and reopens only below open + (closed - open) / 2 (hysteresis, no flicker). `shown`: the last answer (null: fresh). */
export function zoomFixedShown(gv, d, shown = null) {
  const p = zoomCutParams(gv);
  if (shown === false) return d < p.open + 0.5 * (p.closed - p.open);
  return d <= p.closed;
}

/** Merge a fetched config over the defaults (missing keys keep the defaults). */
export function normaliseGodview(raw) {
  const d = structuredClone(DEFAULT_GODVIEW);
  if (!raw || typeof raw !== 'object') return d;
  const out = { ...d, ...raw };
  out.viewpoints = { ...d.viewpoints, ...(raw.viewpoints || {}) };
  out.cutaway = { ...d.cutaway, ...(raw.cutaway || {}) };
  out.cutaway.zoom = { ...d.cutaway.zoom, ...(raw.cutaway?.zoom || {}) };
  out.lights = { ...d.lights, ...(raw.lights || {}) };
  out.display = { ...d.display, ...(raw.display || {}) };
  out.bench = { ...d.bench, ...(raw.bench || {}) };
  return out;
}
