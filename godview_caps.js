// Caps of the god's-eye view's fixed section (configs/twin/godview.yaml cutaway.modes.fixed, docs/WEB_TWIN.md §19).
// scripts/build_godview_caps.py intersects the PLA shell with the fixed plane, chains the section into closed loops and
// triangulates the solid wall between them: assets/godview_caps.json + assets/meshes/cap_<mode>_*.bin, static meshes
// of pla_matte_white with a flat normal facing the removed half. A renderer draws them only in the fixed mode.
//
//   import { mergeCaps, CAP_FLAG, markCaps } from '../godview_caps.js';
//   const bundle2 = await mergeCaps(bundle);            // appends the caps as static meshes named cap_*
//   const S = prepareScene(...bundle2...); markCaps(S);  // sets CAP_FLAG on their mesh-table entries and triangles
//
// The input is not mutated. Meshes already present (same name) are not added twice.

import { decodeMesh } from './gpu/scene_gpu.js';

/** mesh flag bit of a cap (gpu/scene_gpu.js MESH_FLAG uses bits 0..7) */
export const CAP_FLAG = 256;
/** the fixed plane when godview.yaml has no cutaway.modes.fixed (the same numbers as the yaml) */
export const FIXED_DEFAULT = Object.freeze({ n: [-1, 0, 0], offset: 0.055 });

export const isCapName = (name) => typeof name === 'string' && name.startsWith('cap_');

/** the fixed plane of godview (n unit, offset): points with dot(p - centre, n) > offset are removed */
export function fixedPlane(godview, mode = 'fixed') {
  const m = godview?.cutaway?.modes?.[mode];
  const n = (m?.n || FIXED_DEFAULT.n).map(Number), l = Math.hypot(n[0], n[1], n[2]) || 1;
  return { n: [n[0] / l, n[1] / l, n[2] / l], offset: Number(m?.offset ?? FIXED_DEFAULT.offset) };
}

/** bundle {scene, meshes, base} -> a new bundle with the cap meshes appended as static meshes */
export async function mergeCaps(bundle, baseUrl, fetchFn = fetch) {
  let root = String(baseUrl || bundle.base || new URL('./assets/', import.meta.url).href);
  if (!root.endsWith('/')) root += '/';
  const r = await fetchFn(root + 'godview_caps.json', { cache: 'no-store' });
  if (!r.ok) throw new Error(`godview_caps.json: HTTP ${r.status}`);
  const caps = await r.json();
  const have = new Set(bundle.scene.meshes.map((m) => m.name));
  const add = (caps.meshes || []).filter((m) => !have.has(m.name));
  const decoded = {};
  await Promise.all(add.map(async (m) => {
    const res = await fetchFn(root + m.file);
    if (!res.ok) throw new Error(`${m.file}: HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    if (m.bytes != null && buf.byteLength !== m.bytes) throw new Error(`${m.file}: ${buf.byteLength} bytes, expected ${m.bytes}`);
    decoded[m.name] = decodeMesh(m, buf);
  }));
  const entries = add.map((m) => ({ name: m.name, file: m.file, material: m.material, group: 'cap', dynamic: false,
    cap_mode: m.cap_mode, vertex_count: m.vertex_count, triangle_count: m.triangle_count, bbox: m.bbox, layout: m.layout, bytes: m.bytes }));
  return { ...bundle, scene: { ...bundle.scene, meshes: [...bundle.scene.meshes, ...entries] }, meshes: { ...bundle.meshes, ...decoded } };
}

/** Set CAP_FLAG on the prepared scene's cap meshes (meshTable) and their triangles (tris word 13). Idempotent.
 *  Returns the number of cap triangles. */
export function markCaps(S) {
  const capMesh = new Set();
  S.meshTable.forEach((m, i) => { if (isCapName(m.name)) { m.flags = (m.flags | CAP_FLAG) >>> 0; capMesh.add(i); } });
  if (!capMesh.size) return 0;
  const U = new Uint32Array(S.tris.buffer, S.tris.byteOffset, S.tris.length);
  let n = 0;
  for (let o = 0; o + 15 < U.length; o += 16) if (capMesh.has(U[o + 12])) { U[o + 13] = (U[o + 13] | CAP_FLAG) >>> 0; n++; }
  return n;
}

// ---------------------------------------------------------------- the page's cutaway modes (key C)
/** the order key C cycles through: the eye-following plane (default), the fixed section, no cut */
export const CUT_CYCLE = Object.freeze(['view', 'fixed', false]);

/** ?cut= -> a mode: '0' / 'off' -> false, 'fixed' -> 'fixed', '1' / 'on' / 'view' -> 'view'; absent -> null */
export function cutModeFromQuery(q) {
  if (q == null || q === '') return null;
  if (q === '0' || q === 'off' || q === 'false') return false;
  return q === 'fixed' ? 'fixed' : 'view';
}

/** the next mode after `mode`; a renderer without the fixed section (no cutawayMode getter) cycles view <-> off */
export function nextCutMode(mode, fixedSupported = true) {
  const cyc = fixedSupported ? CUT_CYCLE : CUT_CYCLE.filter((m) => m !== 'fixed');
  const cur = mode === true ? 'view' : mode || false;
  const i = cyc.indexOf(cur);
  return cyc[(i + 1) % cyc.length];
}

/** the quiet hint line for a mode */
export function cutModeNote(mode) {
  return mode === 'fixed' ? 'Cutaway: fixed section' : mode ? 'Cutaway: follows the view' : 'Cutaway off';
}
