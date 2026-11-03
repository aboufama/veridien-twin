// God's-eye-view extras: meshes only the god's-eye view draws. scripts/build_godview_extras.py writes them to
// assets/godview_extras.json + assets/meshes/extra_*.bin. Two kinds of entry:
//   - added: the parts of the rig beyond the dome that the scene leaves out (grasper handles, the scope's eyepiece,
//     light post, light cable and camera head), each a dynamic mesh in the frame of an existing body ("L/shaft",
//     "R/shaft", "scope"), so setPoses carries them with no other change;
//   - replacing ("replaces": "<scene mesh name>"): a more detailed stand-in for a scene mesh, e.g. the AR0144 camera
//     modules for the lens_front_CAMk discs, or detailed jaws for L_jaw_a. Several entries may replace one mesh (one
//     per material). The named mesh leaves the merged scene; the replacements take its place in every list that names
//     it (instrument parts, scope mesh_parts, port cannula_meshes, camera lens_mesh), with the replaced part's body.
// An entry is dynamic (in its body's frame; the body must exist) unless it says "dynamic": false (then it is static,
// in D, with no body).
//
//   import { mergeExtras } from '../godview_extras.js';
//   const godBundle = await mergeExtras(bundle, bundle.base);      // bundle = await loadScene(base)
//   GodView.create({ scene: godBundle, ... });                     // the camera simulations keep `bundle`
//
// Only the god's-eye view gets them: the tracking cameras and the scope keep the scene's own meshes (truth, owner-lens
// skipping and the kinematics never change). The input is not mutated. The meshes are marked overview_only (like the
// trocar housings), so gpu/scene_gpu.js leaves them out of the in-cavity occlusion capsules. A replacing entry whose
// target is not in the scene is skipped (a stale extras file never doubles a part).
// A top-level "bench" object (ADR-136: {"z": the stand's underside}) is merged over the scene's bench, so the god's-eye
// table drops under the stand; the camera simulations keep scene.json's bench (they never load this file).

import { decodeMesh } from './gpu/scene_gpu.js';

/** Plan the merge on parsed JSON only (pure; exported for the tests): {add, removed: Set, renames: Map old -> [new]}. */
export function planExtras(sj, extras) {
  // the bodies the scene already poses: instrument parts, the scope, any dynamic mesh's body key
  const bodies = new Set();
  for (const inst of sj.instruments || []) for (const p of inst.parts || []) bodies.add(`${inst.id}/${p.body}`);
  if (sj.scope) bodies.add(sj.scope.body || 'scope');
  for (const m of sj.meshes) if (m.dynamic && m.body) bodies.add(m.body);
  const have = new Map(sj.meshes.map((m) => [m.name, m]));
  const add = [], removed = new Set(), renames = new Map(), names = new Set();
  for (const m of extras.meshes || []) {
    if (have.has(m.name) || names.has(m.name)) continue;
    const isStatic = m.dynamic === false;
    // "prop/<name>" bodies are new movable props the page poses itself (e.g. the practice module, P key)
    if (!isStatic && !bodies.has(m.body) && !String(m.body ?? '').startsWith('prop/')) continue;
    if (m.replaces != null) {
      const old = have.get(m.replaces);
      if (!old) continue;
      // a dynamic stand-in rides on the replaced mesh's body; a static one replaces a static mesh
      if (isStatic ? !!old.dynamic : !(old.dynamic && old.body === m.body)) continue;
      removed.add(m.replaces);
      if (!renames.has(m.replaces)) renames.set(m.replaces, []);
      renames.get(m.replaces).push(m.name);
    }
    names.add(m.name);
    add.push(isStatic ? { ...m, dynamic: false, body: null, overview_only: true } : { ...m, dynamic: true, overview_only: true });
  }
  const bench = extras.bench && Number.isFinite(Number(extras.bench.z)) ? { z: Number(extras.bench.z) } : null;
  return { add, removed, renames, bench };
}

/** The scene.json with a plan applied (pure; exported for the tests). */
export function applyPlan(sj, { add, removed, renames, bench = null }) {
  const swap = (list) => (list || []).flatMap((n) => (renames.has(n) ? renames.get(n) : [n]));
  const out = { ...sj, meshes: [...sj.meshes.filter((m) => !removed.has(m.name)), ...add] };
  if (bench && sj.bench) out.bench = { ...sj.bench, ...bench };
  if (sj.instruments) {
    out.instruments = sj.instruments.map((inst) => ({
      ...inst,
      parts: (inst.parts || []).flatMap((p) => (renames.has(p.mesh) ? renames.get(p.mesh).map((n) => ({ ...p, mesh: n })) : [p])),
    }));
  }
  if (sj.scope) {
    out.scope = { ...sj.scope, mesh_parts: swap(sj.scope.mesh_parts) };
    if (sj.scope.emitter_mesh && renames.has(sj.scope.emitter_mesh)) out.scope.emitter_mesh = renames.get(sj.scope.emitter_mesh)[0];
  }
  if (sj.ports) out.ports = sj.ports.map((p) => (p.cannula_meshes ? { ...p, cannula_meshes: swap(p.cannula_meshes) } : p));
  if (sj.cameras) {
    out.cameras = sj.cameras.map((c) => (c.lens_mesh && renames.has(c.lens_mesh)
      ? { ...c, lens_mesh: renames.get(c.lens_mesh)[0], godview_lens_meshes: renames.get(c.lens_mesh) } : c));
  }
  return out;
}

/**
 * scene: a loadScene bundle {scene, meshes, base} (returns a new bundle) or a parsed scene.json (returns a new
 * scene.json with the entries merged; the caller decodes the meshes). baseUrl: the assets directory holding
 * godview_extras.json (default: the bundle's base, else ./assets/ beside this module).
 */
export async function mergeExtras(scene, baseUrl, fetchFn = fetch) {
  const isBundle = !!(scene && scene.scene && Array.isArray(scene.scene.meshes));
  const sj = isBundle ? scene.scene : scene;
  let root = String(baseUrl || (isBundle && scene.base) || new URL('./assets/', import.meta.url).href);
  if (!root.endsWith('/')) root += '/';
  const r = await fetchFn(root + 'godview_extras.json', { cache: 'no-store' });
  if (!r.ok) throw new Error(`godview_extras.json: HTTP ${r.status}`);
  const extras = await r.json();
  const plan = planExtras(sj, extras);

  const decoded = {};
  if (isBundle) {
    await Promise.all(plan.add.map(async (m) => {
      const res = await fetchFn(root + m.file);
      if (!res.ok) throw new Error(`${m.file}: HTTP ${res.status}`);
      const buf = await res.arrayBuffer();
      if (m.bytes != null && buf.byteLength !== m.bytes) throw new Error(`${m.file}: ${buf.byteLength} bytes, expected ${m.bytes}`);
      decoded[m.name] = decodeMesh(m, buf);
    }));
  }
  const merged = applyPlan(sj, plan);
  if (!isBundle) return merged;
  const meshes = { ...scene.meshes, ...decoded };
  for (const n of plan.removed) delete meshes[n];
  return { ...scene, scene: merged, meshes };
}
