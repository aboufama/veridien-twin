// node web/twin/godview_pt3/tools/build_cache_extras.mjs  -> web/twin/godview_pt3/cache/prepared.{json,bin} for the scene
// the page actually gives the god view: scene.json + assets/godview_extras.json (handles, scope head, camera modules,
// detailed tools; twinpage/main.js). CPU only. Same output format and key as build_cache.mjs, so loadPrepared hits on
// the merged scene. Writes into the god-view directory this script sits in (godview_pt3).
import { readFile, writeFile } from 'node:fs/promises';
const root = new URL('../../', import.meta.url).pathname;           // web/twin/
const gvDir = new URL('../', import.meta.url).pathname;             // web/twin/godview_pt3/
const { loadScene, prepareScene } = await import(root + 'gpu/scene_gpu.js');
const { serializePrepared, cacheKey } = await import(gvDir + 'cache.js');
const { mergeExtras } = await import(root + 'godview_extras.js');
const fetchFn = async (p) => { const b = await readFile(p); return { ok: true, json: async () => JSON.parse(b.toString()), arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) }; };
const src0 = await loadScene(root + 'assets/', fetchFn);
const src1 = await mergeExtras(src0, root + 'assets/', fetchFn);
// + the fixed section's caps (assets/godview_caps.json), merged after the extras exactly as godview_pt3/godview.js does
const { mergeCaps } = await import(root + 'godview_caps.js');
const src = await mergeCaps(src1, root + 'assets/', fetchFn);
const gv = JSON.parse((await readFile(gvDir + 'godview.json')).toString());
const spacing = Number(process.argv[2] || 0.008);
const extent = gv.bench?.extent || src.scene.bench?.extent;
const sc = { ...src.scene, bench: { ...(src.scene.bench || {}), extent } };
const t0 = performance.now();
const S = prepareScene(sc, src.meshes, {}, { nodeSpacing: spacing, leafSize: 4 });
const { json, bin } = serializePrepared(S, cacheKey(src.scene, extent, spacing));
await writeFile(gvDir + 'cache/prepared.json', JSON.stringify(json));
await writeFile(gvDir + 'cache/prepared.bin', bin);
console.log(`prepared (with extras and caps) in ${(performance.now() - t0).toFixed(0)} ms: ${src.scene.meshes.length} meshes, ${S.triCount} tris, bin ${(bin.length / 1e6).toFixed(1)} MB, key ${JSON.stringify(json.key)} -> ${gvDir}cache/`);
