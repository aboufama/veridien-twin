// node web/twin/lite_pt2/tools/build_cache.mjs [factor=3]  -> web/twin/lite_pt2/cache/prepared.{json,bin} (CPU only)
// PHYSICS_JSON=<file>: the physics exactly as the app server serves /configs/twin/physics.json, i.e.
//   .venv/bin/python -c "import yaml,json; print(json.dumps(yaml.safe_load(open('configs/twin/physics.yaml')), default=str))"
// Regenerate after any change of scene.json, the meshes, physics.yaml or the tracer's nodeSpacing / leafSize (the
// renderer logs a key mismatch and computes the scene instead). The judge of the cache: expand(compact(S)) must equal
// prepareScene's S word for word in every array the tracer reads, checked here; the build fails otherwise.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
const root = new URL('../../', import.meta.url).pathname;           // web/twin/
const { loadScene, prepareScene } = await import(root + 'gpu/scene_gpu.js');
const { pack, unpack, cacheKey, compactPrepared, expandPrepared } = await import(root + 'lite_pt2/cache.js');
const { innerScene } = await import(root + 'lite_pt2/cams.js');
const fetchFn = async (p) => { const b = await readFile(p); return { ok: true, json: async () => JSON.parse(b.toString()), arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) }; };
const src = await loadScene(root + 'assets/', fetchFn);
if (!process.env.PHYSICS_JSON) throw new Error('set PHYSICS_JSON (see the header)');
const physics = JSON.parse((await readFile(process.env.PHYSICS_JSON)).toString());
const F = Number(process.argv[2] || 3);
const nodeSpacing = 0.003, leafSize = 4;            // gpu/renderer.js DEFAULTS (the live preset keeps them)
const t0 = performance.now();
const { scene } = innerScene(src.scene, F);
const S = prepareScene(scene, src.meshes, physics, { nodeSpacing, leafSize });
const t1 = performance.now();
const key = cacheKey(src.scene, physics, { F, nodeSpacing, leafSize });
const C = compactPrepared(S);
const { json, bin } = pack(C, key);
// read-back judge: what the page will load equals prepareScene's output
const R = expandPrepared(unpack(json, bin.buffer).data);
const words = (a) => new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4);
let bad = 0;
for (const k of Object.keys(S)) {
  if (k === 'adjacency') continue;                  // not read by the tracer
  const a = S[k], b = R[k];
  if (ArrayBuffer.isView(a)) {
    const wa = words(a), wb = words(b);
    if (wa.length !== wb.length || wa.some((x, i) => x !== wb[i])) { console.error('MISMATCH', k); bad++; }
  } else if (k !== 'stats' && JSON.stringify(a, (kk, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v)) !== JSON.stringify(b, (kk, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v))) { console.error('MISMATCH', k); bad++; }
}
if (bad) throw new Error(`${bad} fields differ: cache not written`);
await mkdir(root + 'lite_pt2/cache', { recursive: true });
await writeFile(root + 'lite_pt2/cache/prepared.json', json);
await writeFile(root + 'lite_pt2/cache/prepared.bin', bin);
console.log(`prepareScene ${(t1 - t0).toFixed(0)} ms; ${S.triCount} tris, ${S.nodeCount} nodes; exceptions: ${C.triExc.idx.length} tris, ${C.nodeExc.idx.length} nodes; read-back exact; json ${(json.length / 1e6).toFixed(2)} MB, bin ${(bin.length / 1e6).toFixed(1)} MB; key ${JSON.stringify(key)}`);
