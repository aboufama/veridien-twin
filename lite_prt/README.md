# lite_prt: raster + precomputed indirect light vs scope pose

`renderer.js` exports `LitePRTRenderer` (also the default export) with the §8/§18.1 API: `create`, `setPoses`,
`renderCamera`, `sensor`, `readback`, `visible`, `timings`, `present(name, canvasContext)`, and `cameras` with `SCOPE`.
It subclasses `gpu/renderer.js` `TwinRenderer` (imported, never edited), so the sensor chain, readback, `visible` and
the GPU budget timer are the path tracer's own.

Per frame:
1. Raster visibility (4x MSAA `rg16uint`: BVH triangle slot | body), per-vertex KB4 (or the SCOPE pinhole). Long CAD
   triangles of the posed bodies are bisected (raster only) to <= 3 mm edges and keep their slot.
2. One compute pass shades each pixel: MSAA samples grouped by triangle, the exact KB4 ray per group intersected with
   that one triangle, then the tracer's shading (area-emitter samples, node-culled shadow rays, §8 BSDF, node indirect
   x capsule occlusion, posed bodies' BSDF bounce). Culling uses the bodies' in-cavity capsules (`liteVis`).
3. Indirect: `E_free = scale x chroma x (mean + B c(pose))`, B = 48 PCA components of node luminance over 384 Sobol scope
   poses (path tracer's converged node solver, tools at default), c = thin-plate RBF of the emitter position and axis.
   `scale` = first-bounce flux now / with the default tools (judge path: two node-direct passes in `converge`).

Assets: `assets/prt.bin` + `prt.json` (25 MB, made by `tools/run.py gen` + `fit` on the pod), `assets/physics.json`
(written by the driver from configs/twin/physics.yaml). Without the asset (`prt: false`, `?prt=0`) it falls back to
the tracer's gather table + Jacobi.

Judge: `judge.html` exposes `window.twinJudge` (`render`, `renderB64`, `perf`, `gen`, `info`). Driver:
`python3 web/twin/lite_prt/tools/run.py {smoke|gen|fit|judge|perf|contact}` under `gpu_guard run` on the pod.
