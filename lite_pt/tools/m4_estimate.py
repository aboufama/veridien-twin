"""M4 GPU-ms estimate for lite_pt from pod (RTX 4090) per-frame pass timings, with per-family ratios calibrated on
identical work measured on both machines (2026-10-31):
  trace      15.0 = M4 14.43 ms (runs/webtwin/20261031_prof_live_01/renderer_bench_control.json: CAM1 live trace,
             1280x720, cached) / pod 0.964 ms (runs/webtwin/race/lite_pt/cal1: the same pass, factor 1)
  pixel      11.1 = M4 2.17 ms / pod 0.195 ms, the full §9 sensor chain at 1280x720 (the nearest full-res pass to the
             G-buffer raster, upsample and display sensor; the raster is the least certain)
  radiosity   6.1 = M4 9.08 ms per update (direct 3.24 + Jacobi gather 5.03 + update 0.81) / pod 1.49 ms per update
             (0.745 ms per frame at radiosityEvery 2, cal1 CAM1)
The 4090 is therefore 6-15x an M4 on these kernels, not 4-6x. usage: python m4_estimate.py <run>/bench.json ..."""
import json
import sys
from pathlib import Path

RATIO = {"trace": 15.0, "pixel": 11.1, "radiosity": 6.1}


def fam(k):
    return "trace" if "@lo" in k else "radiosity" if k.startswith("radiosity") else "pixel"


def main():
    for path in sys.argv[1:]:
        res = json.loads(Path(path).read_text())
        print(f"== {path}  query {res.get('query')}")
        for cam, v in res["bench"].items():
            est = {k: ms * RATIO[fam(k)] for k, ms in v["passes_median"].items()}
            view = sum(x for k, x in est.items() if fam(k) != "radiosity")
            rad = sum(x for k, x in est.items() if fam(k) == "radiosity")
            pod = sum(v["passes_median"].values())
            print(f"  {cam:20s} pod {pod:6.3f} ms/frame -> M4 view {view:6.2f} + radiosity {rad:6.2f} = {view + rad:6.2f} ms   "
                  + " ".join(f"{k.split('.')[-1] if '.' in k else k}:{x:.2f}" for k, x in est.items()))


if __name__ == "__main__":
    main()
