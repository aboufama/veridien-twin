"""Smoke: open lite_nodes/judge.html?<query> headless for each query given, render the clamped default state on CAM1
and print level / median ratio against its cached Mitsuba reference (quick diagnostics; the real judge is
judge_lite_nodes.py)."""
import json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
# the outer `gpu_guard run` already holds the machine-wide GPU lock for this whole process tree; TwinPage would wait on
# it forever from this child process, so the page takes a private, nested lock instead
os.environ.setdefault("VERIDIEN_GPU_LOCK", "/tmp/veridien_lite_nodes_inner.lock")
import numpy as np
from veridien.webtwin.headless import TwinPage

cache = Path(os.environ.get("VERIDIEN_REF_CACHE", "/workspace/refs"))
st = json.loads((cache / "clamped_states.json").read_text())["default"]["state"]
refs = {c: np.load(next(cache.glob(f"ref_*_060ce8568ccb_{c}_4096_*.npz")))["mean"].astype(np.float64) for c in ("CAM1", "CAM2", "CAM3")}
Y = lambda a: a[..., 0] * .2126 + a[..., 1] * .7152 + a[..., 2] * .0722
for q in (sys.argv[1:] or [""]):
    page = TwinPage(path="lite_nodes/judge.html" + ("?" + q if q else ""), harness=False, ready_timeout=240, say=lambda *a: None)
    try:
        page.start()
        r = page.render(st, cameras=["CAM1", "CAM2", "CAM3"], linear=True)
        for c in ("CAM1", "CAM2", "CAM3"):
            t = np.asarray(r[c]["linear"]).reshape(int(r[c]["height"]), int(r[c]["width"]), 4)[..., :3].astype(np.float64)
            Yt, Yr = Y(t), Y(refs[c])
            lit = Yr > 0.02 * np.percentile(Yr, 99)
            rel = (Yt[lit] - Yr[lit]) / Yr[lit]
            print(f"{q or 'base'} {c}: level {Yt[lit].sum() / Yr[lit].sum():.4f} median|rel| {np.median(np.abs(rel)):.4f} p95 {np.percentile(np.abs(rel), 95):.4f} signed {np.median(rel):+.4f} mean twin {Yt.mean():.3f} ref {Yr.mean():.3f}", flush=True)
            np.save(f"/root/smoke_{q.replace('=', '').replace('&', '_') or 'base'}_{c}.npy", t.astype(np.float32))
    finally:
        errs = [l for l in page.console if "error" in l.lower()]
        print("\n".join(errs[-10:]))
        page.close()
