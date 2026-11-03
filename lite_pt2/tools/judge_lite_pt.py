"""Judge the lite_pt2 candidate (web/twin/lite_pt2/judge.html) with the shared Mitsuba judge (judges.judge_mitsuba) and
measure its GPU ms per frame. Run on the pod under gpu_guard:

  cd /root/twin && VERIDIEN_GPU_HEADLESS=1 VERIDIEN_REF_CACHE=/workspace/refs python3 -m veridien.webtwin.gpu_guard run \
      --timeout 1500 -- python3 web/twin/lite_pt2/tools/judge_lite_pt.py --query factor=2 --out runs/webtwin/race/lite_pt/f2

The cached 4096-spp references are linked (never written) from $VERIDIEN_REF_CACHE into runs/webtwin/cache, where
judge_mitsuba looks. --smoke renders one state x CAM1 at the default only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))


def link_refs():
    src = Path(os.environ.get("VERIDIEN_REF_CACHE") or "/workspace/refs")
    dst = REPO / "runs" / "webtwin" / "cache"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.glob("ref_*.npz"):
        t = dst / f.name
        if not t.exists():
            t.symlink_to(f)
            n += 1
    return n


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="factor=2")
    ap.add_argument("--out", default="runs/webtwin/fix/lite_pt2/judge")
    ap.add_argument("--states", default="")
    ap.add_argument("--cameras", default="CAM1,CAM2,CAM3")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--bench", type=int, default=40)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--scope-png", action="store_true")
    a = ap.parse_args()
    out = (REPO / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[lite_pt2] linked {link_refs()} refs; query {a.query}; commit {git_commit()}", flush=True)
    from veridien.webtwin import judges
    from veridien.webtwin.headless import TwinPage

    try:
        import matplotlib  # noqa: F401
    except ImportError:          # the pod has no matplotlib: skip the per-image error figures (numbers are unaffected)
        judges.error_figure = lambda *a, **k: None

    scene_path = REPO / "web" / "twin" / "assets" / "scene.json"
    physics_path = REPO / "configs" / "twin" / "physics.yaml"
    res = {"query": a.query, "commit": git_commit(), "seed": 1}
    t0 = time.time()
    page = TwinPage(path="lite_pt2/judge.html?" + a.query, harness=False)
    with page:
        res["adapter"] = page.adapter
        res["info"] = page.evaluate("() => window.__judgePack(window.twinJudge.info())")
        print("[lite_pt2] info", json.dumps(res["info"])[:400], flush=True)
        if a.bench:
            b = {}
            for cam, mv in [("SCOPE", True), ("SCOPE", "scope"), ("CAM1", True)]:
                r = page.evaluate("async ({c, n, mv}) => window.__judgePack(await window.twinJudge.bench(c, n, {moving: mv}))", {"c": cam, "n": a.bench, "mv": mv})
                cam = cam + ("" if mv is True else "_scope_moving")
                tot = [f["total"] for f in r["frames"]][5:]
                passes = {}
                for f in r["frames"][5:]:
                    for k, v in f["passes"].items():
                        passes.setdefault(k, []).append(v)
                b[cam] = {"gpu_ms_median": float(np.median(tot)) if tot else None, "gpu_ms_mean": float(np.mean(tot)) if tot else None, "gpu_ms_p95": float(np.percentile(tot, 95)) if tot else None,
                          "wall_ms": r["wallMs"], "passes_median": {k: float(np.sum(v) / len(tot)) for k, v in passes.items()},
                          "fallback_samples": r["fallback"], "timestamp_query": r["gpu"], "n": len(tot), "rates": r.get("rates")}
                print(f"[lite_pt2] bench {cam}: GPU {b[cam]['gpu_ms_median']} ms median, mean {b[cam]['gpu_ms_mean']}, p95 {b[cam]['gpu_ms_p95']}, wall {r['wallMs']:.2f} ms; "
                      f"passes {json.dumps(b[cam]['passes_median'])}", flush=True)
            res["bench"] = b
            (out / "bench.json").write_text(json.dumps(res, indent=1, default=str))
        if a.scope_png or True:
            st = judges.mitsuba_states(judges.TwinScene.load(scene_path)) if hasattr(judges, "TwinScene") else None
            r = page.render({}, cameras=["SCOPE", "CAM1", "CAM2", "CAM3"], linear=False, rgb8=True, seed=1)
            for c in ["SCOPE", "CAM1", "CAM2", "CAM3"]:
                rgb = np.asarray(r[c]["rgb8"]).reshape(r[c]["height"], r[c]["width"], 3)
                np.save(out / f"default_{c}_rgb8.npy", rgb)
        try:
            sc = page.evaluate("async () => window.__judgePack(await window.twinJudge.scopeCheck({}, 1))")
            res["scope_check_vs_inner_tracer"] = sc
            print("[lite_pt2] scope check vs full-res tracer", json.dumps(sc), flush=True)
        except Exception as e:  # noqa: BLE001
            print("[lite_pt2] scope check failed", e, flush=True)
        if not a.no_judge:
            states = [s for s in a.states.split(",") if s] or None
            jr = judges.judge_mitsuba(page, scene_path, physics_path, out, spp=4096, passes=64, seed=1,
                                      cameras=a.cameras.split(","), states=states, twin_repeat=a.repeat, noise_frames=0,
                                      spectral=False, depth=False)
            res["judge"] = jr
    res["seconds"] = time.time() - t0
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    if "judge" in res:
        summarize(res["judge"], out)


def summarize(jr, out):
    rows = []
    for sn, s in jr["states"].items():
        for cam, c in s["cameras"].items():
            for kind in ("live", "twin_mean"):
                o = c[kind]["overall"]
                reg = c[kind].get("regions", {})
                e = (c[kind].get("edges") or {}).get("all", {})
                rows.append({"state": sn, "cam": cam, "kind": kind, "level": o.get("level_ratio"), "median": o.get("median_rel"),
                             "p95": o.get("p95_rel"), "dome": (reg.get("dome_interior") or {}).get("median_rel"),
                             "bench": (reg.get("bench") or {}).get("median_rel"), "edges": e.get("median_abs_px")})
    (out / "summary.json").write_text(json.dumps({"rows": rows, "pooled": jr.get("pooled"), "verdict": jr.get("verdict")}, indent=1, default=str))
    for r in rows:
        print("[lite_pt2] {state:16s} {cam} {kind:9s} level {level} median {median} p95 {p95} dome {dome} bench {bench} edges {edges}".format(**{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}))
    print("[lite_pt2] pooled", json.dumps(jr.get("pooled"), default=str))
    print("[lite_pt2] verdict", json.dumps(jr.get("verdict"), default=str))


if __name__ == "__main__":
    main()
