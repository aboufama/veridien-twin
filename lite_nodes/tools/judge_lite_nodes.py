"""Judge for web/twin/lite_nodes (docs/WEB_TWIN.md §18.3), run on the pod under gpu_guard:

  cd /root/twin && VERIDIEN_GPU_HEADLESS=1 VERIDIEN_REF_CACHE=/workspace/refs python3 -m veridien.webtwin.gpu_guard run \
     --timeout 1500 -- python3 web/twin/lite_nodes/tools/judge_lite_nodes.py --out runs/webtwin/race/lite_nodes/<tag>

1. capture: the clamped judge states (refs/clamped_states.json, exactly the states the cached Mitsuba references were
   rendered for) x CAM1-3 through lite_nodes/judge.html in headless Chromium, one live frame each (the renderer is
   deterministic), plus SCOPE at the default state and the speed bench (GPU timestamps, wall times, load time);
2. compare: judges.judge_mitsuba_offline against the cached 4096-spp references (never re-rendered: a missing
   reference is an error);
3. contact.png: Mitsuba | lite_nodes for CAM1-3 at the default state, and the SCOPE display image.
Prints seed, data hash and git commit (judges.header)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
# the outer `gpu_guard run` already holds the machine-wide GPU lock for this whole process tree; TwinPage would wait on
# it forever from this child process, so the page takes a private, nested lock instead
os.environ.setdefault("VERIDIEN_GPU_LOCK", "/tmp/veridien_lite_nodes_inner.lock")

from veridien.webtwin import judges as J  # noqa: E402

SCENE = REPO / "web" / "twin" / "assets" / "scene.json"
PHYS = REPO / "configs" / "twin" / "physics.yaml"


def say(*a):
    print(*a, flush=True)


def capture(out: Path, query: str, cams: list[str], bench: bool) -> dict:
    from veridien.webtwin.headless import TwinPage

    cache = Path(os.environ.get("VERIDIEN_REF_CACHE", "/workspace/refs"))
    clamped = json.loads((cache / "clamped_states.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    page = TwinPage(path="lite_nodes/judge.html" + (("?" + query) if query else ""), harness=False, say=say)
    meta = {"source": "lite_nodes/judge.html", "query": query, "frames": 1, "warmup": 0, "states": {}}
    t0 = time.time()
    with page:
        meta["adapter"] = page.adapter
        meta["info"] = page.evaluate("() => window.twinJudge.info()")
        say(f"[capture] page ready in {time.time() - t0:.1f} s: {meta['info']}")
        for name, v in clamped.items():
            st0, st1 = v["state"], v.get("state1")
            t = time.time()
            r = page.render(st0, cameras=cams, linear=True, state1=st1, seed=1)
            arrs = {}
            for c in cams:
                W, H = int(r[c]["width"]), int(r[c]["height"])
                arrs[c] = np.asarray(r[c]["linear"], np.float32).reshape(1, H, W, 4)[..., :3]
            np.savez_compressed(out / f"{name}.npz", **arrs)
            meta["states"][name] = {"state": st0, "state1": st1, "seconds": time.time() - t}
            say(f"[capture] {name}: {len(cams)} cameras in {time.time() - t:.1f} s; mean {[float(arrs[c].mean()) for c in cams]}")
        d = clamped.get("default") or next(iter(clamped.values()))
        r = page.render(d["state"], cameras=["SCOPE", *cams], linear=True, rgb8=True, seed=1)
        extra = {}
        for c in ["SCOPE", *cams]:
            W, H = int(r[c]["width"]), int(r[c]["height"])
            extra[f"{c}_rgb8"] = np.asarray(r[c]["rgb8"], np.uint8).reshape(H, W, 3)
            extra[f"{c}_linear"] = np.asarray(r[c]["linear"], np.float32).reshape(H, W, 4)[..., :3]
        np.savez_compressed(out / "default_views.npz", **extra)
        if bench:
            meta["bench"] = page.evaluate("async () => window.__judgePack(await window.twinJudge.bench({cameras: ['SCOPE', 'CAM1', 'CAM2', 'CAM3'], frames: 60}))")
            say("[bench] " + json.dumps(meta["bench"], default=str)[:3000])
        meta["console_tail"] = page.console[-40:]
    J.write_json(out / "capture.json", meta)
    return meta


def compare(cap: Path, out: Path) -> dict:
    orig = J.reference_cached

    def cached_only(ref, camera, state, poses0, poses1, spp, passes, seed, cache_dir=None, say=print):
        cache_dir = cache_dir or J.ref_cache_dir()
        key = (f"{ref.scene.sha()}_{ref.physics.sha()}_{J.poses_key(poses0, poses1)}_{camera}_{spp}_{seed}_{ref.variant}"
               f"_d{ref.max_depth}")
        if not (cache_dir / f"ref_{key}.npz").is_file():
            raise RuntimeError(f"reference not cached (refusing to render): ref_{key}.npz")
        return orig(ref, camera, state, poses0, poses1, spp, passes, seed, cache_dir, say)

    J.reference_cached = cached_only
    try:
        import matplotlib  # noqa: F401
    except ImportError:            # the pod has no matplotlib: no error figures, the statistics are unchanged
        J.error_figure = lambda *a, **k: None
    res = J.judge_mitsuba_offline({"lite_nodes": cap}, SCENE, PHYS, out, spp=4096, seed=1, noise=False, say=say)
    J.write_json(out / "mitsuba.json", res)
    return res


def summarize(res: dict) -> dict:
    rows, reg = [], {}
    for sname, s in res["states"].items():
        for cam, c in s["cameras"].items():
            v = c["lite_nodes"]["live"]
            o = v["overall"]
            rows.append({"state": sname, "camera": cam, "level": o.get("level_ratio"), "median": o.get("median_rel"),
                         "p95": o.get("p95_rel"), "edges_px": ((v.get("edges") or {}).get("all") or {}).get("median_abs_px")})
            for rn, rv in (v.get("regions") or {}).items():
                if rv.get("median_rel") is not None:
                    reg.setdefault(rn, []).append(rv["median_rel"])
    pooled = res["pooled"]["lite_nodes"]
    out = {"pooled": {"median_abs_rel": pooled["live"]["median_abs_rel"], "p95_abs_rel": pooled["live"]["p95_abs_rel"],
                      "level_min_max": [min(pooled["live"]["level_ratio_images"]), max(pooled["live"]["level_ratio_images"])],
                      "edges": pooled.get("edges")},
           "regions_median_of_medians": {k: float(np.median(v)) for k, v in reg.items()}, "per_image": rows}
    return out


def contact(cap: Path, out_png: Path):
    import cv2
    from veridien.webtwin.mitsuba_ref import Physics, Reference, TwinScene

    scene, phys = TwinScene.load(SCENE), Physics.load(PHYS)
    ref = Reference(scene, phys, max_depth=1024, rr_depth=16)
    meta = json.loads((cap / "capture.json").read_text())
    sname = "default" if "default" in meta["states"] else next(iter(meta["states"]))
    st0 = meta["states"][sname]["state"]
    p0 = J.poses_for_state(scene, st0, say)[0]
    z = np.load(cap / f"{sname}.npz")
    dv = np.load(cap / "default_views.npz")
    tiles = []

    def tm(x, s):
        y = np.clip(x * s, 0, 1)
        y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
        return (np.clip(y, 0, 1) * 255).astype(np.uint8)

    for cam in ["CAM1", "CAM2", "CAM3"]:
        r = J.reference_cached(ref, cam, {"name": sname, **st0}, p0, None, 4096, 64, 1, None, say)
        Yr = J.rgb_to_luminance(r.mean)
        s = 0.8 / max(np.percentile(Yr, 99), 1e-30)
        a = np.ascontiguousarray(tm(r.mean, s)[..., ::-1])
        b = np.ascontiguousarray(tm(z[cam][0], s)[..., ::-1])
        for im, lab in ((a, f"Mitsuba 4096 spp {cam}"), (b, f"lite_nodes {cam}")):
            cv2.putText(im, lab, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.concatenate([a, b], 1))
    sc = np.ascontiguousarray(dv["SCOPE_rgb8"][..., ::-1])
    cv2.putText(sc, "lite_nodes SCOPE (display look)", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    pad = np.zeros_like(sc)
    lin = dv["SCOPE_linear"]
    s = 0.8 / max(np.percentile(J.rgb_to_luminance(lin), 99), 1e-30)
    sl = np.ascontiguousarray(tm(lin, s)[..., ::-1])
    cv2.putText(sl, "lite_nodes SCOPE (linear, same tone map)", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(np.concatenate([sc, sl], 1))
    img = np.concatenate(tiles, 0)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    say(f"[contact] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--query", default="")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--skip-capture", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--contact-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h = J.header("lite_nodes", 1, SCENE, PHYS, say)
    cap = out / "capture"
    if a.contact_only:
        contact(cap, out / "contact.png")
        return
    if not a.skip_capture:
        capture(cap, a.query, ["CAM1", "CAM2", "CAM3"], not a.no_bench)
    if a.no_compare:
        return
    res = compare(cap, out / "compare")
    sm = summarize(res)
    sm["header"] = h
    J.write_json(out / "summary.json", sm)
    say("[summary] " + json.dumps({k: v for k, v in sm.items() if k != "per_image"}, default=float))
    for r in sm["per_image"]:
        say(f"  {r['state']:16s} {r['camera']}: level {r['level']:.4f} median {r['median']:.4f} p95 {r['p95']:.4f} edges {r['edges_px']}")
    contact(cap, out / "contact.png")


if __name__ == "__main__":
    main()
