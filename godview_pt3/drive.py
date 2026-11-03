"""godview_pt2 driver (pod): smoke / capture / bench / conv through web/twin/godview_pt3/judge.html with the repo's TwinPage.

    python3 web/twin/godview_pt3/drive.py smoke --out /workspace/godview_pt/smoke
    python3 web/twin/godview_pt3/drive.py capture --out /workspace/godview_pt/cap1 --spp 512 --query "gain=..&exposure=.."
    python3 web/twin/godview_pt3/drive.py bench --out /workspace/godview_pt/bench.json
Outputs: <out>/<view>__<state>__{traced,raster}.npy (float32 HxWx3 linear), PNG previews (Khronos PBR Neutral), summary.json.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from veridien.webtwin.headless import TwinPage  # noqa: E402

VIEWS = ["V1_surgeon", "V2_left", "V3_top", "V4_far"]


def say(s):
    print(s, flush=True)


def neutral(c):
    sc, des = 0.76, 0.15
    x = c.min(axis=-1, keepdims=True)
    off = np.where(x < 0.08, x - 6.25 * x * x, 0.04)
    c = c - off
    peak = c.max(axis=-1, keepdims=True)
    d = 1 - sc
    npk = 1 - d * d / (peak + d - sc)
    g = 1 - 1 / (des * (peak - npk) + 1)
    comp = c * (npk / np.maximum(peak, 1e-9))
    comp = comp * (1 - g) + npk * g
    return np.where(peak < sc, c, comp)


def png(path, lin, exposure):
    from PIL import Image
    t = neutral(np.clip(lin * exposure, 0, None))
    t = np.clip(t, 0, 1)
    s = np.where(t <= 0.0031308, 12.92 * t, 1.055 * t ** (1 / 2.4) - 0.055)
    Image.fromarray((255 * np.clip(s, 0, 1) + 0.5).astype(np.uint8)).save(path)


def states():
    p = Path(os.environ.get("VERIDIEN_REF_CACHE", "/workspace/refs")) / "clamped_states.json"
    d = json.loads(p.read_text())
    return {k: d[k]["state"] for k in ("default", "crossing") if k in d}


def cap(page, view, state, w, h, spp, mode):
    t0 = time.time()
    s = page.page.evaluate("async ({v, s, o}) => window.godJudge.capture(v, s, o)",
                           {"v": view, "s": state, "o": {"spp": spp, "mode": mode, "size": [w, h]}})
    a = np.frombuffer(base64.b64decode(s), dtype=np.float32).reshape(h, w, 3)
    return a, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--query", default="")
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--size", default="")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--states", default="default,crossing")
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--png", default="")
    ap.add_argument("--modes", default="traced,raster")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pngdir = Path(a.png) if a.png else out
    pngdir.mkdir(parents=True, exist_ok=True)
    path = "godview_pt3/judge.html" + ("?" + a.query if a.query else "")
    summ = {"query": a.query, "spp": a.spp}
    with TwinPage(harness=False, path=path, say=say) as page:
        info = page.page.evaluate("() => window.godJudge.info()")
        summ["info"] = info
        say(f"[godview_pt] info {json.dumps(info)}")
        W, H = info["image"]
        if a.size:
            W, H = map(int, a.size.split("x"))
        st = states()
        if a.cmd == "smoke":
            img, dt = cap(page, "V1_surgeon", st["default"], 320, 200, 1, "traced")
            say(f"[smoke] traced 320x200 1spp {dt:.2f}s mean {img.mean(axis=(0, 1))} nan {int(np.isnan(img).sum())}")
            img2, dt2 = cap(page, "V1_surgeon", st["default"], 320, 200, 16, "traced")
            say(f"[smoke] traced 16spp {dt2:.2f}s mean {img2.mean(axis=(0, 1))}")
            r, dt3 = cap(page, "V1_surgeon", st["default"], 320, 200, 1, "raster")
            say(f"[smoke] raster {dt3:.2f}s mean {r.mean(axis=(0, 1))} nan {int(np.isnan(r).sum())}")
            say(f"[smoke] debug {page.page.evaluate('async () => window.godJudge.debug()')}")
            e = a.exposure or 0.8 / max(float(np.percentile(img2.mean(axis=-1), 90)), 1e-9)
            png(pngdir / "smoke_traced.png", img2, e)
            png(pngdir / "smoke_raster.png", r, e)
            summ["smoke"] = {"traced_mean": img2.mean(axis=(0, 1)).tolist(), "raster_mean": r.mean(axis=(0, 1)).tolist(), "exposure": e}
        elif a.cmd == "stale":
            res = {}
            for v in a.views.split(","):
                res[v] = page.page.evaluate("async ({v, s}) => window.godJudge.staleProbe(v, s, [30, 90])", {"v": v, "s": st["default"]})
                say(f"[stale] {v} {json.dumps(res[v])}")
            summ["stale"] = res
        elif a.cmd == "capture":
            summ["captures"] = {}
            for v in a.views.split(","):
                for sn in a.states.split(","):
                    for mode in a.modes.split(","):
                        img, dt = cap(page, v, st[sn], W, H, a.spp, mode)
                        np.save(out / f"{v}__{sn}__{mode}.npy", img)
                        e = a.exposure or info.get("exposure") or 1.0
                        png(pngdir / f"{v}__{sn}__{mode}.png", img, e)
                        summ["captures"][f"{v}__{sn}__{mode}"] = {"seconds": dt, "mean": img.mean(axis=(0, 1)).tolist()}
                        say(f"[capture] {v} {sn} {mode} {W}x{H} spp {a.spp if mode == 'traced' else '-'}: {dt:.1f}s mean {img.mean(axis=(0, 1))}")
            summ["debug"] = page.page.evaluate("async () => window.godJudge.debug()")
        elif a.cmd == "shots":
            summ["shots"] = {}
            for v in a.views.split(","):
                for n in (0, 1, 4, 16, 64, 512):
                    if v != "V1_surgeon" and n not in (0, 512):
                        continue
                    t = page.page.evaluate("async ({v, s, n}) => window.godJudge.live(v, s, n)", {"v": v, "s": st["default"], "n": n})
                    page.page.locator("#view").screenshot(path=str(pngdir / f"live_{v}_{n:03d}spp.png"))
                    summ["shots"][f"{v}_{n}"] = t
                    say(f"[shots] {v} {n} spp: {json.dumps(t)}")
        elif a.cmd == "bench":
            page.page.set_viewport_size({"width": 1600, "height": 1000})
            r = page.page.evaluate("async () => window.godJudge.bench(60)")
            summ["bench"] = r
            say(f"[bench] {json.dumps(r)}")
        elif a.cmd == "conv":
            r = page.page.evaluate("async ({s, w, h}) => window.godJudge.convergence('V1_surgeon', s, [1, 2, 4, 8, 16, 32, 64, 128, 256], 2048, [w, h])",
                                   {"s": st["default"], "w": W, "h": H})
            summ["conv"] = r
            say(f"[conv] {json.dumps(r)}")
        summ["console_tail"] = page.console[-30:]
    (out / f"summary_{a.cmd}.json").write_text(json.dumps(summ, indent=1, default=str))
    if pngdir != out:
        (pngdir / f"summary_{a.cmd}.json").write_text(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
