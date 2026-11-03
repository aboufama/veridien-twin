"""lite_rt judge driver (pod): capture every cached judge state x CAM1-3 (+ SCOPE for the contact sheet) through
web/twin/lite_rt/judge.html with the repo's own TwinPage/twin_frames, in the capture layout judges.py expects, then
speed. Compare with: scripts/judge_web_twin.py --compare <out> (cached Mitsuba refs, $VERIDIEN_REF_CACHE).

    python3 web/twin/lite_rt/drive.py capture --out runs/webtwin/race/lite_rt/cap_acc --query "" --warmup 12 --frames 4
    python3 web/twin/lite_rt/drive.py speed --out runs/webtwin/race/lite_rt/speed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from veridien.webtwin import judges as J  # noqa: E402
from veridien.webtwin.headless import TwinPage  # noqa: E402

CAMS = ["CAM1", "CAM2", "CAM3"]


def say(s):
    print(s, flush=True)


def frames(page, st0, st1, cams, seeds, keep=True, accumulate=None):
    imgs = {c: [] for c in cams}
    for sd in seeds:
        o = {"seed": int(sd), "rgb8": False}
        if accumulate is not None:
            o["accumulate"] = accumulate
        if not keep:
            page.page.evaluate("async ({s, o}) => { await window.twinJudge.render(s, o); return 1; }",
                               {"s": st0, "o": {"cameras": cams, "linear": False, "state1": st1, **o}})
            continue
        r = page.render(st0, cameras=cams, linear=True, state1=st1, **o)
        for c in cams:
            lin = np.asarray(r[c]["linear"])
            W = int(r[c]["width"])
            n = lin.size // 4
            imgs[c].append(lin.reshape(n // W, W, 4)[..., :3].astype(np.float32))
    return imgs


def capture(a):
    out = Path(a.out)
    (out / "cull1").mkdir(parents=True, exist_ok=True)
    states = json.loads((Path(os.environ.get("VERIDIEN_REF_CACHE", "/workspace/refs")) / "clamped_states.json").read_text())
    if a.states:
        states = {k: v for k, v in states.items() if k in a.states.split(",")}
    path = "lite_rt/judge.html" + (("?" + a.query) if a.query else "")
    meta = {"source": path, "query": a.query, "frames": a.frames, "warmup": a.warmup, "states": {}}
    with TwinPage(harness=False, path=path, say=say) as page:
        meta["adapter"] = page.adapter
        meta["info"] = page.evaluate("() => window.twinJudge.info()")
        say(f"[lite_rt] info {json.dumps(meta['info'], default=str)[:600]}")
        for name, v in states.items():
            cams = CAMS + (["SCOPE"] if a.scope and name == "default" else [])
            st0, st1 = v["state"], v.get("state1")
            t0 = time.time()
            frames(page, st0, st1, cams, [a.seed + 100 + k for k in range(a.warmup)], keep=False)
            tw = frames(page, st0, st1, cams, [a.seed + 7919 * k for k in range(a.frames)])
            np.savez_compressed(out / "cull1" / f"{name}.npz", **{c: np.stack(tw[c]) for c in CAMS})
            if "SCOPE" in cams:
                np.save(out / f"SCOPE_{name}.npy", tw["SCOPE"][0])
            meta["states"][name] = {"state": st0, "state1": st1, "seconds": time.time() - t0}
            say(f"[capture] {name}: {a.frames} frames x {len(cams)} cameras in {time.time() - t0:.1f} s")
        meta["console_tail"] = page.console[-30:]
    J.write_json(out / "cull1" / "capture.json", meta)
    J.write_json(out / "capture_summary.json", meta)


def speed(a):
    path = "lite_rt/judge.html" + (("?" + a.query) if a.query else "")
    res = {}
    with TwinPage(harness=False, path=path, say=say) as page:
        res["adapter"] = page.adapter
        res["info"] = page.evaluate("() => window.twinJudge.info()")
        for label, name, mv in (("scope_moving", "SCOPE", True), ("scope_still", "SCOPE", False), ("cam1_moving", "CAM1", True),
                                ("cams3_moving", ["CAM1", "CAM2", "CAM3"], True)):
            r = page.evaluate("async ({n, s, m}) => window.__judgePack(await window.twinJudge.speed(n, s, {moving: m}))",
                              {"n": name, "s": a.seconds, "m": mv})
            fr = r["frames"]
            keys = sorted({k for f in fr for k in f if k not in ("frame", "submits", "maxSubmit")})
            per = {k: float(np.median([f.get(k, 0.0) for f in fr])) for k in keys}
            tot = [sum(v for k, v in f.items() if k not in ("frame", "submits", "maxSubmit")) for f in fr]
            res[label] = {"n_frames": r["n"], "gpu_ms_median_by_pass": per, "gpu_ms_total_median": float(np.median(tot)) if tot else None,
                          "gpu_ms_total_p95": float(np.percentile(tot, 95)) if tot else None,
                          "wall_ms_median": float(np.median(r["wall"])), "max_submit_ms": float(max([f.get("maxSubmit", 0) for f in fr] or [0]))}
            say(f"[speed] {label}: {json.dumps(res[label])}")
        res["console_tail"] = page.console[-20:]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    J.write_json(Path(a.out), res)


def shot(a):
    """The display chain (present) on the SCOPE: tools moving (per-pixel short history) and then still."""
    path = "lite_rt/judge.html" + (("?" + a.query) if a.query else "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with TwinPage(harness=False, path=path, say=say) as page:
        page.page.evaluate("() => { const c = document.getElementById('view'); c.style.width = '1280px'; c.style.height = '720px'; }")
        r = page.evaluate("async () => window.__judgePack(await window.twinJudge.speed('SCOPE', 2, {moving: true, present: true}))")
        page.page.locator("#view").screenshot(path=str(out / "scope_moving.png"))
        r2 = page.evaluate("async () => window.__judgePack(await window.twinJudge.speed('SCOPE', 1.5, {moving: false, present: true}))")
        page.page.locator("#view").screenshot(path=str(out / "scope_still.png"))
        for k, rr in (("moving", r), ("still", r2)):
            fr = rr["frames"]
            tot = [sum(v for kk, v in f.items() if kk not in ("frame", "submits", "maxSubmit")) for f in fr]
            say(f"[shot] {k}: {rr['n']} frames, GPU ms median {np.median(tot) if tot else None}, wall median {np.median(rr['wall']):.2f}")
        say("\n".join(page.console[-8:]))


def contact(a):
    """Mitsuba | lite_rt, CAM1-3 default state (+ SCOPE, lite_rt only), same exposure per row (the reference's p99)."""
    from PIL import Image, ImageDraw
    from veridien.webtwin.mitsuba_ref import Physics, Reference, TwinScene

    cap = Path(a.out)
    scene, phys = TwinScene.load(J.DEFAULT_SCENE), Physics.load(J.DEFAULT_PHYSICS)
    ref = Reference(scene, phys, max_depth=1024, rr_depth=16)
    meta = json.loads((cap / "cull1" / "capture.json").read_text())
    st0 = meta["states"]["default"]["state"]
    p0 = J.poses_for_state(scene, st0, say)[0]
    z = np.load(cap / "cull1" / "default.npz")
    oetf = lambda x: np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - 0.055)  # noqa: E731
    tm = lambda img, g: (255 * np.clip(oetf(np.clip(img * g, 0, 1)), 0, 1)).astype(np.uint8)  # noqa: E731
    rows = []
    for cam in CAMS:
        r = J.reference_cached(ref, cam, {"name": "default", **st0}, p0, None, 4096, 64, 1, None, say)
        Y = J.rgb_to_luminance(r.mean)
        g = 0.8 / max(float(np.percentile(Y, 99)), 1e-30)
        rows.append((cam, tm(r.mean, g), tm(z[cam][0], g)))
    sc = cap / "SCOPE_default.npy"
    W, H = 640, 360
    sheet = Image.new("RGB", (2 * W + 30, (len(rows) + 1) * (H + 22) + 10), (246, 244, 239))
    d = ImageDraw.Draw(sheet)
    for i, (cam, a_, b_) in enumerate(rows):
        y = 10 + i * (H + 22)
        sheet.paste(Image.fromarray(a_).resize((W, H), Image.BILINEAR), (10, y + 16))
        sheet.paste(Image.fromarray(b_).resize((W, H), Image.BILINEAR), (20 + W, y + 16))
        d.text((10, y), f"{cam} default  Mitsuba 4096 spp", fill=(30, 30, 30))
        d.text((20 + W, y), f"{cam} default  lite_rt (judge capture frame 0)", fill=(30, 30, 30))
    if sc.is_file():
        s = np.load(sc)
        Y = J.rgb_to_luminance(s)
        g = 0.8 / max(float(np.percentile(Y[Y > 0], 99)), 1e-30) if (Y > 0).any() else 1.0
        y = 10 + len(rows) * (H + 22)
        sheet.paste(Image.fromarray(tm(s, g)).resize((W, H), Image.BILINEAR), (20 + W, y + 16))
        d.text((10, y + 16), "SCOPE: no Mitsuba reference\n(the judge refs cover CAM1-3)", fill=(30, 30, 30))
        d.text((20 + W, y), "SCOPE default  lite_rt", fill=(30, 30, 30))
    sheet.save(cap / "contact.png")
    say(f"[contact] {cap / 'contact.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["capture", "speed", "contact", "shot"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--query", default="")
    ap.add_argument("--states")
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--scope", action="store_true")
    a = ap.parse_args()
    {"capture": capture, "speed": speed, "contact": contact, "shot": shot}[a.cmd](a)
