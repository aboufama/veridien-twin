"""Capture the three.js god's-eye candidate on a pod (headless Chromium, hardware WebGPU) for every viewpoint x state.

    cd /root/twin && VERIDIEN_GPU_HEADLESS=1 python3 -m veridien.webtwin.gpu_guard run --timeout 600 -- \
        python3 web/twin/godview_three/run_capture.py --out /workspace/godview_three/<tag> [--size 320x200] [--bench]

Writes <out>/<viewpoint>__<state>.npy (float32 h x w x 3 linear radiance), <out>/png/<viewpoint>__<state>.png (the
tone-mapped display frame, a screenshot of the page canvas), and <out>/summary.json (load time, GPU ms, sources).
States come from /workspace/refs/clamped_states.json (the reference agent's clamped judge states) when present.
"""
from __future__ import annotations

import argparse
import base64
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
POD_FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--use-angle=vulkan",
             "--use-vulkan=native", "--ignore-gpu-blocklist", "--disable-vulkan-surface", "--enable-gpu",
             "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--no-first-run",
             "--enable-dawn-features=allow_unsafe_apis"]


def neutral(x: np.ndarray) -> np.ndarray:
    """Khronos PBR Neutral tone map (linear in, linear out), then sRGB 8-bit."""
    start, desat = 0.8 - 0.04, 0.15
    x = np.maximum(x, 0).astype(np.float64)
    m = x.min(-1, keepdims=True)
    off = np.where(m < 0.08, m - 6.25 * m * m, 0.04)
    x = x - off
    peak = x.max(-1, keepdims=True)
    d = 1 - start
    newpeak = 1 - d * d / (peak + d - start)
    y = np.where(peak < start, x, x * newpeak / np.maximum(peak, 1e-9))
    g = 1 - 1 / (desat * (peak - newpeak) + 1)
    y = np.where(peak < start, y, y * (1 - g) + newpeak * g)
    y = np.clip(y, 0, 1)
    s = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (s * 255 + 0.5).astype(np.uint8)


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default=None, help="WxH override (small-first runs)")
    ap.add_argument("--states", default="/workspace/godrefs/states.json")
    ap.add_argument("--query", default="")
    ap.add_argument("--state-names", default=None, help="comma list; default godview.yaml states")
    ap.add_argument("--views", default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--no-npy", action="store_true")
    a = ap.parse_args()
    import yaml

    out = Path(a.out); (out / "png").mkdir(parents=True, exist_ok=True)
    harness = REPO / "runs" / "webtwin" / "harness"; harness.mkdir(parents=True, exist_ok=True)
    (harness / "physics.json").write_text(json.dumps(yaml.safe_load((REPO / "configs/twin/physics.yaml").read_text())))
    gv = yaml.safe_load((REPO / "configs/twin/godview.yaml").read_text())
    views = a.views.split(",") if a.views else list(gv["viewpoints"])
    names = a.state_names.split(",") if a.state_names else list(gv.get("states", ["default"]))
    states = {}
    if Path(a.states).is_file():
        d = json.loads(Path(a.states).read_text())
        d = d.get("states", d)
        states = {k: (v.get("state", v) if isinstance(v, dict) else v) for k, v in d.items()}
    size = [int(x) for x in a.size.split("x")] if a.size else gv.get("image", [1280, 800])

    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(REPO)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from playwright.sync_api import sync_playwright
    summary = {"size": size, "views": views, "states": names, "captures": {}}
    console = []
    try:
        time.sleep(0.5)
        with sync_playwright() as pw:
            br = pw.chromium.launch(headless=True, args=POD_FLAGS)
            try:
                pg = br.new_page(viewport={"width": max(1460, size[0] + 20), "height": max(920, size[1] + 20)})
                pg.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
                pg.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
                t0 = time.time()
                pg.goto(f"http://127.0.0.1:{port}/web/twin/godview_three/judge.html{a.query}", wait_until="domcontentloaded", timeout=60000)
                pg.wait_for_function("window.godJudge !== undefined", timeout=60000)
                pg.evaluate("window.godJudge.ready")
                summary["load_s_wall"] = time.time() - t0
                summary["info"] = pg.evaluate("window.godJudge.info()")
                print("ready", json.dumps(summary["info"])[:400], flush=True)
                for v in views:
                    for s in names:
                        st = states.get(s)
                        t1 = time.time()
                        b = pg.evaluate("([v, s, o]) => window.godJudge.capture(v, s, o)", [v, st, {"size": size, "ss": a.ss}])
                        arr = np.frombuffer(base64.b64decode(b), dtype="<f4").reshape(size[1], size[0], 3)
                        if not a.no_npy:
                            np.save(out / f"{v}__{s}.npy", arr)
                        from PIL import Image
                        Image.fromarray(neutral(arr * float(summary["info"]["exposure"]))).save(out / "png" / f"lin_{v}__{s}.png")
                        pg.evaluate("([v, s, z]) => window.godJudge.show(v, s, z)", [v, st, size])
                        pg.locator("#c").screenshot(path=str(out / "png" / f"{v}__{s}.png"))
                        summary["captures"][f"{v}__{s}"] = {"s": time.time() - t1, "mean": float(arr.mean()),
                                                           "max": float(arr.max()), "finite": bool(np.isfinite(arr).all())}
                        print(v, s, summary["captures"][f"{v}__{s}"], flush=True)
                if a.bench:
                    summary["bench_1440x900"] = pg.evaluate("window.godJudge.bench('V1_surgeon', [1440, 900], 240)")
                    print("bench", summary["bench_1440x900"], flush=True)
                    summary["bench_1440x900_orbit"] = pg.evaluate("window.godJudge.bench('V1_surgeon', [1440, 900], 240, true)")
                    print("bench orbit", summary["bench_1440x900_orbit"], flush=True)
            finally:
                br.close()
    finally:
        srv.terminate()
        summary["console"] = console[-80:]
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    errs = [c for c in console if c.startswith(("error", "pageerror"))]
    if errs:
        print("\n".join(errs[:20]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
