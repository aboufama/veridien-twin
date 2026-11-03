"""Renderer judges and benchmark in headless Brave on the Mac's Metal GPU (not SwiftShader).

Always under the machine-wide GPU lock with a hard timeout (docs/WEB_TWIN.md §12), small first:
    .venv/bin/python -m veridien.webtwin.gpu_guard run --timeout 300 -- .venv/bin/python web/twin/tools/gpu_bench.py --no-lock --scale 0.25 --cams CAM1 --spp 1
(without gpu_guard, the script takes the lock itself; it always closes the browser in a finally).

    .venv/bin/python web/twin/tools/gpu_bench.py --test             # WGSL vs JS, GPU vs CPU BVH, furnace, disc emitter
    .venv/bin/python web/twin/tools/gpu_bench.py                    # per-pass GPU ms on the real scene (3 cameras + overview)
    .venv/bin/python web/twin/tools/gpu_bench.py --noise            # render noise vs modelled sensor noise
    .venv/bin/python web/twin/tools/gpu_bench.py --images --out DIR # also write the frames as PNG

Serves the worktree on 127.0.0.1 (a secure context, so WebGPU works), converts configs/twin/physics.yaml to JSON at
/__physics.json, and drives web/twin/tests/gpu_page.html with Playwright (Apache-2.0) using the installed Brave
with a fresh temporary profile. Prints the adapter, the git commit and the scene provenance with every result.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import http.server
import json
import mimetypes
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BRAVE = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] == "/__physics.json":
            data = json.dumps(yaml.safe_load((ROOT / "configs/twin/physics.yaml").read_text())).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


def serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_page(query: str, timeout_s: float, headless: bool = True, no_lock: bool = False) -> tuple[dict, list[str]]:
    from playwright.sync_api import sync_playwright

    from veridien.webtwin.gpu_guard import gpu_lock

    srv, port = serve()
    logs: list[str] = []
    # docs/WEB_TWIN.md §12: one GPU process machine-wide (the lock, held for the browser's whole life, unless the
    # caller already holds it through `gpu_guard run`), a hard timeout, and the browser closed in a finally
    lock = contextlib.nullcontext() if no_lock else gpu_lock(say=lambda m: print(m, file=sys.stderr))
    try:
        with lock, sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                tempfile.mkdtemp(prefix="twin-gpu-"), executable_path=BRAVE, headless=headless,
                args=["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal",
                      "--disable-background-timer-throttling", "--disable-renderer-backgrounding"])
            try:
                page = ctx.new_page()
                page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
                page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
                page.goto(f"http://127.0.0.1:{port}/web/twin/tests/gpu_page.html?{query}")
                page.wait_for_function("window.gpuResult !== undefined", timeout=timeout_s * 1000)
                res = page.evaluate("window.gpuResult")
            finally:
                ctx.close()
    finally:
        srv.shutdown()
    return res, logs


def write_images(res: dict, out: Path) -> None:
    import cv2
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    for name, im in (res.get("bench", {}).get("images") or {}).items():
        w, h = im["w"], im["h"]
        if "rgb8_b64" in im:
            a = np.frombuffer(base64.b64decode(im["rgb8_b64"]), np.uint8).reshape(h, w, 3)
            cv2.imwrite(str(out / f"{name}_rgb8.png"), a[:, :, ::-1])
        if "rgba8_b64" in im:
            a = np.frombuffer(base64.b64decode(im["rgba8_b64"]), np.uint8).reshape(h, w, 4)
            cv2.imwrite(str(out / f"{name}.png"), a[:, :, 2::-1])
        if "linear_b64" in im:
            a = np.frombuffer(base64.b64decode(im["linear_b64"]), np.float32).reshape(h, w, 4)
            np.save(out / f"{name}_linear.npy", a[:, :, :3])
            v = a[:, :, :3] / max(1e-9, float(np.percentile(a[:, :, 1], 99.5)))
            v = np.clip(v, 0, 1) ** (1 / 2.2)
            cv2.imwrite(str(out / f"{name}_linear_preview.png"), (v[:, :, ::-1] * 255).astype(np.uint8))
        im.pop("rgb8_b64", None); im.pop("rgba8_b64", None); im.pop("linear_b64", None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="GPU judges (WGSL vs JS, traversal, furnace, disc)")
    ap.add_argument("--noise", action="store_true", help="render noise against the modelled sensor noise")
    ap.add_argument("--cull", action="store_true", help="shadow-ray culling against tracing every shadow ray")
    ap.add_argument("--mode", default=None, help="page mode (test, bench, noise, cull, disc)")
    ap.add_argument("--spp", type=int, default=None, help="pass-1 sub-samples per pixel (renderer default when omitted)")
    ap.add_argument("--refine", type=int, default=None, help="pass-2 extra sub-samples")
    ap.add_argument("--nl", type=int, default=None, help="light samples per sub-sample")
    ap.add_argument("--nlr", type=int, default=None, help="light samples per refinement sub-sample")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--node", type=float, default=0.003, help="radiosity node spacing for --test (m)")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write result JSON (and PNGs with --images) here")
    ap.add_argument("--scene", default=None, help="scene directory URL relative to the page (default ../assets/)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print every console line")
    ap.add_argument("--extra", default="", help="extra query string for the page (e.g. 'camopts=normals')")
    ap.add_argument("--scale", type=float, default=1.0, help="camera resolution scale (small-first runs: 0.25 = 320x180)")
    ap.add_argument("--cams", default=None, help="comma list of cameras (default all)")
    ap.add_argument("--no-lock", action="store_true", help="the caller holds the GPU lock (gpu_guard run)")
    a = ap.parse_args()

    mode = a.mode or ("test" if a.test else "noise" if a.noise else "cull" if a.cull else "bench")
    q = f"mode={mode}&frames={a.frames}&seeds={a.seeds}&node={a.node}"
    for k in ("spp", "refine", "nl", "nlr"):
        if getattr(a, k) is not None:
            q += f"&{k}={getattr(a, k)}"
    if a.images:
        q += "&images=1"
    if a.scene:
        q += f"&scene={a.scene}"
    if a.extra:
        q += "&" + a.extra
    if a.scale != 1.0:
        q += f"&scale={a.scale}"
    if a.cams:
        q += f"&cams={a.cams}"
    t0 = time.time()
    res, logs = run_page(q, a.timeout, headless=not a.headed, no_lock=a.no_lock)
    scene = json.loads((ROOT / "web/twin/assets/scene.json").read_text()) if (ROOT / "web/twin/assets/scene.json").exists() else {}
    res["meta"] = {"git_commit": git_commit(), "scene_provenance": {k: scene.get("provenance", {}).get(k) for k in ("git_commit", "built", "seed")},
                   "physics_sha16": __import__("hashlib").sha256((ROOT / "configs/twin/physics.yaml").read_bytes()).hexdigest()[:16],
                   "browser": "Brave (headless, Playwright)", "wall_s": round(time.time() - t0, 1), "query": q}
    if a.out and a.images:
        write_images(res, a.out)
    for line in logs:
        if a.verbose or "[error]" in line or "pageerror" in line or "ERROR" in line or "GPU error" in line or "lost" in line:
            print(line, file=sys.stderr)
    txt = json.dumps(res, indent=1, default=float)
    print(txt)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"gpu_{mode}.json").write_text(txt)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
