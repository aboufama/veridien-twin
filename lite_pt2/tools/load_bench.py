#!/usr/bin/env python3
"""Load bench of lite_pt2 (web/twin/lite_pt2/loadbench.html) in headless Brave: create() time, first frame, the
deferred gather table, and the browser's long-task record (main-thread tasks > 50 ms) while four tiles render.

    .venv/bin/python -m veridien.webtwin.gpu_guard run --timeout 200 -- env VERIDIEN_GPU_LOCK=<inner.lock> nice -n 10 \
        .venv/bin/python web/twin/lite_pt2/tools/load_bench.py --renderer lite_pt2 --out runs/webtwin/fix/lite_pt2/load
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default="lite_pt2")
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--out", default="runs/webtwin/fix/lite_pt2/load")
    a = ap.parse_args()
    from veridien.webtwin.headless import TwinPage
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    page = TwinPage(path=f"lite_pt2/loadbench.html?renderer={a.renderer}&seconds={a.seconds}", harness=False)
    with page:
        res = page.evaluate("async () => await window.twinJudge.result()")
        res["adapter"] = page.adapter
        try:
            page.page.screenshot(path=str(out / "tiles.png"))
        except Exception as e:  # noqa: BLE001
            res["screenshot_error"] = str(e)
    res["wall_s"] = time.time() - t0
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    keys = ("marks", "create_ms", "first_frame_ms", "gather_ready_ms", "long_task_max_ms", "long_tasks_over_100",
            "raf_max_gap_during_create_ms", "raf_max_gap_after_create_ms", "fps", "error")
    print(f"[load_bench] {a.renderer}: " + json.dumps({k: res.get(k) for k in keys}), flush=True)
    print(f"[load_bench] long tasks [start ms, duration ms]: {res.get('long_tasks')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
