"""Judge a run_capture.py output directory against the Mitsuba god-view references (src/veridien/webtwin/godview_judge.py).

    python3 web/twin/godview_three/judge_three.py --cand /workspace/godview_three/<tag> [--refs /workspace/godrefs]

Writes <cand>/judge.json (per viewpoint x state: verdict rows, regions, edges, SSIM) and <cand>/sbs/*.png
(reference | candidate | relative error), and prints one line per pair plus the worst-case summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    from veridien.webtwin import godview_judge as G

    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--refs", default="/workspace/godrefs")
    a = ap.parse_args()
    cand_dir, ref_dir = Path(a.cand), Path(a.refs)
    gv = G.load_godview()
    scene_data = json.loads((REPO / "web/twin/assets/scene.json").read_text())
    states = json.loads((ref_dir / "states.json").read_text())["states"]
    res, rows = {}, []
    for f in sorted(cand_dir.glob("*__*.npy")):
        v, s = f.stem.split("__")
        rp = ref_dir / G.ref_name(v, s)
        if not rp.is_file():
            continue
        ref = G.load_ref(rp, gain=gv["lights"]["scope"].get("gain"))
        cand = np.load(f).astype(np.float64)
        poses = states[s]["poses"]
        r = G.compare(cand, ref, gv, v, scene_data, poses)
        vd = G.verdict(r, gv)
        r["verdict"] = vd
        res[f"{v}__{s}"] = r
        G.side_by_side(cand_dir / "sbs" / f"{v}__{s}.png", ref["mean"], cand, ref["hit"].astype(bool),
                       float(gv["display"]["exposure"]), title=f"{v} {s}")
        rr = vd["rows"]
        line = (f"{v:11s} {s:9s} level {rr['level']['value']:.3f}  med {rr['median_rel']['value']:.3f}  "
                f"p95 {rr['p95_rel']['value']:.3f}  ssim {rr['ssim']['value']:.3f}  edges {rr['edges_median_px']['value']}  "
                f"pass {vd['pass']}")
        reg = {k: (round(x.get('median_rel', float('nan')), 3), round(x.get('signed_median_rel', float('nan')), 3))
               for k, x in r["regions"].items() if "median_rel" in x}
        print(line, flush=True)
        print("    regions (median, signed):", reg, flush=True)
        rows.append(rr)
    (cand_dir / "judge.json").write_text(json.dumps(res, indent=1, default=float))
    if rows:
        worst = {k: (max if k in ("median_rel", "p95_rel", "edges_median_px") else min)(
            [r[k]["value"] for r in rows if r[k]["value"] is not None]) for k in ("median_rel", "p95_rel", "ssim", "edges_median_px")}
        lv = [r["level"]["value"] for r in rows]
        print("worst:", worst, "level range", (min(lv), max(lv)), "all pass", all(r_["pass"] for r_ in
              [res[k]["verdict"] for k in res]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
