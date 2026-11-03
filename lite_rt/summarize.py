"""Summarise a judges.py --compare result (mitsuba.json): pooled verdict + per-region medians (pixel-weighted mean of
per-image medians and the worst image), per state x camera overall numbers."""
import json
import sys

import numpy as np

r = json.load(open(sys.argv[1]))
out = {"pooled": {}, "regions": {}, "images": []}
for var, pr in r["pooled"].items():
    out["pooled"][var] = {"verdict": pr["verdict"], "live": {k: v for k, v in pr["live"].items() if k != "level_ratio_images"},
                          "twin_mean": {k: v for k, v in pr["twin_mean"].items() if k != "level_ratio_images"},
                          "edges": pr.get("edges")}
reg = {}
for s, sr in r["states"].items():
    for cam, cr in sr["cameras"].items():
        for var, vr in cr.items():
            if not isinstance(vr, dict) or "live" not in vr:
                continue
            o = vr["live"]["overall"]
            out["images"].append({"state": s, "cam": cam, "median": o.get("median_rel"), "p95": o.get("p95_rel"), "level": o.get("level_ratio"),
                                  "edges_mean_px": (vr["twin_mean"].get("edges") or {}).get("all", {}).get("median_abs_px")})
            for name, st in (vr["live"].get("regions") or {}).items():
                if st.get("median_rel") is None:
                    continue
                reg.setdefault(name, []).append((st["n_lit"], st["median_rel"], st.get("signed_median_rel"), f"{s}/{cam}"))
for name, L in reg.items():
    n = np.array([x[0] for x in L], float)
    m = np.array([x[1] for x in L], float)
    sg = np.array([x[2] if x[2] is not None else np.nan for x in L], float)
    w = int(np.argmax(m))
    out["regions"][name] = {"weighted_median_rel": float((n * m).sum() / n.sum()), "median_of_images": float(np.median(m)),
                            "signed_weighted": float(np.nansum(n * sg) / n.sum()), "worst": [float(m[w]), L[w][3]], "images": len(L)}
print(json.dumps(out, indent=1, default=float))
