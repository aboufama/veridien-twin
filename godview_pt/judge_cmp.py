"""godview_pt vs the Mitsuba god's-eye reference (/workspace/godrefs/god_<view>_<state>.npz, scripts/godview_ref.py):
level, median / p95 |rel| of luminance on lit pixels, SSIM on the tone-mapped image (Khronos PBR Neutral, sRGB, 7x7 box
windows, luminance), and the same for the raster-only image. numpy only (the pods have no skimage).

    python3 web/twin/godview_pt/judge_cmp.py --cap /workspace/godview_pt/cap1 --refs /workspace/godrefs --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def neutral(c):
    c = np.asarray(c, np.float64)
    sc, des = 0.76, 0.15
    x = c.min(axis=-1, keepdims=True)
    off = np.where(x < 0.08, x - 6.25 * x * x, 0.04)
    c = c - off
    peak = c.max(axis=-1, keepdims=True)
    d = 1 - sc
    npk = 1 - d * d / np.maximum(peak + d - sc, 1e-9)
    g = 1 - 1 / (des * (peak - npk) + 1)
    comp = c * (npk / np.maximum(peak, 1e-9))
    comp = comp * (1 - g) + npk * g
    return np.where(peak < sc, c, comp)


def display(lin, exposure):
    t = np.clip(neutral(np.clip(lin * exposure, 0, None)), 0, 1)
    return np.where(t <= 0.0031308, 12.92 * t, 1.055 * t ** (1 / 2.4) - 0.055)


def lum(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def box(a, r=3):
    p = np.pad(np.asarray(a, np.float64), r, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def ssim(a, b):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ma, mb = box(a), box(b)
    va, vb, cab = box(a * a) - ma * ma, box(b * b) - mb * mb, box(a * b) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))
    return float(s.mean())


def stats(img, ref, exposure):
    Y, R = lum(img), lum(ref)
    p99 = float(np.percentile(R, 99))
    m = R > 0.02 * p99
    rel = np.abs(Y[m] - R[m]) / R[m]
    return {"level": float(Y[m].sum() / R[m].sum()), "median_rel": float(np.median(rel)), "p95_rel": float(np.percentile(rel, 95)),
            "ssim_tonemapped": ssim(lum(display(img, exposure)), lum(display(ref, exposure))), "n_px": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--refs", default="/workspace/godrefs")
    ap.add_argument("--out", default="")
    ap.add_argument("--diff", default="")
    a = ap.parse_args()
    cap, refs = Path(a.cap), Path(a.refs)
    res = {}
    for f in sorted(refs.glob("god_*.npz")):
        tag = f.stem[4:]
        view, state = tag.rsplit("_", 1)
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        gain, exposure = float(meta.get("gain_yaml") or 1.0), float(meta.get("exposure_yaml") or 1.0)
        ref = z["studio_mean"] + gain * z["scope_mean"]
        row = {"gain": gain, "exposure": exposure, "ref_spp": meta.get("spp")}
        for mode in ("traced", "raster"):
            p = cap / f"{view}__{state}__{mode}.npy"
            if not p.exists():
                continue
            img = np.load(p)
            row[mode] = stats(img, ref, exposure)
            if a.diff:
                from PIL import Image
                d = Path(a.diff)
                d.mkdir(parents=True, exist_ok=True)
                Yi, Yr = lum(img), lum(ref)
                r = np.clip((Yi - Yr) / np.maximum(Yr, 1e-6), -0.25, 0.25) / 0.25
                rgb = np.stack([np.clip(r, 0, 1), np.zeros_like(r), np.clip(-r, 0, 1)], -1) * 0.8 + 0.2 * display(ref, exposure)
                Image.fromarray((255 * np.clip(rgb, 0, 1)).astype(np.uint8)).save(d / f"diff_{mode}_{view}_{state}.png")
                Image.fromarray((255 * display(ref, exposure)).astype(np.uint8)).save(d / f"ref_{view}_{state}.png")
        if "traced" in row or "raster" in row:
            res[tag] = row
            print(tag, json.dumps({k: v for k, v in row.items() if k in ("traced", "raster")}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
