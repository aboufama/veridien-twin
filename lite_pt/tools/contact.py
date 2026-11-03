"""Contact sheet: Mitsuba | lite_pt live frame for CAM1-3 at the default state, plus the SCOPE view (rgb8).
usage: python contact.py <run_dir> <out.png> [state]
Both radiance images share one display mapping per camera (exposure from the Mitsuba image's 99.5th percentile
luminance, sRGB), so they are comparable; the SCOPE panel is the candidate's own sensor output."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def tone(img, scale):
    return (srgb(img * scale) * 255 + 0.5).astype(np.uint8)


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    state = sys.argv[3] if len(sys.argv) > 3 else "default"
    tiles = []
    for cam in ("CAM1", "CAM2", "CAM3"):
        d = np.load(run / "images" / f"{state}_{cam}.npz")
        ref, live = d["ref"][..., :3], d["twin_live"][..., :3]
        Y = ref @ np.array([0.2126, 0.7152, 0.0722])
        s = 0.85 / max(np.percentile(Y, 99.5), 1e-12)
        rel = np.abs((live @ [0.2126, 0.7152, 0.0722]) - Y) / np.maximum(Y, 1e-30)
        lit = Y > 0.02 * np.percentile(Y, 99)
        tiles.append((cam, tone(ref, s), tone(live, s), float(np.median(rel[lit]))))
    W, H = 640, 360
    pad, lab = 8, 22
    sheet = Image.new("RGB", (2 * W + 3 * pad, 4 * (H + lab) + 5 * pad), (243, 243, 239))
    dr = ImageDraw.Draw(sheet)
    for i, (cam, a, b, med) in enumerate(tiles):
        y = pad + i * (H + lab + pad)
        for j, (img, t) in enumerate(((a, f"{cam} Mitsuba 4096 spp"), (b, f"{cam} lite_pt live frame (median |rel| {100 * med:.1f} %)"))):
            x = pad + j * (W + pad)
            sheet.paste(Image.fromarray(img).resize((W, H), Image.LANCZOS), (x, y + lab))
            dr.text((x, y + 4), t, fill=(29, 29, 27))
    sc = run / "default_SCOPE_rgb8.npy"
    if sc.exists():
        y = pad + 3 * (H + lab + pad)
        s = np.load(sc)
        sheet.paste(Image.fromarray(s.astype(np.uint8)).resize((W, H), Image.LANCZOS), (pad, y + lab))
        dr.text((pad, y + 4), "SCOPE (lite_pt, display sensor), default state", fill=(29, 29, 27))
        c1 = run / "default_CAM1_rgb8.npy"
        if c1.exists():
            sheet.paste(Image.fromarray(np.load(c1).astype(np.uint8)).resize((W, H), Image.LANCZOS), (2 * pad + W, y + lab))
            dr.text((2 * pad + W, y + 4), "CAM1 (lite_pt, display sensor), default state", fill=(29, 29, 27))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
