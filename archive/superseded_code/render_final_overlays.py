"""
Render the final crack overlay for every image using the DEPLOYED model.

The paint templates already encode exactly this result -- they were regenerated
after the new model went live -- but they also tint every REJECTED candidate
cyan, which is what a reviewer needs and not what a result should look like.
This renders crack-only: original greyscale with accepted cracks in red.

The mask is read out of the template rather than recomputed, so what is shown
is provably what the app is using. build_simple_overlay alpha-blends, so the
mask is recovered algebraically (crack pixels satisfy G == B and R-G in
{140,141}) rather than by an RGB threshold -- verified against
run_unified_pipeline on one MAR and one steel image at 0 pixels difference.

Corrections are included: run_unified_pipeline applies the correction mask
before the template is written, so every human verdict is already reflected.

    python3 render_final_overlays.py [--full-res]
Writes ../../figures/final_overlays/<image>_cracks.png + an index sheet
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, contrast_kwargs_for
from detect_cracks import load_as_uint8, find_field_of_view

Image.MAX_IMAGE_PIXELS = None
OUT_DIR = os.path.join(PROJECT_ROOT, "figures", "final_overlays")
VIEW_MAX = 2048
THUMB = 300


def crack_mask_from_template(name):
    p = os.path.join(PAINT_DIR, f"{name}_paint_template.png")
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p).convert("RGB")).astype(np.int16)
    d = a[..., 0] - a[..., 1]
    return (a[..., 1] == a[..., 2]) & ((d == 140) | (d == 141))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-res", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR)
                    if f.lower().endswith(".tif"))
    rows, thumbs = [], []
    for i, name in enumerate(names, 1):
        m = crack_mask_from_template(name)
        if m is None:
            print(f"{name}: no template, skipped"); continue
        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{name}.tif"),
                             **contrast_kwargs_for(name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        if img8.shape != m.shape:
            print(f"{name}: shape mismatch, skipped"); continue

        rgb = np.stack([img8] * 3, -1).astype(np.uint8)
        rgb[m] = (225, 25, 25)
        im = Image.fromarray(rgb)
        if not a.full_res and max(im.size) > VIEW_MAX:
            s = VIEW_MAX / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        im.save(os.path.join(OUT_DIR, f"{name}_cracks.png"))

        pct = 100.0 * m.sum() / img8.size
        rows.append({"SourceImage": name, "crack_px": int(m.sum()),
                     "image_px": int(img8.size), "crack_area_pct": pct})
        t = im.copy(); t.thumbnail((THUMB, THUMB))
        thumbs.append((t, f"{name[-24:]} {pct:.2f}%"))
        print(f"[{i}/{len(names)}] {name:32s} crack area {pct:6.2f}%", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT_DIR, "final_crack_area.csv"), index=False)

    # index sheet so all 63 can be scanned at once
    ncol = 8
    nrow = int(np.ceil(len(thumbs) / ncol))
    PAD, LBL = 4, 12
    W = ncol * (THUMB + PAD)
    H = nrow * (THUMB + PAD + LBL)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (t, cap) in enumerate(thumbs):
        r, c = divmod(i, ncol)
        x, y = c * (THUMB + PAD), r * (THUMB + PAD + LBL)
        sheet.paste(t, (x, y))
        dr.text((x + 2, y + THUMB + 1), cap, fill=(0, 0, 0))
    idx = os.path.join(PROJECT_ROOT, "figures", "final_overlays_index.png")
    sheet.save(idx)

    print(f"\n{len(d)} overlays in {OUT_DIR}")
    print(f"index sheet: {idx}")
    print(f"crack area: median {d.crack_area_pct.median():.2f}%  "
          f"mean {d.crack_area_pct.mean():.2f}%  "
          f"min {d.crack_area_pct.min():.2f}%  max {d.crack_area_pct.max():.2f}%")


if __name__ == "__main__":
    main()
