"""
Contact sheet of the MARGINAL calls -- regions the live threshold (0.40)
accepts that a stricter 0.45 would reject.

This is the only set that decides whether the threshold is too permissive.
Regions scoring far above 0.45 are kept either way, and regions below 0.40 are
dropped either way; the disagreement lives entirely in the 0.40-0.45 band. So
instead of scrolling a 25-megapixel image hunting for small red specks, judge
these directly: if they are real microcracks, 0.40 is right, and if they are
grain-boundary contrast or polishing noise, 0.45 is right.

Each tile is a native-resolution window with the region's ACTUAL outline drawn,
not its bounding box. An earlier sheet in this session drew bounding boxes, and
for regions larger than the window that rendered as straight lines spanning the
whole crop, which made several tiles unreadable.

    python3 marginal_calls_sheet.py [--images A B C] [--n 40]
Writes ../../figures/marginal_calls_<threshold band>.png
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import joblib
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure, morphology

from common import ORIGINAL_DIR, PROJECT_ROOT, PROD_MODEL_PATH, contrast_kwargs_for
from detect_cracks import (load_as_uint8, find_field_of_view, flatten_background,
                            segment_dark_regions, clean_mask, compute_vesselness,
                            exclude_border_background, extract_candidates)

Image.MAX_IMAGE_PIXELS = None
LOW, HIGH = 0.40, 0.45
CROP = 220
DEFAULT_IMAGES = ["MAR_Amb_AS_ETD_0003", "MAR_Amb_HIP_ETD_0010", "MAR_Amb_Cast_CBS_0004"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=DEFAULT_IMAGES)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    bundle = joblib.load(PROD_MODEL_PATH)
    tiles = []
    for name in a.images:
        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{name}.tif"),
                             **contrast_kwargs_for(name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        flat = flatten_background(img8)
        clean = clean_mask(segment_dark_regions(flat, img8=img8), min_area_px=13)
        ves = compute_vesselness(flat)
        clean = exclude_border_background(clean, ves)
        labeled, df = extract_candidates(clean, flat, ves, min_area_px=40)
        p = bundle["clf"].predict_proba(
            bundle["scaler"].transform(df[bundle["feature_names"]].values))[:, 1]
        band = (p >= LOW) & (p < HIGH) & (df["Area"].values < 50000)
        print(f"{name:30s} {len(df):>5d} candidates | "
              f"accepted@{LOW} {int(((p>=LOW)).sum()):>5d} | "
              f"marginal {LOW}-{HIGH}: {int(band.sum()):>4d}", flush=True)
        if not band.any():
            continue
        idx = np.argsort(-df["Area"].values[band])
        chosen = df["Label"].values[band][idx][: max(a.n // len(a.images), 1)]
        probs = p[band][idx][: len(chosen)]
        for lbl, pr in zip(chosen, probs):
            m = labeled == int(lbl)
            ys, xs = np.nonzero(m)
            cy, cx = int(ys.mean()), int(xs.mean())
            ty = max(0, min(cy - CROP // 2, img8.shape[0] - CROP))
            tx = max(0, min(cx - CROP // 2, img8.shape[1] - CROP))
            crop = img8[ty:ty + CROP, tx:tx + CROP]
            if crop.shape != (CROP, CROP):
                continue
            sub = m[ty:ty + CROP, tx:tx + CROP]
            # actual outline, not a bounding box
            outline = sub & ~morphology.binary_erosion(sub)
            rgb = np.stack([crop] * 3, -1).astype(np.uint8)
            rgb[outline] = (255, 40, 40)
            tiles.append((Image.fromarray(rgb),
                          f"{name[-16:]} p={pr:.3f} {int(m.sum())}px",
                          {"SourceImage": name, "Label": int(lbl), "proba": float(pr),
                           "area": int(m.sum()), "cy": cy, "cx": cx}))

    if not tiles:
        print("no marginal regions found"); return
    ncol = 8
    nrow = int(np.ceil(len(tiles) / ncol))
    PAD, LBL = 4, 13
    sheet = Image.new("RGB", (ncol * (CROP + PAD), nrow * (CROP + PAD + LBL)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    manifest = []
    for i, (im, cap, meta) in enumerate(tiles):
        r, c = divmod(i, ncol)
        x, y = c * (CROP + PAD), r * (CROP + PAD + LBL)
        sheet.paste(im, (x, y))
        # big index in the corner so the sheet can be answered by number
        n = str(i + 1)
        dr.rectangle([x + 2, y + 2, x + 2 + 13 * len(n) + 8, y + 24], fill=(255, 235, 0))
        dr.text((x + 8, y + 7), n, fill=(0, 0, 0))
        dr.text((x + 2, y + CROP + 1), f"{n}. {cap}", fill=(0, 0, 0))
        manifest.append({"index": i + 1, **meta})
    out = os.path.join(PROJECT_ROOT, "figures", f"marginal_calls_{LOW}_{HIGH}.png")
    sheet.save(out)
    mf = os.path.join(PROJECT_ROOT, "figures", f"marginal_calls_{LOW}_{HIGH}_manifest.json")
    with open(mf, "w") as fh:
        json.dump({"low": LOW, "high": HIGH, "regions": manifest}, fh, indent=2)
    print(f"\nwrote {out} ({len(tiles)} regions, {CROP}x{CROP} native px)")
    print(f"wrote {mf}")
    print(f"red outline = region accepted at {LOW} but rejected at {HIGH}")
    print("numbers are stable identifiers -- reply with which are cracks and the "
          "verdicts can be ingested straight into the correction masks")


if __name__ == "__main__":
    main()
