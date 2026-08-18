"""
Contact sheet of the SAM-only additions the pipeline has nothing at, shown at
NATIVE resolution so they can actually be judged.

These 789 regions (>50 px from any pipeline crack) are the crux: the union's
value rests entirely on whether they are real cracks, and no measurement here
can say, because 99.7% of the added area lies in never-reviewed pixels. Rather
than argue from statistics, put them on screen.

Sampling is stratified by size, not random. Tiny regions dominate by count
(median 8 px in view pixels) but carry little area and are hard to judge by
eye; large ones are both judgeable and consequential. Both bands are shown and
labelled so the sheet is not silently biased toward the flattering half.

Coordinates come from the downscaled overlays, so they are scaled back up to
native resolution per image before cropping.

    python3 sam_addition_contactsheet.py [--n 48]
Writes ../../figures/sam_additions_contactsheet.png
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
from skimage import measure

from common import ORIGINAL_DIR, PROJECT_ROOT, contrast_kwargs_for
from detect_cracks import load_as_uint8, find_field_of_view

Image.MAX_IMAGE_PIXELS = None
OVL_DIR = os.path.join(PROJECT_ROOT, "figures", "sam_union")
OUT = os.path.join(PROJECT_ROOT, "figures", "sam_additions_contactsheet.png")
YELLOW = (255, 240, 20)
CROP = 260          # native px window around each region
FAR_PX = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()

    d = pd.read_csv(os.path.join(OVL_DIR, "sam_addition_character.csv"))
    far = d[d.min_dist > FAR_PX].copy()
    print(f"{len(far)} separate additions available")

    # stratified: half from the largest, half from the middle of the size range
    far = far.sort_values("area", ascending=False)
    big = far.head(a.n // 2)
    mid = far.iloc[a.n // 2: len(far) // 2].sample(min(a.n - len(big), max(len(far) // 2 - a.n // 2, 0)),
                                                    random_state=0) if len(far) > a.n else far.iloc[len(big):]
    sel = pd.concat([big, mid]).head(a.n)
    print(f"selected {len(sel)} ({len(big)} largest, {len(sel)-len(big)} mid-size) "
          f"across {sel.SourceImage.nunique()} images")

    tiles = []
    for img_name, grp in sel.groupby("SourceImage"):
        ovl = np.array(Image.open(os.path.join(OVL_DIR, f"{img_name}_union.png")).convert("RGB"))
        ymask = ((ovl[..., 0] == YELLOW[0]) & (ovl[..., 1] == YELLOW[1]) &
                 (ovl[..., 2] == YELLOW[2]))
        lab = measure.label(ymask, connectivity=2)
        props = {p.area: p for p in measure.regionprops(lab)}

        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{img_name}.tif"),
                             **contrast_kwargs_for(img_name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        sy = img8.shape[0] / ovl.shape[0]
        sx = img8.shape[1] / ovl.shape[1]

        for _, r in grp.iterrows():
            p = props.get(int(r.area))
            if p is None:
                continue
            cy, cx = p.centroid
            ny, nx = int(cy * sy), int(cx * sx)
            by0, by1 = int(p.bbox[0] * sy), int(p.bbox[2] * sy)
            bx0, bx1 = int(p.bbox[1] * sx), int(p.bbox[3] * sx)
            ty0 = max(0, min(ny - CROP // 2, img8.shape[0] - CROP))
            tx0 = max(0, min(nx - CROP // 2, img8.shape[1] - CROP))
            crop = img8[ty0:ty0 + CROP, tx0:tx0 + CROP]
            if crop.shape != (CROP, CROP):
                continue
            im = Image.fromarray(np.stack([crop] * 3, -1))
            dr = ImageDraw.Draw(im)
            dr.rectangle([bx0 - tx0 - 2, by0 - ty0 - 2, bx1 - tx0 + 2, by1 - ty0 + 2],
                         outline=YELLOW, width=2)
            tiles.append((im, f"{img_name[-18:]}  {int(r.area*sy*sx)}px"))

    if not tiles:
        print("no tiles"); return
    ncol = 8
    nrow = int(np.ceil(len(tiles) / ncol))
    PAD, LBL = 4, 12
    W = ncol * (CROP + PAD)
    H = nrow * (CROP + PAD + LBL)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (im, cap) in enumerate(tiles):
        r, c = divmod(i, ncol)
        x, y = c * (CROP + PAD), r * (CROP + PAD + LBL)
        sheet.paste(im, (x, y))
        dr.text((x + 2, y + CROP + 1), cap, fill=(0, 0, 0))
    sheet.save(OUT)
    print(f"\nwrote {OUT}  ({len(tiles)} crops, {CROP}x{CROP} native px each)")
    print("yellow box = what SAM added and the pipeline has nothing at")


if __name__ == "__main__":
    main()
