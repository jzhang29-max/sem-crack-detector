"""
Break the union's additions down by what they physically are, to find the
improvement with the largest payoff.

The contact sheet showed the biggest additions are the near-black empty
background beyond the specimen edge -- precisely the artifact
exclude_border_background() strips from the pipeline. SAM has no equivalent
notion of "outside the specimen", so it segments that void as enthusiastically
as it segments a crack, and because those regions are enormous they dominate
the 6.6M added pixels.

Classifies every added region by its NATIVE-resolution mean brightness in the
original image:
    void        < 25   empty background / off-specimen. Never a crack.
    dark        25-90  plausible crack or void
    mid        90-170  striations, grain contrast, topographic shading
    bright      > 170  ridges and edge highlights. Never a crack.
Brightness is read from the original image rather than the overlay, since the
overlay has recoloured exactly the pixels of interest.

    python3 sam_addition_breakdown.py
Writes ../../figures/sam_union/sam_addition_breakdown.csv
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

from common import ORIGINAL_DIR, PROJECT_ROOT, contrast_kwargs_for
from detect_cracks import load_as_uint8, find_field_of_view

Image.MAX_IMAGE_PIXELS = None
OVL_DIR = os.path.join(PROJECT_ROOT, "figures", "sam_union")
OUT = os.path.join(OVL_DIR, "sam_addition_breakdown.csv")
RED, YELLOW, ORANGE = (220, 30, 30), (255, 240, 20), (255, 140, 0)

BANDS = [("void (<25) -- off-specimen background", -1, 25),
         ("dark (25-90) -- plausible crack/void", 25, 90),
         ("mid (90-170) -- striations / shading", 90, 170),
         ("bright (>170) -- ridges, highlights", 170, 256)]


def main():
    files = sorted(f for f in os.listdir(OVL_DIR) if f.endswith("_union.png"))
    rows = []
    for i, f in enumerate(files, 1):
        name = f[:-len("_union.png")]
        ovl = np.array(Image.open(os.path.join(OVL_DIR, f)).convert("RGB"))
        eq = lambda c: (ovl[..., 0] == c[0]) & (ovl[..., 1] == c[1]) & (ovl[..., 2] == c[2])
        yellow, pipe = eq(YELLOW), eq(RED) | eq(ORANGE)
        if not yellow.any():
            continue

        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{name}.tif"),
                             **contrast_kwargs_for(name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        # bring the overlay-resolution masks up to native resolution
        zy, zx = img8.shape[0] / yellow.shape[0], img8.shape[1] / yellow.shape[1]
        yn = np.asarray(Image.fromarray(yellow.astype(np.uint8) * 255).resize(
            (img8.shape[1], img8.shape[0]), Image.NEAREST)) > 127
        pn = np.asarray(Image.fromarray(pipe.astype(np.uint8) * 255).resize(
            (img8.shape[1], img8.shape[0]), Image.NEAREST)) > 127
        dist = ndi.distance_transform_edt(~pn) if pn.any() else np.full(img8.shape, 1e6)

        lab = measure.label(yn, connectivity=2)
        for p in measure.regionprops(lab, intensity_image=img8):
            rows.append({"SourceImage": name, "area": int(p.area),
                         "mean_brightness": float(p.intensity_mean),
                         "min_dist_to_pipeline": float(dist[tuple(np.array(p.coords).T)].min())})
        if i % 12 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    tot = d.area.sum()
    print(f"\n{len(d):,} added regions, {tot:,} native px total\n")
    print("=" * 82)
    print("WHAT IS THE UNION ACTUALLY ADDING? (by native mean brightness)")
    print("=" * 82)
    print(f"{'band':40s} {'regions':>9s} {'area px':>12s} {'% area':>8s}")
    for lbl, lo, hi in BANDS:
        m = (d.mean_brightness > lo) & (d.mean_brightness <= hi)
        print(f"{lbl:40s} {int(m.sum()):>9,d} {int(d.loc[m,'area'].sum()):>12,d} "
              f"{100.0*d.loc[m,'area'].sum()/tot:>7.1f}%")

    void = d.mean_brightness <= 25
    bright = d.mean_brightness > 170
    junk = void | bright
    print(f"\n  void + bright = {100.0*d.loc[junk,'area'].sum()/tot:.1f}% of all added area "
          f"in {int(junk.sum()):,} regions -- neither can be a crack")
    dark = (d.mean_brightness > 25) & (d.mean_brightness <= 90)
    print(f"  dark, plausibly crack   = {100.0*d.loc[dark,'area'].sum()/tot:.1f}% "
          f"({int(dark.sum()):,} regions)")

    far = d.min_dist_to_pipeline > 50
    print(f"\n  AFTER removing void+bright, additions >50 px from any pipeline crack:")
    keep = far & ~junk
    print(f"    {int(keep.sum()):,} regions, {int(d.loc[keep,'area'].sum()):,} px "
          f"({100.0*d.loc[keep,'area'].sum()/tot:.1f}% of current additions)")
    print(f"    median area {d.loc[keep,'area'].median():.0f} px, "
          f"median brightness {d.loc[keep,'mean_brightness'].median():.0f}")
    print(f"\n  => filtering void+bright alone would cut the union's added area by "
          f"{100.0*d.loc[junk,'area'].sum()/tot:.0f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
