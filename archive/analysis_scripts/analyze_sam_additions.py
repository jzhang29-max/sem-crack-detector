"""
Characterise WHAT the union adds, to decide how to improve it.

"SAM is adding real cracks" and "0 of SAM's additions land on human-confirmed
crack pixels" are both true at once, because 99.7% of the added pixels sit in
never-reviewed territory. So the useful question is not whether the additions
are real but what KIND of thing they are, since the two possibilities call for
opposite fixes:

  A. HALO -- SAM traces a crack the pipeline already found, only wider. Then
     the additions hug existing crack regions, they are boundary refinement
     rather than detection, and the fix is to merge them into the parent region
     instead of counting them as new cracks. This would also explain 0 hits on
     confirmed-crack pixels: a human painting a crack marks the crack, not the
     1-3 px of grey either side of it, so the halo lands on unreviewed pixels
     by construction.

  B. SEPARATE -- SAM finds crack-like features at locations the pipeline has
     nothing at. Then the pipeline is genuinely missing cracks, the fix belongs
     upstream in segmentation or in the classifier's threshold, and these
     regions are the most valuable thing in the dataset to get labelled.

Measured by distance from each added region to the nearest pipeline crack.
Reads the saved overlays rather than re-running SAM (~4 min/image); they encode
the three classes as flat, exact colours, so extraction is an equality test and
not a brightness threshold. They are downscaled to 2048 px, so distances are in
view pixels -- fine for separating "touching" from "far", which is all that is
being asked.

    python3 analyze_sam_additions.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

from common import PROJECT_ROOT

Image.MAX_IMAGE_PIXELS = None
OVL_DIR = os.path.join(PROJECT_ROOT, "figures", "sam_union")
OUT = os.path.join(PROJECT_ROOT, "figures", "sam_union", "sam_addition_character.csv")

# exact colours written by make_overlay()
RED, YELLOW, ORANGE = (220, 30, 30), (255, 240, 20), (255, 140, 0)


def classes(path):
    a = np.array(Image.open(path).convert("RGB"))
    eq = lambda c: (a[..., 0] == c[0]) & (a[..., 1] == c[1]) & (a[..., 2] == c[2])
    return eq(RED), eq(YELLOW), eq(ORANGE)


def main():
    files = sorted(f for f in os.listdir(OVL_DIR) if f.endswith("_union.png"))
    rows = []
    for i, f in enumerate(files, 1):
        name = f[:-len("_union.png")]
        red, yellow, orange = classes(os.path.join(OVL_DIR, f))
        pipe = red | orange                      # everything the pipeline called crack
        if not yellow.any():
            continue
        # distance from every pixel to the nearest pipeline crack pixel
        dist = ndi.distance_transform_edt(~pipe) if pipe.any() else np.full(red.shape, 1e6)
        lab = measure.label(yellow, connectivity=2)
        for p in measure.regionprops(lab):
            d = dist[tuple(np.array(p.coords).T)]
            rows.append({"SourceImage": name, "area": int(p.area),
                         "min_dist": float(d.min()), "mean_dist": float(d.mean()),
                         "elongation": (p.axis_major_length / max(p.axis_minor_length, 0.5)),
                         "solidity": p.solidity})
        if i % 12 == 0:
            print(f"  {i}/{len(files)} overlays read", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    n = len(d)
    print(f"\n{n:,} separate SAM-added regions across {d.SourceImage.nunique()} images\n")

    print("=" * 76)
    print("HOW FAR IS EACH ADDITION FROM THE NEAREST PIPELINE CRACK?")
    print("=" * 76)
    bins = [(0, 0, "touching (0 px) -- halo on an existing crack"),
            (1, 2, "1-2 px -- effectively the same crack, wider"),
            (3, 10, "3-10 px -- adjacent, arguably same feature"),
            (11, 50, "11-50 px -- nearby but distinct"),
            (51, 10 ** 9, ">50 px -- genuinely separate location")]
    for lo, hi, lbl in bins:
        m = (d.min_dist >= lo) & (d.min_dist <= hi)
        print(f"  {lbl:46s} {int(m.sum()):>6,d} regions ({100.0*m.sum()/n:5.1f}%)  "
              f"{int(d.loc[m,'area'].sum()):>10,d} px ({100.0*d.loc[m,'area'].sum()/d.area.sum():5.1f}% of added area)")

    touching = d.min_dist <= 2
    print(f"\n  => {100.0*touching.sum()/n:.1f}% of added regions, carrying "
          f"{100.0*d.loc[touching,'area'].sum()/d.area.sum():.1f}% of the added area, "
          f"are within 2 px of a crack the pipeline already found.")
    far = d.min_dist > 50
    print(f"  => {int(far.sum()):,} regions ({100.0*far.sum()/n:.1f}%) are >50 px away "
          f"-- these are the only true candidate DETECTIONS.")

    if far.sum():
        print("\n" + "=" * 76)
        print("THE GENUINELY SEPARATE ONES -- worth labelling first")
        print("=" * 76)
        t = d[far].groupby("SourceImage").agg(n=("area", "size"), px=("area", "sum"),
                                              med_area=("area", "median")).sort_values("px", ascending=False)
        print(t.head(12).to_string())
        print(f"\nshape of separate additions: median elongation "
              f"{d.loc[far,'elongation'].median():.2f}, median solidity "
              f"{d.loc[far,'solidity'].median():.2f}")
        print(f"shape of halo additions:     median elongation "
              f"{d.loc[touching,'elongation'].median():.2f}, median solidity "
              f"{d.loc[touching,'solidity'].median():.2f}")
        print("(real cracks are elongated and low-solidity; blobs are the opposite)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
