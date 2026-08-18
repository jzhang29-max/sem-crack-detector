"""
Improved union: drop the SAM additions that cannot physically be cracks.

Two filters, neither needing a single new label:

  1. VOID. A region whose native mean brightness is under 25 is the empty
     background beyond the specimen edge. The pipeline already excludes this
     via exclude_border_background(); SAM has no concept of "off-specimen" and
     segments the void as readily as a crack. Because those regions are huge
     they were 31.6% of all added area by themselves.

  2. NOT DARKER THAN THE IMAGE. A crack is locally dark, so a region whose mean
     brightness is at or above the image median is a ridge, a striation or
     grain contrast. This is deliberately RELATIVE rather than a fixed ceiling:
     these images have quite different exposures, so any absolute bright cutoff
     would be too strict on some and useless on others.

Runs on the masks already stored in the sweep's overlays instead of re-running
SAM (~4 min/image, ~1.5 h for the set). Consequence worth stating: those
overlays are 2048 px, so mask edges are quantised to that grid, and the area
figures here can differ by a fraction of a percent from a native re-run. The
filter thresholds themselves are evaluated on NATIVE brightness, so the
accept/reject decisions are exact.

    python3 sam_union_v2.py
Writes ../../figures/sam_union_v2/ and sam_union_v2_stats.csv
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
from skimage import measure

from common import (ORIGINAL_DIR, PROJECT_ROOT, contrast_kwargs_for,
                     load_correction_mask)
from detect_cracks import load_as_uint8, find_field_of_view

Image.MAX_IMAGE_PIXELS = None
OVL_DIR = os.path.join(PROJECT_ROOT, "figures", "sam_union")
OUT_DIR = os.path.join(PROJECT_ROOT, "figures", "sam_union_v2")
STATS = os.path.join(OUT_DIR, "sam_union_v2_stats.csv")
RED, YELLOW, ORANGE = (220, 30, 30), (255, 240, 20), (255, 140, 0)
VOID_MAX = 25
VIEW_MAX = 2048


def up(mask, shape):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (shape[1], shape[0]), Image.NEAREST)) > 127


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(OVL_DIR) if f.endswith("_union.png"))
    rows = []
    for i, f in enumerate(files, 1):
        name = f[:-len("_union.png")]
        ovl = np.array(Image.open(os.path.join(OVL_DIR, f)).convert("RGB"))
        eq = lambda c: (ovl[..., 0] == c[0]) & (ovl[..., 1] == c[1]) & (ovl[..., 2] == c[2])
        yellow, pipe = eq(YELLOW), eq(RED) | eq(ORANGE)

        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{name}.tif"),
                             **contrast_kwargs_for(name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        pipe_n = up(pipe, img8.shape)
        yell_n = up(yellow, img8.shape)
        med = float(np.median(img8))

        kept = np.zeros_like(yell_n)
        n_void = n_bright = n_kept = 0
        a_void = a_bright = 0
        if yell_n.any():
            lab = measure.label(yell_n, connectivity=2)
            for p in measure.regionprops(lab, intensity_image=img8):
                b = p.intensity_mean
                if b <= VOID_MAX:
                    n_void += 1; a_void += p.area; continue
                if b >= med:
                    n_bright += 1; a_bright += p.area; continue
                kept[tuple(np.array(p.coords).T)] = True
                n_kept += 1

        tot = img8.size
        union2 = pipe_n | kept
        rec = {"SourceImage": name, "img_median": med,
               "pipeline_px": int(pipe_n.sum()),
               "sam_only_before": int(yell_n.sum()), "sam_only_after": int(kept.sum()),
               "rej_void_regions": n_void, "rej_void_px": int(a_void),
               "rej_bright_regions": n_bright, "rej_bright_px": int(a_bright),
               "kept_regions": n_kept,
               "pipeline_area_pct": 100.0 * pipe_n.sum() / tot,
               "union2_area_pct": 100.0 * union2.sum() / tot,
               "sam_only_area_pct_after": 100.0 * kept.sum() / tot}

        cm = load_correction_mask(name, img8.shape)
        for k in ("on_crack", "on_notcrack", "unreviewed"):
            rec[k] = np.nan
        if cm is not None:
            rec["on_crack"] = int((kept & (cm == 1)).sum())
            rec["on_notcrack"] = int((kept & (cm == 2)).sum())
            rec["unreviewed"] = int((kept & (cm == 0)).sum())

        rgb = np.stack([img8] * 3, -1).astype(np.uint8)
        rgb[pipe_n & ~kept] = (220, 30, 30)
        rgb[kept & ~pipe_n] = (255, 240, 20)
        rgb[kept & pipe_n] = (255, 140, 0)
        im = Image.fromarray(rgb)
        if max(im.size) > VIEW_MAX:
            s = VIEW_MAX / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        im.save(os.path.join(OUT_DIR, f"{name}_union_v2.png"))

        rows.append(rec)
        if i % 10 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(STATS, index=False)
    b, a = int(d.sam_only_before.sum()), int(d.sam_only_after.sum())
    print("\n" + "=" * 80)
    print(f"{len(d)} images")
    print(f"SAM-only added area   before {b:>12,d} px   after {a:>12,d} px   "
          f"({100.0*(b-a)/max(b,1):.1f}% removed)")
    print(f"  rejected as void    {int(d.rej_void_px.sum()):>12,d} px "
          f"({int(d.rej_void_regions.sum()):,} regions)")
    print(f"  rejected as bright  {int(d.rej_bright_px.sum()):>12,d} px "
          f"({int(d.rej_bright_regions.sum()):,} regions)")
    print(f"  kept                {a:>12,d} px ({int(d.kept_regions.sum()):,} regions)")
    print(f"mean crack area%  pipeline {d.pipeline_area_pct.mean():.3f}  "
          f"union-v2 {d.union2_area_pct.mean():.3f}")

    g = d.dropna(subset=["on_crack"])
    oc, on_, ou = int(g.on_crack.sum()), int(g.on_notcrack.sum()), int(g.unreviewed.sum())
    print(f"\nground truth on what v2 still adds ({len(g)} reviewed images):")
    print(f"  on confirmed CRACK      {oc:>10,d} px")
    print(f"  on confirmed NOT-crack  {on_:>10,d} px")
    print(f"  never reviewed          {ou:>10,d} px")
    if oc + on_:
        print(f"  -> {100.0*oc/(oc+on_):.1f}% of adjudicated additions are real crack")
    print(f"\noverlays in {OUT_DIR}")


if __name__ == "__main__":
    main()
