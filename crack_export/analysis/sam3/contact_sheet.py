#!/usr/bin/env python3
"""Every frame's detector overlay on one page, ordered by score.

Frames that carry a hand label are captioned with their clIoU_adapt; the rest show only the
predicted area fraction, because there is nothing to score them against and a blank where a
number would go is more honest than a number that means something different.

Ordering is by score descending, then unscored frames by predicted area. That puts the
failure cases at the bottom where they are easy to inspect, which is the point of the sheet --
the two lowest-scoring frames in the 5-frame sample turned out to be one label-completeness
artefact and one genuine false-positive mode, and only looking at them distinguished the two.
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SC = os.path.dirname(os.path.abspath(__file__))
# read the quantile from the config rather than hardcoding it in the title: the
# first version of this sheet said "top 2%" while rendering a top-1% run.
CFG = json.load(open(f"{SC}/best_config.json"))
COLS = 7
THUMB = 460


def main():
    rows = json.load(open(f"{SC}/detector_all.json"))
    scored = sorted([r for r in rows if "clIoU_adapt" in r],
                    key=lambda r: -r["clIoU_adapt"])
    unscored = sorted([r for r in rows if "clIoU_adapt" not in r],
                      key=lambda r: -r["pred_frac"])
    allr = scored + unscored
    n = len(allr)
    nrow = math.ceil(n / COLS)
    fig, ax = plt.subplots(nrow, COLS, figsize=(COLS * 2.35, nrow * 2.62))
    ax = np.atleast_2d(ax)
    for i, r in enumerate(allr):
        a = ax[i // COLS, i % COLS]
        p = f"{SC}/detector_overlays/{r['frame']}.png"
        if os.path.exists(p):
            im = Image.open(p)
            im.thumbnail((THUMB, THUMB), Image.LANCZOS)
            a.imshow(np.array(im))
        cap = r["frame"][:26]
        if "clIoU_adapt" in r:
            cap += f"\nclIoU {r['clIoU_adapt']:.3f}  PAR {r.get('PAR', float('nan')):.2f}"
            col = "#1d9e75" if r["clIoU_adapt"] >= 0.30 else (
                "#ba7517" if r["clIoU_adapt"] >= 0.15 else "#a32d2d")
        else:
            cap += f"\nno label · pred {100*r['pred_frac']:.1f}%"
            col = "#5f5e5a"
        a.set_title(cap, fontsize=6.2, color=col, pad=2)
        a.axis("off")
    for j in range(n, nrow * COLS):
        ax[j // COLS, j % COLS].axis("off")
    ns = [r["clIoU_adapt"] for r in scored]
    fig.suptitle(
        f"Best detector on all 62 frames — Meijering ridge, sigmas 1-4, "
        f"top {100-CFG['quantile']:g}% of response\n"
        f"orange = prediction · {len(scored)} frames scorable against a hand label "
        f"(median clIoU_adapt {np.median(ns):.3f}), {len(unscored)} unlabelled · "
        "ordered best to worst", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(f"{SC}/detector_contact_sheet.png", dpi=110)
    print(f"wrote detector_contact_sheet.png  ({n} frames, {nrow}x{COLS})")
    print(f"  scored median clIoU_adapt {np.median(ns):.4f}   "
          f"range {min(ns):.3f}-{max(ns):.3f}")


if __name__ == "__main__":
    main()
