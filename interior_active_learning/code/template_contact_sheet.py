#!/usr/bin/env python3
"""Every paint template matching a prefix, on one page, grouped by specimen.

This is the APP's output -- interior_active_learning/paint/*_paint_template.png, what the
two-pass pipeline plus any hand corrections actually render. It is deliberately not
crack_export/analysis/sam3/contact_sheet.py, which sheets the offline ridge-filter
evaluation over the 62-frame corpus from detector_overlays/. Two different detectors over
two different frame sets; keeping one script for both would invite exactly the comparison
the repo spends several documents warning against.

    python3 template_contact_sheet.py --prefix MAR_H_ --prefix MAR_AmbB_ --out sheet.png

Ordering is by specimen_key, then name, so a 2x2 design reads as four blocks rather than
as 80 unrelated thumbnails. Each caption carries the frame's predicted area fraction --
measured from the template's own red channel, not read from a cache, because the cache is
deliberately not written for unreleased frames.
"""
import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import template_writer  # noqa: E402
from common import PAINT_DIR  # noqa: E402

try:
    from aggregate import specimen_key
except Exception:                                    # pragma: no cover
    def specimen_key(_):
        return None

THUMB = 420


def red_fraction(arr):
    """Share of the frame the renderer tinted as crack.

    The overlay writes (225,25,25) over a grey base, so R-G separates tint from base. The
    same >40 margin the rest of this repo uses; a bare R>threshold would count the bright
    speckle that covers these frames.
    """
    return float(((arr[:, :, 0].astype(np.int16) - arr[:, :, 1]) > 40).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", action="append", required=True,
                    help="frame-name prefix; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    names = sorted(
        f[:-len("_paint_template.png")]
        for f in os.listdir(PAINT_DIR)
        if f.endswith("_paint_template.png")
        and f.startswith(tuple(a.prefix))
    )
    if not names:
        sys.exit(f"no templates match {a.prefix} in {PAINT_DIR}. Render them first "
                 f"(regenerate_templates.py --only NAME), then re-run.")

    names.sort(key=lambda n: (str(specimen_key(n)), n))
    print(f"{len(names)} templates", flush=True)

    cells = []
    for i, n in enumerate(names, 1):
        # path_for_read, not a bare join: the overlay write is DEFERRED, so a template
        # whose render is still queued would be read stale or mid-write. This is a read,
        # and the barrier belongs to the read -- the same rule flip_region learned the
        # hard way. The suite scans for exactly this and caught the bare join here.
        _tp = template_writer.path_for_read(
            os.path.join(PAINT_DIR, f"{n}_paint_template.png"))
        with Image.open(_tp) as im:
            im = im.convert("RGB")
            # reducing_gap does the expensive shrink with a cheap box filter first; these
            # are 25 MP each and a straight LANCZOS over 80 of them is minutes of nothing.
            im.thumbnail((THUMB, THUMB), Image.LANCZOS, reducing_gap=3.0)
            arr = np.asarray(im)
        cells.append((n, arr, red_fraction(arr)))
        if i % 20 == 0:
            print(f"  {i}/{len(names)}", flush=True)

    ncol = a.cols
    nrow = math.ceil(len(cells) / ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 2.30, nrow * 2.60))
    ax = np.atleast_2d(ax)

    # One colour per specimen, so the blocks are visible without reading every caption.
    keys = sorted({str(specimen_key(n)) for n, _, _ in cells})
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    colour = {k: palette[i % len(palette)] for i, k in enumerate(keys)}

    for i, (n, arr, frac) in enumerate(cells):
        axi = ax[i // ncol][i % ncol]
        axi.imshow(arr)
        k = str(specimen_key(n))
        axi.set_title(f"{n}\n{100 * frac:.2f}% predicted", fontsize=5.6,
                      color=colour[k], pad=2.4)
        axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_edgecolor(colour[k]); s.set_linewidth(1.5)
    for j in range(len(cells), nrow * ncol):
        ax[j // ncol][j % ncol].axis("off")

    med = 100 * float(np.median([f for _, _, f in cells]))
    sub = "  ".join(f"{k}: n={sum(1 for n,_,_ in cells if str(specimen_key(n))==k)}"
                    for k in keys)
    fig.suptitle(
        (a.title or "Paint templates") +
        f"\n{len(cells)} frames, median {med:.2f}% predicted   |   {sub}"
        "\nDETECTOR OUTPUT, NOT REVIEWED LABELS -- these are candidates to correct",
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(a.out, dpi=145)
    print(f"wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
