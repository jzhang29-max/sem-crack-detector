#!/usr/bin/env python3
"""Human-guided, image-bounded masks: keep the operator's judgement, drop the brush geometry.

THE PROBLEM. This app applies corrections one way only. interior_candidates.
apply_pixel_corrections treats a painted pixel as "force CRACK" -- an absolute override -- so
a stroke's disc geometry lands in the exported mask. That is where the round scallops come
from, and on MAR_Amb_AS_ETD_0003 it means the export is 49.2% brush against the detector's
own 4.7%. Suppressing corrections entirely removes the scallops but throws away the reason
they were painted: the operator was filling in crack the detector missed.

So there were only two options, and both were wrong. This is the third.

THE IDEA IS NOT MINE -- it is the sibling TXM app's, whose export has used it all along
(app/core/pipeline.py:effective_mask, corrections="gate"): inside a crack stroke the
threshold DROPS rather than being forced True, so the stroke means "believe weaker evidence
here" and the boundary is still drawn by the image. Ported here, with the per-pixel evidence
test that app uses for its `tight` step -- darker than a large local box mean:

    gated = machine_prediction
            OR  (painted crack AND darker than the local box mean)
            AND NOT painted not-crack

Inside a stroke the operator decides WHERE to look; the image decides WHERE THE EDGE IS.
A stroke over bright matrix contributes nothing, which is the point -- it cannot manufacture
crack out of a brush sweep, and the 49.2%-painted frame collapses back toward what is
actually dark.

READ painted_kept_pct AGAINST 50%, NOT AGAINST 0%. "Darker than the local box mean" splits a
symmetric local distribution in half by construction, so ANY textured region -- a rough
fracture surface especially -- endorses about 50% of itself whatever is there. 50% is the null
expectation and means the paint carried no darkness information; it does NOT mean half the
stroke was real. Measured over 47 reviewed frames the median is 67.2%, so paint is darker than
chance on average -- but the two most heavily painted frames (MAR_Amb_AS_CBS_0003 and
_ETD_0003, 50.8% and 49.2% of the frame painted) sit at 50.8% and 52.3%, i.e. exactly at the
null. On those two the gate is uninformative and the residual 26% it keeps is texture, not
evidence. This is the same trap as a percentile threshold, which marks a fixed share dark
however clean the image is.

WHAT THIS IS NOT. It is not a claim that gating is more accurate. It is not scored against
the labels here, because the labels ARE the strokes -- scoring a stroke-derived mask against
the strokes that made it is the circularity this project keeps tripping over. What is
reported is geometry: how much area each convention produces, and how much of the painted
region survives the image's verdict.

Usage:  python3 crack_export/tools/gated_masks.py [--box 301] [--limit N]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

Image.MAX_IMAGE_PIXELS = None
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
DERIVED = os.environ.get("SEMCRACK_DERIVED", os.path.join(REPO, "crack_export", "derived"))
sys.path.insert(0, os.path.join(REPO, "interior_active_learning", "code"))
sys.path.insert(0, os.path.join(REPO, "code"))
from common import PAINT_DIR, contrast_kwargs_for            # noqa: E402
from detect_cracks import load_as_uint8, find_field_of_view   # noqa: E402

MACH = os.path.join(DERIVED, "machine_masks")
OUT = os.path.join(DERIVED, "gated_masks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=int, default=301,
                    help="local box-mean window, in px. 301 matches the TXM app's tighten step.")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    stems = sorted(os.path.basename(p).replace("_machine.png", "")
                   for p in glob.glob(f"{MACH}/*_machine.png"))
    if a.limit:
        stems = stems[:a.limit]
    if not stems:
        sys.exit(f"no machine masks under {MACH} -- run machine_only_masks.py first")

    rows = []
    for i, stem in enumerate(stems, 1):
        machine = np.array(Image.open(f"{MACH}/{stem}_machine.png").convert("L")) < 128
        cmp_ = os.path.join(PAINT_DIR, f"{stem}_correction_mask.png")
        if not os.path.exists(cmp_):
            # Nothing painted: gated IS machine. Write it anyway so the directory is complete
            # and a consumer never has to know which frames were reviewed.
            Image.fromarray(np.where(machine, 0, 255).astype(np.uint8), "L").save(
                os.path.join(OUT, f"{stem}_gated.png"), optimize=True)
            rows.append(dict(frame=stem, reviewed=False, machine_pct=100 * float(machine.mean()),
                             paste_pct=100 * float(machine.mean()),
                             gated_pct=100 * float(machine.mean()), painted_kept_pct=None))
            continue

        cm = np.array(Image.open(cmp_))
        if cm.ndim > 2:
            cm = cm[..., 0]
        if cm.shape != machine.shape:
            rows.append(dict(frame=stem, error=f"{cm.shape} vs {machine.shape}"))
            continue
        painted, notc = cm == 1, cm == 2

        img8 = load_as_uint8(os.path.join(REPO, "original", f"{stem}.tif"),
                             **contrast_kwargs_for(stem))
        if img8.shape != machine.shape:
            x0, y0, x1, y1 = find_field_of_view(img8)
            img8 = img8[y0:y1, x0:x1]
        if img8.shape != machine.shape:
            rows.append(dict(frame=stem, error=f"image {img8.shape} vs {machine.shape}"))
            continue

        # Local box mean over the image. Taken on float32: a uint8 accumulator wraps, and a
        # wrapped mean would silently invert the comparison on bright frames.
        boxmean = uniform_filter(img8.astype(np.float32), size=a.box, mode="nearest")
        darker = img8.astype(np.float32) < boxmean

        gated = (machine | (painted & darker)) & ~notc
        paste = (machine | painted) & ~notc            # what the app ships today

        Image.fromarray(np.where(gated, 0, 255).astype(np.uint8), "L").save(
            os.path.join(OUT, f"{stem}_gated.png"), optimize=True)

        rows.append(dict(
            frame=stem, reviewed=True,
            machine_pct=100 * float(machine.mean()),
            paste_pct=100 * float(paste.mean()),
            gated_pct=100 * float(gated.mean()),
            painted_pct=100 * float(painted.mean()),
            # of the painted region, how much the image endorses
            painted_kept_pct=100 * float(darker[painted].mean()) if painted.any() else None,
            # what gating ADDS over the machine alone -- the operator's real contribution
            gated_adds_pct=100 * float((gated & ~machine).mean()),
            paste_adds_pct=100 * float((paste & ~machine).mean())))
        print(f"  [{i}/{len(stems)}] {stem[:42]:<42} machine {rows[-1]['machine_pct']:5.2f}%"
              f"  gated {rows[-1]['gated_pct']:5.2f}%  paste {rows[-1]['paste_pct']:5.2f}%",
              flush=True)

    json.dump(rows, open(os.path.join(DERIVED, "gated_masks.json"), "w"))
    rev = [r for r in rows if r.get("reviewed") and "error" not in r]
    if rev:
        g = np.array([r["gated_pct"] for r in rev])
        p = np.array([r["paste_pct"] for r in rev])
        m = np.array([r["machine_pct"] for r in rev])
        k = np.array([r["painted_kept_pct"] for r in rev if r["painted_kept_pct"] is not None])
        ga = np.array([r["gated_adds_pct"] for r in rev])
        pa = np.array([r["paste_adds_pct"] for r in rev])
        print(f"\n  {len(rev)} reviewed frames")
        print(f"    machine  median {np.median(m):5.2f}% of frame")
        print(f"    gated    median {np.median(g):5.2f}%   adds {np.median(ga):5.2f} pp over machine")
        print(f"    paste    median {np.median(p):5.2f}%   adds {np.median(pa):5.2f} pp over machine")
        print(f"    of the painted region, the image endorses median {np.median(k):.1f}%")
    err = [r for r in rows if "error" in r]
    print(f"  errors: {len(err)}")
    for r in err[:5]:
        print(f"    {r['frame']}: {r['error']}")
    print(f"  masks: {OUT}")


if __name__ == "__main__":
    main()
