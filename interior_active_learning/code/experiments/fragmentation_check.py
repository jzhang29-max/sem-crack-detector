"""The measurement that reversed the shipping decision, made reproducible.

SAM 2 refinement was briefly the default because on adjudicated pixels -- the ~8% of a frame a
human has marked -- it beat the bare detector on all four metrics. A user then looked at the
overlays and said they had got worse. They were right, and the reason the metric missed it is
that the metric never looked at the other 92% of the frame.

This script measures what the overlays showed, on the whole frame, for one image, and writes
the numbers to a JSON artifact so the table in README.md ("What refinement does to the mask")
traces to something a reader can re-run rather than to a claim.

    ./.venv/bin/python3 interior_active_learning/code/experiments/fragmentation_check.py

Roughly 8 minutes on the default frame: SAM 2 is prompted once per accepted region. Output:
    interior_active_learning/code/experiments/fragmentation_check.json

What it reports, per arm (bare vs refined) and per region:
    predicted_px   how much crack the arm claims
    components     how many separate pieces that claim is in
    skeleton_px    total centreline length -- the tool's headline "crack length"

A refinement that trims boundaries fairly would cut pixels a little and leave components and
skeleton roughly alone. Fragmentation looks the opposite: FEWER pixels but MORE components and
a LONGER skeleton, because solid regions become lacy and break into pieces.

HUMAN CORRECTIONS ARE NEUTRALISED for this measurement, so both arms see the same detector
output and the difference is refinement's alone. That is deliberate, and it means the absolute
pixel counts here are LOWER than what an overlay shows for the same frame -- an overlay
includes the human's added pixels. The percentages are the comparable quantity; the absolutes
are only comparable to each other.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, _HERE)

import detector_config as _dc
import sam2_hybrid as _sh
import sam2_refine as _sr
import unified_pipeline as up

#: The frame the original ad-hoc diagnostic used, kept so the numbers are comparable. It is a
#: large through-crack frame -- the case where fragmentation matters most, because the headline
#: length comes from one long connected crack.
FRAME = "260708_316_H_b2_front_CBS_002"

#: Beside the other experiment artifacts, which is where every other script writes.
OUT = os.path.join(_HERE, "fragmentation_check.json")


def _per_region(lab, accepted, refined):
    """Pixel change region by region, so the effect cannot hide in a whole-frame average."""
    out = []
    for lbl in accepted:
        before = (lab == lbl)
        n0 = int(before.sum())
        if n0 == 0:
            continue
        n1 = int((refined & before).sum())
        out.append({"label": int(lbl), "px_before": n0, "px_after": n1,
                    "pct_change": round(100.0 * (n1 - n0) / n0, 2)})
    return sorted(out, key=lambda r: -r["px_before"])


def run(frame=FRAME):
    # PINNED, not inherited: this script's whole purpose is to compare two named detectors, so
    # neither arm may depend on whatever the module default happens to be.
    with _dc.pinned("off"):
        with _sh._without_human_input():
            st = up.run_unified_pipeline(frame)
    lab, df, img8 = st["labeled"], st["df"], st["img8"]
    accepted = df.loc[df["IsCrack"], "Label"].tolist()
    bare = np.isin(lab, accepted)
    print(f"  {frame}: {len(df)} candidates, {len(accepted)} accepted, "
          f"{int(bare.sum()):,} crack px", flush=True)

    # THE MODEL THE APP ACTUALLY USES, read from sam2_refine rather than named here. A first
    # version of this script hard-coded hiera-large while `--sam2 refine` ships hiera-tiny, so
    # it measured a configuration no user can get. Reading the constant means the two cannot
    # drift apart again.
    proc, model, dev, torch = _sh._load_sam2(_sr.DEFAULT_MODEL)
    print(f"  SAM 2 on {dev}: prompting {len(accepted)} boxes", flush=True)
    boxes = _sh._boxes_from(lab, accepted)
    refined = _sh._sam2_mask_for_boxes(img8, boxes, proc, model, dev, torch)

    arms = {"bare": _sh._shape_cost(bare), "refined": _sh._shape_cost(refined)}
    regions = _per_region(lab, accepted, refined)

    # WHERE THE PIXELS GO. The per-region table only sees refined pixels that fall inside an
    # original candidate's footprint, so it cannot explain a frame total that rises while
    # almost every region shrinks. The difference is spill: area SAM 2 claims outside the
    # boxes' original masks. Without this the two views of the same run look contradictory.
    inside = int((refined & bare).sum())
    outside = int((refined & ~bare).sum())
    trimmed = int((bare & ~refined).sum())

    def delta(k):
        a, b = arms["bare"][k], arms["refined"][k]
        return round(100.0 * (b - a) / a, 1) if a else None

    payload = {
        "frame": frame,
        "detector_arms": {"bare": _dc.stamp("off"), "refined": _dc.stamp("refine")},
        "sam2_model": _sr.DEFAULT_MODEL,
        "whole_frame": arms,
        "pct_change": {k: delta(k) for k in ("predicted_px", "components", "skeleton_px")},
        "per_region": regions,
        "regions_that_lost_pixels": sum(1 for r in regions if r["pct_change"] < 0),
        "regions_total": len(regions),
        "pixel_flow": {"kept_inside_original": inside,
                       "spilled_outside_original": outside,
                       "trimmed_from_original": trimmed},
        "reading": (
            "Fewer pixels with more components and a longer skeleton is fragmentation, not "
            "a cleaner boundary: the same crack is being reported as more, shorter pieces. "
            "Crack count and crack length are the two numbers this tool exists to produce, "
            "and an adjudicated-pixel f1 cannot see either of them move."),
    }
    out = OUT
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n  {'':14}{'bare':>12}{'refined':>12}{'change':>10}")
    for k in ("predicted_px", "components", "skeleton_px"):
        print(f"  {k:14}{arms['bare'][k]:>12,}{arms['refined'][k]:>12,}"
              f"{payload['pct_change'][k]:>9}%")
    print(f"\n  {payload['regions_that_lost_pixels']} of {payload['regions_total']} regions "
          f"lost pixels")
    print(f"  pixel flow: {inside:,} kept, {trimmed:,} trimmed away, "
          f"{outside:,} spilled outside the original regions")
    print(f"  wrote {out}")
    return payload


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print(f"usage: fragmentation_check.py [FRAME_NAME]\n"
              f"  FRAME_NAME  a frame in original/, without the .tif (default {FRAME})")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        print(f"unknown option {sys.argv[1]!r}. Pass a frame name, or --help.")
        sys.exit(2)
    run(sys.argv[1] if len(sys.argv) > 1 else FRAME)
