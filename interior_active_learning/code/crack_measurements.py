"""
Answers "does the pipeline generate a CSV of crack size/measurements?" for
the UNIFIED pipeline specifically: the OLD production pipeline
(detect_cracks.py) already saves a per-CANDIDATE CSV (training_data/
<image>_cracks.csv -- Area, Elongation, MeanDarkness, IsCrack, etc.), but
that's per-candidate-Label, computed BEFORE interior candidates fold in, and
never regenerated for the unified pipeline's final result.

This is genuinely new: one row per FINAL CRACK (connected component of the
final mask, after both passes + MST merge -- i.e. what a person looking at
the overlay would call "one crack"), with physical measurements a
materials-science reader wants: skeleton length (the actual curved path
length, not just a bounding-box diagonal), mean/max width, tortuosity,
branch-point count (is this a simple crack or a branching network?),
orientation, plus the raw shape/brightness features already used for
classification.

Usage
-----
    python3 crack_measurements.py <image_name>          # one image
    python3 crack_measurements.py --all                 # all 25 images
Writes: interior_active_learning/measurements/<image>_crack_measurements.csv
"""
import json as _json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP_ROOT, MEASUREMENTS_DIR, ORIGINAL_DIR, PROD_MODEL_PATH
from unified_pipeline import run_unified_pipeline
from extended_features import crack_shape_measurements

OUT_DIR = MEASUREMENTS_DIR

def all_images():
    """Every image in original/, from the filesystem.

    This was a frozen 25-name literal from an earlier round of the project. The MAR
    superalloy frames merged in later, the 24hr side captures, and anything the user
    uploads through the app were all absent, so `--all` printed a clean report for 25
    files while silently measuring none of the other 20 hand-corrected images -- the
    researcher got no CSV for the image they had just spent an hour labelling, and no
    warning either.
    """
    if not os.path.isdir(ORIGINAL_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR)
                  if f.lower().endswith(".tif") and not f.startswith("apptest"))



def measure_stage(image_name, stage):
    """Turn one pipeline stage into measurement rows. No file I/O, no calibration.

    Split out of measure_image so a sensitivity sweep can re-measure the SAME geometry
    under a different decision threshold without either duplicating this logic or writing
    62 CSVs it does not want. The split matters more than it looks: if a sweep carried its
    own copy of the merge rule, the censoring test or the fragment count, it would drift
    from what measure_image actually writes, and it would then be reporting the
    sensitivity of a measurement nobody ships.

    Returns (rows, n_bridge_px).
    """
    labeled, df, img8 = stage["labeled"], stage["df"], stage["img8"]
    crack_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
    img_area = crack_mask.size

    # HONOUR THE MERGE DECISION THE PIPELINE ALREADY PAID FOR.
    #
    # merge_large_cracks exists because "the main crack is often segmented into several
    # large fragments ... each counts and displays as a separate crack even though they're
    # obviously one". It returns the connector geometry as stage["bridge_mask"], and
    # detect_cracks.save_bw_image ORs it into its own export -- but this CSV re-labelled
    # the bare crack_mask, so it reported one row per FRAGMENT. That understates
    # SkeletonLength_px for the main crack, which is the headline number in a fatigue or
    # creep write-up, and gives a crack count that disagrees with the bw export for the
    # same image.
    #
    # A bridge is only allowed to join things, never to become a crack on its own: after
    # the union, components that contain no originally-detected crack pixel are dropped,
    # so a connector whose endpoints both fell below threshold cannot invent a region.
    bridge = stage.get("bridge_mask")
    n_bridge_px = 0
    if bridge is not None:
        bridge = np.asarray(bridge, dtype=bool)
        if bridge.shape == crack_mask.shape and bridge.any():
            joined = measure.label(crack_mask | bridge, connectivity=2)
            keep = np.unique(joined[crack_mask])
            keep = keep[keep != 0]
            merged_mask = np.isin(joined, keep.tolist())
            n_bridge_px = int((merged_mask & ~crack_mask).sum())
            crack_mask = merged_mask

    fragments = measure.label(
        np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist()), connectivity=2)
    groups = measure.label(crack_mask, connectivity=2)
    n_groups = groups.max()
    rows = []
    for group_id in range(1, n_groups + 1):
        mask = groups == group_id
        if mask.sum() < 5:
            continue  # skeletonize/regionprops need a few pixels to be meaningful
        m = crack_shape_measurements(mask)
        # How many separately-detected fragments this row merges. 1 means the pipeline
        # found it whole; >1 means merge_large_cracks joined it, and the reader can see
        # that rather than having to trust it.
        m["NFragmentsMerged"] = int(len({int(v) for v in np.unique(fragments[mask])} - {0}))
        # RIGHT-CENSORING. A crack that reaches the frame edge continues outside it, so its
        # measured length is a LOWER BOUND, not a length. Pooling censored and uncensored
        # cracks into one mean -- or into "longest crack per frame" -- silently mixes two
        # different quantities, and the bias is not random: the longest cracks are exactly the
        # ones most likely to run off the edge, so the statistic most affected is the one a
        # fatigue study reports. Flagging it is the minimum; a survival-style estimator would
        # be the proper treatment and is not attempted here.
        _touch = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
        m["TouchesBoundary"] = bool(_touch > 0)
        m["BoundaryPx"] = _touch
        m["LengthIsCensored"] = bool(_touch > 0)
        ys, xs = np.where(mask)
        m.update({
            "SourceImage": image_name,
            "CrackID": group_id,
            "CentroidX_px": round(float(xs.mean()), 1),
            "CentroidY_px": round(float(ys.mean()), 1),
            "AreaPct_of_image": round(100 * m["Area_px"] / img_area, 4),
        })
        rows.append(m)
    return rows, n_bridge_px


def measure_image(image_name):
    rows, n_bridge_px = measure_stage(image_name, run_unified_pipeline(image_name))

    cols = ["SourceImage", "CrackID", "Area_px", "AreaPct_of_image", "SkeletonLength_px",
            "MeanWidth_px", "MaxWidth_px", "Tortuosity", "BranchPointCount",
            "EllipseMajorAxis_px", "EllipseMinorAxis_px", "Orientation_deg",
            "BoundaryRoughness", "CentroidX_px", "CentroidY_px", "NFragmentsMerged",
            "TouchesBoundary", "BoundaryPx", "LengthIsCensored"]

    # Physical units when this image has been calibrated. A crack length in PIXELS is not
    # a publishable quantity, so a CSV that only offers _px columns cannot be used for the
    # paper the tool exists to support. Uncalibrated images still export pixel columns and
    # say so in the sidecar -- they do NOT get um columns filled with a 1.0 default, which
    # would be indistinguishable from a real measurement.
    import calibration as _cal
    umpx = _cal.get_um_per_px(image_name)
    if umpx:
        rows = [_cal.convert_row(r, umpx) for r in rows]
        cols = cols + [_cal.um_column_name(c) for c in
                       ["Area_px", "SkeletonLength_px", "MeanWidth_px", "MaxWidth_px",
                        "EllipseMajorAxis_px", "EllipseMinorAxis_px"]]

    out_df = pd.DataFrame(rows, columns=cols).sort_values("Area_px", ascending=False).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{image_name}_crack_measurements.csv")
    out_df.to_csv(out_path, index=False)

    # Provenance beside every table. Without this, once a CSV leaves the machine no number
    # in it can be traced to a model, a threshold, or a calibration.
    prov = _cal.provenance_header(image_name, PROD_MODEL_PATH, None)
    # Where the mask came from. A CSV whose regions were segmented by another tool but does
    # not say so is worse than no CSV: the numbers read as native to this pipeline.
    import external_mask as _em
    prov.update(_em.provenance_for(image_name))
    prov["n_cracks"] = int(len(out_df))
    prov["bridge_px_added"] = n_bridge_px
    _cens = int(out_df["LengthIsCensored"].sum()) if len(out_df) else 0
    prov["cracks_censored_by_frame_edge"] = _cens
    prov["censoring_note"] = (
        "A crack touching the frame edge continues outside it, so its length is a "
        "LOWER BOUND. Do not pool censored and uncensored lengths, and do not "
        "report longest-crack-per-frame across conditions without accounting for "
        "it: the longest cracks are the most likely to be censored.")
    prov["columns"] = list(out_df.columns)
    _json.dump(prov, open(os.path.join(OUT_DIR, f"{image_name}_provenance.json"), "w"),
               indent=1)

    units = f"{umpx:.5f} um/px" if umpx else "PIXELS ONLY (uncalibrated)"
    print(f"{image_name}: {len(out_df)} distinct cracks measured [{units}] -> {out_path}")
    return out_df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        imgs = all_images()
        failed = []
        for img in imgs:
            try:
                measure_image(img)
            except Exception as e:
                failed.append((img, f"{type(e).__name__}: {e}"))
                print(f"{img}: FAILED ({type(e).__name__}: {e})")
        print(f"\n{len(imgs) - len(failed)}/{len(imgs)} measured, {len(failed)} failed")
        for img, why in failed:
            print(f"  FAILED {img}: {why}")
        # A run that measured nothing used to exit 0 and read as success.
        if failed:
            sys.exit(1)
    elif len(sys.argv) > 1:
        measure_image(sys.argv[1])
    else:
        print(__doc__)
