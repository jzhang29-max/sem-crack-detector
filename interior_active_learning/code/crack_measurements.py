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
from common import EXP_ROOT, ORIGINAL_DIR, PROD_MODEL_PATH
from unified_pipeline import run_unified_pipeline
from extended_features import crack_shape_measurements

OUT_DIR = os.path.join(EXP_ROOT, "measurements")

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



def measure_image(image_name):
    stage = run_unified_pipeline(image_name)
    labeled, df, img8 = stage["labeled"], stage["df"], stage["img8"]
    crack_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
    img_area = crack_mask.size

    groups = measure.label(crack_mask, connectivity=2)
    n_groups = groups.max()
    rows = []
    for group_id in range(1, n_groups + 1):
        mask = groups == group_id
        if mask.sum() < 5:
            continue  # skeletonize/regionprops need a few pixels to be meaningful
        m = crack_shape_measurements(mask)
        ys, xs = np.where(mask)
        m.update({
            "SourceImage": image_name,
            "CrackID": group_id,
            "CentroidX_px": round(float(xs.mean()), 1),
            "CentroidY_px": round(float(ys.mean()), 1),
            "AreaPct_of_image": round(100 * m["Area_px"] / img_area, 4),
        })
        rows.append(m)

    cols = ["SourceImage", "CrackID", "Area_px", "AreaPct_of_image", "SkeletonLength_px",
            "MeanWidth_px", "MaxWidth_px", "Tortuosity", "BranchPointCount",
            "EllipseMajorAxis_px", "EllipseMinorAxis_px", "Orientation_deg",
            "BoundaryRoughness", "CentroidX_px", "CentroidY_px"]

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
    prov["n_cracks"] = int(len(out_df))
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
