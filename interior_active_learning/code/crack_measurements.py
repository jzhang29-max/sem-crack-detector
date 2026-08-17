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
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP_ROOT
from unified_pipeline import run_unified_pipeline
from extended_features import crack_shape_measurements

OUT_DIR = os.path.join(EXP_ROOT, "measurements")

IMAGES = [
    '260622_316_H_b2_back_CBS_01', '260622_316_H_b2_front_CBS_01', '260622_316_H_b2_front_CBS_02',
    '260622_316_H_b2_front_CBS_03', '260622_316_H_b2_front_CBS_04', '260622_316_H_b4_CBS_01',
    '260622_316_H_b4_CBS_02', '260622_316_amb_b3_CBS_01', '260622_316_amb_b3_CBS_02',
    '260708_316_H_b2_front_CBS_001', '260708_316_H_b2_front_CBS_002', '260708_316_H_b2_front_CBS_003',
    '260708_316_H_b2_front_CBS_004', '260708_316_H_b2_front_CBS_005', '260708_316_H_b2_front_CBS_006',
    '260708_316_H_b2_front_CBS_007', '260708_316_H_b2_front_CBS_008', '260708_316_H_b2_front_CBS_009',
    '260708_316_H_b2_front_CBS_010', '260708_316_H_b2_front_CBS_011', '260708_316_H_b2_front_CBS_012',
    '260708_316_H_b2_front_CBS_013', '260708_316_H_b2_front_CBS_014', '260708_316_H_b2_front_CBS_015',
    '260708_316_H_b2_front_CBS_016',
]


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
            "MajorAxisLength_px", "MinorAxisLength_px", "Orientation_deg",
            "BoundaryRoughness", "CentroidX_px", "CentroidY_px"]
    out_df = pd.DataFrame(rows, columns=cols).sort_values("Area_px", ascending=False).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{image_name}_crack_measurements.csv")
    out_df.to_csv(out_path, index=False)
    print(f"{image_name}: {len(out_df)} distinct cracks measured -> {out_path}")
    return out_df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        for img in IMAGES:
            try:
                measure_image(img)
            except Exception as e:
                print(f"{img}: FAILED ({e})")
    elif len(sys.argv) > 1:
        measure_image(sys.argv[1])
    else:
        print(__doc__)
