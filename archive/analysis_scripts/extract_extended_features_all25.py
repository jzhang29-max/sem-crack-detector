"""
Recompute BoundaryRoughness + BranchPointDensity (extended_features.py) for
every CURRENTLY LABELED candidate across all 25 images -- both the 71
original-ledger (Step-E) candidates and the ~250 interior (Step-H)
candidates -- so they can be pooled into a 13-feature dataset and tested
against the deployed 11-feature unified model via the same CV methodology.

Original candidates: reuses build_original_ledger_unified_features.py's own
mask-recovery logic (raw pre-correction-stage fallback for Labels absorbed
into a merged crack).

Interior candidates: regenerates the same concavity/bridge_corridor/
interior_fill candidate list build_interior_candidates_for_image() would,
then matches each freshly-regenerated candidate back to the ALREADY-LABELED
row in candidates/<image>_interior.csv using the exact same
(CandidateType, position, area) tolerance matching interior_candidates.py's
own _carry_forward_labels() uses for label persistence across regenerations
-- this is the established, already-relied-upon mechanism for this project,
not a new invented one.

Writes: candidates/extended_features_pooled.csv
  columns: SourceImage, Label, Source (original_ledger/interior),
           CandidateType, BoundaryRoughness, BranchPointDensity
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology, measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LEDGER_PATH, CANDIDATES_DIR, ORIGINAL_DIR, PROD_MODEL_PATH, contrast_kwargs_for
from interior_candidates import (
    run_production_pipeline, concavity_candidates, bridge_corridor_candidates,
    interior_fill_candidates,
)
from detect_cracks import (
    load_as_uint8, find_field_of_view, flatten_background, segment_dark_regions,
    clean_mask, compute_vesselness, extract_candidates, classify_with_model,
)
from extended_features import boundary_roughness, branch_point_density

OUT_CSV = os.path.join(CANDIDATES_DIR, "extended_features_pooled.csv")
WINDOW_MARGIN_PX = 400  # unused here (kept for parity/reference -- extended_features crops its own bbox)

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


def _raw_pipeline_labeled(image_name):
    """Same as build_original_ledger_unified_features.py's own helper --
    recovers a Label that no longer exists post-merge (absorbed into a
    bigger confirmed crack by a paint-app correction)."""
    image_path = os.path.join(ORIGINAL_DIR, f"{image_name}.tif")
    img8 = load_as_uint8(image_path, **contrast_kwargs_for(image_name))
    x0, y0, x1, y1 = find_field_of_view(img8)
    img8 = img8[y0:y1, x0:x1]
    flat = flatten_background(img8)
    dark_mask = segment_dark_regions(flat, img8=img8)
    clean = clean_mask(dark_mask, min_area_px=13)
    vesselness = compute_vesselness(flat)
    labeled, df = extract_candidates(clean, flat, vesselness, min_area_px=40)
    df = classify_with_model(df, PROD_MODEL_PATH)
    return labeled


def extract_original_ledger(image_name, stage, rows_out):
    ledger = pd.read_csv(LEDGER_PATH)
    ledger["Label"] = ledger["Label"].astype(int)
    group = ledger[ledger["SourceImage"] == image_name]
    if len(group) == 0:
        return

    labeled, df = stage["labeled"], stage["df"]
    raw_labeled = None
    for _, row in group.iterrows():
        label = int(row["Label"])
        candidate_mask = labeled == label
        if not candidate_mask.any():
            if raw_labeled is None:
                try:
                    raw_labeled = _raw_pipeline_labeled(image_name)
                except Exception:
                    raw_labeled = False
            if raw_labeled is not False:
                candidate_mask = raw_labeled == label
            if not candidate_mask.any():
                continue
        try:
            br = boundary_roughness(candidate_mask)
            bpd = branch_point_density(candidate_mask)
        except Exception as e:
            print(f"  SKIP {image_name} Label {label} (original_ledger): {e}")
            continue
        rows_out.append(dict(SourceImage=image_name, Label=label, Source="original_ledger",
                              CandidateType="original", BoundaryRoughness=br, BranchPointDensity=bpd))


def extract_interior(image_name, stage, rows_out):
    csv_path = os.path.join(CANDIDATES_DIR, f"{image_name}_interior.csv")
    if not os.path.exists(csv_path):
        return
    existing = pd.read_csv(csv_path)
    as_str = existing["IsCrack"].astype(str).str.strip().str.upper()
    is_true = as_str.isin(["TRUE", "1", "1.0"])
    is_false = as_str.isin(["FALSE", "0", "0.0"])
    labeled_rows = existing[is_true | is_false].copy()
    if len(labeled_rows) == 0:
        return

    labeled, df, flat, img8 = stage["labeled"], stage["df"], stage["flat"], stage["img8"]
    crack_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
    dist_to_crack = ndi.distance_transform_edt(~crack_mask)

    raw = concavity_candidates(crack_mask)
    raw += bridge_corridor_candidates(labeled, df, flat, crack_mask)
    raw += interior_fill_candidates(labeled, df, img8, crack_mask, dist_to_crack)

    fresh = []
    for mask_bool, ctype, parent in raw:
        ys, xs = np.where(mask_bool)
        fresh.append(dict(mask=mask_bool, ctype=ctype, x=xs.mean(), y=ys.mean(), area=int(mask_bool.sum())))

    n_matched = 0
    for _, old_row in labeled_rows.iterrows():
        same_type = [f for f in fresh if f["ctype"] == old_row["CandidateType"]]
        best = None
        for f in same_type:
            if abs(f["x"] - old_row["X"]) <= 5 and abs(f["y"] - old_row["Y"]) <= 5 \
                    and abs(f["area"] - old_row["Area"]) <= max(old_row["Area"] * 0.15, 5):
                best = f
                break
        if best is None:
            continue
        try:
            br = boundary_roughness(best["mask"])
            bpd = branch_point_density(best["mask"])
        except Exception as e:
            print(f"  SKIP {image_name} Label {int(old_row['Label'])} (interior): {e}")
            continue
        rows_out.append(dict(SourceImage=image_name, Label=int(old_row["Label"]), Source="interior",
                              CandidateType=old_row["CandidateType"], BoundaryRoughness=br,
                              BranchPointDensity=bpd))
        n_matched += 1
    print(f"  interior: matched {n_matched}/{len(labeled_rows)} labeled candidates to fresh regeneration")


def main():
    rows = []
    for image_name in IMAGES:
        print(f"=== {image_name} ===", flush=True)
        try:
            stage = run_production_pipeline(image_name)
        except Exception as e:
            print(f"  PIPELINE FAILED: {e}")
            continue
        extract_original_ledger(image_name, stage, rows)
        extract_interior(image_name, stage, rows)

    out_df = pd.DataFrame.from_records(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out_df)} rows)")
    print(out_df["Source"].value_counts())


if __name__ == "__main__":
    main()
