"""
Give the ORIGINAL (Step E) candidates listed in manual_corrections_ledger.csv
the SAME 11-feature schema used by the interior-candidate model
(INTERIOR_FEATURE_COLUMNS in active_learning_select.py), using ONLY
genuinely human-verified labels (the ledger), so they can be pooled with the
existing interior-candidate examples into one honest, unified dataset.

Run once, standalone (not imported elsewhere):
    python3 build_original_ledger_unified_features.py

Writes:
    ../candidates/original_ledger_unified_features.csv
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology, measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LEDGER_PATH, CANDIDATES_DIR, ORIGINAL_DIR, PROD_MODEL_PATH, contrast_kwargs_for
from active_learning_select import INTERIOR_FEATURE_COLUMNS
from interior_candidates import run_production_pipeline
from detect_cracks import (
    load_as_uint8, find_field_of_view, flatten_background, segment_dark_regions,
    clean_mask, compute_vesselness, extract_candidates, classify_with_model,
)

OUT_CSV = os.path.join(CANDIDATES_DIR, "original_ledger_unified_features.csv")

# Same scale as interior_fill_candidates' max_reach_px=500 and the task's own
# "400px margin" guidance -- generous enough that a genuinely isolated
# candidate's window still covers a meaningful local neighborhood, but far
# smaller than a whole-image distance-transform recomputation per candidate
# (the perf trap _region_features's own docstring warns about).
WINDOW_MARGIN_PX = 400


def _raw_pipeline_labeled(image_name):
    """Recompute candidate Labels the way run_production_pipeline() does,
    but stopping right after classify_with_model() -- i.e. BEFORE
    apply_pixel_corrections()/merge_large_cracks() can split, merge, or drop
    a Label. A ledger Label number refers to *this* raw numbering (assigned
    once by extract_candidates() and never reused), so a Label that a later
    paint-app correction subsequently merged into a bigger confirmed crack
    (and which therefore no longer exists as its own Label by the time
    run_production_pipeline() finishes) can still be recovered from here.

    img8/flat/vesselness are pure functions of the source image and are
    identical to run_production_pipeline()'s stage dict for the same image
    (nothing about corrections touches them) -- only `labeled`/`df` differ,
    so only `labeled` from this raw pass is actually needed by the caller."""
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


def _original_candidate_features(candidate_mask_full, flat, img8, vesselness, crack_mask_full,
                                  margin=WINDOW_MARGIN_PX):
    """Mirrors interior_candidates._region_features(), but for an ORIGINAL
    candidate that may itself be part of crack_mask_full (if the ledger says
    it's a confirmed crack). Naively measuring "distance to nearest confirmed
    crack pixel" would then trivially be 0 for every real crack (it IS part
    of crack_mask) -- so this always excludes the candidate's OWN pixels
    (candidate_mask_full, a plain boolean footprint -- robust even when the
    candidate's original Label number no longer exists in the FINAL labeled
    array post-merge; see _raw_pipeline_labeled) from the crack mask before
    measuring distance/boundary-touching to any OTHER confirmed crack.

    For performance, distance-transform + dilation are computed on a local
    crop (bbox +/- margin, clipped to image bounds) rather than the whole
    frame -- same class of perf lesson already documented on
    _region_features itself (avoid whole-image ops per candidate).

    Returns (feats_dict, window_was_crack_free: bool).
    """
    ys, xs = np.where(candidate_mask_full)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    area = int(candidate_mask_full.sum())

    props = measure.regionprops(candidate_mask_full[y0:y1, x0:x1].astype(np.uint8))[0]
    minor = props.axis_minor_length if props.axis_minor_length > 0 else 0.5
    elongation = props.axis_major_length / minor
    perim = props.perimeter if props.perimeter > 0 else 1.0
    circularity = min(4 * np.pi * props.area / (perim ** 2), 1.0)

    H, W = candidate_mask_full.shape
    wy0, wy1 = max(0, y0 - margin), min(H, y1 + margin)
    wx0, wx1 = max(0, x0 - margin), min(W, x1 + margin)

    # Exclude this candidate's OWN pixels from the crack mask -- the whole
    # point of this function (see docstring / task spec). Cheap boolean ops
    # over the full array; only the expensive distance transform + dilation
    # below are restricted to the local window.
    crack_mask_excl_self_full = crack_mask_full & ~candidate_mask_full
    crack_excl_win = crack_mask_excl_self_full[wy0:wy1, wx0:wx1]
    candidate_win = candidate_mask_full[wy0:wy1, wx0:wx1]

    window_is_crack_free = not crack_excl_win.any()
    if window_is_crack_free:
        # No OTHER confirmed crack pixel anywhere in this generous local
        # window -- a real, valid data point (this candidate is genuinely
        # isolated), not an error. distance_transform_edt on an all-True
        # input (no zero/background pixel present) falls back to measuring
        # distance to the ARRAY EDGE (an artifact of the algorithm, verified
        # directly), which would be a meaningless number here -- so instead
        # fall back to the window's own max possible distance (its
        # diagonal), a value at least as large as anything genuinely
        # measurable within this window.
        fallback_dist = float(np.hypot(wy1 - wy0, wx1 - wx0))
        mean_dist_to_crack = fallback_dist
        frac_touching_crack = 0.0
    else:
        dist_local = ndi.distance_transform_edt(~crack_excl_win)
        mean_dist_to_crack = float(dist_local[candidate_win].mean())

        dil_local = morphology.binary_dilation(crack_excl_win, morphology.disk(1))
        boundary_win = candidate_win & ~morphology.binary_erosion(candidate_win)
        by, bx = np.where(boundary_win)
        touching = int(dil_local[by, bx].sum()) if len(by) else 0
        frac_touching_crack = touching / max(1, len(by))

    feats = {
        "LogArea": float(np.log10(max(area, 1))),
        "Elongation": elongation,
        "Solidity": props.solidity,
        "Eccentricity": props.eccentricity,
        "Extent": props.extent,
        "Circularity": circularity,
        "MeanRawBrightness": float(img8[candidate_mask_full].mean()),
        "MeanFlatBrightness": float(flat[candidate_mask_full].mean()),
        "MeanVesselness": float(vesselness[candidate_mask_full].mean()),
        "FracBoundaryTouchingCrack": frac_touching_crack,
        "MeanDistToCrack": mean_dist_to_crack,
    }
    return feats, window_is_crack_free


def main():
    ledger = pd.read_csv(LEDGER_PATH)
    ledger["Label"] = ledger["Label"].astype(int)

    records = []
    skipped = []
    n_empty_window = 0

    for image_name, group in ledger.groupby("SourceImage", sort=False):
        print(f"=== {image_name}: running production pipeline ({len(group)} ledger rows) ===")
        try:
            stage = run_production_pipeline(image_name)
        except Exception as e:
            for _, row in group.iterrows():
                skipped.append((image_name, int(row["Label"]), f"pipeline failed: {e}"))
            print(f"  PIPELINE FAILED for {image_name}: {e}")
            continue

        labeled, df, flat, img8, vesselness = (
            stage["labeled"], stage["df"], stage["flat"], stage["img8"], stage["vesselness"]
        )
        crack_mask_full = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
        raw_labeled = None  # lazily computed only if some Label isn't found post-correction

        for _, row in group.iterrows():
            label = int(row["Label"])
            corrected_to = str(row["CorrectedTo"]).strip().lower()
            is_crack = corrected_to in ("true", "1")

            candidate_mask = labeled == label
            if not candidate_mask.any():
                # This Label may have been absorbed into a bigger confirmed
                # crack by a paint-app pixel correction (e.g. a force-crack
                # stroke that touched this candidate AND an adjacent crack
                # got merged into that crack's Label, per
                # interior_candidates._merge_blank_force_crack_into_touching_cracks)
                # -- it no longer exists as its own Label in the FINAL
                # labeled array, but it still exists at the raw
                # extract_candidates()/classify_with_model() stage the
                # ledger's Label numbering actually refers to. Recover the
                # candidate's own pixel footprint from there instead of
                # silently dropping a genuine, human-verified label.
                if raw_labeled is None:
                    try:
                        raw_labeled = _raw_pipeline_labeled(image_name)
                    except Exception as e:
                        raw_labeled = False
                        print(f"  raw-stage recovery unavailable for {image_name}: {e}")
                if raw_labeled is not False:
                    candidate_mask = raw_labeled == label
                if not candidate_mask.any():
                    skipped.append((image_name, label, "Label not present in labeled array (missing), "
                                                         "including at raw pre-correction stage"))
                    print(f"  SKIP {image_name} Label {label}: not present in labeled array "
                          f"(checked both final and raw pre-correction stage)")
                    continue
                print(f"  RECOVERED {image_name} Label {label} from raw pre-correction stage "
                      f"(absorbed into a merged crack Label by a paint-app correction)")

            try:
                feats, was_empty = _original_candidate_features(
                    candidate_mask, flat, img8, vesselness, crack_mask_full
                )
            except Exception as e:
                skipped.append((image_name, label, f"feature computation failed: {e}"))
                print(f"  SKIP {image_name} Label {label}: feature computation failed: {e}")
                continue

            if was_empty:
                n_empty_window += 1

            rec = dict(feats)
            rec.update({
                "IsCrack": is_crack,
                "SourceImage": image_name,
                "Label": label,
                "Source": "original_ledger",
            })
            records.append(rec)

    cols = INTERIOR_FEATURE_COLUMNS + ["IsCrack", "SourceImage", "Label", "Source"]
    out_df = pd.DataFrame.from_records(records, columns=cols)
    out_df.to_csv(OUT_CSV, index=False)

    print("\n=== SUMMARY ===")
    print(f"Ledger rows: {len(ledger)}")
    print(f"Successfully processed: {len(records)}")
    print(f"Skipped: {len(skipped)}")
    for img, lbl, reason in skipped:
        print(f"  - {img} Label {lbl}: {reason}")
    print(f"Label counts: {out_df['IsCrack'].value_counts().to_dict()}")
    print(f"Empty-local-search-window fallback used: {n_empty_window} / {len(records)}")
    print(f"Wrote {OUT_CSV} ({len(out_df)} rows, columns: {list(out_df.columns)})")


if __name__ == "__main__":
    main()
