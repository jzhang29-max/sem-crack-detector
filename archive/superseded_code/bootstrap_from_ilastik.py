"""
Auto-generate training labels (Label,IsCrack CSVs) for train_classifier.py
from existing Ilastik probability maps, instead of manual review sheets.

Why: detect_cracks.py's --mode train workflow normally needs a human to open
a review sheet and flip TRUE/FALSE on individual candidates. But if an
Ilastik project has already been trained (interactively, pixel-by-pixel) on
an image, its probability map IS a ground-truth crack mask for that image --
so we can skip the manual step entirely: run the same candidate segmentation
this tool always uses, overlap each candidate region against the
thresholded Ilastik mask, and auto-assign IsCrack from majority overlap.
The result is a normal Label,IsCrack CSV, usable as a manifest row for
train_classifier.py exactly like a hand-reviewed one.

This only works for an (image, probability-map) pair whose pixel grids
match exactly (same crop, no resize) -- mismatched pairs are skipped with a
warning rather than silently misaligned.

Usage
-----
    python3 bootstrap_from_ilastik.py manifest_raw.csv --out-dir bootstrapped/

manifest_raw.csv columns: image,probability,channel
    image        - path to the raw TIFF (what detect_cracks.py will run on)
    probability  - path to the Ilastik Probabilities TIFF (1 or N channels)
    channel      - which channel is the crack class (0-indexed)

Writes one <name>_cracks.csv per row (Label,IsCrack + all feature columns,
same schema as a manual review CSV) plus a ready-to-use train_classifier.py
manifest (image,labels_csv) at --out-dir/manifest.csv.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_cracks import (
    clean_mask,
    compute_vesselness,
    extract_candidates,
    find_field_of_view,
    flatten_background,
    load_as_uint8,
    segment_dark_regions,
)


def load_probability_channel(path, channel):
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[..., channel]
    arr = arr.astype(np.float64)
    if arr.max() > 1.5:  # looks like 0-255 rather than 0-1
        arr = arr / 255.0
    return arr


def bootstrap_one(image_path, prob_path, channel, out_csv, bg_sigma=40, denoise_sigma=1.0,
                   open_radius=1, close_radius=3, min_area_px=40,
                   vessel_sigma_min=1, vessel_sigma_max=6, autocrop=False,
                   overlap_threshold=0.5, ground_truth_threshold=0.5):
    img8 = load_as_uint8(image_path)
    gt = load_probability_channel(prob_path, channel)

    if gt.shape != img8.shape:
        raise ValueError(
            f"pixel grid mismatch: {os.path.basename(image_path)} is {img8.shape} "
            f"but {os.path.basename(prob_path)} is {gt.shape} -- skipping, "
            f"can't safely align labels to candidates."
        )

    if autocrop:
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        gt = gt[y0:y1, x0:x1]

    flat = flatten_background(img8, sigma=bg_sigma)
    dark_mask = segment_dark_regions(flat, denoise_sigma=denoise_sigma)
    clean = clean_mask(dark_mask, open_radius=open_radius, close_radius=close_radius,
                        min_area_px=max(5, min_area_px // 3))
    vesselness = compute_vesselness(flat, sigma_min=vessel_sigma_min, sigma_max=vessel_sigma_max)
    labeled, df = extract_candidates(clean, flat, vesselness, min_area_px=min_area_px)

    gt_mask = gt >= ground_truth_threshold
    is_crack = []
    overlap_fracs = []
    for label in df["Label"]:
        region = labeled == label
        frac = gt_mask[region].mean() if region.any() else 0.0
        overlap_fracs.append(frac)
        is_crack.append(frac >= overlap_threshold)

    df = df.copy()
    df["GTOverlapFraction"] = overlap_fracs
    df["IsCrack"] = is_crack

    n_pos = int(sum(is_crack))
    print(f"{os.path.basename(image_path)}: {len(df)} candidates, "
          f"{n_pos} auto-labeled crack / {len(df) - n_pos} artifact "
          f"(ground truth coverage: {gt_mask.mean()*100:.2f}% of image)")

    df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="CSV with columns: image,probability,channel")
    ap.add_argument("--out-dir", required=True, help="where to write per-image label CSVs and the training manifest")
    ap.add_argument("--overlap-threshold", type=float, default=0.5,
                     help="min fraction of a candidate's pixels that must fall inside the Ilastik crack mask to auto-label it IsCrack=True")
    ap.add_argument("--ground-truth-threshold", type=float, default=0.5,
                     help="probability threshold to binarize the Ilastik channel into a crack mask")
    args = ap.parse_args()

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    manifest = pd.read_csv(args.manifest)
    manifest.columns = [c.strip() for c in manifest.columns]
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for _, r in manifest.iterrows():
        image_path = os.path.join(manifest_dir, r["image"])
        prob_path = os.path.join(manifest_dir, r["probability"])
        channel = int(r["channel"])
        name = os.path.splitext(os.path.basename(image_path))[0]
        out_csv = os.path.join(args.out_dir, f"{name}_cracks.csv")
        try:
            bootstrap_one(
                image_path, prob_path, channel, out_csv,
                overlap_threshold=args.overlap_threshold,
                ground_truth_threshold=args.ground_truth_threshold,
            )
            rows.append({"image": image_path, "labels_csv": out_csv})
        except Exception as e:
            print(f"SKIPPING {os.path.basename(image_path)}: {e}")

    if not rows:
        print("No image/probability pairs could be bootstrapped.")
        sys.exit(1)

    train_manifest_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(rows).to_csv(train_manifest_path, index=False)
    print(f"\nWrote training manifest: {train_manifest_path}")
    print("Train with:")
    print(f'  python3 train_classifier.py "{train_manifest_path}" --model crack_clf.joblib')


if __name__ == "__main__":
    main()
