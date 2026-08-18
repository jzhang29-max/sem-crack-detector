"""
Train (or extend) a crack/artifact classifier from MULTIPLE reviewed images.

Why multiple images: a classifier trained on one image's corrections learns
that image's quirks (its specific pore size, its specific contrast). Training
across several images -- ideally spanning different detector modes (this
dataset has both CBS and ETD) -- makes it generalize instead of overfit.

Workflow
--------
1. For each image you want to contribute training data:
     python3 detect_cracks.py IMAGE.tif --review-sheet
   This writes IMAGE_cracks.csv and IMAGE_review_page*.png.

2. Open the review page(s), find thumbnails whose red/cyan border is wrong,
   and flip that row's IsCrack value (TRUE/FALSE) in IMAGE_cracks.csv. Save
   your edits (in place, or as IMAGE_cracks_reviewed.csv -- either is fine,
   the filename doesn't matter).

3. Repeat for a handful of images -- 3-6 images covering both CBS and ETD is
   a reasonable start. Note: don't change --bg-sigma/--min-area/etc between
   generating a CSV and using it here; those flags affect how candidates are
   numbered, and a labels CSV must line up with a fresh segmentation of the
   SAME image under the SAME settings.

4. Build a manifest CSV with two columns, one row per reviewed image:
     image,labels_csv
     ../260708_316_H_b2_front_CBS_002.tif,../260708_316_H_b2_front_CBS_002_cracks.csv
     ../260708_316_H_b2_front_ETD_003.tif,../260708_316_H_b2_front_ETD_003_cracks.csv
   (paths are resolved relative to the manifest file's own folder)

5. Run:
     python3 train_classifier.py manifest.csv --model crack_clf.joblib

6. Apply the trained model to any other image in the series:
     python3 detect_cracks.py OTHER_IMAGE.tif --mode apply --model crack_clf.joblib
   or process the whole batch at once with batch_apply.py.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_cracks import (
    FEATURE_COLUMNS,
    clean_mask,
    compute_vesselness,
    extract_candidates,
    find_field_of_view,
    flatten_background,
    load_as_uint8,
    segment_dark_regions,
)


def build_candidates_for_image(image_path, bg_sigma=40, denoise_sigma=1.0,
                                open_radius=1, close_radius=3, min_area_px=40,
                                vessel_sigma_min=1, vessel_sigma_max=6, autocrop=True):
    img8 = load_as_uint8(image_path)
    if autocrop:
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
    flat = flatten_background(img8, sigma=bg_sigma)
    dark_mask = segment_dark_regions(flat, denoise_sigma=denoise_sigma)
    clean = clean_mask(dark_mask, open_radius=open_radius, close_radius=close_radius,
                        min_area_px=max(5, min_area_px // 3))
    vesselness = compute_vesselness(flat, sigma_min=vessel_sigma_min, sigma_max=vessel_sigma_max)
    _, df = extract_candidates(clean, flat, vesselness, min_area_px=min_area_px)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="CSV with columns: image,labels_csv (see module docstring).")
    parser.add_argument("--model", required=True, help="Where to save the trained classifier (.joblib).")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    manifest = pd.read_csv(args.manifest)
    manifest.columns = [c.strip() for c in manifest.columns]

    all_rows = []
    for _, r in manifest.iterrows():
        image_path = os.path.join(manifest_dir, r["image"])
        labels_path = os.path.join(manifest_dir, r["labels_csv"])

        df = build_candidates_for_image(image_path)
        labels_df = pd.read_csv(labels_path)
        labels_df.columns = [c.strip() for c in labels_df.columns]
        if "Label" not in labels_df.columns or "IsCrack" not in labels_df.columns:
            print(f"SKIPPING {labels_path}: missing 'Label' or 'IsCrack' column")
            continue

        merged = df.merge(labels_df[["Label", "IsCrack"]], on="Label", how="inner")
        merged["SourceImage"] = os.path.basename(image_path)
        n_pos = int(merged["IsCrack"].astype(bool).sum())
        print(f"{os.path.basename(image_path)}: {len(merged)} labeled candidates "
              f"({n_pos} crack / {len(merged) - n_pos} artifact)")
        all_rows.append(merged)

    if not all_rows:
        print("No usable labeled rows found -- nothing to train on.")
        sys.exit(1)

    full = pd.concat(all_rows, ignore_index=True)
    y = full["IsCrack"].astype(bool).values
    X_raw = full[FEATURE_COLUMNS].values

    if len(np.unique(y)) < 2:
        print("Need both crack and artifact examples across the manifest to train a classifier.")
        sys.exit(1)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    import joblib

    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    groups = full["SourceImage"].values

    def make_clf():
        return RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            class_weight="balanced", random_state=0,
        )

    # Leave-one-IMAGE-out cross-validation, not a row-level split. Candidates
    # from the same image are correlated (shared lighting, shared crack
    # morphology), so a random row split would leak information between
    # "train" and "test" and overstate accuracy. Holding out one whole image
    # per fold answers the question that actually matters: does the model
    # generalize to an image it has never seen?
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        print("\nOnly one image in the manifest -- can't do leave-one-image-out CV. Fitting without it.")
    else:
        logo = LeaveOneGroupOut()
        print(f"\nLeave-one-image-out cross-validation ({n_groups} images):")
        fold_accs, fold_bal_accs, fold_ns = [], [], []
        for train_idx, test_idx in logo.split(X, y, groups):
            held_out = groups[test_idx][0]
            clf_fold = make_clf()
            clf_fold.fit(X[train_idx], y[train_idx])
            pred = clf_fold.predict(X[test_idx])
            acc = accuracy_score(y[test_idx], pred)
            bal_acc = balanced_accuracy_score(y[test_idx], pred) if len(np.unique(y[test_idx])) > 1 else float("nan")
            n_test = len(test_idx)
            n_pos_test = int(y[test_idx].sum())
            fold_accs.append(acc)
            fold_bal_accs.append(bal_acc)
            fold_ns.append(n_test)
            bal_str = f"{bal_acc:.3f}" if not np.isnan(bal_acc) else "n/a (single class)"
            print(f"  held out {held_out:45s} n={n_test:4d} (crack={n_pos_test:4d})  "
                  f"accuracy={acc:.3f}  balanced_accuracy={bal_str}")

        mean_acc = float(np.mean(fold_accs))
        weighted_acc = float(np.average(fold_accs, weights=fold_ns))
        valid_bal = [b for b in fold_bal_accs if not np.isnan(b)]
        print(f"  ---")
        print(f"  mean per-image accuracy      : {mean_acc:.3f} +/- {np.std(fold_accs):.3f}")
        print(f"  candidate-weighted accuracy  : {weighted_acc:.3f}")
        if valid_bal:
            print(f"  mean balanced accuracy       : {np.mean(valid_bal):.3f} +/- {np.std(valid_bal):.3f}")

    # Final model trained on ALL labeled images -- the LOGO loop above is
    # purely for the honest generalization estimate printed above; none of
    # those fold-specific models are kept.
    clf = make_clf()
    clf.fit(X, y)
    importances = sorted(zip(FEATURE_COLUMNS, clf.feature_importances_), key=lambda t: -t[1])
    print("Feature importances:")
    for name, imp in importances:
        print(f"  {name:16s} {imp:.3f}")

    joblib.dump({"scaler": scaler, "clf": clf, "feature_names": FEATURE_COLUMNS}, args.model)
    print(f"\nSaved trained model: {args.model}")
    print("Apply it to another image with:")
    print(f'  python3 detect_cracks.py OTHER_IMAGE.tif --mode apply --model "{args.model}"')


if __name__ == "__main__":
    main()
