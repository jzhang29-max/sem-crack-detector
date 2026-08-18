"""
Compare three different ways of training the crack/artifact classifier, all
evaluated so the numbers are actually comparable to each other.

Why: with only 15-16 images of labeled data, how you split train/test matters
a lot -- leave-one-image-out (each fold = 1 image) showed huge fold-to-fold
variance (some images are "easy", one was a real outlier). This script tries
three alternatives and reports honest, image-grouped-and-held-out accuracy
for each, so you can see the actual tradeoffs instead of taking one number
on faith.

Experiments
-----------
1. Logistic Regression, same 5-fold grouped split as #1: a completely
   different (linear, far fewer effective parameters) model family, using
   the identical fold assignment as experiment 1 so the comparison isolates
   "does the model family matter" rather than "was the split different."

All three are evaluated with the SAME metric (per-image held-out accuracy)
so the comparison table at the end is apples-to-apples.

Usage
-----
    python3 compare_training_strategies.py LABELS_DIR --out-dir OUT_DIR

LABELS_DIR should contain one CSV per image (Label, <feature columns>,
IsCrack) -- e.g. the auto_baseline/ folder from a previous auto-mode run.
Filenames are used as the per-image group identifier.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_cracks import FEATURE_COLUMNS

FIXED_4 = [
    "260708_316_H_b2_front_CBS_002",
    "260708_316_H_b2_front_CBS_005",
    "260708_316_H_b2_front_CBS_010",
    "260708_316_H_b2_front_CBS_012",
]


def load_all_labels(labels_dir, baseline_dir=None, correction_weight=1.0):
    """
    Load per-image label CSVs. If baseline_dir is given, any row whose
    IsCrack differs from the baseline (e.g. the original auto-clustering
    guess) is flagged IsManualCorrection=True and given `correction_weight`
    instead of 1.0 -- a manual correction is a much stronger signal than an
    unreviewed auto-generated label, and a handful of them get outvoted by
    thousands of ordinary rows unless the training explicitly weights them
    more heavily.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(labels_dir, "*_cracks.csv"))):
        name = os.path.basename(f).replace("_cracks.csv", "")
        df = pd.read_csv(f)
        if len(df) == 0:
            continue
        df = df.copy()
        df["SourceImage"] = name
        df["IsManualCorrection"] = False
        df["SampleWeight"] = 1.0

        baseline_f = os.path.join(baseline_dir, f"{name}_cracks.csv") if baseline_dir else None
        if baseline_f and os.path.exists(baseline_f):
            baseline = pd.read_csv(baseline_f)[["Label", "IsCrack"]].rename(columns={"IsCrack": "BaselineIsCrack"})
            merged = df.merge(baseline, on="Label", how="left")
            differs = (merged["IsCrack"].astype(bool) != merged["BaselineIsCrack"].astype(bool)).fillna(False)
            df["IsManualCorrection"] = differs.values
            df.loc[df["IsManualCorrection"], "SampleWeight"] = correction_weight

        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def per_image_accuracy(df_test, pred):
    df_test = df_test.copy()
    df_test["pred"] = pred
    out = {}
    for name, g in df_test.groupby("SourceImage"):
        out[name] = float((g["IsCrack"].astype(bool).values == g["pred"].astype(bool).values).mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels_dir", help="Folder of per-image label CSVs (Label, features..., IsCrack).")
    parser.add_argument("--out-dir", required=True, help="Where to save the 3 models + comparison CSV.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--baseline-dir", default=None,
                         help="Folder of original (pre-correction) label CSVs, e.g. auto_baseline/. "
                              "Rows that differ from this baseline are treated as manual corrections "
                              "and up-weighted during training.")
    parser.add_argument("--correction-weight", type=float, default=15.0,
                         help="How many ordinary auto-labeled rows one manual correction should "
                              "outweigh during training (default 15).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    full = load_all_labels(args.labels_dir, baseline_dir=args.baseline_dir,
                            correction_weight=args.correction_weight)
    images = sorted(full["SourceImage"].unique())
    n_corrections = int(full["IsManualCorrection"].sum())
    print(f"Loaded {len(full)} labeled candidates from {len(images)} images "
          f"({n_corrections} flagged as manual corrections, weight={args.correction_weight}).")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    import joblib

    X_all = full[FEATURE_COLUMNS].values
    y_all = full["IsCrack"].astype(bool).values
    groups_all = full["SourceImage"].values
    weights_all = full["SampleWeight"].values

    scaler = StandardScaler().fit(X_all)
    Xs_all = scaler.transform(X_all)

    per_image_results = {img: {} for img in images}

    # ---------------- Experiment 1: RandomForest, grouped 5-fold ----------------
    print(f"\n=== Experiment 1: RandomForest, {args.n_folds}-fold grouped CV (all {len(images)} images) ===")
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(Xs_all, y_all, groups_all)):
        clf = RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=0)
        clf.fit(Xs_all[train_idx], y_all[train_idx], sample_weight=weights_all[train_idx])
        pred = clf.predict(Xs_all[test_idx])
        accs = per_image_accuracy(full.iloc[test_idx], pred)
        held_out_imgs = ", ".join(sorted(accs))
        print(f"  fold {fold_i}: held out [{held_out_imgs}]")
        for img, acc in accs.items():
            per_image_results[img]["Exp1_GroupKFold_RF"] = acc

    # ---------------- Experiment 2: fixed 4-image training ----------------
    print(f"\n=== Experiment 2: RandomForest, trained ONLY on the original 4 images ===")
    train_mask = full["SourceImage"].isin(FIXED_4).values
    clf_fixed = RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=0)
    clf_fixed.fit(Xs_all[train_mask], y_all[train_mask], sample_weight=weights_all[train_mask])
    test_idx = np.where(~train_mask)[0]
    pred = clf_fixed.predict(Xs_all[test_idx])
    accs = per_image_accuracy(full.iloc[test_idx], pred)
    for img, acc in accs.items():
        per_image_results[img]["Exp2_Fixed4_RF"] = acc
    print(f"  trained on: {FIXED_4}")
    print(f"  evaluated on the other {len(accs)} images (not used in training)")

    # ---------------- Experiment 3: Logistic Regression, same folds as Exp 1 ----------------
    print(f"\n=== Experiment 3: Logistic Regression, same {args.n_folds}-fold grouped CV ===")
    sgkf2 = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    for fold_i, (train_idx, test_idx) in enumerate(sgkf2.split(Xs_all, y_all, groups_all)):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xs_all[train_idx], y_all[train_idx], sample_weight=weights_all[train_idx])
        pred = clf.predict(Xs_all[test_idx])
        accs = per_image_accuracy(full.iloc[test_idx], pred)
        for img, acc in accs.items():
            per_image_results[img]["Exp3_GroupKFold_LogReg"] = acc

    # ---------------- Comparison table ----------------
    comp = pd.DataFrame.from_dict(per_image_results, orient="index")
    comp.index.name = "image"
    comp = comp.sort_index()
    print("\n=== Comparison (per-image held-out accuracy) ===")
    print(comp.to_string())

    print("\n=== Summary ===")
    summary_rows = []
    for col in ["Exp1_GroupKFold_RF", "Exp2_Fixed4_RF", "Exp3_GroupKFold_LogReg"]:
        vals = comp[col].dropna()
        n_below_80 = int((vals < 0.80).sum())
        row = {
            "experiment": col,
            "n_images_evaluated": len(vals),
            "mean_accuracy": round(vals.mean(), 3),
            "std_accuracy": round(vals.std(), 3),
            "min_accuracy": round(vals.min(), 3),
            "n_images_below_80pct": n_below_80,
        }
        summary_rows.append(row)
        print(f"  {col:26s} n={row['n_images_evaluated']:2d}  mean={row['mean_accuracy']:.3f}  "
              f"std={row['std_accuracy']:.3f}  min={row['min_accuracy']:.3f}  "
              f"(images <80% acc: {n_below_80})")

    comp.to_csv(os.path.join(args.out_dir, "model_comparison_per_image.csv"))
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "model_comparison_summary.csv"), index=False)

    # ---------------- Fit + save final deployable models ----------------
    print("\n=== Saving final models (fit on their full respective training data) ===")

    clf1 = RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=0)
    clf1.fit(Xs_all, y_all, sample_weight=weights_all)
    joblib.dump({"scaler": scaler, "clf": clf1, "feature_names": FEATURE_COLUMNS},
                os.path.join(args.out_dir, "model_exp1_groupkfold_rf.joblib"))

    joblib.dump({"scaler": scaler, "clf": clf_fixed, "feature_names": FEATURE_COLUMNS},
                os.path.join(args.out_dir, "model_exp2_fixed4_rf.joblib"))

    clf3 = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf3.fit(Xs_all, y_all, sample_weight=weights_all)
    joblib.dump({"scaler": scaler, "clf": clf3, "feature_names": FEATURE_COLUMNS},
                os.path.join(args.out_dir, "model_exp3_logreg.joblib"))

    print("Saved:")
    print(f"  {args.out_dir}/model_exp1_groupkfold_rf.joblib")
    print(f"  {args.out_dir}/model_exp2_fixed4_rf.joblib")
    print(f"  {args.out_dir}/model_exp3_logreg.joblib")

    importances = sorted(zip(FEATURE_COLUMNS, clf1.feature_importances_), key=lambda t: -t[1])
    print("\nExp1 (RandomForest, all-data) feature importances:")
    for name, imp in importances:
        print(f"  {name:16s} {imp:.3f}")

    coefs = sorted(zip(FEATURE_COLUMNS, clf3.coef_[0]), key=lambda t: -abs(t[1]))
    print("\nExp3 (Logistic Regression, all-data) standardized coefficients:")
    for name, coef in coefs:
        print(f"  {name:16s} {coef:+.3f}")


if __name__ == "__main__":
    main()
