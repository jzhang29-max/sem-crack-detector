"""
Train the crack classifier on every human-verified region and evaluate it
honestly with grouped cross-validation.

Methodology matches the rest of this project: StratifiedGroupKFold grouped
by SourceImage over several seeds, so no image ever appears in both the
train and test half of a fold. That grouping is the point -- candidates from
one image are highly correlated (same material, same imaging conditions,
often the same physical crack), so a random split would leak and inflate
every number.

Guard retained from the earlier MAR-only version: grouped CV is only
meaningful when negatives are spread across several images. If they are
concentrated in one or two, holding those out leaves single-class folds and
any score printed would be an artifact. The script says so and skips CV
rather than printing a misleading number.

Compares the same model families the benchmark already uses, then saves the
best-by-AUC as the deployed model.

    python3 train_and_evaluate.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from common import PROJECT_ROOT

CSV = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")
OUT_MODEL = os.path.join(PROJECT_ROOT, "models", "crack_classifier_v2.joblib")
OUT_JSON = os.path.join(PROJECT_ROOT, "models", "crack_classifier_v2_metrics.json")

FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity",
            "Extent", "Circularity", "MeanDarkness", "MeanVesselness"]

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=0),
    "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=200, random_state=0),
    "SVC (RBF)": lambda: SVC(kernel="rbf", probability=True, class_weight="balanced",
                              random_state=0),
}
SEEDS = [0, 1, 2, 3, 4]


def main():
    df = pd.read_csv(CSV)
    X, y = df[FEATURES].values, df["IsCrack"].astype(bool).values
    groups = df["SourceImage"].values
    n_pos, n_neg = int(y.sum()), int((~y).sum())

    per = df.groupby("SourceImage")["IsCrack"].agg(["size", "sum"])
    per["neg"] = per["size"] - per["sum"]
    imgs_neg = int((per["neg"] > 0).sum())
    imgs_pos = int((per["sum"] > 0).sum())

    print(f"Dataset: {len(df)} regions, {n_pos} crack / {n_neg} not-crack, "
          f"{df['SourceImage'].nunique()} images")
    print(f"images with negatives: {imgs_neg}   with positives: {imgs_pos}\n")
    print(per.sort_values("neg", ascending=False).head(12).to_string(), "\n")

    cv_ok = imgs_neg >= 3 and imgs_pos >= 3
    results = {}
    if not cv_ok:
        print(f"!! Grouped CV NOT reportable: negatives appear in only {imgs_neg} image(s).")
        print("!! Any score would be an artifact of single-class folds. Skipping.\n")
    else:
        n_splits = min(5, imgs_neg)
        print(f"StratifiedGroupKFold(n_splits={n_splits}) x {len(SEEDS)} seeds, grouped by image\n")
        print(f"{'model':22s} {'accuracy':>18s} {'AUC-ROC':>18s} {'folds':>6s}")
        for name, factory in MODELS.items():
            accs, aucs = [], []
            for seed in SEEDS:
                sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                for tr, te in sgkf.split(X, y, groups):
                    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                        continue
                    sc = StandardScaler().fit(X[tr])
                    clf = factory().fit(sc.transform(X[tr]), y[tr])
                    accs.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
                    aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
            if not accs:
                print(f"{name:22s} {'no usable folds':>18s}")
                continue
            results[name] = {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                              "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                              "n_folds": len(accs)}
            print(f"{name:22s} {np.mean(accs):.4f} +/- {np.std(accs):.4f} "
                  f"  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} {len(accs):>6d}")

    best = max(results, key=lambda k: results[k]["auc"]) if results else "LogisticRegression"
    print(f"\nbest by AUC: {best}" if results else f"\nno CV -- defaulting to {best}")

    scaler = StandardScaler().fit(X)
    clf = MODELS[best]().fit(scaler.transform(X), y)
    cm = confusion_matrix(y, clf.predict(scaler.transform(X)), labels=[False, True])
    print("\nIn-sample confusion matrix (rows=true [not-crack, crack]):")
    print(cm)
    print("NOTE: in-sample -- the model saw every row. The grouped-CV numbers above are the "
          "honest estimate.")

    if hasattr(clf, "coef_"):
        print("\nStandardized coefficients:")
        for f, c in sorted(zip(FEATURES, clf.coef_[0]), key=lambda t: -abs(t[1])):
            print(f"  {f:18s} {c:+.4f}")

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": clf, "feature_names": FEATURES,
                  "model_family": best, "n_train": len(df), "n_pos": n_pos, "n_neg": n_neg,
                  "images": sorted(df["SourceImage"].unique().tolist()),
                  "cv_reportable": cv_ok, "cv_results": results}, OUT_MODEL)
    with open(OUT_JSON, "w") as f:
        json.dump({"n_train": len(df), "n_pos": n_pos, "n_neg": n_neg,
                    "n_images": int(df["SourceImage"].nunique()),
                    "images_with_negatives": imgs_neg,
                    "cv_reportable": cv_ok, "cv_results": results, "best": best}, f, indent=2)
    print(f"\nSaved {OUT_MODEL}")


if __name__ == "__main__":
    main()
