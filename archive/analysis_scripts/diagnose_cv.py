"""
Diagnose WHY grouped-CV AUC on the full corrected dataset looks poor, and
compute an estimator that is actually valid for this label distribution.

The problem: 1023 of 1079 negatives (95%) come from a single image,
AS_24hr_BSE_Side_008 -- the one image that was exhaustively hand-labelled
region-by-region. Every other image was reviewed by confirming cracks, so it
contributes almost only positives. Under StratifiedGroupKFold that image sits
in the test half of exactly one fold; the other four folds are tested on a
handful of negatives against hundreds of positives, where per-fold AUC is
dominated by sampling noise. Averaging those five numbers is not a meaningful
point estimate, and its large sd is the tell.

Reported here instead:
  1. per-fold composition + AUC, so the noise is visible rather than averaged away
  2. POOLED out-of-fold AUC -- concatenate every held-out prediction, then
     compute one AUC over all 4110. Each row is still predicted by a model
     that never saw its image, so it is honest, but the metric is computed on
     the full negative set at once instead of five unequal slices.
  3. leave-one-image-out AUC for each image that has both classes
  4. the CURRENT PRODUCTION model applied to all 4110 rows, for reference

    python3 diagnose_cv.py
"""
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
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              average_precision_score)

from common import PROJECT_ROOT, PROD_MODEL_PATH

CSV = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")
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

    print("=" * 78)
    print("1. PER-FOLD COMPOSITION (LogisticRegression, seed 0) -- shows the noise")
    print("=" * 78)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    print(f"{'fold':>4} {'test rows':>10} {'test pos':>9} {'test neg':>9} {'AUC':>8}  images in test")
    for i, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        npos, nneg = int(y[te].sum()), int((~y[te]).sum())
        auc = roc_auc_score(y[te], p) if nneg and npos else float("nan")
        big = "AS_24hr_BSE_Side_008" in set(groups[te])
        print(f"{i:>4} {len(te):>10} {npos:>9} {nneg:>9} {auc:>8.4f}  "
              f"{len(set(groups[te])):>2} imgs{'  <-- holds the 1023-negative image' if big else ''}")

    print()
    print("=" * 78)
    print("2. POOLED OUT-OF-FOLD -- every row predicted by a model blind to its image")
    print("=" * 78)
    print(f"{'model':22s} {'pooled AUC':>22s} {'pooled acc':>18s} {'avg precision':>15s}")
    pooled = {}
    for name, factory in MODELS.items():
        aucs, accs, aps = [], [], []
        for seed in SEEDS:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            oof = np.full(len(y), np.nan)
            for tr, te in sgkf.split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                clf = factory().fit(sc.transform(X[tr]), y[tr])
                oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y, oof))
            accs.append(accuracy_score(y, oof >= 0.5))
            aps.append(average_precision_score(y, oof))
        pooled[name] = (float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)),
                         float(np.mean(aps)))
        print(f"{name:22s} {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}        "
              f"{np.mean(accs):.4f}          {np.mean(aps):.4f}")

    best = max(pooled, key=lambda k: pooled[k][0])
    print(f"\nbest by pooled OOF AUC: {best}  ({pooled[best][0]:.4f})")

    # confusion matrix of the best model's pooled OOF predictions, seed 0
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in sgkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = MODELS[best]().fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    cm = confusion_matrix(y, oof >= 0.5, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    print(f"\nOUT-OF-FOLD confusion matrix, {best}, seed 0 (rows=true [not-crack, crack]):")
    print(cm)
    print(f"  recall (cracks found)      {tp / max(tp + fn, 1):.1%}")
    print(f"  specificity (non-cracks rejected) {tn / max(tn + fp, 1):.1%}")
    print(f"  precision                  {tp / max(tp + fp, 1):.1%}")

    print()
    print("=" * 78)
    print("3. LEAVE-ONE-IMAGE-OUT (only images with both classes are scorable)")
    print("=" * 78)
    per = df.groupby("SourceImage")["IsCrack"].agg(["size", "sum"])
    per["neg"] = per["size"] - per["sum"]
    both = per[(per["sum"] > 0) & (per["neg"] > 0)].index.tolist()
    print(f"{'image':34s} {'rows':>6} {'pos':>6} {'neg':>6} {'AUC':>8} {'recall':>8} {'spec':>7}")
    for img in both:
        te = groups == img
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = MODELS[best]().fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        yt, pred = y[te], p >= 0.5
        tp = int((yt & pred).sum()); fn = int((yt & ~pred).sum())
        tn = int((~yt & ~pred).sum()); fp = int((~yt & pred).sum())
        print(f"{img:34s} {int(te.sum()):>6} {int(yt.sum()):>6} {int((~yt).sum()):>6} "
              f"{roc_auc_score(yt, p):>8.4f} {tp / max(tp + fn, 1):>7.1%} {tn / max(tn + fp, 1):>6.1%}")

    print()
    print("=" * 78)
    print("4. CURRENT PRODUCTION MODEL on all 4110 human-verified rows")
    print("=" * 78)
    b = joblib.load(PROD_MODEL_PATH)
    p = b["clf"].predict_proba(b["scaler"].transform(df[b["feature_names"]].values))[:, 1]
    pred = p >= 0.5
    cm = confusion_matrix(y, pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    print(cm)
    print(f"  AUC {roc_auc_score(y, p):.4f}   accuracy {accuracy_score(y, pred):.4f}")
    print(f"  recall {tp / max(tp + fn, 1):.1%}   specificity {tn / max(tn + fp, 1):.1%}")
    print("  CAVEAT: not out-of-sample -- some of these rows informed this model.")

    print()
    print("=" * 78)
    print("5. WHAT DRIVES THE VARIANCE: AUC restricted to the hand-labelled image")
    print("=" * 78)
    img = "AS_24hr_BSE_Side_008"
    te = groups == img
    sc = StandardScaler().fit(X[~te])
    clf = MODELS[best]().fit(sc.transform(X[~te]), y[~te])
    p = clf.predict_proba(sc.transform(X[te]))[:, 1]
    print(f"trained on the other 31 images, tested on {img} ({int(te.sum())} rows, "
          f"{int(y[te].sum())} pos / {int((~y[te]).sum())} neg)")
    print(f"  AUC {roc_auc_score(y[te], p):.4f}")
    print("  This is the only image with enough negatives for a stable single-image AUC.")


if __name__ == "__main__":
    main()
