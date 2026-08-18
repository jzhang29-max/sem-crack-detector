"""
Test whether the corrected label set is biased by HOW it was collected, and
whether that bias is fixable by reweighting rather than by more labelling.

The concern: of 32 reviewed images, 23 contributed positives only. That is a
property of the review workflow, not of the material -- the paint app is used
to confirm cracks, so a region only becomes a training row if someone clicked
it, and nobody clicks the thousands of obviously-uninteresting dark specks.
One image, AS_24hr_BSE_Side_008, was instead labelled exhaustively region by
region, and it supplies 1023 of the 1079 negatives (95%).

If that matters, a model trained on the other 31 images should do measurably
WORSE on the exhaustively-labelled image than the current production model
does -- because it has been taught that dark regions are cracks by default.

Conditions, all evaluated on AS_24hr_BSE_Side_008 (the only image with a
trustworthy positive/negative ratio):
  A  current production model                      (contaminated: saw this image)
  B  train on all 31 other images                  (the naive "train on everything")
  C  train only on other images that have BOTH classes
  D  train on all 31 others, per-image sample weights so no image dominates
  E  reverse direction: train on AS_24hr only, test on the 8 other both-class images

    python3 label_bias_experiment.py
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
from sklearn.metrics import roc_auc_score, confusion_matrix

from common import PROJECT_ROOT, PROD_MODEL_PATH

CSV = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")
FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity",
            "Extent", "Circularity", "MeanDarkness", "MeanVesselness"]
HELD = "AS_24hr_BSE_Side_008"


def report(tag, y, p):
    pred = p >= 0.5
    tp = int((y & pred).sum()); fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum()); fp = int((~y & pred).sum())
    auc = roc_auc_score(y, p) if y.any() and (~y).any() else float("nan")
    print(f"{tag:52s} AUC {auc:6.4f}   recall {tp/max(tp+fn,1):6.1%}   "
          f"spec {tn/max(tn+fp,1):6.1%}")
    return auc


def fit_eval(tr_df, te_df, weights=None):
    Xtr, ytr = tr_df[FEATURES].values, tr_df["IsCrack"].astype(bool).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr, sample_weight=weights)
    p = clf.predict_proba(sc.transform(te_df[FEATURES].values))[:, 1]
    return te_df["IsCrack"].astype(bool).values, p, clf


def main():
    df = pd.read_csv(CSV)
    per = df.groupby("SourceImage")["IsCrack"].agg(["size", "sum"])
    per["neg"] = per["size"] - per["sum"]
    both = set(per[(per["sum"] > 0) & (per["neg"] > 0)].index)
    pos_only = sorted(set(per.index) - both)

    print(f"32 reviewed images: {len(both)} have both classes, {len(pos_only)} are positives-only")
    print(f"negatives: {int(per['neg'].sum())} total, "
          f"{int(per.loc[HELD, 'neg'])} of them ({per.loc[HELD,'neg']/per['neg'].sum():.0%}) "
          f"from {HELD} alone\n")

    held = df[df.SourceImage == HELD]
    rest = df[df.SourceImage != HELD]
    y_held = held["IsCrack"].astype(bool).values
    print(f"evaluating on {HELD}: {len(held)} rows, "
          f"{int(y_held.sum())} crack / {int((~y_held).sum())} not-crack\n")
    print("-" * 92)

    b = joblib.load(PROD_MODEL_PATH)
    p = b["clf"].predict_proba(b["scaler"].transform(held[b["feature_names"]].values))[:, 1]
    a_auc = report("A  production model (SAW this image -- optimistic)", y_held, p)

    y, p, clf_b = fit_eval(rest, held)
    b_auc = report("B  trained on all 31 other images", y, p)

    tr_c = rest[rest.SourceImage.isin(both)]
    y, p, _ = fit_eval(tr_c, held)
    c_auc = report(f"C  trained on the {tr_c.SourceImage.nunique()} other both-class images "
                    f"({len(tr_c)} rows)", y, p)

    counts = rest.groupby("SourceImage")["IsCrack"].size()
    w = rest["SourceImage"].map(lambda s: 1.0 / counts[s]).values
    y, p, _ = fit_eval(rest, held, weights=w)
    d_auc = report("D  all 31 others, per-image weights (no image dominates)", y, p)

    print("-" * 92)
    others_both = df[df.SourceImage.isin(both - {HELD})]
    y, p, _ = fit_eval(held, others_both)
    print(f"E  reverse: train on {HELD} only, test on the "
          f"{others_both.SourceImage.nunique()} other both-class images ({len(others_both)} rows)")
    report("   ", y, p)

    print()
    print("=" * 92)
    print("WHAT THE MODEL LEARNS FROM EACH LABEL SET (standardized LR coefficients)")
    print("=" * 92)
    _, _, clf_all = fit_eval(df, held)
    _, _, clf_held = fit_eval(held, held)
    print(f"{'feature':18s} {'all 4110 rows':>16s} {'exhaustive image only':>23s}   agree?")
    for i, f in enumerate(FEATURES):
        ca, ch = clf_all.coef_[0][i], clf_held.coef_[0][i]
        flag = "" if np.sign(ca) == np.sign(ch) else "  <-- OPPOSITE SIGN"
        print(f"{f:18s} {ca:+16.4f} {ch:+23.4f}{flag}")

    print()
    print("mean feature value by class, all 4110 rows:")
    print(df.groupby("IsCrack")[["LogArea", "MeanDarkness", "MeanVesselness"]].mean().to_string())

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    best_new = max(b_auc, c_auc, d_auc)
    if best_new < a_auc - 0.02:
        print(f"Training on the corrected set makes the model WORSE on the one image with")
        print(f"trustworthy ground truth: best new variant {best_new:.4f} vs production {a_auc:.4f}.")
        print("Do NOT deploy. The corrected labels are ~74% positives-only-by-construction,")
        print("so the model learns 'dark region => crack'. The fix is not-crack (blue) marks")
        print("on more images, not more crack confirmations.")
    else:
        print(f"Best new variant {best_new:.4f} vs production {a_auc:.4f} -- retraining is safe.")


if __name__ == "__main__":
    main()
