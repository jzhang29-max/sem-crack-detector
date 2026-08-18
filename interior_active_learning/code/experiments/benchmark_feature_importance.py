"""
Benchmark: feature importance (Akbari et al. 2022 style, Fig. 12).

Two legitimate, complementary views of "which features matter", computed on
the REAL 243-example labeled interior-candidate dataset (11 features, grouped
by SourceImage):

(a) RandomForestClassifier(n_estimators=300, random_state=0).feature_importances_
    fit on the full dataset -- matches the paper's own convention exactly
    (mean decrease in impurity, Gini importance).

(b) The ACTUALLY DEPLOYED model's own signal: LogisticRegression(
    class_weight="balanced") fit on StandardScaler-normalized features (same
    recipe as train_interior_model.py's main()), plotting |standardized
    coefficient| -- this is what the deployed model itself actually weighs.

Both are fit on the FULL dataset (no held-out split) since the goal here is
descriptive (which features drive the decision), not a generalization
estimate -- exactly as in the reference paper's Fig. 12.

Outputs:
  - benchmark_figures/feature_importance.png (two-panel horizontal bar chart)
  - prints exact top-5 features + values for both methods
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
sys.path.insert(0, str(CODE_DIR))

from train_interior_model import load_labeled_interior
from active_learning_select import INTERIOR_FEATURE_COLUMNS

OUT_DIR = HERE.parents[1] / "benchmark_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0


def main():
    df = load_labeled_interior()
    X = df[INTERIOR_FEATURE_COLUMNS].values
    y = df["IsCrack"].astype(bool).values
    groups = df["SourceImage"].values

    print(f"Loaded {len(df)} labeled interior candidates from "
          f"{df['SourceImage'].nunique()} source images "
          f"({int(y.sum())} True, {int((~y).sum())} False)")

    # (a) Random Forest feature_importances_ (paper's own convention, Fig. 12)
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    rf.fit(X, y)
    rf_importances = rf.feature_importances_
    rf_order = np.argsort(rf_importances)[::-1]

    # (b) Deployed model: LogisticRegression(class_weight="balanced") on
    # StandardScaler-normalized features -- same recipe as
    # train_interior_model.py's main().
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(Xs, y)
    abs_coefs = np.abs(logreg.coef_[0])
    lr_order = np.argsort(abs_coefs)[::-1]

    print("\n=== (a) RandomForestClassifier(n_estimators=300, random_state=0) "
          "feature_importances_, full dataset ===")
    for rank, idx in enumerate(rf_order, start=1):
        print(f"  {rank}. {INTERIOR_FEATURE_COLUMNS[idx]:28s} {rf_importances[idx]:.4f}")

    print("\n=== (b) Deployed LogisticRegression(class_weight='balanced') on "
          "StandardScaler features, |standardized coef|, full dataset ===")
    for rank, idx in enumerate(lr_order, start=1):
        print(f"  {rank}. {INTERIOR_FEATURE_COLUMNS[idx]:28s} {abs_coefs[idx]:.4f}"
              f"  (signed coef={logreg.coef_[0][idx]:+.4f})")

    # ---- Plot: two-panel horizontal bar chart, ranked ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    names_sorted = [INTERIOR_FEATURE_COLUMNS[i] for i in rf_order][::-1]
    vals_sorted = [rf_importances[i] for i in rf_order][::-1]
    ax.barh(names_sorted, vals_sorted, color="#4C72B0")
    ax.set_xlabel("Gini importance")
    ax.set_title("(a) RandomForestClassifier(n_estimators=300)\nfeature_importances_ "
                  "(paper's Fig. 12 convention)")

    ax = axes[1]
    names_sorted2 = [INTERIOR_FEATURE_COLUMNS[i] for i in lr_order][::-1]
    vals_sorted2 = [abs_coefs[i] for i in lr_order][::-1]
    ax.barh(names_sorted2, vals_sorted2, color="#DD8452")
    ax.set_xlabel("|standardized coefficient|")
    ax.set_title("(b) Deployed LogisticRegression\n(class_weight='balanced'), |coef_|")

    fig.suptitle("Feature importance: interior crack-candidate classifier "
                  f"(n={len(df)}, {df['SourceImage'].nunique()} source images)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = OUT_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path} ({out_path.stat().st_size} bytes)")

    return {
        "rf_top5": [(INTERIOR_FEATURE_COLUMNS[i], float(rf_importances[i])) for i in rf_order[:5]],
        "lr_top5": [(INTERIOR_FEATURE_COLUMNS[i], float(abs_coefs[i])) for i in lr_order[:5]],
        "lr_order_full": [(INTERIOR_FEATURE_COLUMNS[i], float(abs_coefs[i])) for i in lr_order],
    }


if __name__ == "__main__":
    main()
