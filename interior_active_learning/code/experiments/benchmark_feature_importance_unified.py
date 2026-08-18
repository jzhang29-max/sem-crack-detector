"""
UNIFIED-MODEL version of benchmark_feature_importance.py -- same two views
(RandomForest Gini importance + deployed LogisticRegression |standardized
coef|), fit on the FULL pooled 329-example unified dataset (both Step-E and
Step-H candidates, one 11-feature schema) instead of the 243-example
interior-only dataset.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from unified_data import load_unified_pooled
from active_learning_select import INTERIOR_FEATURE_COLUMNS

OUT_DIR = HERE.parents[1] / "benchmark_figures_unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0


def main():
    df = load_unified_pooled()
    X = df[INTERIOR_FEATURE_COLUMNS].values
    y = df["IsCrack"].astype(bool).values

    print(f"Loaded {len(df)} pooled unified examples from {df['SourceImage'].nunique()} source images "
          f"({int(y.sum())} True, {int((~y).sum())} False)")

    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    rf.fit(X, y)
    rf_importances = rf.feature_importances_
    rf_order = np.argsort(rf_importances)[::-1]

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(Xs, y)
    abs_coefs = np.abs(logreg.coef_[0])
    lr_order = np.argsort(abs_coefs)[::-1]

    print("\n=== (a) RandomForestClassifier(n_estimators=300, random_state=0), full dataset ===")
    for rank, idx in enumerate(rf_order, start=1):
        print(f"  {rank}. {INTERIOR_FEATURE_COLUMNS[idx]:28s} {rf_importances[idx]:.4f}")

    print("\n=== (b) Deployed LogisticRegression(class_weight='balanced'), |standardized coef| ===")
    for rank, idx in enumerate(lr_order, start=1):
        print(f"  {rank}. {INTERIOR_FEATURE_COLUMNS[idx]:28s} {abs_coefs[idx]:.4f}"
              f"  (signed coef={logreg.coef_[0][idx]:+.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    names_sorted = [INTERIOR_FEATURE_COLUMNS[i] for i in rf_order][::-1]
    vals_sorted = [rf_importances[i] for i in rf_order][::-1]
    ax.barh(names_sorted, vals_sorted, color="#4C72B0")
    ax.set_xlabel("Gini importance")
    ax.set_title("(a) RandomForestClassifier(n_estimators=300)\nfeature_importances_")

    ax = axes[1]
    names_sorted2 = [INTERIOR_FEATURE_COLUMNS[i] for i in lr_order][::-1]
    vals_sorted2 = [abs_coefs[i] for i in lr_order][::-1]
    ax.barh(names_sorted2, vals_sorted2, color="#DD8452")
    ax.set_xlabel("|standardized coefficient|")
    ax.set_title("(b) Deployed UNIFIED LogisticRegression\n(class_weight='balanced'), |coef_|")

    fig.suptitle("Feature importance: UNIFIED single-model classifier "
                  f"(n={len(df)}, {df['SourceImage'].nunique()} source images)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = OUT_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
