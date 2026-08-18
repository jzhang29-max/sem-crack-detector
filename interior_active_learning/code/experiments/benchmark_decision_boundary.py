"""
Benchmark: 2-feature decision boundary (Akbari et al. 2022 style, Fig. 10a-c).

Uses the TWO most important features per the deployed LogisticRegression's
own |coef_| ranking (computed in benchmark_feature_importance.py): the top-2
by |standardized coefficient| are MeanVesselness and LogArea.

A FRESH LogisticRegression(class_weight="balanced") is fit using ONLY these
2 (StandardScaler-normalized) features -- purely for visualization. This is
necessarily a simplification of the real, deployed 11-feature model (same
caveat as the reference paper's own Fig. 10, which also visualizes only 2 of
several input features at a time).

The 2-feature model's in-sample accuracy is reported alongside a grouped
5-fold CV accuracy (StratifiedGroupKFold, grouped by SourceImage) so the
reader can see both the boundary-fit accuracy and an honest generalization
estimate for this deliberately-simplified 2D model.

Outputs:
  - benchmark_figures/decision_boundary.png
  - prints the 2 features used, in-sample accuracy, and CV accuracy
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
sys.path.insert(0, str(CODE_DIR))

from train_interior_model import load_labeled_interior
from active_learning_select import INTERIOR_FEATURE_COLUMNS

OUT_DIR = HERE.parents[1] / "benchmark_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Top-2 features by |standardized coefficient| from the deployed
# LogisticRegression (see benchmark_feature_importance.py output):
#   1. MeanVesselness  |coef|=2.1876
#   2. LogArea         |coef|=1.1147
FEAT_X = "MeanVesselness"
FEAT_Y = "LogArea"


def main():
    df = load_labeled_interior()
    y = df["IsCrack"].astype(bool).values
    groups = df["SourceImage"].values

    ix = INTERIOR_FEATURE_COLUMNS.index(FEAT_X)
    iy = INTERIOR_FEATURE_COLUMNS.index(FEAT_Y)
    X2 = df[[FEAT_X, FEAT_Y]].values

    print(f"Using top-2 features by deployed LogisticRegression |coef_|: "
          f"{FEAT_X} (rank 1), {FEAT_Y} (rank 2)")
    print(f"n={len(df)} ({int(y.sum())} True / {int((~y).sum())} False)")

    scaler = StandardScaler().fit(X2)
    X2s = scaler.transform(X2)

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X2s, y)
    pred_full = clf.predict(X2s)
    acc_insample = accuracy_score(y, pred_full)
    print(f"In-sample accuracy (fit and evaluated on all {len(df)} points): "
          f"{acc_insample:.4f}")

    # Honest grouped CV accuracy for this deliberately simplified 2-feature model
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    cv_accs = []
    for train_idx, test_idx in sgkf.split(X2, y, groups):
        sc = StandardScaler().fit(X2[train_idx])
        clf_cv = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf_cv.fit(sc.transform(X2[train_idx]), y[train_idx])
        pred = clf_cv.predict(sc.transform(X2[test_idx]))
        cv_accs.append(accuracy_score(y[test_idx], pred))
    cv_acc_mean, cv_acc_std = np.mean(cv_accs), np.std(cv_accs)
    print(f"Grouped 5-fold CV accuracy (StratifiedGroupKFold, random_state=0): "
          f"{cv_acc_mean:.4f} +/- {cv_acc_std:.4f}")

    # ---- Decision boundary mesh (in standardized-feature space, labeled
    # back in original units for readability) ----
    x_min, x_max = X2s[:, 0].min() - 0.5, X2s[:, 0].max() + 0.5
    y_min, y_max = X2s[:, 1].min() - 0.5, X2s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                          np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # map standardized mesh back to original units for plotting axes
    xx_orig = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_orig = yy * scaler.scale_[1] + scaler.mean_[1]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap_bg = ListedColormap(["#FFDDDD", "#DDEEFF"])  # False=red-ish, True=blue-ish
    ax.contourf(xx_orig, yy_orig, Z, alpha=0.6, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5])
    ax.contour(xx_orig, yy_orig, Z, levels=[0.5], colors="k", linewidths=1.5)

    colors = np.where(y, "#1f4e9c", "#c0392b")
    ax.scatter(X2[~y, 0], X2[~y, 1], c="#c0392b", edgecolors="k", s=40,
               label="Not crack (False)", zorder=3)
    ax.scatter(X2[y, 0], X2[y, 1], c="#1f4e9c", edgecolors="k", s=40,
               label="Crack (True)", zorder=3)

    ax.set_xlabel(FEAT_X)
    ax.set_ylabel(FEAT_Y)
    ax.set_title(
        f"2-feature decision boundary: {FEAT_X} vs {FEAT_Y}\n"
        f"LogisticRegression(class_weight='balanced'), n={len(df)}\n"
        f"In-sample acc={acc_insample:.3f}, grouped 5-fold CV acc="
        f"{cv_acc_mean:.3f}±{cv_acc_std:.3f}\n"
        f"(2D simplification of the real 11-feature deployed model)"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = OUT_DIR / "decision_boundary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path} ({out_path.stat().st_size} bytes)")

    return {
        "feat_x": FEAT_X, "feat_y": FEAT_Y,
        "acc_insample": float(acc_insample),
        "cv_acc_mean": float(cv_acc_mean), "cv_acc_std": float(cv_acc_std),
    }


if __name__ == "__main__":
    main()
