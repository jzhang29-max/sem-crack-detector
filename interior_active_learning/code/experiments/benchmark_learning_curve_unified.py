"""
UNIFIED-MODEL version of benchmark_learning_curve.py -- same methodology
(20/40/60/80/100% of data, 5x StratifiedGroupKFold(5) AUC-ROC), run on the
pooled 329-example unified dataset instead of the 243-example interior-only
dataset.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from unified_data import load_unified_pooled
from active_learning_select import INTERIOR_FEATURE_COLUMNS

OUT_DIR = HERE.parents[1] / "benchmark_figures_unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
CV_SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5


def subsample_indices(y, groups, frac, seed):
    n = len(y)
    if frac >= 1.0:
        return np.arange(n)
    target_n = max(int(round(n * frac)), 4)
    idx_all = np.arange(n)
    sub_idx, _ = train_test_split(idx_all, train_size=target_n, stratify=y, random_state=seed, shuffle=True)
    return np.sort(sub_idx)


def run_cv_auc(X, y, groups, seed):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in sgkf.split(X, y, groups):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        scaler = StandardScaler().fit(X[train_idx])
        Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y_train)
        proba = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))
    return aucs


def main():
    df = load_unified_pooled()
    X_full = df[INTERIOR_FEATURE_COLUMNS].values
    y_full = df["IsCrack"].astype(bool).values
    groups_full = df["SourceImage"].values
    n_total = len(df)

    print(f"Full pooled dataset: n={n_total}, {int(y_full.sum())} True / "
          f"{int((~y_full).sum())} False, {df['SourceImage'].nunique()} source images")

    results = []
    for frac in FRACTIONS:
        sub_idx = subsample_indices(y_full, groups_full, frac, seed=int(round(frac * 100)))
        X, y, groups = X_full[sub_idx], y_full[sub_idx], groups_full[sub_idx]
        n_sub = len(sub_idx)
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        n_groups_sub = len(set(groups))

        run_means = []
        skipped_folds_total = 0
        for seed in CV_SEEDS:
            aucs = run_cv_auc(X, y, groups, seed)
            skipped_folds_total += (N_FOLDS - len(aucs))
            if len(aucs) > 0:
                run_means.append(np.mean(aucs))
        run_means = np.array(run_means)
        mean_auc = float(np.mean(run_means))
        std_auc = float(np.std(run_means))

        print(f"frac={frac:.0%}  n={n_sub:3d} ({n_pos} True/{n_neg} False, "
              f"{n_groups_sub} groups)  AUC-ROC = {mean_auc:.4f} +/- {std_auc:.4f}  "
              f"({skipped_folds_total} single-class folds skipped)")

        results.append({"frac": frac, "n_samples": n_sub, "n_pos": n_pos, "n_neg": n_neg,
                         "auc_mean": mean_auc, "auc_std": std_auc, "n_runs": len(run_means)})

    xs = [r["n_samples"] for r in results]
    means = [r["auc_mean"] for r in results]
    stds = [r["auc_std"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=5, color="#4C72B0", linewidth=2, markersize=7)
    ax.set_xlabel("Training set size (# labeled examples)")
    ax.set_ylabel("AUC-ROC (mean +/- std over 5 StratifiedGroupKFold(5) runs)")
    ax.set_title("Learning curve: UNIFIED single-model LogisticRegression\n"
                  "(class_weight='balanced'), grouped by SourceImage")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    for x, m in zip(xs, means):
        ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "learning_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path} ({out_path.stat().st_size} bytes)")

    last_gain = means[-1] - means[-2]
    first_gain = means[1] - means[0]
    print(f"\nGain from 80%->100% of data: {last_gain:+.4f}")
    print(f"Gain from 20%->40% of data: {first_gain:+.4f}")


if __name__ == "__main__":
    main()
