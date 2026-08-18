"""
UNIFIED-MODEL version of benchmark_model_comparison.py: same methodology
(5x independent shuffled StratifiedGroupKFold(5), grouped by SourceImage,
accuracy + AUC-ROC), but trained/evaluated on the pooled 329-example unified
dataset (load_unified_pooled() -- original Step-E candidates AND Step-H
interior candidates in one 11-feature schema) instead of the two-stage
system's 243-example interior-only dataset.

Writes to benchmark_figures_unified/ (kept separate from the two-stage
benchmark_figures/ so neither set of figures is overwritten -- both remain
available for a side-by-side comparison in the paper).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from unified_data import load_unified_pooled
from active_learning_select import INTERIOR_FEATURE_COLUMNS

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

OUT_DIR = HERE.parents[1] / "benchmark_figures_unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = HERE / "benchmark_results_unified.json"

N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]

# Two NN architectures, both regularized (alpha) and early-stopped, to give
# a neural network a fair shot despite the small n=329 -- one small/shallow
# (less prone to overfitting on this little data), one wider, so a "NN just
# needs more capacity" objection can't be raised without evidence.
MODEL_FACTORIES = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0),
    "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=200, random_state=0),
    "SVC (RBF)": lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0),
    "GaussianNB": lambda: GaussianNB(),
    "KNeighbors (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
    "NeuralNet (8,)": lambda: MLPClassifier(hidden_layer_sizes=(8,), alpha=1e-2, max_iter=3000,
                                             early_stopping=True, random_state=0),
    "NeuralNet (32,16)": lambda: MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=3000,
                                                early_stopping=True, random_state=0),
}


def load_data():
    df = load_unified_pooled()
    X = df[INTERIOR_FEATURE_COLUMNS].values
    y = df["IsCrack"].astype(bool).values
    groups = df["SourceImage"].values
    return X, y, groups, df


def run_cv_for_model(model_name, X, y, groups):
    per_run_acc, per_run_auc, oof_store = [], [], {}
    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        fold_accs, fold_aucs = [], []
        oof_y_true = np.full(len(y), np.nan)
        oof_y_score = np.full(len(y), np.nan)
        oof_y_pred = np.full(len(y), np.nan)

        for train_idx, test_idx in sgkf.split(X, y, groups):
            scaler = StandardScaler().fit(X[train_idx])
            Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
            ytr, yte = y[train_idx], y[test_idx]

            clf = MODEL_FACTORIES[model_name]()
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            proba = clf.predict_proba(Xte)[:, 1]

            fold_accs.append(accuracy_score(yte, pred))
            if len(np.unique(yte)) == 2:
                fold_aucs.append(roc_auc_score(yte, proba))

            oof_y_true[test_idx] = yte.astype(float)
            oof_y_score[test_idx] = proba
            oof_y_pred[test_idx] = pred.astype(float)

        per_run_acc.append(float(np.mean(fold_accs)))
        per_run_auc.append(float(np.mean(fold_aucs)))
        oof_store[seed] = {"y_true": oof_y_true.tolist(), "y_score": oof_y_score.tolist(),
                            "y_pred": oof_y_pred.tolist()}
    return per_run_acc, per_run_auc, oof_store


def main():
    X, y, groups, df = load_data()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    n_groups = len(set(groups))
    print(f"Loaded {len(df)} pooled unified examples from {n_groups} source images "
          f"({n_pos} True / {n_neg} False)")

    results, oof_all = {}, {}
    for name in MODEL_FACTORIES:
        print(f"\n=== {name} ===")
        per_run_acc, per_run_auc, oof_store = run_cv_for_model(name, X, y, groups)
        acc_mean, acc_std = float(np.mean(per_run_acc)), float(np.std(per_run_acc))
        auc_mean, auc_std = float(np.mean(per_run_auc)), float(np.std(per_run_auc))
        print(f"  accuracy = {acc_mean:.4f} +/- {acc_std:.4f}")
        print(f"  AUC-ROC  = {auc_mean:.4f} +/- {auc_std:.4f}")
        results[name] = {"per_run_acc": per_run_acc, "per_run_auc": per_run_auc,
                          "acc_mean": acc_mean, "acc_std": acc_std,
                          "auc_mean": auc_mean, "auc_std": auc_std}
        oof_all[name] = oof_store

    with open(RESULTS_JSON, "w") as f:
        json.dump({"results": results, "oof": oof_all, "n_examples": len(df),
                    "n_pos": n_pos, "n_neg": n_neg, "n_groups": n_groups}, f)
    print(f"\nSaved raw results + OOF predictions to {RESULTS_JSON}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(MODEL_FACTORIES.keys())
    acc_means = [results[m]["acc_mean"] for m in model_names]
    acc_stds = [results[m]["acc_std"] for m in model_names]
    auc_means = [results[m]["auc_mean"] for m in model_names]
    auc_stds = [results[m]["auc_std"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, acc_means, width, yerr=acc_stds, capsize=4, label="Accuracy")
    ax.bar(x + width / 2, auc_means, width, yerr=auc_stds, capsize=4, label="AUC-ROC")
    ax.set_ylabel("Score")
    ax.set_title(
        "UNIFIED single-model: 5x StratifiedGroupKFold(5), grouped by SourceImage\n"
        f"Pooled Step-E + Step-H candidate dataset (n={len(df)}, {n_pos} pos / {n_neg} neg, "
        f"{n_groups} source images)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "model_comparison_bars.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
