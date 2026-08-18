"""
Benchmark 2: ROC curves (Akbari et al. 2022 style, Fig. 8) for the 4
best-performing models identified in benchmark_model_comparison.py.

"Best-performing" = highest combined score = mean(accuracy_mean, auc_mean)
across the 5 independent StratifiedGroupKFold(5) runs computed in step 1.

For each of the 4 models we take ONE representative run (seed=0's
StratifiedGroupKFold(5) split) and pool the out-of-fold predicted
probabilities across all 5 folds of that run (each example scored exactly
once, by the fold whose model never saw it during training), then plot the
ROC curve from those pooled out-of-fold scores.

Requires benchmark_model_comparison.py to have been run first (reads its
benchmark_results.json for the OOF predictions).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "benchmark_results.json"
OUT_DIR = HERE.parents[1] / "benchmark_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_SEED = "0"  # JSON keys are strings after round-trip


def main():
    with open(RESULTS_JSON) as f:
        data = json.load(f)

    results = data["results"]

    # Select 4 best-performing models by combined score.
    combined = {
        name: 0.5 * (r["acc_mean"] + r["auc_mean"])
        for name, r in results.items()
    }
    ranked = sorted(combined.items(), key=lambda kv: -kv[1])
    top4 = [name for name, _ in ranked[:4]]
    print("Combined score ranking (0.5*acc_mean + 0.5*auc_mean):")
    for name, score in ranked:
        print(f"  {name:22s} {score:.4f}  (acc={results[name]['acc_mean']:.4f}, "
              f"auc={results[name]['auc_mean']:.4f})")
    print(f"\nTop 4 selected for ROC plot: {top4}")

    fig, ax = plt.subplots(figsize=(7, 7))
    legend_aucs = {}

    for name in top4:
        oof = data["oof"][name][REPRESENTATIVE_SEED]
        y_true = np.array(oof["y_true"])
        y_score = np.array(oof["y_score"])
        assert not np.isnan(y_true).any(), f"{name}: OOF pooling left NaNs (a sample was never in a test fold)"

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        legend_aucs[name] = float(auc)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "ROC curves: 4 best-performing models\n"
        "(pooled out-of-fold predictions, StratifiedGroupKFold(5), seed=0)"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "roc_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    print("\nAUC values shown in legend:")
    for name, auc in legend_aucs.items():
        print(f"  {name:22s} AUC = {auc:.4f}")


if __name__ == "__main__":
    main()
