"""
UNIFIED-MODEL version of benchmark_roc_curves.py -- reads
benchmark_results_unified.json (must run benchmark_model_comparison_unified.py
first) and plots ROC curves for the 4 best-performing models on the pooled
329-example unified dataset.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "benchmark_results_unified.json"
OUT_DIR = HERE.parents[1] / "benchmark_figures_unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_SEED = "0"


def main():
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    results = data["results"]

    combined = {name: 0.5 * (r["acc_mean"] + r["auc_mean"]) for name, r in results.items()}
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
        assert not np.isnan(y_true).any(), f"{name}: OOF pooling left NaNs"
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        legend_aucs[name] = float(auc)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "UNIFIED single-model: ROC curves, 4 best-performing models\n"
        "(pooled out-of-fold predictions, StratifiedGroupKFold(5), seed=0)"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "roc_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")
    for name, auc in legend_aucs.items():
        print(f"  {name:22s} AUC = {auc:.4f}")


if __name__ == "__main__":
    main()
