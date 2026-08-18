"""
UNIFIED-MODEL version of benchmark_confusion_matrix.py -- confusion matrix
(proportions) for LogisticRegression(class_weight='balanced') (the same
recipe as the deployed unified_model.joblib), pooled out-of-fold predictions,
StratifiedGroupKFold(5), seed=0, on the pooled 329-example unified dataset.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "benchmark_results_unified.json"
OUT_DIR = HERE.parents[1] / "benchmark_figures_unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_SEED = "0"
MODEL_NAME = "LogisticRegression"


def main():
    with open(RESULTS_JSON) as f:
        data = json.load(f)

    oof = data["oof"][MODEL_NAME][REPRESENTATIVE_SEED]
    y_true_raw = np.array(oof["y_true"])
    assert not np.isnan(y_true_raw).any()
    y_true = y_true_raw.astype(int)
    y_pred = np.array(oof["y_pred"]).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    tn, fp = cm_norm[0]
    fn, tp = cm_norm[1]

    print("Raw counts confusion matrix (rows=true, cols=pred), labels=[No-crack(0), Crack(1)]:")
    print(cm)
    print(f"\nTPR (recall) = {tp:.4f}  FNR = {fn:.4f}  TNR = {tn:.4f}  FPR = {fp:.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels = ["No crack (0)", "Crack (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(
        "UNIFIED single-model: confusion matrix (proportions)\n"
        "LogisticRegression, pooled out-of-fold, StratifiedGroupKFold(5), seed=0",
        fontsize=11,
    )
    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, label="Proportion within true class")
    fig.tight_layout()
    out_path = OUT_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
