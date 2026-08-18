"""
Benchmark 3: Confusion matrix (proportions, Akbari et al. 2022 style, Fig. 10d-f)
for the actually-deployed LogisticRegression(max_iter=2000, class_weight='balanced')
model, using out-of-fold predictions pooled across one representative
StratifiedGroupKFold(5) run (seed=0) -- every example is scored exactly once,
by a fold whose model never trained on it.

Requires benchmark_model_comparison.py to have been run first (reads its
benchmark_results.json for the OOF predictions).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "benchmark_results.json"
OUT_DIR = HERE.parents[1] / "benchmark_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_SEED = "0"
MODEL_NAME = "LogisticRegression"


def main():
    with open(RESULTS_JSON) as f:
        data = json.load(f)

    oof = data["oof"][MODEL_NAME][REPRESENTATIVE_SEED]
    y_true_raw = np.array(oof["y_true"])
    assert not np.isnan(y_true_raw).any(), "OOF pooling left NaNs (a sample was never in a test fold)"
    y_true = y_true_raw.astype(int)
    y_pred = np.array(oof["y_pred"]).astype(int)
    assert len(y_true) == len(y_pred)

    # labels=[0, 1] -> 0 = "no crack" (negative), 1 = "crack" (positive)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalized: rate within each true class

    tn, fp = cm_norm[0]
    fn, tp = cm_norm[1]

    print("Raw counts confusion matrix (rows=true, cols=pred), labels=[No-crack(0), Crack(1)]:")
    print(cm)
    print("\nRow-normalized (proportions within each true class):")
    print(cm_norm)
    print(f"\nTrue Positive Rate (recall, crack correctly caught)   = {tp:.4f}")
    print(f"False Negative Rate (crack missed)                     = {fn:.4f}")
    print(f"True Negative Rate (no-crack correctly cleared)        = {tn:.4f}")
    print(f"False Positive Rate (no-crack wrongly flagged)         = {fp:.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels = ["No crack (0)", "Crack (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        "Confusion matrix (proportions): LogisticRegression (deployed)\n"
        "pooled out-of-fold, StratifiedGroupKFold(5), seed=0",
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
