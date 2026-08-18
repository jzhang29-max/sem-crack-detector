"""
Empirical test: do the 2 new generalizable shape/topology features
(BoundaryRoughness, BranchPointDensity -- see extended_features.py) actually
improve the unified model, or not? Same CV methodology as every other
benchmark in this project (StratifiedGroupKFold(5) x 5 seeds, grouped by
SourceImage) run TWICE on the identical rows/splits -- once with the
current 11 features, once with 11+2=13 -- so any difference is attributable
to the 2 new features, not to a different train/test split or sample.

Requires extract_extended_features_all25.py to have been run first
(candidates/extended_features_pooled.csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from unified_data import load_unified_pooled
from active_learning_select import INTERIOR_FEATURE_COLUMNS

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

EXTENDED_CSV = HERE.parents[1] / "candidates" / "extended_features_pooled.csv"
EXTRA_COLS = ["BoundaryRoughness", "BranchPointDensity"]

N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]

MODEL_FACTORIES = {
    "LogisticRegression (deployed)": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0),
    "SVC (RBF)": lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0),
}


def load_data():
    pooled = load_unified_pooled()
    extended = pd.read_csv(EXTENDED_CSV)
    merged = pooled.merge(extended[["SourceImage", "Label"] + EXTRA_COLS],
                           on=["SourceImage", "Label"], how="inner")
    dropped = len(pooled) - len(merged)
    print(f"Pooled dataset: {len(pooled)} rows. Matched to extended features: {len(merged)} "
          f"({dropped} rows dropped -- couldn't be re-matched to a freshly regenerated candidate).")
    return merged


def run_cv(X, y, groups, model_name):
    accs, aucs = [], []
    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        fold_accs, fold_aucs = [], []
        for train_idx, test_idx in sgkf.split(X, y, groups):
            scaler = StandardScaler().fit(X[train_idx])
            Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
            clf = MODEL_FACTORIES[model_name]()
            clf.fit(Xtr, y[train_idx])
            pred = clf.predict(Xte)
            proba = clf.predict_proba(Xte)[:, 1]
            fold_accs.append(accuracy_score(y[test_idx], pred))
            if len(np.unique(y[test_idx])) == 2:
                fold_aucs.append(roc_auc_score(y[test_idx], proba))
        accs.append(np.mean(fold_accs))
        aucs.append(np.mean(fold_aucs))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(aucs)), float(np.std(aucs))


def main():
    df = load_data()
    y = df["IsCrack"].astype(bool).values
    groups = df["SourceImage"].values
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    print(f"n={len(df)} ({n_pos} pos / {n_neg} neg), {df['SourceImage'].nunique()} source images\n")

    X11 = df[INTERIOR_FEATURE_COLUMNS].values
    X13 = df[INTERIOR_FEATURE_COLUMNS + EXTRA_COLS].values

    print(f"{'Model':30s} {'Features':10s} {'Accuracy':18s} {'AUC-ROC':18s}")
    print("-" * 80)
    results = {}
    for name in MODEL_FACTORIES:
        acc11, accstd11, auc11, aucstd11 = run_cv(X11, y, groups, name)
        acc13, accstd13, auc13, aucstd13 = run_cv(X13, y, groups, name)
        print(f"{name:30s} {'11 (base)':10s} {acc11:.4f} +/- {accstd11:.4f}   {auc11:.4f} +/- {aucstd11:.4f}")
        print(f"{name:30s} {'13 (ext.)':10s} {acc13:.4f} +/- {accstd13:.4f}   {auc13:.4f} +/- {aucstd13:.4f}")
        delta_auc = auc13 - auc11
        verdict = "HELPS" if delta_auc > 0.005 else ("HURTS" if delta_auc < -0.005 else "NO CHANGE")
        print(f"{'':30s} {'':10s} AUC delta: {delta_auc:+.4f}  ({verdict})\n")
        results[name] = dict(acc11=acc11, auc11=auc11, acc13=acc13, auc13=auc13, delta_auc=delta_auc)

    # Feature-importance check: where do the 2 new features rank among all 13,
    # in the model that's actually deployed (LogReg)?
    from sklearn.preprocessing import StandardScaler as SS
    scaler = SS().fit(X13)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X13), y)
    coefs = clf.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    all_cols = INTERIOR_FEATURE_COLUMNS + EXTRA_COLS
    print("Full-dataset LogisticRegression |standardized coef| ranking (13 features):")
    for rank, idx in enumerate(order, start=1):
        marker = "  <-- NEW" if all_cols[idx] in EXTRA_COLS else ""
        print(f"  {rank:2d}. {all_cols[idx]:28s} {abs(coefs[idx]):.4f}{marker}")

    return results


if __name__ == "__main__":
    main()
