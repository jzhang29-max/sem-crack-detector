"""
THE retrain entry point for the current pipeline: retrains
models/unified_model.joblib on every human-verified label this project has
-- every original-candidate correction in ../manual_corrections_ledger.csv
PLUS every labeled interior candidate in candidates/*_interior.csv --
pooled into one 11-feature dataset (unified_data.load_unified_pooled()).

Run this any time after painting new corrections (via paint_server.py) to
fold them into the model. The paint app auto-detects the newer model file
and refreshes any stale template the next time you open that image.

Usage
-----
    python3 train_unified_model.py

Writes models/unified_model.joblib (overwrites the previous one -- run
`python3 checkpoint_model.py` first, or just copy the .joblib file by hand,
if you want to keep the old one for comparison).
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS_DIR
from active_learning_select import INTERIOR_FEATURE_COLUMNS
from train_interior_model import calibrate_interior_fill_rule
from unified_data import load_unified_pooled
import build_original_ledger_unified_features


def main():
    # Step 0: regenerate the original-candidate feature CSV fresh, so it
    # reflects the CURRENT manual_corrections_ledger.csv (including anything
    # corrected since the last time this ran) rather than a stale snapshot.
    print("=== Regenerating original-candidate features from the ledger ===")
    build_original_ledger_unified_features.main()

    print("\n=== Pooling + training ===")
    pooled = load_unified_pooled()
    X = pooled[INTERIOR_FEATURE_COLUMNS].values
    y = pooled["IsCrack"].astype(bool).values
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    print(f"Training on {len(pooled)} pooled examples ({n_pos} pos / {n_neg} neg, "
          f"{pooled['SourceImage'].nunique()} source images)")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, y)

    # Neutral Pass-1 placeholders for the two crack-context features (see
    # unified_pipeline.score_pass1_candidates()'s docstring for why the
    # training pool's own MEAN, not an arbitrary value or extreme
    # percentile, is the mathematically correct choice here).
    dist_placeholder = float(pooled["MeanDistToCrack"].mean())
    touching_placeholder = float(pooled["FracBoundaryTouchingCrack"].mean())
    print(f"Pass-1 placeholders: MeanDistToCrack={dist_placeholder:.2f}, "
          f"FracBoundaryTouchingCrack={touching_placeholder:.4f}")

    rule = calibrate_interior_fill_rule(pooled, scaler, clf)
    if rule is not None:
        print(f"interior_fill hybrid rule: dist_thr={rule['dist_thr']:.2f} bri_thr={rule['bri_thr']:.2f} "
              f"floor={rule['floor']} -- recall={rule['recall_on_known_positives']:.1%} on known positives, "
              f"{rule['full_pool_accept_rate']:.1%} accept rate on the full candidate pool")
    else:
        print("interior_fill hybrid rule: not calibrated (fewer than 2 known negatives) -- "
              "falls back to the plain ML threshold for this candidate type")

    bundle = {
        "scaler": scaler,
        "clf": clf,
        "feature_names": INTERIOR_FEATURE_COLUMNS,
        "interior_fill_rule": rule,
        "threshold_default": 0.5,
        "pass1_dist_placeholder": dist_placeholder,
        "pass1_touching_placeholder": touching_placeholder,
    }
    out_path = os.path.join(MODELS_DIR, "unified_model.joblib")
    joblib.dump(bundle, out_path)
    print(f"\nSaved {out_path}")
    print("Open the paint app (paint_server.py, http://127.0.0.1:8767) -- it will "
          "auto-detect this newer model and refresh any stale image template.")


if __name__ == "__main__":
    # --help USED TO DO THE WORK. main() ignored argv, so asking for help ran a fit
    # and could write a bundle over the deployed one. Refuse before any of that.
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__ or "")
        print("usage: train_unified_model.py\n"
              "  no arguments. Fits the one shared model both passes score with, and writes its bundle.")
        sys.exit(0)
    if len(sys.argv) > 1:
        print(f"unknown option {sys.argv[1]!r}. This script takes no arguments. "
              f"Use --help.")
        sys.exit(2)
    main()
