"""
Experiment: per_type_models
============================
Compare a pooled single LogisticRegression baseline against THREE separate
per-CandidateType LogisticRegression models (concavity, bridge_corridor,
interior_fill), with extra-strong regularization variants tried for the
interior_fill model specifically (since it only has 2 negative examples).

All CV is StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
grouped by SourceImage, computed on the POOLED set of predictions (i.e. for
the per-type approach, each fold's held-out predictions from whichever
per-type model applies to that row are concatenated back into one pooled
prediction vector before computing balanced_accuracy), so the numbers are
directly comparable to the pooled baseline's 0.729.
"""

import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Project root derived from this file's location, not hardcoded: these scripts
# shipped with an absolute /Users/... path and could only run on the machine that
# wrote them, while archive/README.md advertises them as rerunnable.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CAND_GLOB = os.path.join(_ROOT, "interior_active_learning", "candidates", "*_interior.csv")

FEATURES = [
    "LogArea", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
    "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
    "FracBoundaryTouchingCrack", "MeanDistToCrack",
]

TYPES = ["concavity", "bridge_corridor", "interior_fill"]


def load_labeled_interior():
    rows = []
    for f in sorted(glob.glob(CAND_GLOB)):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        labeled = d[d["IsCrack"].astype(str).str.strip().str.upper().isin(["TRUE", "FALSE"])].copy()
        if len(labeled) == 0:
            continue
        labeled["IsCrack"] = labeled["IsCrack"].astype(str).str.strip().str.upper() == "TRUE"
        rows.append(labeled)
    return pd.concat(rows, ignore_index=True)


def load_full_pool():
    """Load ALL rows (labeled or not) from the same CSVs, for acceptance-rate checks."""
    rows = []
    for f in sorted(glob.glob(CAND_GLOB)):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    df = load_labeled_interior()
    # Drop user_painted -- always-true, no discriminative signal.
    df = df[df["CandidateType"] != "user_painted"].reset_index(drop=True)
    print(f"Loaded {len(df)} labeled non-user_painted rows.")
    print(df.groupby("CandidateType")["IsCrack"].agg(["count", "sum"]))
    print()

    X_all = df[FEATURES].values
    y_all = df["IsCrack"].values.astype(int)
    groups_all = df["SourceImage"].values
    types_all = df["CandidateType"].values

    # -----------------------------------------------------------------
    # 1) POOLED BASELINE: single LogisticRegression(class_weight="balanced")
    #    trained on all types pooled, StratifiedGroupKFold(5), threshold 0.5.
    # -----------------------------------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    baseline_oof_pred = np.zeros(len(df), dtype=int)
    baseline_oof_proba = np.zeros(len(df), dtype=float)
    for train_idx, test_idx in sgkf.split(X_all, y_all, groups_all):
        scaler = StandardScaler().fit(X_all[train_idx])
        Xtr = scaler.transform(X_all[train_idx])
        Xte = scaler.transform(X_all[test_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(Xtr, y_all[train_idx])
        proba = clf.predict_proba(Xte)[:, 1]
        baseline_oof_proba[test_idx] = proba
        baseline_oof_pred[test_idx] = (proba >= 0.5).astype(int)

    baseline_bacc = balanced_accuracy_score(y_all, baseline_oof_pred)
    print(f"POOLED BASELINE 5-fold balanced_accuracy (user_painted excluded): {baseline_bacc:.4f}")
    for t in TYPES:
        m = types_all == t
        acc_rate = baseline_oof_pred[m].mean()
        print(f"  baseline OOF accept-rate for {t}: {acc_rate:.3f}  (n={m.sum()})")
    print()

    # Reconciliation note: the task context states the production baseline
    # gets ~0.729. Re-deriving it under the EXACT rules given here (drop
    # user_painted, StandardScaler + LogisticRegression(class_weight=
    # "balanced"), StratifiedGroupKFold(5, shuffle=True, random_state=0))
    # gives 0.6486, not 0.729. If user_painted rows are INCLUDED in the
    # pooled fit/eval (they are always-True and inflate the positive class
    # with easy points), the number moves to ~0.736, much closer to the
    # stated 0.729 -- suggesting the original production baseline may not
    # have excluded user_painted. We follow the task's explicit instruction
    # to drop user_painted for BOTH the baseline and the per-type
    # experiment here, so the two are compared apples-to-apples even though
    # neither matches 0.729 exactly. Printed for transparency:
    full_labeled_incl_painted = load_labeled_interior()
    Xwp = full_labeled_incl_painted[FEATURES].values
    ywp = full_labeled_incl_painted["IsCrack"].values.astype(int)
    gwp = full_labeled_incl_painted["SourceImage"].values
    oof_wp = np.zeros(len(full_labeled_incl_painted), dtype=int)
    for train_idx, test_idx in sgkf.split(Xwp, ywp, gwp):
        scaler = StandardScaler().fit(Xwp[train_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(Xwp[train_idx]), ywp[train_idx])
        oof_wp[test_idx] = (clf.predict_proba(scaler.transform(Xwp[test_idx]))[:, 1] >= 0.5).astype(int)
    bacc_wp = balanced_accuracy_score(ywp, oof_wp)
    print(f"  [reconciliation only, not used further] pooled baseline WITH "
          f"user_painted included: {bacc_wp:.4f}  (context claims ~0.729)")
    print()

    # -----------------------------------------------------------------
    # 2) PER-TYPE MODELS, StratifiedGroupKFold(5) done PER TYPE (since each
    #    type's rows are a disjoint subset with its own SourceImage groups),
    #    but predictions recombined into one pooled OOF vector for the
    #    overall balanced_accuracy metric, directly comparable to baseline.
    #
    #    For interior_fill we grid over several C values (including strong
    #    regularization) to find the most conservative (lowest full-pool
    #    acceptance rate) option that still separates its 2 known negatives.
    # -----------------------------------------------------------------
    interior_fill_C_grid = [1.0, 0.5, 0.1, 0.05, 0.01]

    # IMPORTANT DESIGN CHOICE: we reuse the exact SAME pooled
    # StratifiedGroupKFold(5) fold assignment (computed once on the full
    # pooled df, stratified on IsCrack, grouped by SourceImage -- identical
    # to what the baseline uses) for all three per-type sub-models, rather
    # than re-running StratifiedGroupKFold independently per type.
    #
    # Why: interior_fill has only 2 negatives living in 2 distinct
    # SourceImage groups. An independent per-type StratifiedGroupKFold on
    # interior_fill alone (even with n_splits shrunk to 2) can and does
    # place BOTH negative-containing groups into the same test fold by
    # chance, leaving a training fold with ZERO negatives (verified this
    # crashes LogisticRegression with "only one class" -- confirmed via a
    # standalone probe before writing this final version). Reusing the
    # pooled fold assignment (driven mostly by the much larger, more
    # balanced concavity/bridge_corridor rows) empirically keeps at least
    # 1 negative in interior_fill's training partition in every one of the
    # 5 folds (verified: min-train-neg per fold was 1, 1, 2, 2, 1 across
    # the 5 folds). This is also the more natural reading of "recombine
    # predictions across the 3 per-type models for the pooled
    # StratifiedGroupKFold evaluation" -- one shared partition of the data,
    # three sub-models trained/evaluated within it.
    pooled_fold_assignment = list(sgkf.split(X_all, y_all, groups_all))

    def run_per_type_cv(C_by_type):
        """Run per-type models within the shared pooled fold assignment,
        return pooled OOF preds/proba."""
        oof_pred = np.full(len(df), -1, dtype=int)
        oof_proba = np.full(len(df), np.nan, dtype=float)
        for train_idx, test_idx in pooled_fold_assignment:
            for t in TYPES:
                type_mask = types_all == t
                tr_mask_t = type_mask[train_idx]
                te_mask_t = type_mask[test_idx]
                tr_idx_t = train_idx[tr_mask_t]
                te_idx_t = test_idx[te_mask_t]
                if len(te_idx_t) == 0:
                    continue
                C = C_by_type[t]
                scaler = StandardScaler().fit(X_all[tr_idx_t])
                Xtr = scaler.transform(X_all[tr_idx_t])
                Xte = scaler.transform(X_all[te_idx_t])
                clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=C)
                clf.fit(Xtr, y_all[tr_idx_t])
                proba = clf.predict_proba(Xte)[:, 1]
                oof_proba[te_idx_t] = proba
                oof_pred[te_idx_t] = (proba >= 0.5).astype(int)
        assert (oof_pred >= 0).all(), "some rows never scored OOF"
        return oof_pred, oof_proba

    print("=== Per-type model CV sweep over interior_fill C ===")
    results = {}
    for C_if in interior_fill_C_grid:
        C_by_type = {"concavity": 1.0, "bridge_corridor": 1.0, "interior_fill": C_if}
        oof_pred, oof_proba = run_per_type_cv(C_by_type)
        bacc = balanced_accuracy_score(y_all, oof_pred)
        results[C_if] = (bacc, oof_pred, oof_proba)
        print(f"  interior_fill C={C_if:<5} -> pooled 5-fold balanced_accuracy={bacc:.4f}")
        for t in TYPES:
            m = types_all == t
            print(f"      OOF accept-rate {t:<16}: {oof_pred[m].mean():.3f} (n={m.sum()})")
    print()

    # -----------------------------------------------------------------
    # 3) Fit FINAL per-type models on ALL labeled data (no held-out split)
    #    for each candidate C, then score the FULL interior_fill pool
    #    (all rows, not just labeled) to get true acceptance rate at
    #    threshold 0.5 -- directly comparable to baseline's 64%.
    # -----------------------------------------------------------------
    full_pool = load_full_pool()
    full_pool = full_pool[full_pool["CandidateType"] != "user_painted"].reset_index(drop=True)
    print(f"Full candidate pool (all types, all rows incl. unlabeled): {len(full_pool)} rows")
    print(full_pool.groupby("CandidateType").size())
    print()

    # fit final concavity & bridge_corridor models (C=1.0, standard) on all labeled data
    final_models = {}
    final_scalers = {}
    for t in ["concavity", "bridge_corridor"]:
        m = types_all == t
        scaler = StandardScaler().fit(X_all[m])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
        clf.fit(scaler.transform(X_all[m]), y_all[m])
        final_models[t] = clf
        final_scalers[t] = scaler

    # For interior_fill, fit one final model per C in the grid.
    if_mask = types_all == "interior_fill"
    X_if = X_all[if_mask]
    y_if = y_all[if_mask]
    df_if = df[if_mask].reset_index(drop=True)
    neg_if_idx = np.where(y_if == 0)[0]
    print(f"interior_fill labeled negatives (n={len(neg_if_idx)}):")
    for i in neg_if_idx:
        print(f"   row -> SourceImage={df_if.iloc[i]['SourceImage']}, "
              f"features={dict(zip(FEATURES, X_if[i]))}")
    print()

    interior_fill_final = {}
    for C_if in interior_fill_C_grid:
        scaler = StandardScaler().fit(X_if)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=C_if)
        clf.fit(scaler.transform(X_if), y_if)
        interior_fill_final[C_if] = (clf, scaler)

        # leave-one-out-ish check on the 2 known negatives: refit leaving
        # each negative out individually, score it with the resulting model.
        loo_scores = []
        for neg_i in neg_if_idx:
            train_mask = np.ones(len(X_if), dtype=bool)
            train_mask[neg_i] = False
            sc = StandardScaler().fit(X_if[train_mask])
            c = LogisticRegression(class_weight="balanced", max_iter=2000, C=C_if)
            c.fit(sc.transform(X_if[train_mask]), y_if[train_mask])
            p = c.predict_proba(sc.transform(X_if[neg_i:neg_i+1]))[0, 1]
            loo_scores.append(p)
        # also score the 2 negatives using the model trained on ALL data (in-sample, weaker check)
        insample_scores = clf.predict_proba(scaler.transform(X_if[neg_if_idx]))[:, 1]

        # score full interior_fill pool
        if_pool = full_pool[full_pool["CandidateType"] == "interior_fill"]
        Xp = if_pool[FEATURES].values
        proba_pool = clf.predict_proba(scaler.transform(Xp))[:, 1]
        accept_rate = (proba_pool >= 0.5).mean()

        print(f"interior_fill C={C_if}: full-pool accept-rate(thr=0.5)={accept_rate:.3f}  "
              f"(n_pool={len(if_pool)})")
        print(f"    known-negative in-sample proba: {np.round(insample_scores, 3)}")
        print(f"    known-negative leave-one-out proba: {np.round(loo_scores, 3)}")
        print()

    # -----------------------------------------------------------------
    # 4) Also compute concavity/bridge_corridor full-pool acceptance rate
    #    with the per-type models (should stay near baseline 12%, not
    #    collapse to ~0%).
    # -----------------------------------------------------------------
    for t in ["concavity", "bridge_corridor"]:
        pool_t = full_pool[full_pool["CandidateType"] == t]
        Xp = pool_t[FEATURES].values
        proba_pool = final_models[t].predict_proba(final_scalers[t].transform(Xp))[:, 1]
        accept_rate = (proba_pool >= 0.5).mean()
        print(f"{t}: full-pool accept-rate(thr=0.5) with per-type model = {accept_rate:.3f} "
              f"(n_pool={len(pool_t)})")
    print()

    # Also compute combined concavity+bridge_corridor pooled acceptance rate
    cb_pool = full_pool[full_pool["CandidateType"].isin(["concavity", "bridge_corridor"])]
    n_accept = 0
    for t in ["concavity", "bridge_corridor"]:
        pool_t = cb_pool[cb_pool["CandidateType"] == t]
        Xp = pool_t[FEATURES].values
        proba_pool = final_models[t].predict_proba(final_scalers[t].transform(Xp))[:, 1]
        n_accept += (proba_pool >= 0.5).sum()
    cb_accept_rate = n_accept / len(cb_pool)
    print(f"combined concavity+bridge_corridor full-pool accept-rate = {cb_accept_rate:.3f} "
          f"(n_pool={len(cb_pool)})")
    print()

    # -----------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Pooled single-model baseline balanced_accuracy: {baseline_bacc:.4f}")
    for C_if in interior_fill_C_grid:
        bacc = results[C_if][0]
        print(f"Per-type models (interior_fill C={C_if}): pooled balanced_accuracy={bacc:.4f}")


if __name__ == "__main__":
    main()
