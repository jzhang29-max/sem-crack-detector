"""
Round 3 experiment: simple_untried_algorithms
-----------------------------------------------
Rounds 1-2 tried LogisticRegression (production baseline), RandomForest,
GradientBoosting, SVC(rbf), a smooth 3-feature logistic score, and hard
percentile-gate rules. All ensemble/kernel methods overfit the 2 known
interior_fill negatives and made full-pool acceptance WORSE (toward 100%).

This experiment tries algorithms with genuinely different bias/variance
behavior that have NOT yet been tried:

  1. GaussianNB on all 11 features (pooled).
  2. k-NN, k=3 and k=5, distance-weighted, on all 11 features (pooled,
     standardized -- k-NN needs scaling for distance to be meaningful).
  3. A single-feature decision stump (DecisionTreeClassifier(max_depth=1),
     i.e. one threshold) independently for EACH of the 11 features -- the
     simplest possible model, essentially zero capacity to memorize 2 points
     beyond one number.
  4. A depth-2 decision tree (max_leaf_nodes in {3, 4}) restricted to just
     MeanDistToCrack and MeanFlatBrightness -- the same two features the
     production hybrid rule already uses as hard percentile gates, but let a
     tree pick the split points instead of grid-searching percentiles.

Protocol (kept identical to alternative_algorithms.py / production baseline
for comparability):
  - Pool all 3 candidate types (concavity, bridge_corridor, interior_fill),
    dropping user_painted (always True, no discriminative signal).
  - StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) grouped by
    SourceImage -> mean balanced_accuracy across folds (production baseline
    LogReg pooled CV: ~0.729 balanced_acc from train_interior_model.py runs).
  - Refit each model on ALL labeled (non-user_painted) data:
      - full interior_fill candidate POOL (labeled + unlabeled) acceptance
        rate @ predict_proba >= 0.5 (production plain-floor baseline: ~96%;
        production hybrid rule: 57.6%).
      - recall on the 39 KNOWN interior_fill positives @ predict_proba >= 0.5
        (production hybrid rule: 74.4%) -- for apples-to-apples comparison.
      - full-pool acceptance for concavity / bridge_corridor (must stay near
        their current, already-reasonable behavior -- large drop-to-zero or
        jump-to-one is a red flag).
  - LEAVE-ONE-OUT check on the 2 known interior_fill negatives: refit the
    WHOLE pipeline excluding one known negative at a time (all other labeled
    data, all types, retained) and check whether the held-out negative is
    still scored < 0.5. This is the generalization check requested -- not
    in-sample probability.
  - For the single-feature stumps, also report the actual chosen threshold
    and direction per feature so the rule is inspectable/interpretable.
  - For the 2-feature tree, print the learned tree structure (thresholds)
    directly.

Honesty note: with only 2 known interior_fill negatives, "passing" the LOO
check is a necessary-but-not-sufficient condition, not proof of
generalization. Prior rounds found recall swings of 56-72% across small
image subsets purely from resampling noise -- treat small point differences
here the same way.
"""
import glob
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# `os` is imported HERE because the block below needs it. It was missing entirely: the
# hardcoded /Users/... path was replaced with a __file__-derived one and the import was
# never added, so this script raised NameError before doing any work.
import os

# Project root derived from this file's location, not hardcoded: these scripts
# shipped with an absolute /Users/... path and could only run on the machine that
# wrote them, while archive/README.md advertises them as rerunnable.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CANDIDATES_GLOB = os.path.join(_ROOT, "interior_active_learning", "candidates", "*_interior.csv")

FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
            "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
            "FracBoundaryTouchingCrack", "MeanDistToCrack"]
TWO_GATE_FEATURES = ["MeanDistToCrack", "MeanFlatBrightness"]

RNG_SEED = 0


def load_labeled_interior():
    rows = []
    for f in sorted(glob.glob(CANDIDATES_GLOB)):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        labeled = d[d["IsCrack"].astype(str).str.strip().str.upper().isin(["TRUE", "FALSE"])].copy()
        if len(labeled) == 0:
            continue
        labeled["IsCrack"] = labeled["IsCrack"].astype(str).str.strip().str.upper() == "TRUE"
        rows.append(labeled)
    return pd.concat(rows, ignore_index=True)


def load_all_rows():
    rows = []
    for f in sorted(glob.glob(CANDIDATES_GLOB)):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def pooled_cv_balanced_acc(pipe_template, X, y, groups, ctype, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    fold_bal_accs = []
    fold_details = []
    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        pipe = clone(pipe_template)
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        bal_acc = balanced_accuracy_score(y[test_idx], preds)
        fold_bal_accs.append(bal_acc)
        test_types = ctype[test_idx]
        detail = {}
        for t in np.unique(test_types):
            mask = test_types == t
            if mask.sum() == 0:
                continue
            detail[t] = (int(mask.sum()), int(y[test_idx][mask].sum()))
        fold_details.append((fold_i, bal_acc, detail))
    return fold_bal_accs, fold_details


def full_pool_acceptance(pipe, df_full_pool, feat_cols):
    acceptance = {}
    for t in ["interior_fill", "concavity", "bridge_corridor"]:
        pool_t = df_full_pool[df_full_pool["CandidateType"] == t].copy()
        if len(pool_t) == 0:
            continue
        Xt = pool_t[feat_cols].values
        probs = pipe.predict_proba(Xt)[:, 1]
        accept_rate = float((probs >= 0.5).mean())
        acceptance[t] = (accept_rate, len(pool_t))
    return acceptance


def recall_on_known_positives(pipe, df_labeled, feat_cols):
    pos = df_labeled[(df_labeled["CandidateType"] == "interior_fill") & (df_labeled["IsCrack"])]
    if len(pos) == 0:
        return float("nan"), 0
    Xp = pos[feat_cols].values
    probs = pipe.predict_proba(Xp)[:, 1]
    return float((probs >= 0.5).mean()), len(pos)


def loo_negative_check(pipe_template, df, feat_cols):
    neg_mask = (df["CandidateType"] == "interior_fill") & (~df["IsCrack"])
    neg_idx = df.index[neg_mask].tolist()
    loo_results = []
    for hold_idx in neg_idx:
        train_mask = df.index != hold_idx
        X_loo_train = df.loc[train_mask, feat_cols].values
        y_loo_train = df.loc[train_mask, "IsCrack"].values.astype(int)
        X_held = df.loc[[hold_idx], feat_cols].values

        loo_pipe = clone(pipe_template)
        loo_pipe.fit(X_loo_train, y_loo_train)
        prob_held = loo_pipe.predict_proba(X_held)[:, 1][0]
        correct = prob_held < 0.5
        loo_results.append((hold_idx, float(prob_held), bool(correct)))
    return loo_results


def run_model(name, pipe_template, df, df_full_pool, feat_cols, verbose_tree=None):
    print("=" * 70)
    print(f"MODEL: {name}  (features: {feat_cols if len(feat_cols) <= 2 else f'all {len(feat_cols)}'})")
    print("=" * 70)

    X = df[feat_cols].values
    y = df["IsCrack"].values.astype(int)
    groups = df["SourceImage"].values
    ctype = df["CandidateType"].values

    fold_bal_accs, fold_details = pooled_cv_balanced_acc(pipe_template, X, y, groups, ctype)
    mean_bal_acc = float(np.mean(fold_bal_accs))
    std_bal_acc = float(np.std(fold_bal_accs))
    print(f"Per-fold balanced_accuracy: {[round(a, 3) for a in fold_bal_accs]}")
    for fold_i, bal_acc, detail in fold_details:
        print(f"  fold {fold_i}: bal_acc={bal_acc:.3f}  test-set (n, n_true) per type: {detail}")
    print(f"Mean balanced_accuracy across 5 folds: {mean_bal_acc:.4f} (std {std_bal_acc:.4f})")

    pipe = clone(pipe_template)
    pipe.fit(X, y)
    train_preds = pipe.predict(X)
    train_acc = accuracy_score(y, train_preds)
    train_bal_acc = balanced_accuracy_score(y, train_preds)
    print(f"In-sample (train-on-all) accuracy: {train_acc:.4f}, balanced_accuracy: {train_bal_acc:.4f}")

    acceptance = full_pool_acceptance(pipe, df_full_pool, feat_cols)
    for t, (rate, n) in acceptance.items():
        print(f"Full-pool acceptance rate @0.5 for {t}: {rate:.3f} (n={n})")

    recall_pos, n_pos = recall_on_known_positives(pipe, df, feat_cols)
    print(f"Recall on {n_pos} known interior_fill positives @0.5: {recall_pos:.3f}")

    loo_results = loo_negative_check(pipe_template, df, feat_cols)
    print(f"Known interior_fill negatives LOO check (refit excl. each, want prob<0.5):")
    for hold_idx, prob_held, correct in loo_results:
        print(f"  LOO holdout row {hold_idx}: P(crack)={prob_held:.3f} -> "
              f"{'CORRECT (rejected)' if correct else 'WRONG: accepted (>=0.5)'}")

    if verbose_tree is not None:
        print(verbose_tree)

    return {
        "mean_bal_acc": mean_bal_acc, "std_bal_acc": std_bal_acc,
        "train_acc": train_acc, "acceptance": acceptance,
        "recall_pos": recall_pos, "loo_results": loo_results,
    }


def main():
    df_all_labeled = load_labeled_interior()
    df_full_pool = load_all_rows()

    print(f"Total labeled rows (all types incl. user_painted): {len(df_all_labeled)}")
    print(f"CandidateType counts (labeled):\n{df_all_labeled['CandidateType'].value_counts()}\n")

    df = df_all_labeled[df_all_labeled["CandidateType"] != "user_painted"].copy().reset_index(drop=True)
    print(f"Rows after dropping user_painted: {len(df)}")
    print(df.groupby("CandidateType")["IsCrack"].agg(["count", "sum"]))
    print()

    results = {}

    # ---------------------------------------------------------------
    # 1. GaussianNB, all 11 features
    # ---------------------------------------------------------------
    results["GaussianNB"] = run_model(
        "GaussianNB",
        Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        df, df_full_pool, FEATURES)

    # ---------------------------------------------------------------
    # 2. k-NN, k=3 and k=5, distance-weighted, all 11 features
    # ---------------------------------------------------------------
    for k in (3, 5):
        results[f"kNN_k{k}_distance"] = run_model(
            f"kNN_k{k}_distance",
            Pipeline([("scaler", StandardScaler()),
                      ("clf", KNeighborsClassifier(n_neighbors=k, weights="distance"))]),
            df, df_full_pool, FEATURES)

    # ---------------------------------------------------------------
    # 3. Single-feature decision stumps -- one per feature, all 11
    # ---------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# SINGLE-FEATURE DECISION STUMPS (max_depth=1), one per feature")
    print("#" * 70)
    stump_results = {}
    for feat in FEATURES:
        stump_template = Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=RNG_SEED)),
        ])
        res = run_model(f"Stump[{feat}]", stump_template, df, df_full_pool, [feat])
        # extract the actual learned threshold on the full-data refit for interpretability
        full_pipe = clone(stump_template)
        Xf = df[[feat]].values
        yf = df["IsCrack"].values.astype(int)
        full_pipe.fit(Xf, yf)
        tree = full_pipe.named_steps["clf"]
        if tree.tree_.node_count > 1:
            thr = tree.tree_.threshold[0]
            print(f"  Learned stump rule (fit on all data): {feat} <= {thr:.3f}  ->  "
                  f"{export_text(tree, feature_names=[feat]).strip()}")
        else:
            print(f"  Stump for {feat} did not split (root is a leaf) -- feature has no useful threshold.")
        stump_results[feat] = res
        results[f"Stump[{feat}]"] = res

    best_stump_feat = max(stump_results, key=lambda f: stump_results[f]["mean_bal_acc"])
    print(f"\nBest single-feature stump by pooled mean balanced_accuracy: {best_stump_feat} "
          f"({stump_results[best_stump_feat]['mean_bal_acc']:.4f})")

    # ---------------------------------------------------------------
    # 4. 2-feature decision tree (MeanDistToCrack, MeanFlatBrightness),
    #    max_leaf_nodes in {3, 4}
    # ---------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# 2-FEATURE DECISION TREE (MeanDistToCrack, MeanFlatBrightness only)")
    print("#" * 70)
    for max_leaves in (3, 4):
        tree_template = Pipeline([
            ("clf", DecisionTreeClassifier(max_leaf_nodes=max_leaves, class_weight="balanced",
                                            random_state=RNG_SEED)),
        ])
        full_pipe = clone(tree_template)
        Xf = df[TWO_GATE_FEATURES].values
        yf = df["IsCrack"].values.astype(int)
        full_pipe.fit(Xf, yf)
        tree_text = export_text(full_pipe.named_steps["clf"], feature_names=TWO_GATE_FEATURES)
        name = f"Tree2Feat_maxleaf{max_leaves}"
        res = run_model(name, tree_template, df, df_full_pool, TWO_GATE_FEATURES,
                         verbose_tree=f"Learned tree structure (fit on all data):\n{tree_text}")
        results[name] = res

    # ---------------------------------------------------------------
    # Reference: production baseline pooled LogisticRegression, same protocol
    # ---------------------------------------------------------------
    results["LogisticRegression_baseline"] = run_model(
        "LogisticRegression_baseline",
        Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RNG_SEED))]),
        df, df_full_pool, FEATURES)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (plain @0.5 threshold, pooled model, NO percentile gating)")
    print("=" * 70)
    header = (f"{'Model':30s} {'PooledBalAcc':>13s} {'std':>6s} {'TrainAcc':>9s} "
              f"{'IF_accept':>10s} {'IF_recall':>10s} {'CC_accept':>10s} {'BC_accept':>10s} {'LOO_both_reject':>16s}")
    print(header)
    display_order = ["LogisticRegression_baseline", "GaussianNB", "kNN_k3_distance", "kNN_k5_distance"]
    display_order += [f"Stump[{f}]" for f in FEATURES]
    display_order += ["Tree2Feat_maxleaf3", "Tree2Feat_maxleaf4"]
    for name in display_order:
        r = results[name]
        if_acc = r["acceptance"].get("interior_fill", (float("nan"), 0))[0]
        cc_acc = r["acceptance"].get("concavity", (float("nan"), 0))[0]
        bc_acc = r["acceptance"].get("bridge_corridor", (float("nan"), 0))[0]
        loo_both = all(c for _, _, c in r["loo_results"]) if r["loo_results"] else False
        print(f"{name:30s} {r['mean_bal_acc']:13.4f} {r['std_bal_acc']:6.3f} {r['train_acc']:9.4f} "
              f"{if_acc:10.3f} {r['recall_pos']:10.3f} {cc_acc:10.3f} {bc_acc:10.3f} {str(loo_both):>16s}")

    print("\nProduction reference (hybrid percentile-gated rule, for comparison, not re-derived here):")
    print("  recall_on_known_positives=0.744  full_pool_accept=0.576  LOO both rejected=True")
    print("  Plain floor-only baseline (no gating) reference: accept~0.96 (over-permissive)")

    print("\nDone.")


if __name__ == "__main__":
    main()
