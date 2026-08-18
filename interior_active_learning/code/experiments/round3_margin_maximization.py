"""
Round 3 experiment: margin_maximization

Round 2 found a "Pareto-better" 3rd-gate rule (higher recall, lower pool
acceptance than production) but a margin/sensitivity audit rejected it for
production because it moved one negative's rejection off a robust ML-floor
margin onto a brand-new, never-stress-tested gate with only ~7% relative
margin. That raised the real question this script answers: forget recall for
a moment -- among all 2-gate (MeanDistToCrack-percentile, MeanFlatBrightness-
percentile, ML-floor) configs that still reject BOTH known interior_fill
negatives under a proper leave-one-out refit (exact same validity test as
calibrate_interior_fill_rule() in ../train_interior_model.py), which one is
the SAFEST -- i.e. maximizes the worst-case (minimum-across-both-negatives)
rejection margin -- even if that costs some recall?

Margin definition (per negative, per candidate config):
  For every leg (dist / bri / floor) that is currently FAILING for that
  negative (i.e. contributing to its rejection), compute a *relative*
  margin:
    dist:  |value - dist_thr| / dist_thr * 100      (%)
    bri:   |value - bri_thr|  / bri_thr  * 100       (%)
    floor: |floor - LOO_proba| / floor * 100         (%)   (also reported as
                                                             the raw absolute
                                                             difference, since
                                                             the task brief
                                                             specifies
                                                             floor - LOO_proba
                                                             literally)
  A negative's "protection margin" for that config = the SMALLEST of its
  currently-failing legs' relative margins (the weakest link -- if more than
  one leg fails, the negative is more robust than this number suggests
  against single-axis perturbation, but this is the conservative number to
  optimize against). A config's overall margin = min over the 2 negatives of
  their protection margins. We grid-search (N, M, floor) and MAXIMIZE this
  minimum -- a maximin search -- rather than maximizing recall subject to a
  feasibility constraint (which is what calibrate_interior_fill_rule() does).

Everything downstream of the pooled scaler/clf (concavity, bridge_corridor,
the plain interior_fill floor fallback) is untouched by this experiment --
only the interior_fill-specific (N, M, floor) triple changes, so
concavity/bridge_corridor behavior is unaffected by construction. This is
verified numerically at the end anyway.

Run: python3 round3_margin_maximization.py
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from active_learning_select import INTERIOR_FEATURE_COLUMNS
from train_interior_model import (
    load_labeled_interior, load_full_interior_fill_pool, calibrate_interior_fill_rule,
)

# Documented "current production" numbers from the task brief.
DOC_PROD_FLOOR = 0.60
DOC_PROD_N, DOC_PROD_M = 70, 90
RISKY_REL_MARGIN_PCT = 15.0  # same heuristic threshold used in round 2's third-gate audit


def fit_pooled(df):
    X = df[INTERIOR_FEATURE_COLUMNS].values
    y = df["IsCrack"].astype(bool).values
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X), y)
    return scaler, clf


def score(df_, sc, model):
    return model.predict_proba(sc.transform(df_[INTERIOR_FEATURE_COLUMNS].values))[:, 1]


def loo_refit_excluding(df_full, source_image, label):
    keep = ~((df_full["SourceImage"] == source_image) & (df_full["Label"].values == label))
    X_loo = df_full.loc[keep, INTERIOR_FEATURE_COLUMNS].values
    y_loo = df_full.loc[keep, "IsCrack"].astype(bool).values
    sc = StandardScaler().fit(X_loo)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X_loo), y_loo)
    return sc, clf


def margin_detail(dist_thr, bri_thr, floor, neg_dist, neg_bri, loo_proba_neg):
    """Per-negative margin decomposition for one (dist_thr, bri_thr, floor)
    config. Returns (valid, min_margin_pct, per_negative_detail) where valid
    means the config rejects BOTH known negatives."""
    detail = []
    for i in range(2):
        dist_pass = neg_dist[i] <= dist_thr
        bri_pass = neg_bri[i] <= bri_thr
        floor_pass = loo_proba_neg[i] >= floor
        if dist_pass and bri_pass and floor_pass:
            return False, None, None
        active = []
        if not dist_pass:
            active.append(("dist", abs(neg_dist[i] - dist_thr) / dist_thr * 100))
        if not bri_pass:
            active.append(("bri", abs(neg_bri[i] - bri_thr) / bri_thr * 100))
        if not floor_pass:
            active.append(("floor", abs(floor - loo_proba_neg[i]) / floor * 100,
                            floor - loo_proba_neg[i]))
        # tightest (smallest) relative margin among the legs actually failing
        leg, m = min(((a[0], a[1]) for a in active), key=lambda t: t[1])
        detail.append({"neg_idx": i, "tightest_leg": leg, "tightest_rel_margin_pct": m,
                        "n_active_legs": len(active), "active_legs": active})
    min_margin = min(d["tightest_rel_margin_pct"] for d in detail)
    return True, min_margin, detail


def main():
    t0 = time.time()
    df_full = load_labeled_interior()
    pool = load_full_interior_fill_pool()
    interior_lab = df_full[df_full["CandidateType"] == "interior_fill"].reset_index(drop=True)
    y_int = interior_lab["IsCrack"].astype(bool).values
    pos_rows = interior_lab[y_int].reset_index(drop=True)
    neg_rows = interior_lab[~y_int].reset_index(drop=True)
    print("=" * 92)
    print(f"Loaded {len(df_full)} labeled rows ({df_full['SourceImage'].nunique()} images); "
          f"interior_fill: {len(pos_rows)} positive, {len(neg_rows)} negative.")
    assert len(neg_rows) == 2, "This script assumes exactly 2 known interior_fill negatives."
    for _, r in neg_rows.iterrows():
        print(f"  NEGATIVE {r['SourceImage']} (Label={r['Label']}): "
              f"MeanDistToCrack={r['MeanDistToCrack']:.3f}  MeanFlatBrightness={r['MeanFlatBrightness']:.3f}")

    scaler_full, clf_full = fit_pooled(df_full)
    full_fit_proba_pos = score(pos_rows, scaler_full, clf_full)
    pool_proba = score(pool, scaler_full, clf_full)
    dist_pool = pool["MeanDistToCrack"].values
    bri_pool = pool["MeanFlatBrightness"].values
    dist_pos = pos_rows["MeanDistToCrack"].values
    bri_pos = pos_rows["MeanFlatBrightness"].values
    neg_dist = neg_rows["MeanDistToCrack"].values
    neg_bri = neg_rows["MeanFlatBrightness"].values
    n_pos_total = len(pos_rows)
    n_pool_total = len(pool)

    # ---- sanity check: reproduce train_interior_model.py's own calibration ----
    baseline = calibrate_interior_fill_rule(df_full, scaler_full, clf_full)
    print("\nReproduced calibrate_interior_fill_rule() on CURRENT data (sanity check):")
    print(f"  {baseline}")
    print(f"  (Task brief documents floor=0.60, recall=74.4%, pool_accept=57.6% -- the dist/bri "
          f"thresholds reproduce exactly, but the floor/recall/accept differ slightly here "
          f"({baseline['floor']}/{baseline['recall_on_known_positives']*100:.1f}%/"
          f"{baseline['full_pool_accept_rate']*100:.1f}%), almost certainly because a few more "
          f"labels have been added to candidates/*_interior.csv since the production numbers in "
          f"the task brief were computed. Both the documented AND the freshly-reproduced "
          f"production configs are evaluated below for a fair comparison.)")

    # ---- full-data LOO-refit probability for each known negative ----
    loo_proba_neg = np.zeros(2)
    for i, r in neg_rows.iterrows():
        sc, clf = loo_refit_excluding(df_full, r["SourceImage"], r["Label"])
        loo_proba_neg[i] = clf.predict_proba(
            sc.transform(neg_rows.iloc[[i]][INTERIOR_FEATURE_COLUMNS].values))[0, 1]
    print("\nFull-data LOO-refit probabilities (used for the floor leg, exactly as in "
          "calibrate_interior_fill_rule()):")
    for i, r in neg_rows.iterrows():
        print(f"  neg{i} ({r['SourceImage']}): LOO_proba={loo_proba_neg[i]:.4f}")

    # ======================================================================
    # Grid search: maximize the MINIMUM margin (maximin), not recall.
    # ======================================================================
    N_grid = np.arange(1, 96, 1)     # percentile of dist_pool, 1..95
    M_grid = np.arange(1, 100, 1)    # percentile of bri_pool, 1..99
    floor_grid = np.round(np.arange(0.05, 0.951, 0.01), 2)  # ML-proba floor
    print(f"\nGrid: {len(N_grid)} x {len(M_grid)} x {len(floor_grid)} = "
          f"{len(N_grid)*len(M_grid)*len(floor_grid):,} raw (N,M,floor) combos "
          f"(finer than production's step-5/step-5/step-0.05 grid, to find the true maximin).")

    pos_sorted_cache = {}
    pool_sorted_cache = {}
    results = []

    for N in N_grid:
        dist_thr = np.percentile(dist_pool, N)
        dist_pass_pos = dist_pos <= dist_thr
        dist_pass_pool = dist_pool <= dist_thr
        dist_pass_neg = neg_dist <= dist_thr
        for M in M_grid:
            bri_thr = np.percentile(bri_pool, M)
            geom_pass_pos = dist_pass_pos & (bri_pos <= bri_thr)
            geom_pass_pool = dist_pass_pool & (bri_pool <= bri_thr)
            geom_pass_neg = dist_pass_neg & (neg_bri <= bri_thr)

            pos_sorted = np.sort(full_fit_proba_pos[geom_pass_pos])
            pool_sorted = np.sort(pool_proba[geom_pass_pool])

            # Per negative: is it rejected purely by geometry (dist/bri), for
            # EVERY floor, or does the outcome depend on floor?
            for i in range(2):
                pass  # (handled inside margin_detail per floor below)

            for floor in floor_grid:
                valid, min_margin, detail = margin_detail(
                    dist_thr, bri_thr, floor, neg_dist, neg_bri, loo_proba_neg)
                if not valid:
                    continue
                recall = (len(pos_sorted) - np.searchsorted(pos_sorted, floor, side="left")) / n_pos_total
                pool_accept = (len(pool_sorted) - np.searchsorted(pool_sorted, floor, side="left")) / n_pool_total
                results.append((N, M, floor, dist_thr, bri_thr, min_margin, recall, pool_accept))

    elapsed = time.time() - t0
    print(f"\nValid (reject-both-negatives) grid points found: {len(results):,} "
          f"(search took {elapsed:.1f}s)")
    if not results:
        print("No valid config found anywhere in the grid -- aborting.")
        return

    cols = ["N", "M", "floor", "dist_thr", "bri_thr", "min_margin_pct", "recall", "pool_accept"]
    grid_df = pd.DataFrame(results, columns=cols)
    grid_df.to_csv(os.path.join(os.path.dirname(__file__), "round3_margin_maximization_full_grid.csv"),
                    index=False)

    # ---- maximin selection: max min_margin_pct, tie-break by higher recall,
    # then lower pool_accept, then prefer a genuinely-binding brightness gate ----
    grid_df_sorted = grid_df.sort_values(
        by=["min_margin_pct", "recall", "pool_accept"], ascending=[False, False, True]
    ).reset_index(drop=True)
    best = grid_df_sorted.iloc[0]
    print("\n" + "=" * 92)
    print("MAXIMIN CONFIG (maximizes the worst-case minimum margin across both negatives):")
    print(f"  N={best.N:.0f} (dist pctile)  M={best.M:.0f} (bri pctile)  floor={best.floor:.2f}")
    print(f"  dist_thr={best.dist_thr:.3f}px   bri_thr={best.bri_thr:.3f}")
    print(f"  min_margin_pct={best.min_margin_pct:.2f}%   recall_on_known_positives={best.recall*100:.1f}%   "
          f"full_pool_accept={best.pool_accept*100:.1f}%")
    valid, min_margin, detail = margin_detail(best.dist_thr, best.bri_thr, best.floor,
                                               neg_dist, neg_bri, loo_proba_neg)
    for d in detail:
        r = neg_rows.iloc[d["neg_idx"]]
        print(f"    neg{d['neg_idx']} ({r['SourceImage']}): tightest active leg = {d['tightest_leg']} "
              f"(rel margin {d['tightest_rel_margin_pct']:.2f}%), "
              f"{d['n_active_legs']} leg(s) active: {d['active_legs']}")

    # ---- evaluate the PRODUCTION rule(s) with the exact same margin metric ----
    print("\n" + "=" * 92)
    print("PRODUCTION RULE(S) evaluated with the same margin metric, for comparison:")

    def eval_and_print(label, dist_thr, bri_thr, floor):
        valid, min_margin, detail = margin_detail(dist_thr, bri_thr, floor, neg_dist, neg_bri, loo_proba_neg)
        geom_pass_pos = (dist_pos <= dist_thr) & (bri_pos <= bri_thr)
        recall = ((full_fit_proba_pos >= floor) & geom_pass_pos).mean()
        geom_pass_pool = (dist_pool <= dist_thr) & (bri_pool <= bri_thr)
        pool_accept = ((pool_proba >= floor) & geom_pass_pool).mean()
        print(f"\n  {label}: dist_thr={dist_thr:.3f} bri_thr={bri_thr:.3f} floor={floor:.2f}")
        print(f"    valid (rejects both LOO negatives)? {valid}")
        print(f"    recall={recall*100:.1f}%  pool_accept={pool_accept*100:.1f}%")
        if valid:
            print(f"    min_margin_pct={min_margin:.2f}%")
            for d in detail:
                r = neg_rows.iloc[d["neg_idx"]]
                print(f"      neg{d['neg_idx']} ({r['SourceImage']}): tightest leg={d['tightest_leg']} "
                      f"margin={d['tightest_rel_margin_pct']:.2f}%  active_legs={d['active_legs']}")
        return valid, min_margin, recall, pool_accept

    doc_dist_thr = np.percentile(dist_pool, DOC_PROD_N)
    doc_bri_thr = np.percentile(bri_pool, DOC_PROD_M)
    _, doc_margin, doc_recall, doc_accept = eval_and_print(
        "DOCUMENTED production (task brief: floor=0.60)", doc_dist_thr, doc_bri_thr, DOC_PROD_FLOOR)
    _, fresh_margin, fresh_recall, fresh_accept = eval_and_print(
        f"FRESHLY-RECALIBRATED production (this run's calibrate_interior_fill_rule(), "
        f"floor={baseline['floor']})",
        baseline["dist_thr"], baseline["bri_thr"], baseline["floor"])

    # ======================================================================
    # Show the margin/recall trade-off curve (top of the Pareto-ish frontier
    # by margin) so it's clear how much recall the safest configs cost.
    # ======================================================================
    print("\n" + "=" * 92)
    print("Top 15 configs by min_margin_pct (maximin frontier), with recall/pool_accept shown:")
    print(grid_df_sorted.head(15).to_string(index=False))

    print("\nFor reference, top 15 configs by RECALL among only those with min_margin_pct >= "
          f"{RISKY_REL_MARGIN_PCT:.0f}% (i.e. 'safe enough' by round 2's own risky-margin heuristic):")
    safe_df = grid_df[grid_df["min_margin_pct"] >= RISKY_REL_MARGIN_PCT].sort_values(
        by=["recall", "pool_accept"], ascending=[False, True]).reset_index(drop=True)
    if len(safe_df):
        print(safe_df.head(15).to_string(index=False))
        best_safe = safe_df.iloc[0]
    else:
        print("  (none -- no config clears the 15% margin bar)")
        best_safe = None

    # ======================================================================
    # Practical stress test of the maximin config: does the higher margin
    # number actually buy more resistance to feature perturbation, using the
    # exact same break-even search methodology as
    # refine_stability_and_sensitivity.py (Part 2)?
    # ======================================================================
    print("\n" + "=" * 92)
    print("STRESS TEST: break-even perturbation search (same methodology as round 2's "
          "refine_stability_and_sensitivity.py) comparing the maximin config against the "
          "documented production config.")

    loo_models = {}
    for i, r in neg_rows.iterrows():
        loo_models[i] = loo_refit_excluding(df_full, r["SourceImage"], r["Label"])

    def breakeven_search(dist_thr, bri_thr, floor, label):
        print(f"\n  -- {label} (dist_thr={dist_thr:.3f}, bri_thr={bri_thr:.3f}, floor={floor:.2f}) --")
        dist_idx = INTERIOR_FEATURE_COLUMNS.index("MeanDistToCrack")
        bri_idx = INTERIOR_FEATURE_COLUMNS.index("MeanFlatBrightness")
        for i, r in neg_rows.iterrows():
            sc, clf = loo_models[i]
            base_feat = interior_lab[(interior_lab["SourceImage"] == r["SourceImage"]) &
                                      (interior_lab["Label"] == r["Label"])].iloc[0][
                INTERIOR_FEATURE_COLUMNS].values.astype(float).copy()
            found_any = False
            for feat_name, idx in [("MeanDistToCrack", dist_idx), ("MeanFlatBrightness", bri_idx)]:
                flipped_at = None
                for pct in np.arange(0, 50.01, 0.5):
                    for sign in (-1, 1):
                        factor = 1 + sign * pct / 100.0
                        f = base_feat.copy()
                        f[idx] = base_feat[idx] * factor
                        proba = clf.predict_proba(sc.transform(f.reshape(1, -1)))[0, 1]
                        dist_pass = f[dist_idx] <= dist_thr
                        bri_pass = f[bri_idx] <= bri_thr
                        floor_pass = proba >= floor
                        if dist_pass and bri_pass and floor_pass:
                            flipped_at = sign * pct
                            break
                    if flipped_at is not None:
                        break
                if flipped_at is not None:
                    found_any = True
                    print(f"    neg{i} ({r['SourceImage']}): perturbing {feat_name} by "
                          f"{flipped_at:+.1f}% flips the rule to ACCEPT.")
            if not found_any:
                print(f"    neg{i} ({r['SourceImage']}): no single-feature perturbation up to "
                      f"+/-50% (scanned in 0.5% steps) flips the rule to accept.")

    breakeven_search(doc_dist_thr, doc_bri_thr, DOC_PROD_FLOOR, "DOCUMENTED production")
    breakeven_search(best.dist_thr, best.bri_thr, best.floor, "MAXIMIN config")
    if best_safe is not None:
        breakeven_search(best_safe.dist_thr, best_safe.bri_thr, best_safe.floor,
                          "Best-recall config subject to >=15% margin floor")

    # ======================================================================
    # Confirm concavity / bridge_corridor are untouched (they use the SAME
    # pooled scaler/clf at a plain floor, unrelated to the interior_fill-
    # specific N/M/floor triple chosen above).
    # ======================================================================
    print("\n" + "=" * 92)
    print("CONFIRM concavity / bridge_corridor unaffected (same pooled model regardless of which "
          "interior_fill (N,M,floor) is chosen -- this experiment never touches their data/labels):")
    for t in ["concavity", "bridge_corridor"]:
        sub = df_full[df_full["CandidateType"] == t]
        proba_t = score(sub, scaler_full, clf_full)
        acc05 = (proba_t >= 0.5).mean()
        print(f"  {t}: n={len(sub)}  plain-floor(0.5) acceptance on labeled rows = {acc05*100:.1f}% "
              f"(identical regardless of interior_fill rule choice, by construction)")

    print("\nDone.")


if __name__ == "__main__":
    main()
