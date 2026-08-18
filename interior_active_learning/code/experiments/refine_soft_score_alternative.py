"""
Experiment: soft_score_alternative

The production hybrid rule for interior_fill is a hard AND of 3 independent
threshold cutoffs (ML floor, MeanDistToCrack percentile, MeanFlatBrightness
percentile) -- see calibrate_interior_fill_rule() in ../train_interior_model.py.
A candidate that narrowly misses just ONE leg (e.g. 71st percentile distance
instead of <=70th) gets rejected outright even if the other two legs are
comfortably satisfied.

This experiment tests whether a SMOOTH combined score generalizes at least as
well: fit a small auxiliary LogisticRegression using ONLY 3 inputs --
  1. ML_proba from the pooled model (same pooled model as production)
  2. standardized MeanDistToCrack
  3. standardized MeanFlatBrightness
trained on interior_fill labeled examples only (39 pos / 2 neg),
class_weight="balanced", grid over C in {0.01, 0.05, 0.1, 0.5}.

Validated with the SAME leave-one-out methodology as production: refit
(both the pooled model's scaler/clf AND the auxiliary model) excluding each
known negative in turn, then score that held-out negative and require it
falls below the chosen decision threshold.

Also reports, for each of the 2 known negatives (under LOO refit), the
signed distance from the auxiliary-model decision boundary in probability
terms, and how that compares (qualitatively + numerically) to how far each
negative sits from failing its BEST single leg of the hard-AND rule -- to
judge whether the smooth score gives more or less safety margin.
"""
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import CANDIDATES_DIR
from active_learning_select import INTERIOR_FEATURE_COLUMNS

pd.set_option("display.width", 160)


def load_labeled_interior():
    rows = []
    for f in sorted(glob.glob(os.path.join(CANDIDATES_DIR, "*_interior.csv"))):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        labeled = d[d["IsCrack"].astype(str).str.strip().str.upper().isin(["TRUE", "FALSE"])].copy()
        if len(labeled) == 0:
            continue
        labeled["IsCrack"] = labeled["IsCrack"].astype(str).str.strip().str.upper() == "TRUE"
        rows.append(labeled)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_full_interior_fill_pool():
    rows = []
    for f in sorted(glob.glob(os.path.join(CANDIDATES_DIR, "*_interior.csv"))):
        d = pd.read_csv(f)
        if len(d):
            rows.append(d[d["CandidateType"] == "interior_fill"])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_pooled(df_labeled):
    """The production pooled model: LogisticRegression(class_weight=balanced)
    + StandardScaler over ALL labeled candidates (all 3 types pooled), used
    to produce ML_proba, exactly as production does."""
    X = df_labeled[INTERIOR_FEATURE_COLUMNS].values
    y = df_labeled["IsCrack"].astype(bool).values
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X), y)
    return scaler, clf


def pooled_proba(df_sub, scaler, clf):
    return clf.predict_proba(scaler.transform(df_sub[INTERIOR_FEATURE_COLUMNS].values))[:, 1]


def build_aux_features(df_sub, pooled_scaler, pooled_clf, dist_scaler, bri_scaler):
    ml = pooled_proba(df_sub, pooled_scaler, pooled_clf)
    dist_z = dist_scaler.transform(df_sub[["MeanDistToCrack"]].values).ravel()
    bri_z = bri_scaler.transform(df_sub[["MeanFlatBrightness"]].values).ravel()
    return np.column_stack([ml, dist_z, bri_z])


def hard_and_rule_leg_margins(row, dist_thr, bri_thr, floor, ml_proba):
    """For qualitative comparison: how far (in the cutoff's own units, then
    normalized) each of the 3 legs is from failing, for a single row."""
    return {
        "ml_margin": ml_proba - floor,                       # >=0 means leg satisfied
        "dist_margin": dist_thr - row["MeanDistToCrack"],     # >=0 means leg satisfied
        "bri_margin": bri_thr - row["MeanFlatBrightness"],    # >=0 means leg satisfied
    }


def main():
    df_labeled = load_labeled_interior()
    pool = load_full_interior_fill_pool()
    interior_lab = df_labeled[df_labeled["CandidateType"] == "interior_fill"].reset_index(drop=True)
    y_interior = interior_lab["IsCrack"].astype(bool).values
    pos_mask, neg_mask = y_interior, ~y_interior
    print(f"interior_fill labeled: {pos_mask.sum()} pos / {neg_mask.sum()} neg")
    print(f"full interior_fill pool: {len(pool)} rows")

    # ---- Production pooled model (fit on ALL labeled candidates, all types) ----
    pooled_scaler, pooled_clf = fit_pooled(df_labeled)

    # ---- Baseline hard-AND rule numbers, recomputed here for apples-to-apples ----
    # (mirrors calibrate_interior_fill_rule in train_interior_model.py, but we
    # just need the winning config's numbers, taken from the production run)
    BASE_FLOOR = 0.60
    BASE_DIST_THR = float(np.percentile(pool["MeanDistToCrack"].values, 70))
    BASE_BRI_THR = float(np.percentile(pool["MeanFlatBrightness"].values, 90))
    print(f"\nBaseline hard-AND rule (production): ML>={BASE_FLOOR}, "
          f"MeanDistToCrack<={BASE_DIST_THR:.2f} (70th pct), "
          f"MeanFlatBrightness<={BASE_BRI_THR:.2f} (90th pct)")

    full_fit_ml_interior = pooled_proba(interior_lab, pooled_scaler, pooled_clf)
    base_pos_ok = (interior_lab["MeanDistToCrack"].values[pos_mask] <= BASE_DIST_THR) & \
                  (interior_lab["MeanFlatBrightness"].values[pos_mask] <= BASE_BRI_THR) & \
                  (full_fit_ml_interior[pos_mask] >= BASE_FLOOR)
    base_recall = base_pos_ok.mean()
    full_fit_ml_pool = pooled_proba(pool, pooled_scaler, pooled_clf)
    base_pool_accept = ((pool["MeanDistToCrack"].values <= BASE_DIST_THR) &
                         (pool["MeanFlatBrightness"].values <= BASE_BRI_THR) &
                         (full_fit_ml_pool >= BASE_FLOOR)).mean()
    print(f"  recomputed recall on positives: {base_recall*100:.1f}%  "
          f"(expected ~74.4%), full pool acceptance: {base_pool_accept*100:.1f}% (expected ~57.6%)")

    # ---- Auxiliary smooth-score model ----
    dist_scaler = StandardScaler().fit(interior_lab[["MeanDistToCrack"]].values)
    bri_scaler = StandardScaler().fit(interior_lab[["MeanFlatBrightness"]].values)
    X_aux = build_aux_features(interior_lab, pooled_scaler, pooled_clf, dist_scaler, bri_scaler)
    y_aux = y_interior

    neg_rows = interior_lab[neg_mask].reset_index(drop=True)

    def refit_excluding_negative(neg_row):
        """Refit BOTH the pooled model and the auxiliary model excluding the
        one negative row (matched by SourceImage+Label), exactly like
        production's LOO check. Returns everything needed to score that row."""
        keep = ~((df_labeled["SourceImage"] == neg_row["SourceImage"].values[0]) &
                 (df_labeled["Label"].values == neg_row["Label"].values[0]))
        df_loo_all = df_labeled.loc[keep].reset_index(drop=True)
        pooled_scaler_loo, pooled_clf_loo = fit_pooled(df_loo_all)

        interior_loo = df_loo_all[df_loo_all["CandidateType"] == "interior_fill"].reset_index(drop=True)
        y_loo = interior_loo["IsCrack"].astype(bool).values
        dist_scaler_loo = StandardScaler().fit(interior_loo[["MeanDistToCrack"]].values)
        bri_scaler_loo = StandardScaler().fit(interior_loo[["MeanFlatBrightness"]].values)
        X_loo = build_aux_features(interior_loo, pooled_scaler_loo, pooled_clf_loo, dist_scaler_loo, bri_scaler_loo)
        return pooled_scaler_loo, pooled_clf_loo, dist_scaler_loo, bri_scaler_loo, X_loo, y_loo

    grid_results = []
    C_GRID = [0.01, 0.05, 0.1, 0.5]
    THRESH_GRID = np.arange(0.05, 0.96, 0.01)

    for C in C_GRID:
        # Full-fit auxiliary model (for recall/pool-acceptance reporting)
        aux_clf_full = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X_aux, y_aux)
        full_fit_score = aux_clf_full.predict_proba(X_aux)[:, 1]

        # LOO refit for each known negative -- refit pooled model AND aux model
        # excluding that negative, score the held-out negative under both.
        loo_score_neg = np.zeros(len(neg_rows))
        loo_details = []
        for i in range(len(neg_rows)):
            neg_row = neg_rows.iloc[[i]]
            (ps_loo, pc_loo, ds_loo, bs_loo, X_loo, y_loo) = refit_excluding_negative(neg_row)
            aux_clf_loo = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X_loo, y_loo)
            X_negrow = build_aux_features(neg_row, ps_loo, pc_loo, ds_loo, bs_loo)
            loo_score_neg[i] = aux_clf_loo.predict_proba(X_negrow)[0, 1]
            loo_details.append({
                "SourceImage": neg_row["SourceImage"].values[0],
                "Label": neg_row["Label"].values[0],
                "loo_score": loo_score_neg[i],
                "coef": aux_clf_loo.coef_[0].copy(),
                "intercept": aux_clf_loo.intercept_[0],
            })

        # Full-pool acceptance needs ML_proba/dist/bri for the WHOLE pool under
        # the full-fit pooled+aux models (production analog: full-fit, not LOO,
        # for the pool-acceptance number -- matches how base_pool_accept above
        # was computed with the full-fit pooled model too).
        X_pool_aux = build_aux_features(pool, pooled_scaler, pooled_clf, dist_scaler, bri_scaler)
        full_pool_score = aux_clf_full.predict_proba(X_pool_aux)[:, 1]

        for thresh in THRESH_GRID:
            if (loo_score_neg >= thresh).any():
                continue  # would accept a known negative under LOO -- reject
            recall = (full_fit_score[pos_mask] >= thresh).mean()
            pool_accept = (full_pool_score >= thresh).mean()
            grid_results.append({
                "C": C, "thresh": round(float(thresh), 2),
                "recall": recall, "pool_accept": pool_accept,
                "loo_score_neg": loo_score_neg.copy(),
                "loo_details": loo_details,
            })

    if not grid_results:
        print("\nNo (C, threshold) combination passed the LOO check for the auxiliary "
              "smooth-score model -- it cannot reject both known negatives under refit "
              "at any threshold tried.")
        best = None
    else:
        # Prefer highest recall, then lowest pool acceptance (tightest), then
        # lower C is preferred as a tie-break (more regularized = safer)
        grid_results.sort(key=lambda r: (-r["recall"], r["pool_accept"], r["C"]))
        best = grid_results[0]
        print(f"\nBest LOO-passing auxiliary-model config: C={best['C']}, threshold={best['thresh']}")
        print(f"  recall on known positives: {best['recall']*100:.1f}%  "
              f"(baseline hard-AND: {base_recall*100:.1f}%)")
        print(f"  full interior_fill pool acceptance: {best['pool_accept']*100:.1f}%  "
              f"(baseline hard-AND: {base_pool_accept*100:.1f}%)")
        print(f"  LOO scores of the 2 known negatives (must be < {best['thresh']}): "
              f"{np.round(best['loo_score_neg'], 4)}")

        print("\n  Per-negative LOO margin under the auxiliary model (threshold - loo_score, "
              "bigger = more safety margin):")
        for d in best["loo_details"]:
            margin = best["thresh"] - d["loo_score"]
            print(f"    {d['SourceImage']} / Label={d['Label']}: loo_score={d['loo_score']:.4f}, "
                  f"margin={margin:.4f}, aux_coefs(ML,dist_z,bri_z)={np.round(d['coef'],3)}, "
                  f"intercept={d['intercept']:.3f}")

    # ---- Also show ALL LOO-passing configs briefly for context ----
    print(f"\nTotal (C,threshold) grid points tried: {len(C_GRID)*len(THRESH_GRID)}, "
          f"of which {len(grid_results)} passed LOO on both negatives.")
    if grid_results:
        # show a small table of the top 5 distinct (C) best recall configs
        seen_C = {}
        for r in sorted(grid_results, key=lambda r: (-r["recall"], r["pool_accept"])):
            if r["C"] not in seen_C:
                seen_C[r["C"]] = r
        print("\n  Best config per C (by recall, ties broken by lower pool_accept):")
        for C in C_GRID:
            if C in seen_C:
                r = seen_C[C]
                print(f"    C={C}: thresh={r['thresh']}, recall={r['recall']*100:.1f}%, "
                      f"pool_accept={r['pool_accept']*100:.1f}%")
            else:
                print(f"    C={C}: no threshold passed LOO")

    # ---- Qualitative margin comparison vs hard-AND rule's per-leg margins ----
    print("\n--- Qualitative margin comparison: hard-AND rule's per-leg status for the 2 known "
          "negatives (full-fit pooled ML, NOT LOO -- matches how the hard-AND grid search itself "
          "used LOO only for the floor-vs-loo_proba check, and in-sample dist/bri which are fixed "
          "quantities not model outputs) ---")
    for i in range(len(neg_rows)):
        row = neg_rows.iloc[i]
        ml_p = full_fit_ml_interior[neg_mask][i]
        margins = hard_and_rule_leg_margins(row, BASE_DIST_THR, BASE_BRI_THR, BASE_FLOOR, ml_p)
        print(f"  {row['SourceImage']} / Label={row['Label']}: "
              f"MeanDistToCrack={row['MeanDistToCrack']:.2f} (thr {BASE_DIST_THR:.2f}, "
              f"margin={margins['dist_margin']:.2f}), "
              f"MeanFlatBrightness={row['MeanFlatBrightness']:.2f} (thr {BASE_BRI_THR:.2f}, "
              f"margin={margins['bri_margin']:.2f}), "
              f"ML_proba(full-fit)={ml_p:.3f} (floor {BASE_FLOOR}, margin={margins['ml_margin']:.3f}) "
              f"-> rejected via {'dist' if margins['dist_margin']<0 else ('bri' if margins['bri_margin']<0 else ('ml' if margins['ml_margin']<0 else 'NONE-would-be-accepted'))} leg")

    # ---- Concavity / bridge_corridor sanity check: unaffected by this experiment,
    # since the auxiliary model only ever gates interior_fill decisions; the
    # pooled model used for concavity/bridge_corridor's plain 0.5 threshold is
    # retrained the same way in both baseline and this refinement (all labeled
    # data pooled), so its acceptance behavior is unchanged by construction. We
    # verify this explicitly below by comparing pooled-model predictions on
    # concavity/bridge_corridor before vs after adding the auxiliary model
    # concept (there IS no "after" -- the pooled model itself is untouched).
    other = df_labeled[df_labeled["CandidateType"].isin(["concavity", "bridge_corridor"])]
    if len(other):
        other_ml = pooled_proba(other, pooled_scaler, pooled_clf)
        accept_rate_05 = (other_ml >= 0.5).mean()
        print(f"\nconcavity+bridge_corridor sanity check: n={len(other)}, "
              f"plain 0.5-threshold acceptance = {accept_rate_05*100:.1f}% "
              f"(pooled model itself is not modified by this experiment, so this is "
              f"identical before/after by construction)")

    print("\n=== SUMMARY ===")
    print(f"Baseline hard-AND:      recall={base_recall*100:.1f}%  pool_accept={base_pool_accept*100:.1f}%")
    if best:
        print(f"Soft-score alternative: recall={best['recall']*100:.1f}%  pool_accept={best['pool_accept']*100:.1f}%  "
              f"(C={best['C']}, thresh={best['thresh']})")
    else:
        print("Soft-score alternative: NO CONFIG PASSED LOO -- rejected as an approach.")


if __name__ == "__main__":
    main()
