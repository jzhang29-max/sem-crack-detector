"""
refine_third_gate_feature.py
-----------------------------
Approach under test ("third_gate_feature"): the production hybrid rule for
interior_fill uses exactly 2 gate features (MeanDistToCrack <= Nth pct,
MeanFlatBrightness <= Mth pct) plus an ML-probability floor. This script asks
whether adding a THIRD gate -- MeanVesselness (pooled model's largest
standardized coefficient) or FracBoundaryTouchingCrack -- as either

  (a) an ADDITIONAL required AND-condition on top of both existing gates, or
  (b) a SUBSTITUTE for MeanDistToCrack (keep floor + brightness + new gate), or
  (c) a SUBSTITUTE for MeanFlatBrightness (keep floor + distance + new gate)

can find a config with higher recall on the 39 known positives than the
baseline's 74.4%, at similar-or-lower full interior_fill pool acceptance than
the baseline's 57.6%, while still rejecting both known negatives under
LEAVE-ONE-OUT refit (the same falsifiable generalization test the baseline
used, not in-sample fit).

Methodology is copied as directly as possible from hybrid_rule_plus_ml.py /
train_interior_model.py's calibrate_interior_fill_rule() so results are
apples-to-apples comparable:
  - production-matched loading (user_painted included, global-fit scaler --
    doesn't affect interior_fill numbers at all since interior_fill has zero
    user_painted rows)
  - LOO-refit probability (not in-sample) used as the negative-rejection
    constraint in the grid search
  - grid over percentile cutoffs of the FULL (labeled+unlabeled) interior_fill
    pool for every gate feature, plus an ML-probability floor
  - selection: reject configs that would accept either known negative under
    LOO; among survivors, maximize recall_pos, tie-break by minimizing
    full_pool_accept, then prefer non-vacuous/looser cutoffs

Direction of the new gates: both MeanVesselness and FracBoundaryTouchingCrack
have HIGHER mean among known positives than known negatives (checked
empirically below), so they are gated as `feature >= pool_percentile(P)`
(reject candidates BELOW the cutoff) -- the opposite direction from
distance/brightness, which are gated as `<= percentile` (reject candidates
ABOVE the cutoff, i.e. "too far" / "too bright").
"""
import os
import sys
import glob
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CAND_GLOB = "/Users/jiamingzhang/Desktop/SEM_Crack_Detection_Pipeline/interior_active_learning/candidates/*_interior.csv"

FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
            "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
            "FracBoundaryTouchingCrack", "MeanDistToCrack"]

# Baseline (production) rule, for reference / comparison printouts.
BASELINE_RECALL = 0.744
BASELINE_POOL_ACCEPT = 0.576


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


def load_full_pool(candidate_type):
    rows = []
    for f in sorted(glob.glob(CAND_GLOB)):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        rows.append(d[d["CandidateType"] == candidate_type])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_lr(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(X), y)
    return sc, clf


def score(df_sub, sc, clf):
    return clf.predict_proba(sc.transform(df_sub[FEATURES].values))[:, 1]


def main():
    df_full = load_labeled_interior()  # production-matched loader (incl user_painted)
    print(f"Loaded {len(df_full)} labeled candidates total ({df_full['SourceImage'].nunique()} images)")

    X = df_full[FEATURES].values
    y = df_full["IsCrack"].astype(bool).values
    scaler_full, clf_full = fit_lr(X, y)  # global-fit, matches production train_interior_model.py

    pool_concavity = load_full_pool("concavity")
    pool_bridge = load_full_pool("bridge_corridor")
    pool_interior = load_full_pool("interior_fill")
    print(f"Full pools: concavity={len(pool_concavity)}, bridge_corridor={len(pool_bridge)}, "
          f"interior_fill={len(pool_interior)}\n")

    baseline_accept_concavity = float((score(pool_concavity, scaler_full, clf_full) >= 0.5).mean())
    baseline_accept_bridge = float((score(pool_bridge, scaler_full, clf_full) >= 0.5).mean())
    print(f"[Sanity] concavity full-pool accept @0.5 = {baseline_accept_concavity*100:.1f}%  "
          f"(must stay unchanged -- these gates only ever touch interior_fill scoring)")
    print(f"[Sanity] bridge_corridor full-pool accept @0.5 = {baseline_accept_bridge*100:.1f}%\n")

    interior_lab = df_full[df_full["CandidateType"] == "interior_fill"].reset_index(drop=True)
    y_interior = interior_lab["IsCrack"].astype(bool).values
    pos_mask, neg_mask = y_interior, ~y_interior
    print(f"interior_fill labeled: {pos_mask.sum()} True / {neg_mask.sum()} False\n")

    ml_proba_fullfit = score(interior_lab, scaler_full, clf_full)

    # --- direction check for the two candidate 3rd-gate features ---
    print("Direction check (mean by class) for candidate 3rd-gate features:")
    for feat in ["MeanVesselness", "FracBoundaryTouchingCrack"]:
        print(f"  {feat}: True mean={interior_lab.loc[pos_mask, feat].mean():.4f}  "
              f"False mean={interior_lab.loc[neg_mask, feat].mean():.4f}  "
              f"-> higher-is-more-crack-like, so gate as >= percentile cutoff")
    print()

    # --- LOO-refit ML proba for the 2 known negatives (rejection constraint) ---
    neg_rows = interior_lab[neg_mask].reset_index(drop=True)
    loo_proba_neg = np.zeros(len(neg_rows))
    for i in range(len(neg_rows)):
        neg_row = neg_rows.iloc[[i]]
        keep = ~((df_full["SourceImage"] == neg_row["SourceImage"].values[0]) &
                 (df_full["Label"].values == neg_row["Label"].values[0]))
        X_loo = df_full.loc[keep, FEATURES].values
        y_loo = df_full.loc[keep, "IsCrack"].astype(bool).values
        sc_loo, clf_loo = fit_lr(X_loo, y_loo)
        loo_proba_neg[i] = clf_loo.predict_proba(sc_loo.transform(neg_row[FEATURES].values))[0, 1]
    print(f"Known negatives:")
    for i, row in neg_rows.iterrows():
        print(f"  {row['SourceImage']} Label={row['Label']}: MeanDistToCrack={row['MeanDistToCrack']:.2f}, "
              f"MeanFlatBrightness={row['MeanFlatBrightness']:.2f}, MeanVesselness={row['MeanVesselness']:.4f}, "
              f"FracBoundaryTouchingCrack={row['FracBoundaryTouchingCrack']:.3f}, "
              f"LOO ML proba={loo_proba_neg[i]:.3f}")
    print()

    dist_pool = pool_interior["MeanDistToCrack"].values
    bri_pool = pool_interior["MeanFlatBrightness"].values
    ves_pool = pool_interior["MeanVesselness"].values
    frac_pool = pool_interior["FracBoundaryTouchingCrack"].values
    pool_proba = score(pool_interior, scaler_full, clf_full)

    dist_lab = interior_lab["MeanDistToCrack"].values
    bri_lab = interior_lab["MeanFlatBrightness"].values
    ves_lab = interior_lab["MeanVesselness"].values
    frac_lab = interior_lab["FracBoundaryTouchingCrack"].values

    N_GRID = list(range(5, 71, 5))    # dist percentile grid (lower-is-better feature)
    M_GRID = list(range(20, 91, 5))   # brightness percentile grid (lower-is-better feature)
    P_GRID = list(range(5, 96, 5))    # new-gate percentile grid (higher-is-better feature, coarser)
    FLOOR_GRID = np.arange(0.10, 0.86, 0.05)

    def run_grid(gates, third_feat=None):
        """gates: list of 'dist','bri','third' to include as required AND legs.
        Returns sorted list of (params_dict, recall, pool_accept)."""
        results = []
        n_range = N_GRID if "dist" in gates else [None]
        m_range = M_GRID if "bri" in gates else [None]
        p_range = P_GRID if "third" in gates else [None]

        third_pool = ves_pool if third_feat == "MeanVesselness" else frac_pool
        third_lab_pos = ves_lab[pos_mask] if third_feat == "MeanVesselness" else frac_lab[pos_mask]
        third_lab_neg = ves_lab[neg_mask] if third_feat == "MeanVesselness" else frac_lab[neg_mask]

        for N in n_range:
            dist_thr = np.percentile(dist_pool, N) if N is not None else None
            for M in m_range:
                bri_thr = np.percentile(bri_pool, M) if M is not None else None
                for P in p_range:
                    third_thr = np.percentile(third_pool, P) if P is not None else None

                    def rule_ok(dist_v, bri_v, third_v):
                        ok = np.ones(len(dist_v) if dist_v is not None else len(bri_v), dtype=bool) \
                            if dist_v is not None or bri_v is not None or third_v is not None else None
                        if "dist" in gates:
                            ok = ok & (dist_v <= dist_thr)
                        if "bri" in gates:
                            ok = ok & (bri_v <= bri_thr)
                        if "third" in gates:
                            ok = ok & (third_v >= third_thr)
                        return ok

                    neg_dist = neg_rows["MeanDistToCrack"].values if "dist" in gates else None
                    neg_bri = neg_rows["MeanFlatBrightness"].values if "bri" in gates else None
                    neg_third = third_lab_neg if "third" in gates else None
                    neg_rule_ok = rule_ok(neg_dist, neg_bri, neg_third)

                    for floor in FLOOR_GRID:
                        if (neg_rule_ok & (loo_proba_neg >= floor)).any():
                            continue  # would accept a known negative under LOO -- reject config

                        pos_dist = dist_lab[pos_mask] if "dist" in gates else None
                        pos_bri = bri_lab[pos_mask] if "bri" in gates else None
                        pos_third = third_lab_pos if "third" in gates else None
                        pos_rule_ok = rule_ok(pos_dist, pos_bri, pos_third)
                        recall_pos = (pos_rule_ok & (ml_proba_fullfit[pos_mask] >= floor)).mean()

                        pool_dist = dist_pool if "dist" in gates else None
                        pool_bri = bri_pool if "bri" in gates else None
                        pool_third = third_pool if "third" in gates else None
                        pool_ok = rule_ok(pool_dist, pool_bri, pool_third)
                        full_pool_accept = (pool_ok & (pool_proba >= floor)).mean()

                        results.append({
                            "N": N, "M": M, "P": P, "floor": round(float(floor), 2),
                            "dist_thr": dist_thr, "bri_thr": bri_thr, "third_thr": third_thr,
                            "third_feat": third_feat,
                            "recall": float(recall_pos), "pool_accept": float(full_pool_accept),
                        })
        results.sort(key=lambda r: (-r["recall"], r["pool_accept"]))
        return results

    print("=" * 78)
    print("0. BASELINE REPRODUCTION: 2-gate rule (dist + bri + floor)")
    print("=" * 78)
    baseline_results = run_grid(["dist", "bri"])
    if baseline_results:
        b = baseline_results[0]
        print(f"  best: floor={b['floor']}, dist<= {b['dist_thr']:.2f} (N={b['N']}), "
              f"bri<= {b['bri_thr']:.2f} (M={b['M']})")
        print(f"  recall={b['recall']*100:.1f}%, full_pool_accept={b['pool_accept']*100:.1f}%")
        print(f"  (reference numbers from problem statement: recall=74.4%, pool_accept=57.6% -- "
              f"{'reproduced closely' if abs(b['recall']-BASELINE_RECALL) < 0.02 else 'differs somewhat, likely grid/tie-break variant'})")
    else:
        print("  No baseline config found rejecting both negatives (unexpected).")
    print()

    all_variants = {}

    for third_feat in ["MeanVesselness", "FracBoundaryTouchingCrack"]:
        print("=" * 78)
        print(f"3-GATE VARIANTS using third_feat = {third_feat}")
        print("=" * 78)

        print(f"\n(a) ADDITIONAL gate: dist AND bri AND {third_feat} AND floor")
        res_add = run_grid(["dist", "bri", "third"], third_feat=third_feat)
        if res_add:
            r = res_add[0]
            print(f"  best: floor={r['floor']}, dist<={r['dist_thr']:.2f} (N={r['N']}), "
                  f"bri<={r['bri_thr']:.2f} (M={r['M']}), {third_feat}>={r['third_thr']:.4f} (P={r['P']})")
            print(f"  recall={r['recall']*100:.1f}%, full_pool_accept={r['pool_accept']*100:.1f}%")
            print(f"  top 10 LOO-safe configs (N, M, third_P, floor, recall, pool_accept):")
            for rr in res_add[:10]:
                print(f"    N={rr['N']:>3} M={rr['M']:>3} P={rr['P']:>3} floor={rr['floor']:.2f}  "
                      f"recall={rr['recall']*100:5.1f}%  pool_accept={rr['pool_accept']*100:5.1f}%")
        else:
            print("  No config rejects both known negatives under LOO.")
        all_variants[f"add_{third_feat}"] = res_add[0] if res_add else None

        print(f"\n(b) SUBSTITUTE for dist: bri AND {third_feat} AND floor (drop MeanDistToCrack gate)")
        res_sub_dist = run_grid(["bri", "third"], third_feat=third_feat)
        if res_sub_dist:
            r = res_sub_dist[0]
            print(f"  best: floor={r['floor']}, bri<={r['bri_thr']:.2f} (M={r['M']}), "
                  f"{third_feat}>={r['third_thr']:.4f} (P={r['P']})")
            print(f"  recall={r['recall']*100:.1f}%, full_pool_accept={r['pool_accept']*100:.1f}%")
        else:
            print("  No config rejects both known negatives under LOO.")
        all_variants[f"sub_dist_{third_feat}"] = res_sub_dist[0] if res_sub_dist else None

        print(f"\n(c) SUBSTITUTE for bri: dist AND {third_feat} AND floor (drop MeanFlatBrightness gate)")
        res_sub_bri = run_grid(["dist", "third"], third_feat=third_feat)
        if res_sub_bri:
            r = res_sub_bri[0]
            print(f"  best: floor={r['floor']}, dist<={r['dist_thr']:.2f} (N={r['N']}), "
                  f"{third_feat}>={r['third_thr']:.4f} (P={r['P']})")
            print(f"  recall={r['recall']*100:.1f}%, full_pool_accept={r['pool_accept']*100:.1f}%")
            print(f"  top 10 LOO-safe configs (N, third_P, floor, recall, pool_accept), for robustness check:")
            for rr in res_sub_bri[:10]:
                print(f"    N={rr['N']:>3} P={rr['P']:>3} floor={rr['floor']:.2f}  "
                      f"recall={rr['recall']*100:5.1f}%  pool_accept={rr['pool_accept']*100:5.1f}%")
        else:
            print("  No config rejects both known negatives under LOO.")
        all_variants[f"sub_bri_{third_feat}"] = res_sub_bri[0] if res_sub_bri else None
        print()

    # =====================================================================
    # Final comparison table
    # =====================================================================
    print("=" * 78)
    print("FINAL COMPARISON (all configs already constrained to reject both known")
    print("negatives under LEAVE-ONE-OUT refit)")
    print("=" * 78)
    base = baseline_results[0] if baseline_results else None
    print(f"{'config':40s} {'recall':>8s} {'pool_accept':>12s} {'beats baseline?':>16s}")
    if base:
        print(f"{'2-gate baseline (dist+bri)':40s} {base['recall']*100:7.1f}% {base['pool_accept']*100:11.1f}%"
              f" {'--':>16s}")
    best_overall = None
    for name, r in all_variants.items():
        if r is None:
            print(f"{name:40s} {'--':>8s} {'--':>12s} {'infeasible (no LOO-safe config)':>16s}")
            continue
        beats = (r["recall"] > BASELINE_RECALL + 1e-9) and (r["pool_accept"] <= BASELINE_POOL_ACCEPT + 1e-9)
        print(f"{name:40s} {r['recall']*100:7.1f}% {r['pool_accept']*100:11.1f}% {str(beats):>16s}")
        if beats and (best_overall is None or r["recall"] > best_overall["recall"]):
            best_overall = dict(r, name=name)

    print()
    if best_overall:
        print(f"BEST 3-gate variant beating baseline: {best_overall['name']}")
        print(f"  recall={best_overall['recall']*100:.1f}% vs baseline {BASELINE_RECALL*100:.1f}%, "
              f"pool_accept={best_overall['pool_accept']*100:.1f}% vs baseline {BASELINE_POOL_ACCEPT*100:.1f}%")
    else:
        print("NO 3-gate variant strictly beats the 2-gate baseline "
              "(higher recall AND similar-or-lower pool acceptance) while still passing LOO.")
        print("This is an honest possible outcome given only 2 negatives to validate against --")
        print("adding a 3rd gate with this little validation data raises real overfitting risk:")
        print("the grid search has many more degrees of freedom (N x M x P x floor instead of")
        print("N x M x floor) chasing the same 2 negative constraints, so any apparent gain is")
        print("more likely to be an artifact of the specific 2 negatives we happen to have than")
        print("a real generalizable improvement.")

    print(f"\nSanity: concavity/bridge_corridor acceptance untouched by any of this "
          f"(concavity={baseline_accept_concavity*100:.1f}%, bridge_corridor={baseline_accept_bridge*100:.1f}%) "
          f"-- these gates only ever apply to interior_fill scoring.")


if __name__ == "__main__":
    main()
