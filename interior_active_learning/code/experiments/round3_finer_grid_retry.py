"""
Round 3 -- finer_grid_retry

Redo of the finer-grid Pareto frontier search for calibrate_interior_fill_rule()
(see ../train_interior_model.py) that failed last round due to a technical issue
(no results produced). Same methodology, re-attempted cleanly, PLUS a per-Pareto-point
margin/robustness analysis that round 2 showed is essential (a config can look
Pareto-better on raw recall/acceptance numbers while actually resting on a much
thinner, untested rejection margin for one of the two known negatives).

Methodology (must exactly match production's calibrate_interior_fill_rule so the
grid search is a legitimate refinement, not a different rule):
  - Load ALL labeled interior candidates exactly like train_interior_model.py's
    load_labeled_interior() (every CandidateType, including user_painted -- the
    production pooled LogisticRegression is trained on this exact set; dropping
    user_painted here would silently change the classifier being calibrated).
  - Pooled StandardScaler + LogisticRegression(class_weight="balanced") on all
    labeled rows, all features (INTERIOR_FEATURE_COLUMNS).
  - Two known interior_fill negatives; LEAVE-ONE-OUT refit (excl. just that row)
    to get each negative's out-of-sample ML probability -- this is what the grid
    is constrained against, not in-sample probability.
  - Grid: MeanDistToCrack cutoff at percentile N of the FULL interior_fill pool,
    N in [5, 70] step 1 (production used step 5, same bounds); MeanFlatBrightness
    cutoff at percentile M of the same pool, M in [20, 90] step 1 (production
    step 5, same bounds); floor in [0.10, 0.85] step 0.01 (production step 0.05).
  - A config is valid iff it rejects BOTH known negatives (LOO probability check
    exactly as in production: reject if gate admits AND loo_proba >= floor).
  - Pareto frontier over (recall_on_positives higher-better, full_pool_acceptance
    lower-better) among all valid configs.
  - NEW this round: for each reported Pareto point, decompose *why* each negative
    is rejected (gate exclusion vs. floor exclusion) and report the margin on
    whichever leg is doing the rejecting:
      - floor-caught: floor - LOO_proba (absolute probability units)
      - gate-caught (dist and/or bri): (cutoff - actual_value) / cutoff (relative,
        negative because actual_value > cutoff when caught)
    Flag a point RISKY if the tightest active margin is thin in relative terms
    (heuristic: |relative margin| < 15%, matching the round-2 finding that a 7%
    margin was judged unsafe and a >>15% margin was judged robust).
"""
import os
import sys
import glob
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
from common import CANDIDATES_DIR
from active_learning_select import INTERIOR_FEATURE_COLUMNS

RISKY_REL_MARGIN = 0.15  # flag threshold, see docstring


def load_labeled_interior():
    """EXACTLY train_interior_model.py's loader -- all CandidateType values,
    including user_painted, since that is what the production pooled classifier
    is trained on."""
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
    return pd.concat(rows, ignore_index=True)


def load_full_interior_fill_pool():
    rows = []
    for f in sorted(glob.glob(os.path.join(CANDIDATES_DIR, "*_interior.csv"))):
        d = pd.read_csv(f)
        if len(d):
            rows.append(d[d["CandidateType"] == "interior_fill"])
    return pd.concat(rows, ignore_index=True)


def score(sub_df, sc, model):
    return model.predict_proba(sc.transform(sub_df[INTERIOR_FEATURE_COLUMNS].values))[:, 1]


def main():
    t0 = time.time()
    df = load_labeled_interior()
    n_pos, n_neg = int(df["IsCrack"].sum()), int((~df["IsCrack"]).sum())
    print(f"Loaded {len(df)} labeled candidates ({n_pos} True, {n_neg} False) across "
          f"{df['SourceImage'].nunique()} images (production-matched loader, user_painted included)")
    print(df["CandidateType"].value_counts().to_string())

    # ---- pooled model (identical recipe to production) ----
    X_all = df[INTERIOR_FEATURE_COLUMNS].values
    y_all = df["IsCrack"].astype(bool).values
    scaler = StandardScaler().fit(X_all)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X_all), y_all)

    # ---- sanity: concavity / bridge_corridor own behavior must stay ~unchanged.
    # The interior_fill rule ONLY gates interior_fill candidates -- it can't touch
    # these regardless of what this script finds -- but report their plain-ML
    # acceptance for the record, exactly as before this round's changes. ----
    print("\n[Sanity] concavity / bridge_corridor acceptance is untouched by this rule "
          "(rule only ever gates interior_fill); reported for reference only:")
    for t in ["concavity", "bridge_corridor"]:
        sub = df[df["CandidateType"] == t]
        if len(sub) == 0:
            continue
        p = score(sub, scaler, clf)
        acc = (p >= 0.5).mean()
        print(f"  {t}: n={len(sub)}, plain-0.5 ML acceptance={acc*100:.1f}%")

    interior_lab = df[df["CandidateType"] == "interior_fill"].reset_index(drop=True)
    y_interior = interior_lab["IsCrack"].astype(bool).values
    pos_mask, neg_mask = y_interior, ~y_interior
    print(f"\ninterior_fill labeled: {pos_mask.sum()} True, {neg_mask.sum()} False")
    assert neg_mask.sum() == 2, "expected exactly 2 known negatives per task context"

    pool = load_full_interior_fill_pool()
    dist_pool = pool["MeanDistToCrack"].values
    bri_pool = pool["MeanFlatBrightness"].values
    print(f"full interior_fill pool: {len(pool)} candidates (labeled + unlabeled)")

    full_fit_proba = score(interior_lab, scaler, clf)
    pool_proba = score(pool, scaler, clf)

    neg_rows = interior_lab[neg_mask].reset_index(drop=True)
    print("\nKnown negatives (raw feature values):")
    for i in range(len(neg_rows)):
        r = neg_rows.iloc[i]
        print(f"  #{i}: SourceImage={r['SourceImage']} Label={r['Label']} "
              f"MeanDistToCrack={r['MeanDistToCrack']:.3f}  MeanFlatBrightness={r['MeanFlatBrightness']:.3f}")

    # ---- LOO probability for each known negative ----
    loo_proba_neg = np.zeros(len(neg_rows))
    for i in range(len(neg_rows)):
        neg_row = neg_rows.iloc[[i]]
        keep = ~((df["SourceImage"] == neg_row["SourceImage"].values[0]) &
                 (df["Label"].values == neg_row["Label"].values[0]))
        X_loo = df.loc[keep, INTERIOR_FEATURE_COLUMNS].values
        y_loo = df.loc[keep, "IsCrack"].astype(bool).values
        sc_loo = StandardScaler().fit(X_loo)
        clf_loo = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc_loo.transform(X_loo), y_loo)
        loo_proba_neg[i] = clf_loo.predict_proba(sc_loo.transform(neg_row[INTERIOR_FEATURE_COLUMNS].values))[0, 1]
        print(f"  LOO-refit ML_proba for negative #{i}: {loo_proba_neg[i]:.4f}")

    neg_dist = neg_rows["MeanDistToCrack"].values
    neg_bri = neg_rows["MeanFlatBrightness"].values

    # ---- FINER grid: percentile step 1, floor step 0.01, SAME bounds as production ----
    N_range = list(range(5, 71, 1))       # dist percentile, production: range(5,71,5)
    M_range = list(range(20, 91, 1))      # bri percentile,  production: range(20,91,5)
    floor_range = np.round(np.arange(0.10, 0.86, 0.01), 2)  # production: arange(0.10,0.86,0.05)

    total = len(N_range) * len(M_range) * len(floor_range)
    print(f"\nGrid size: {len(N_range)} x {len(M_range)} x {len(floor_range)} = {total} configs")

    dist_thr_by_N = {N: np.percentile(dist_pool, N) for N in N_range}
    bri_thr_by_M = {M: np.percentile(bri_pool, M) for M in M_range}

    grid = []
    for N in N_range:
        dist_thr = dist_thr_by_N[N]
        neg_dist_ok = neg_dist <= dist_thr
        pos_dist_ok = interior_lab["MeanDistToCrack"].values[pos_mask] <= dist_thr
        pool_dist_ok = dist_pool <= dist_thr
        for M in M_range:
            bri_thr = bri_thr_by_M[M]
            neg_gate_ok = neg_dist_ok & (neg_bri <= bri_thr)          # gate would ADMIT
            pos_rule_ok = pos_dist_ok & (interior_lab["MeanFlatBrightness"].values[pos_mask] <= bri_thr)
            pool_rule_ok = pool_dist_ok & (bri_pool <= bri_thr)
            for floor in floor_range:
                if (neg_gate_ok & (loo_proba_neg >= floor)).any():
                    continue  # would admit a known negative -- invalid
                recall_pos = (pos_rule_ok & (full_fit_proba[pos_mask] >= floor)).mean()
                full_pool_accept = (pool_rule_ok & (pool_proba >= floor)).mean()
                grid.append((N, M, round(float(floor), 2), float(recall_pos), float(full_pool_accept)))

    print(f"Valid (LOO-passing) configs: {len(grid)} / {total}")
    if not grid:
        print("No valid config found.")
        return

    grid_df = pd.DataFrame(grid, columns=["N_pct", "M_pct", "floor", "recall", "pool_accept"])

    # ---- Pareto frontier: higher recall better, lower pool_accept better ----
    grid_sorted = grid_df.sort_values(
        ["recall", "pool_accept", "N_pct", "M_pct", "floor"],
        ascending=[False, True, False, False, True]
    ).reset_index(drop=True)
    pareto_rows = []
    best_accept_so_far = np.inf
    for _, row in grid_sorted.iterrows():
        if row["pool_accept"] < best_accept_so_far - 1e-12:
            pareto_rows.append(row)
            best_accept_so_far = row["pool_accept"]
    pareto_df = pd.DataFrame(pareto_rows).reset_index(drop=True)
    pareto_df = pareto_df.sort_values("recall").reset_index(drop=True)
    print(f"\nFull Pareto frontier: {len(pareto_df)} distinct (recall, pool_accept) points")

    # ---- margin/robustness decomposition for every Pareto point ----
    def margin_report(N, M, floor):
        dist_thr = dist_thr_by_N[N]
        bri_thr = bri_thr_by_M[M]
        legs = []
        for i in range(len(neg_rows)):
            dist_fail = neg_dist[i] > dist_thr
            bri_fail = neg_bri[i] > bri_thr
            gate_caught = dist_fail or bri_fail
            if gate_caught:
                rel_margins = []
                if dist_fail:
                    rel_margins.append(("dist", (dist_thr - neg_dist[i]) / dist_thr))
                if bri_fail:
                    rel_margins.append(("bri", (bri_thr - neg_bri[i]) / bri_thr))
                # tightest (smallest abs) relative margin among active gate legs
                leg_name, rel_margin = min(rel_margins, key=lambda t: abs(t[1]))
                legs.append({
                    "neg_idx": i, "mechanism": f"gate:{leg_name}",
                    "value": rel_margin, "kind": "relative",
                })
            else:
                # must be floor-caught (else config would've been invalid)
                floor_margin = floor - loo_proba_neg[i]
                floor_rel = floor_margin / floor if floor else np.nan
                legs.append({
                    "neg_idx": i, "mechanism": "floor",
                    "value": floor_rel, "kind": "relative(floor)",
                    "abs_value": floor_margin,
                })
        tightest = min(legs, key=lambda l: abs(l["value"]))
        risky = abs(tightest["value"]) < RISKY_REL_MARGIN
        return legs, tightest, risky

    print("\n=== Margin decomposition for every Pareto-optimal point ===")
    detail_rows = []
    for _, row in pareto_df.iterrows():
        N, M, floor = int(row["N_pct"]), int(row["M_pct"]), row["floor"]
        legs, tightest, risky = margin_report(N, M, floor)
        leg_strs = []
        for l in legs:
            if l["mechanism"] == "floor":
                leg_strs.append(f"neg#{l['neg_idx']}:floor(margin={l['abs_value']:+.3f} abs, "
                                 f"{l['value']*100:+.1f}% rel)")
            else:
                leg_strs.append(f"neg#{l['neg_idx']}:{l['mechanism']}(rel_margin={l['value']*100:+.1f}%)")
        flag = " <-- RISKY (thin margin)" if risky else ""
        print(f"  N={N:2d}pct M={M:2d}pct floor={floor:.2f} | recall={row['recall']*100:5.1f}% "
              f"accept={row['pool_accept']*100:5.1f}% | {'; '.join(leg_strs)}{flag}")
        detail_rows.append({
            "N_pct": N, "M_pct": M, "floor": floor,
            "dist_thr": dist_thr_by_N[N], "bri_thr": bri_thr_by_M[M],
            "recall": row["recall"], "pool_accept": row["pool_accept"],
            "tightest_mechanism": tightest["mechanism"],
            "tightest_rel_margin_pct": tightest["value"] * 100,
            "risky": risky,
        })
    detail_df = pd.DataFrame(detail_rows)

    # ---- 8-10 representative points spanning the frontier, for the table ----
    print("\n=== 8-10 representative Pareto points (spanning the frontier) ===")
    n_rep = min(10, len(pareto_df))
    idx = sorted(set(np.linspace(0, len(pareto_df) - 1, n_rep).round().astype(int)))
    rep = detail_df.iloc[idx]
    cols = ["N_pct", "M_pct", "floor", "dist_thr", "bri_thr", "recall", "pool_accept",
            "tightest_mechanism", "tightest_rel_margin_pct", "risky"]
    print(rep[cols].to_string(index=False))

    # ---- comparison to current production point ----
    prod_recall, prod_accept = 0.744, 0.576
    print(f"\n=== Reference: current production point recall=74.4%, pool_accept=57.6% "
          f"(floor=0.60, dist_thr=12.90px [70pct], bri_thr=135.84 [90pct]) ===")
    prod_legs, prod_tightest, prod_risky = margin_report(70, 90, 0.60)
    print("Production point's own margin decomposition (for direct comparison):")
    for l in prod_legs:
        if l["mechanism"] == "floor":
            print(f"  neg#{l['neg_idx']}: floor margin = {l['abs_value']:+.3f} abs ({l['value']*100:+.1f}% rel)")
        else:
            print(f"  neg#{l['neg_idx']}: {l['mechanism']} rel margin = {l['value']*100:+.1f}%")
    print(f"  tightest leg: {prod_tightest['mechanism']} ({prod_tightest['value']*100:+.1f}%) "
          f"-> {'RISKY' if prod_risky else 'not flagged risky by this heuristic'}")

    dominates = pareto_df[
        (pareto_df["recall"] >= prod_recall - 1e-9) & (pareto_df["pool_accept"] <= prod_accept + 1e-9) &
        ((pareto_df["recall"] > prod_recall + 1e-9) | (pareto_df["pool_accept"] < prod_accept - 1e-9))
    ]
    print(f"\nConfigs strictly dominating production on raw numbers: {len(dominates)}")
    if len(dominates):
        dom_detail = detail_df.merge(dominates[["N_pct", "M_pct", "floor"]], on=["N_pct", "M_pct", "floor"])
        print(dom_detail[cols].to_string(index=False))
        n_dom_safe = (~dom_detail["risky"]).sum()
        print(f"\nOf those, {n_dom_safe}/{len(dom_detail)} are NOT flagged risky "
              f"(tightest relative margin >= {RISKY_REL_MARGIN*100:.0f}%).")

    print(f"\nDone in {time.time()-t0:.1f}s")
    grid_df.to_csv(os.path.join(os.path.dirname(__file__), "round3_finer_grid_full_grid.csv"), index=False)
    detail_df.to_csv(os.path.join(os.path.dirname(__file__), "round3_finer_grid_pareto_with_margins.csv"), index=False)
    print("Saved full grid + Pareto-with-margins CSVs alongside this script.")


if __name__ == "__main__":
    main()
