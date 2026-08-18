"""
Experiment: finer_grid_pareto

Refinement of calibrate_interior_fill_rule() (see ../train_interior_model.py) that:
  1. Redoes the exact same grid search + leave-one-out (LOO) rejection methodology
     on the 2 known interior_fill negatives (refit excluding each negative, score
     it out-of-sample, require it stay below the floor whenever the dist/brightness
     gate would otherwise admit it).
  2. Uses a much finer grid: percentile step 1 (vs 5) for both the distance and
     brightness percentile thresholds N, M in [5, 95], and floor step 0.01 (vs 0.05)
     over [0.10, 0.95].
  3. Instead of collapsing to a single "best" point via a fixed tie-break, computes
     the full Pareto frontier over (recall_on_positives, full_pool_acceptance) among
     ALL valid (N, M, floor) configs that still reject both known negatives under LOO
     -- "valid" meaning higher recall AND lower acceptance is strictly better, so a
     config is Pareto-optimal iff no other valid config dominates it on both axes.

Run: python3 refine_finer_grid_pareto.py
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
          f"{df['SourceImage'].nunique()} images")

    # ---- pooled model (same as production: pooled LR + StandardScaler, all types) ----
    X_all = df[INTERIOR_FEATURE_COLUMNS].values
    y_all = df["IsCrack"].astype(bool).values
    scaler = StandardScaler().fit(X_all)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X_all), y_all)

    # ---- sanity check: concavity / bridge_corridor acceptance at plain 0.5, unaffected by
    # this rule since the rule ONLY gates interior_fill ----
    for t in ["concavity", "bridge_corridor"]:
        sub = df[df["CandidateType"] == t]
        if len(sub) == 0:
            continue
        p = score(sub, scaler, clf)
        acc = (p >= 0.5).mean()
        print(f"  [sanity] {t}: n={len(sub)}, plain-0.5 ML acceptance={acc*100:.1f}% "
              f"(unaffected by interior_fill rule -- reported for reference only)")

    interior_lab = df[df["CandidateType"] == "interior_fill"].reset_index(drop=True)
    y_interior = interior_lab["IsCrack"].astype(bool).values
    pos_mask, neg_mask = y_interior, ~y_interior
    print(f"\ninterior_fill labeled: {pos_mask.sum()} True, {neg_mask.sum()} False")
    assert neg_mask.sum() >= 2, "need >=2 known negatives for LOO validation"

    pool = load_full_interior_fill_pool()
    dist_pool = pool["MeanDistToCrack"].values
    bri_pool = pool["MeanFlatBrightness"].values
    print(f"full interior_fill pool: {len(pool)} candidates (labeled + unlabeled)")

    full_fit_proba = score(interior_lab, scaler, clf)

    # ---- LOO probability for each known negative: refit excluding just that row ----
    neg_rows = interior_lab[neg_mask].reset_index(drop=True)
    print("\nKnown negatives:")
    for i in range(len(neg_rows)):
        r = neg_rows.iloc[i]
        dist_pctile = (dist_pool <= r["MeanDistToCrack"]).mean() * 100
        bri_pctile = (bri_pool <= r["MeanFlatBrightness"]).mean() * 100
        print(f"  #{i}: SourceImage={r['SourceImage']} Label={r['Label']} "
              f"MeanDistToCrack={r['MeanDistToCrack']:.2f} (~{dist_pctile:.1f}th pctile of pool), "
              f"MeanFlatBrightness={r['MeanFlatBrightness']:.2f} (~{bri_pctile:.1f}th pctile of pool)")

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
        print(f"  LOO-refit ML_proba for negative #{i}: {loo_proba_neg[i]:.3f}")

    pool_proba = score(pool, scaler, clf)

    # ---- FINE grid search: percentile step=1, floor step=0.01 ----
    N_range = range(5, 96, 1)          # distance percentile
    M_range = range(5, 96, 1)          # brightness percentile
    floor_range = np.round(np.arange(0.10, 0.951, 0.01), 2)

    print(f"\nGrid size: {len(list(N_range))} x {len(list(M_range))} x {len(floor_range)} = "
          f"{len(list(N_range))*len(list(M_range))*len(floor_range)} configs")

    dist_thr_by_N = {N: np.percentile(dist_pool, N) for N in N_range}
    bri_thr_by_M = {M: np.percentile(bri_pool, M) for M in M_range}

    grid = []
    for N in N_range:
        dist_thr = dist_thr_by_N[N]
        neg_dist_ok = neg_rows["MeanDistToCrack"].values <= dist_thr
        pos_dist_ok = interior_lab["MeanDistToCrack"].values[pos_mask] <= dist_thr
        pool_dist_ok = dist_pool <= dist_thr
        for M in M_range:
            bri_thr = bri_thr_by_M[M]
            neg_rule_ok = neg_dist_ok & (neg_rows["MeanFlatBrightness"].values <= bri_thr)
            if not neg_rule_ok.any():
                # neither negative is even admitted by the dist/brightness gate --
                # any floor rejects both trivially; still valid, recall/accept computed below
                pass
            pos_rule_ok = pos_dist_ok & (interior_lab["MeanFlatBrightness"].values[pos_mask] <= bri_thr)
            pool_rule_ok = pool_dist_ok & (bri_pool <= bri_thr)
            for floor in floor_range:
                if (neg_rule_ok & (loo_proba_neg >= floor)).any():
                    continue  # would admit a known negative under LOO -- invalid config
                recall_pos = (pos_rule_ok & (full_fit_proba[pos_mask] >= floor)).mean()
                full_pool_accept = (pool_rule_ok & (pool_proba >= floor)).mean()
                grid.append((int(N), int(M), round(float(floor), 2), float(recall_pos), float(full_pool_accept)))

    print(f"Valid (LOO-passing) configs: {len(grid)} / "
          f"{len(list(N_range))*len(list(M_range))*len(floor_range)}")

    if not grid:
        print("No valid config found -- rule cannot be calibrated with finer grid either.")
        return

    grid_df = pd.DataFrame(grid, columns=["N_pct", "M_pct", "floor", "recall", "pool_accept"])

    # ---- Pareto frontier over (recall higher-better, pool_accept lower-better) ----
    # dedupe on (recall, pool_accept) pairs first to avoid redundant frontier points
    # from many (N,M,floor) combos giving identical outcomes; keep one representative
    # (prefer simplest / loosest: fewest binding constraints) per outcome pair.
    grid_df_sorted = grid_df.sort_values(
        ["recall", "pool_accept", "N_pct", "M_pct", "floor"],
        ascending=[False, True, False, False, True]
    ).reset_index(drop=True)

    pareto_rows = []
    best_pool_accept_so_far = np.inf
    # iterate in descending recall order; a point is Pareto-optimal if its
    # pool_accept is strictly lower than every point seen so far with >= recall
    for _, row in grid_df_sorted.iterrows():
        if row["pool_accept"] < best_pool_accept_so_far - 1e-12:
            pareto_rows.append(row)
            best_pool_accept_so_far = row["pool_accept"]

    pareto_df = pd.DataFrame(pareto_rows).reset_index(drop=True)
    # sort frontier by recall ascending for readability (low-recall/low-accept ... high-recall/high-accept)
    pareto_df = pareto_df.sort_values("recall").reset_index(drop=True)

    print(f"\nFull Pareto frontier: {len(pareto_df)} distinct (recall, pool_accept) points")
    print("\n=== FULL PARETO FRONTIER (recall asc) ===")
    print(pareto_df.to_string(index=False))

    # ---- current production point for reference ----
    prod_recall, prod_accept = 0.744, 0.576

    # ---- highlight interesting regions ----
    print("\n=== Candidates: recall >= 0.85 AND pool_accept <= 0.65 ===")
    hi_recall = pareto_df[(pareto_df["recall"] >= 0.85) & (pareto_df["pool_accept"] <= 0.65)]
    print(hi_recall.to_string(index=False) if len(hi_recall) else "  (none found)")

    print("\n=== Candidates: pool_accept <= 0.40 AND recall >= 0.65 ===")
    lo_accept = pareto_df[(pareto_df["pool_accept"] <= 0.40) & (pareto_df["recall"] >= 0.65)]
    print(lo_accept.to_string(index=False) if len(lo_accept) else "  (none found)")

    # ---- select ~5-10 representative/interesting points across the frontier ----
    print("\n=== 8 representative Pareto points spanning the frontier ===")
    if len(pareto_df) <= 8:
        rep = pareto_df
    else:
        idx = np.linspace(0, len(pareto_df) - 1, 8).round().astype(int)
        idx = sorted(set(idx))
        rep = pareto_df.iloc[idx]
    for _, row in rep.iterrows():
        dist_thr = dist_thr_by_N[int(row["N_pct"])]
        bri_thr = bri_thr_by_M[int(row["M_pct"])]
        print(f"  N={int(row['N_pct']):2d}pct(dist<= {dist_thr:6.2f}px)  "
              f"M={int(row['M_pct']):2d}pct(bri<= {bri_thr:6.2f})  floor={row['floor']:.2f}  "
              f"-> recall={row['recall']*100:5.1f}%  pool_accept={row['pool_accept']*100:5.1f}%")

    # closest point to production recall for direct apples-to-apples comparison
    print(f"\n=== For reference, current production point: recall=74.4%, pool_accept=57.6% ===")
    closest = pareto_df.iloc[(pareto_df["recall"] - prod_recall).abs().argsort()[:3]]
    print("Nearest-recall Pareto points to production recall (74.4%):")
    print(closest.to_string(index=False))

    # does any point dominate production (equal/higher recall, equal/lower accept, at least one strict)?
    dominates = pareto_df[
        (pareto_df["recall"] >= prod_recall - 1e-9) & (pareto_df["pool_accept"] <= prod_accept + 1e-9) &
        ((pareto_df["recall"] > prod_recall + 1e-9) | (pareto_df["pool_accept"] < prod_accept - 1e-9))
    ]
    print(f"\nConfigs strictly dominating production (74.4% recall / 57.6% pool_accept): {len(dominates)}")
    if len(dominates):
        print(dominates.to_string(index=False))

    print(f"\nDone in {time.time()-t0:.1f}s")

    grid_df.to_csv(os.path.join(os.path.dirname(__file__), "finer_grid_pareto_full_grid.csv"), index=False)
    pareto_df.to_csv(os.path.join(os.path.dirname(__file__), "finer_grid_pareto_frontier.csv"), index=False)
    print("Saved full grid + frontier CSVs alongside this script.")


if __name__ == "__main__":
    main()
