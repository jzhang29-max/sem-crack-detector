"""
round3_synthetic_negative_augmentation.py
------------------------------------------
Approach under test ("synthetic_negative_augmentation"): interior_fill has
only 2 real negatives, which is why the production rule
(calibrate_interior_fill_rule() in ../train_interior_model.py) needs 2 extra
gates on top of the plain ML floor and still only reaches 74.4% recall /
57.6% pool acceptance. This script asks whether manufacturing MORE negative
training signal -- without collecting more real labels -- can do better,
via two different flavors of synthetic negative:

(1) GAUSSIAN JITTER around each of the 2 real negatives: add small Gaussian
    noise (scaled to each feature's std within the interior_fill pool) to
    create N synthetic negative points per real negative, down-weighted via
    sample_weight (0.3x a real point) so they inform but don't dominate.
    NOTE: SMOTE/ADASYN-style interpolation is invalid here -- interpolating
    between exactly 2 minority points just draws the line segment joining
    them, which is not a meaningful synthetic-minority manifold assumption
    for 2 points from unrelated images. Jitter (assume local Gaussian
    neighborhoods around each point) is a weaker, more defensible
    assumption but still an assumption, not evidence.

(2) PHYSICS-INFORMED extreme-region negatives: rather than fabricate
    feature vectors, sample a few REAL (unlabeled) interior_fill candidates
    that already sit at the joint 90th-percentile extreme of BOTH
    MeanDistToCrack and MeanFlatBrightness (i.e. genuinely far from the
    crack AND genuinely bright), and assign them a synthetic label of
    False. The features are real measurements; only the label is a fabricated
    domain assumption ("this far + this bright can't be a crack gradient
    tail"). This is checked to not collide with any already-labeled row.

Both are evaluated the same way production's rule was: LOO-refit probability
on the 2 REAL negatives only (synthetic points are training signal only,
NEVER part of the LOO test itself -- and when LOO-ing out a real negative,
its own jittered children are also dropped from that fold's training set, so
the fold never gets to see even noisy copies of the point it's being tested
against). We also check margin (LOO proba value / gate distances relative to
the cutoffs, and how much perturbation would flip a rejection to an
acceptance) rather than just point recall/acceptance, and confirm
concavity/bridge_corridor's own acceptance is not meaningfully disturbed by
adding interior_fill-specific synthetic rows to the POOLED classifier's
training set.
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

CAND_GLOB = "/Users/jiamingzhang/Desktop/SEM_Crack_Detection_Pipeline/interior_active_learning/candidates/*_interior.csv"

FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
            "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
            "FracBoundaryTouchingCrack", "MeanDistToCrack"]

# Valid ranges to clip jittered synthetic points into (avoid physically
# impossible feature vectors, e.g. negative brightness or Solidity > 1).
FEATURE_BOUNDS = {
    "LogArea": (0.0, None), "Elongation": (1.0, None), "Solidity": (0.0, 1.0),
    "Eccentricity": (0.0, 1.0), "Extent": (0.0, 1.0), "Circularity": (0.0, None),
    "MeanRawBrightness": (0.0, None), "MeanFlatBrightness": (0.0, None),
    "MeanVesselness": (0.0, None), "FracBoundaryTouchingCrack": (0.0, 1.0),
    "MeanDistToCrack": (0.0, None),
}

# Reference numbers quoted in the task brief (from an earlier labeled-data
# snapshot). Recomputed fresh below in section 0 -- the live dataset has
# grown since that snapshot, so treat BASELINE_* below as provisional until
# overwritten by the live recomputation.
BASELINE_RECALL = 0.744
BASELINE_POOL_ACCEPT = 0.576
BASELINE_FLOOR, BASELINE_DIST_THR, BASELINE_BRI_THR = 0.60, 12.901030955813528, 135.8445481106745

RNG = np.random.default_rng(0)


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


def fit_lr(X, y, sample_weight=None):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(X), y, sample_weight=sample_weight)
    return sc, clf


def score(df_sub, sc, clf):
    return clf.predict_proba(sc.transform(df_sub[FEATURES].values))[:, 1]


def clip_synthetic(vals):
    out = dict(vals)
    for feat, (lo, hi) in FEATURE_BOUNDS.items():
        v = out[feat]
        if lo is not None:
            v = max(v, lo)
        if hi is not None:
            v = min(v, hi)
        out[feat] = v
    return out


def make_jitter_synthetic(real_neg_rows, pool_std, noise_frac, n_per_neg, rng):
    """real_neg_rows: DataFrame of one or more real negatives to jitter around.
    Returns a DataFrame of synthetic rows with a 'parent_row_key' column
    (SourceImage, Label) identifying which real negative spawned it, so LOO
    folds can drop the right children."""
    synth_rows = []
    for _, row in real_neg_rows.iterrows():
        base = row[FEATURES].astype(float)
        for _ in range(n_per_neg):
            noise = rng.normal(0.0, noise_frac * pool_std.values)
            vals = clip_synthetic(dict(zip(FEATURES, base.values + noise)))
            vals["IsCrack"] = False
            vals["parent_source"] = row["SourceImage"]
            vals["parent_label"] = row["Label"]
            synth_rows.append(vals)
    return pd.DataFrame(synth_rows)


def build_augmented_training(df_real, synth_df, synth_weight):
    """Concatenate real (weight 1.0) + synthetic (weight synth_weight) into
    X, y, sample_weight arrays ready for fit_lr."""
    X_real = df_real[FEATURES].values
    y_real = df_real["IsCrack"].astype(bool).values
    w_real = np.ones(len(df_real))
    if synth_df is None or len(synth_df) == 0:
        return X_real, y_real, w_real
    X_synth = synth_df[FEATURES].values
    y_synth = synth_df["IsCrack"].astype(bool).values
    w_synth = np.full(len(synth_df), synth_weight)
    X = np.vstack([X_real, X_synth])
    y = np.concatenate([y_real, y_synth])
    w = np.concatenate([w_real, w_synth])
    return X, y, w


def grid_search_rule(interior_lab, pos_mask, neg_mask, neg_rows, loo_proba_neg,
                      full_fit_proba, dist_pool, bri_pool, pool_proba):
    """Exact same grid/selection logic as calibrate_interior_fill_rule(), factored
    out so it can be reused for augmented and non-augmented model variants."""
    grid = []
    for N in range(5, 71, 5):
        dist_thr = np.percentile(dist_pool, N)
        for M in range(20, 91, 5):
            bri_thr = np.percentile(bri_pool, M)
            neg_rule_ok = (neg_rows["MeanDistToCrack"].values <= dist_thr) & \
                          (neg_rows["MeanFlatBrightness"].values <= bri_thr)
            pos_rule_ok = (interior_lab["MeanDistToCrack"].values[pos_mask] <= dist_thr) & \
                          (interior_lab["MeanFlatBrightness"].values[pos_mask] <= bri_thr)
            for floor in np.arange(0.10, 0.86, 0.05):
                if (neg_rule_ok & (loo_proba_neg >= floor)).any():
                    continue
                recall_pos = (pos_rule_ok & (full_fit_proba[pos_mask] >= floor)).mean()
                full_pool_accept = ((dist_pool <= dist_thr) & (bri_pool <= bri_thr) & (pool_proba >= floor)).mean()
                grid.append({"N": N, "M": M, "floor": round(float(floor), 2),
                             "dist_thr": dist_thr, "bri_thr": bri_thr,
                             "recall": float(recall_pos), "pool_accept": float(full_pool_accept)})
    if not grid:
        return None
    grid.sort(key=lambda t: (-t["recall"], t["pool_accept"], -t["N"], -t["M"], t["floor"]))
    return grid[0]


def main():
    global BASELINE_RECALL, BASELINE_POOL_ACCEPT, BASELINE_FLOOR, BASELINE_DIST_THR, BASELINE_BRI_THR
    df_full = load_labeled_interior()
    print(f"Loaded {len(df_full)} labeled candidates total ({df_full['SourceImage'].nunique()} images)")

    pool_interior = load_full_pool("interior_fill")
    pool_concavity = load_full_pool("concavity")
    pool_bridge = load_full_pool("bridge_corridor")
    print(f"Full pools: concavity={len(pool_concavity)}, bridge_corridor={len(pool_bridge)}, "
          f"interior_fill={len(pool_interior)}\n")

    interior_lab = df_full[df_full["CandidateType"] == "interior_fill"].reset_index(drop=True)
    y_interior = interior_lab["IsCrack"].astype(bool).values
    pos_mask, neg_mask = y_interior, ~y_interior
    neg_rows = interior_lab[neg_mask].reset_index(drop=True)
    print(f"interior_fill labeled: {pos_mask.sum()} True / {neg_mask.sum()} False")
    print("Known negatives:")
    for _, row in neg_rows.iterrows():
        print(f"  {row['SourceImage']} Label={row['Label']}: MeanDistToCrack={row['MeanDistToCrack']:.2f}, "
              f"MeanFlatBrightness={row['MeanFlatBrightness']:.2f}")
    print("  (NOTE: both real negatives are actually CLOSE and DIM, not 'far and bright' -- "
          "the production gate rejects negative #2 via the distance gate outright, and negative #1 "
          "via the plain ML floor. A 'far AND bright' physics prior targets a different region of "
          "feature space than either known negative, so it cannot directly reinforce either existing "
          "rejection mechanism -- flagged up front as a reason to expect limited benefit from "
          "approach (2) below.)\n")

    dist_pool = pool_interior["MeanDistToCrack"].values
    bri_pool = pool_interior["MeanFlatBrightness"].values
    pool_std = pool_interior[FEATURES].std()

    def loo_refit_excluding_rows(exclude_pairs, extra_synth=None, synth_weight=0.3):
        """Refit pooled scaler+clf excluding the given (SourceImage,Label) real
        rows, optionally adding extra_synth (already built, already excludes
        children of anything in exclude_pairs -- caller's responsibility)."""
        keep = np.ones(len(df_full), dtype=bool)
        for src, lab in exclude_pairs:
            keep &= ~((df_full["SourceImage"] == src) & (df_full["Label"].values == lab))
        df_loo = df_full.loc[keep]
        X, y, w = build_augmented_training(df_loo, extra_synth, synth_weight)
        return fit_lr(X, y, sample_weight=w)

    # =====================================================================
    # 0. BASELINE REPRODUCTION (no augmentation) -- sanity check vs production numbers
    # =====================================================================
    print("=" * 78)
    print("0. BASELINE (production, no synthetic augmentation)")
    print("=" * 78)
    X0, y0 = df_full[FEATURES].values, df_full["IsCrack"].astype(bool).values
    scaler0, clf0 = fit_lr(X0, y0)
    full_fit_proba0 = score(interior_lab, scaler0, clf0)
    pool_proba0 = score(pool_interior, scaler0, clf0)
    loo_proba_neg0 = np.zeros(len(neg_rows))
    for i, row in neg_rows.iterrows():
        sc_loo, clf_loo = loo_refit_excluding_rows([(row["SourceImage"], row["Label"])])
        loo_proba_neg0[i] = clf_loo.predict_proba(sc_loo.transform(row[FEATURES].values.reshape(1, -1)))[0, 1]
    print(f"LOO proba of known negatives (no augmentation): {loo_proba_neg0}")
    base_best = grid_search_rule(interior_lab, pos_mask, neg_mask, neg_rows, loo_proba_neg0,
                                  full_fit_proba0, dist_pool, bri_pool, pool_proba0)
    if base_best:
        print(f"  best: floor={base_best['floor']}, dist<={base_best['dist_thr']:.2f} (N={base_best['N']}), "
              f"bri<={base_best['bri_thr']:.2f} (M={base_best['M']})")
        print(f"  recall={base_best['recall']*100:.1f}%, pool_accept={base_best['pool_accept']*100:.1f}%  "
              f"(task-brief reference snapshot: {BASELINE_RECALL*100:.1f}% / {BASELINE_POOL_ACCEPT*100:.1f}% "
              f"at floor {BASELINE_FLOOR} -- dataset has grown since that snapshot; using the FRESH "
              f"live numbers just computed as the actual baseline for all comparisons below)")
        # Overwrite the provisional task-brief constants with the live, freshly
        # recomputed production rule so every later comparison in this script
        # is against what is ACTUALLY running today, not a stale snapshot.
        BASELINE_RECALL = base_best["recall"]
        BASELINE_POOL_ACCEPT = base_best["pool_accept"]
        BASELINE_FLOOR = base_best["floor"]
        BASELINE_DIST_THR = base_best["dist_thr"]
        BASELINE_BRI_THR = base_best["bri_thr"]
    acc_conc0 = float((score(pool_concavity, scaler0, clf0) >= 0.5).mean())
    acc_bridge0 = float((score(pool_bridge, scaler0, clf0) >= 0.5).mean())
    print(f"  concavity/bridge_corridor @0.5 floor (sanity reference): {acc_conc0*100:.1f}% / {acc_bridge0*100:.1f}%\n")

    # Margin of each known negative under baseline: how far its LOO-proba sits
    # below whatever floor the baseline rule uses, and whether its gate values
    # sit comfortably inside/outside the cutoffs.
    print("Baseline (LIVE, freshly recomputed) per-negative margin detail (cutoffs "
          f"floor={BASELINE_FLOOR}, dist<={BASELINE_DIST_THR:.2f}, bri<={BASELINE_BRI_THR:.2f}):")
    for i, row in neg_rows.iterrows():
        gate_ok = row["MeanDistToCrack"] <= BASELINE_DIST_THR and row["MeanFlatBrightness"] <= BASELINE_BRI_THR
        rel = (BASELINE_FLOOR - loo_proba_neg0[i]) / BASELINE_FLOOR
        print(f"  {row['SourceImage']} Label={row['Label']}: LOO proba={loo_proba_neg0[i]:.3f} "
              f"(floor={BASELINE_FLOOR}, abs margin={BASELINE_FLOOR - loo_proba_neg0[i]:+.3f}, "
              f"rel margin={rel*100:+.1f}%), "
              f"gate_ok={gate_ok} -> rejected via {'ML floor' if gate_ok else 'distance/brightness gate'}")
    print()

    # =====================================================================
    # 1. GAUSSIAN JITTER AUGMENTATION
    # =====================================================================
    print("=" * 78)
    print("1. GAUSSIAN JITTER AUGMENTATION")
    print("=" * 78)
    jitter_results = []
    for noise_frac in [0.05, 0.10, 0.15]:
        for n_per_neg in [10, 25]:
            for synth_weight in [0.3]:
                rng = np.random.default_rng(hash((noise_frac, n_per_neg, synth_weight)) % (2**32))
                synth_all = make_jitter_synthetic(neg_rows, pool_std, noise_frac, n_per_neg, rng)
                X, y, w = build_augmented_training(df_full, synth_all, synth_weight)
                scaler_a, clf_a = fit_lr(X, y, sample_weight=w)
                full_fit_proba_a = score(interior_lab, scaler_a, clf_a)
                pool_proba_a = score(pool_interior, scaler_a, clf_a)

                # LOO: for each real negative, drop it AND its own synthetic
                # children, keep synthetic children of the OTHER negative,
                # regenerate nothing extra (children of the remaining negative
                # already in synth_all with the same rng/seed as full fit --
                # i.e. LOO uses the exact same synthetic cloud minus the
                # held-out point's own children, matching production's LOO
                # methodology of "refit excluding just that row").
                loo_proba_neg_a = np.zeros(len(neg_rows))
                for i, row in neg_rows.iterrows():
                    children_mask = ~((synth_all["parent_source"] == row["SourceImage"]) &
                                       (synth_all["parent_label"] == row["Label"]))
                    synth_loo = synth_all[children_mask]
                    sc_loo, clf_loo = loo_refit_excluding_rows(
                        [(row["SourceImage"], row["Label"])], extra_synth=synth_loo, synth_weight=synth_weight)
                    loo_proba_neg_a[i] = clf_loo.predict_proba(
                        sc_loo.transform(row[FEATURES].values.reshape(1, -1)))[0, 1]

                best = grid_search_rule(interior_lab, pos_mask, neg_mask, neg_rows, loo_proba_neg_a,
                                         full_fit_proba_a, dist_pool, bri_pool, pool_proba_a)
                acc_conc = float((score(pool_concavity, scaler_a, clf_a) >= 0.5).mean())
                acc_bridge = float((score(pool_bridge, scaler_a, clf_a) >= 0.5).mean())
                rec = {
                    "noise_frac": noise_frac, "n_per_neg": n_per_neg, "synth_weight": synth_weight,
                    "n_synth_total": len(synth_all),
                    "loo_proba_neg": loo_proba_neg_a.copy(),
                    "best": best, "acc_conc": acc_conc, "acc_bridge": acc_bridge,
                }
                jitter_results.append(rec)
                if best:
                    print(f"noise={noise_frac:.2f} n_per_neg={n_per_neg:>2} (n_synth={len(synth_all):>2}, w={synth_weight}): "
                          f"LOO neg proba={np.round(loo_proba_neg_a,3)}  "
                          f"best recall={best['recall']*100:5.1f}% pool_accept={best['pool_accept']*100:5.1f}%  "
                          f"floor={best['floor']:.2f} dist<={best['dist_thr']:.2f} bri<={best['bri_thr']:.2f}  "
                          f"[concavity {acc_conc*100:.1f}%, bridge {acc_bridge*100:.1f}%]")
                else:
                    print(f"noise={noise_frac:.2f} n_per_neg={n_per_neg:>2} (n_synth={len(synth_all):>2}, w={synth_weight}): "
                          f"LOO neg proba={np.round(loo_proba_neg_a,3)}  NO LOO-safe grid config found")
    print()

    best_jitter = max((r for r in jitter_results if r["best"]), key=lambda r: r["best"]["recall"], default=None)
    if best_jitter:
        b = best_jitter["best"]
        print(f"Best jitter config: noise_frac={best_jitter['noise_frac']}, n_per_neg={best_jitter['n_per_neg']}, "
              f"synth_weight={best_jitter['synth_weight']} -> recall={b['recall']*100:.1f}%, "
              f"pool_accept={b['pool_accept']*100:.1f}%  (live baseline: {BASELINE_RECALL*100:.1f}% / "
              f"{BASELINE_POOL_ACCEPT*100:.1f}% at floor {BASELINE_FLOOR})")
        beats = (b["recall"] > BASELINE_RECALL + 1e-9) and (b["pool_accept"] <= BASELINE_POOL_ACCEPT + 1e-9)
        print(f"  Strictly beats live baseline (higher recall, <= pool accept)? {beats}")
    print()

    # ---- Seed-variance robustness check on the apparently-best jitter config ----
    # Jitter is stochastic. Before trusting "noise=0.05, n_per_neg=25" as a real
    # improvement, check how much recall/pool_accept/margin move across RNG
    # seeds alone, everything else held fixed. If the spread across seeds is
    # comparable to the apparent gain over baseline, the gain is noise, not signal.
    if best_jitter:
        print(f"Seed-variance check for noise_frac={best_jitter['noise_frac']}, "
              f"n_per_neg={best_jitter['n_per_neg']}, synth_weight={best_jitter['synth_weight']} "
              f"across 8 independent RNG seeds (same hyperparameters, different jitter draws):")
        seed_recalls, seed_accepts, seed_margins_neg1 = [], [], []
        for seed in range(8):
            rng = np.random.default_rng(1000 + seed)
            synth_all = make_jitter_synthetic(neg_rows, pool_std, best_jitter["noise_frac"],
                                               best_jitter["n_per_neg"], rng)
            X, y, w = build_augmented_training(df_full, synth_all, best_jitter["synth_weight"])
            scaler_a, clf_a = fit_lr(X, y, sample_weight=w)
            full_fit_proba_a = score(interior_lab, scaler_a, clf_a)
            pool_proba_a = score(pool_interior, scaler_a, clf_a)
            loo_proba_neg_a = np.zeros(len(neg_rows))
            for i, row in neg_rows.iterrows():
                children_mask = ~((synth_all["parent_source"] == row["SourceImage"]) &
                                   (synth_all["parent_label"] == row["Label"]))
                synth_loo = synth_all[children_mask]
                sc_loo, clf_loo = loo_refit_excluding_rows(
                    [(row["SourceImage"], row["Label"])], extra_synth=synth_loo,
                    synth_weight=best_jitter["synth_weight"])
                loo_proba_neg_a[i] = clf_loo.predict_proba(
                    sc_loo.transform(row[FEATURES].values.reshape(1, -1)))[0, 1]
            best_s = grid_search_rule(interior_lab, pos_mask, neg_mask, neg_rows, loo_proba_neg_a,
                                       full_fit_proba_a, dist_pool, bri_pool, pool_proba_a)
            if best_s:
                seed_recalls.append(best_s["recall"])
                seed_accepts.append(best_s["pool_accept"])
                seed_margins_neg1.append((best_s["floor"] - loo_proba_neg_a[0]) / best_s["floor"])
                print(f"    seed={seed}: recall={best_s['recall']*100:5.1f}% pool_accept={best_s['pool_accept']*100:5.1f}% "
                      f"floor={best_s['floor']:.2f}  LOO_neg={np.round(loo_proba_neg_a,3)}")
            else:
                print(f"    seed={seed}: NO LOO-safe grid config found")
        if seed_recalls:
            print(f"  across seeds: recall {min(seed_recalls)*100:.1f}-{max(seed_recalls)*100:.1f}% "
                  f"(spread {(max(seed_recalls)-min(seed_recalls))*100:.1f} pts), "
                  f"pool_accept {min(seed_accepts)*100:.1f}-{max(seed_accepts)*100:.1f}%, "
                  f"neg#1 rel margin {min(seed_margins_neg1)*100:.1f}-{max(seed_margins_neg1)*100:.1f}%")
            print(f"  for reference, apparent gain over live baseline recall ({BASELINE_RECALL*100:.1f}%) "
                  f"was {(best_jitter['best']['recall']-BASELINE_RECALL)*100:.1f} pts -- "
                  f"{'seed spread is comparable to or larger than the apparent gain (treat as noise)' if (max(seed_recalls)-min(seed_recalls)) >= (best_jitter['best']['recall']-BASELINE_RECALL) else 'seed spread is smaller than the apparent gain'}")
    print()

    # =====================================================================
    # 2. PHYSICS-INFORMED "far AND bright" SYNTHETIC NEGATIVES
    # =====================================================================
    print("=" * 78)
    print("2. PHYSICS-INFORMED EXTREME-REGION SYNTHETIC NEGATIVES")
    print("=" * 78)
    d90 = np.percentile(dist_pool, 90)
    b90 = np.percentile(bri_pool, 90)
    far_bright_pool = pool_interior[(pool_interior["MeanDistToCrack"] >= d90) &
                                     (pool_interior["MeanFlatBrightness"] >= b90)].copy()
    already_labeled = far_bright_pool["IsCrack"].astype(str).str.strip().str.upper().isin(["TRUE", "FALSE"])
    print(f"90th pct dist={d90:.2f}, 90th pct bright={b90:.2f} -> {len(far_bright_pool)} real unlabeled "
          f"pool candidates sit at/above BOTH cutoffs simultaneously "
          f"({int(already_labeled.sum())} of those already carry a real label -- excluded to avoid contradicting real data).")
    far_bright_pool = far_bright_pool[~already_labeled]

    for n_synth_phys, phys_weight in [(3, 1.0), (5, 1.0), (5, 0.3)]:
        rng = np.random.default_rng(42)
        chosen_idx = rng.choice(far_bright_pool.index.values, size=min(n_synth_phys, len(far_bright_pool)), replace=False)
        phys_synth = far_bright_pool.loc[chosen_idx, FEATURES].copy()
        phys_synth["IsCrack"] = False
        phys_synth["parent_source"] = "PHYSICS_PRIOR"
        phys_synth["parent_label"] = -1

        X, y, w = build_augmented_training(df_full, phys_synth, phys_weight)
        scaler_p, clf_p = fit_lr(X, y, sample_weight=w)
        full_fit_proba_p = score(interior_lab, scaler_p, clf_p)
        pool_proba_p = score(pool_interior, scaler_p, clf_p)

        # LOO on the 2 real negatives: physics-prior points are a permanent
        # domain assumption (not derived from either real negative), so they
        # stay in every LOO fold's training set.
        loo_proba_neg_p = np.zeros(len(neg_rows))
        for i, row in neg_rows.iterrows():
            sc_loo, clf_loo = loo_refit_excluding_rows(
                [(row["SourceImage"], row["Label"])], extra_synth=phys_synth, synth_weight=phys_weight)
            loo_proba_neg_p[i] = clf_loo.predict_proba(sc_loo.transform(row[FEATURES].values.reshape(1, -1)))[0, 1]

        best = grid_search_rule(interior_lab, pos_mask, neg_mask, neg_rows, loo_proba_neg_p,
                                 full_fit_proba_p, dist_pool, bri_pool, pool_proba_p)
        acc_conc = float((score(pool_concavity, scaler_p, clf_p) >= 0.5).mean())
        acc_bridge = float((score(pool_bridge, scaler_p, clf_p) >= 0.5).mean())
        # sanity: does the model itself now reject the physics-prior points?
        phys_self_proba = clf_p.predict_proba(scaler_p.transform(phys_synth[FEATURES].values))[:, 1]
        if best:
            print(f"n_phys={n_synth_phys} weight={phys_weight}: LOO neg proba={np.round(loo_proba_neg_p,3)}  "
                  f"best recall={best['recall']*100:5.1f}% pool_accept={best['pool_accept']*100:5.1f}%  "
                  f"floor={best['floor']:.2f} dist<={best['dist_thr']:.2f} bri<={best['bri_thr']:.2f}  "
                  f"[concavity {acc_conc*100:.1f}%, bridge {acc_bridge*100:.1f}%]  "
                  f"phys-point self-proba(max)={phys_self_proba.max():.3f}")
        else:
            print(f"n_phys={n_synth_phys} weight={phys_weight}: LOO neg proba={np.round(loo_proba_neg_p,3)}  "
                  f"NO LOO-safe grid config found")
    print()

    # =====================================================================
    # 3. MARGIN / SENSITIVITY COMPARISON: best jitter config vs baseline
    # =====================================================================
    print("=" * 78)
    print("3. MARGIN COMPARISON (baseline vs best jitter config)")
    print("=" * 78)
    print("For each known negative: LOO-refit probability vs the selected floor, i.e. how much")
    print("the ML score would need to rise (as a fraction of the floor) before this negative")
    print("would slip past the ML-floor leg of the rule. Larger relative margin = more robust.")
    print(f"{'config':30s} {'negative':38s} {'LOO_proba':>10s} {'floor':>7s} {'rel_margin':>11s}")
    for i, row in neg_rows.iterrows():
        rel = (BASELINE_FLOOR - loo_proba_neg0[i]) / BASELINE_FLOOR
        print(f"{'baseline (production)':30s} {row['SourceImage']+' #'+str(row['Label']):38s} "
              f"{loo_proba_neg0[i]:10.3f} {BASELINE_FLOOR:7.2f} {rel*100:10.1f}%")
    if best_jitter and best_jitter["best"]:
        floor_j = best_jitter["best"]["floor"]
        for i, row in neg_rows.iterrows():
            p = best_jitter["loo_proba_neg"][i]
            rel = (floor_j - p) / floor_j if floor_j > 0 else float("nan")
            print(f"{'best jitter aug':30s} {row['SourceImage']+' #'+str(row['Label']):38s} "
                  f"{p:10.3f} {floor_j:7.2f} {rel*100:10.1f}%")
    print()
    print("Interpretation: augmentation changes WHERE the LOO-refit probability of each real")
    print("negative lands, and the grid search may pick a different floor to match, but the")
    print("relative margin is not mechanically guaranteed to widen -- see summary below.\n")

    # =====================================================================
    # 4. PERTURBATION SENSITIVITY: does jitter actually widen the ABSOLUTE
    # safety buffer, or just relocate the same knife-edge gap to a lower
    # floor (where "relative margin" looks bigger purely because the
    # denominator shrank)? Same perturbation-sensitivity methodology as
    # refine_stability_and_sensitivity.py: perturb negative #1 (the one the
    # ML-floor leg actually catches -- negative #2 is caught by the distance
    # gate outright and barely touches the floor leg either way) by
    # +/-5/10/20% on MeanDistToCrack and MeanFlatBrightness together, rescore
    # with each model's own LOO-refit classifier, and find the smallest
    # perturbation that flips it from rejected to accepted under each rule.
    # =====================================================================
    print("=" * 78)
    print("4. PERTURBATION SENSITIVITY on negative #1 (the ML-floor-caught one):")
    print("   baseline rule (floor=%.2f) vs best jitter rule (floor=%.2f)" %
          (BASELINE_FLOOR, best_jitter["best"]["floor"] if best_jitter and best_jitter["best"] else float("nan")))
    print("=" * 78)
    neg1 = neg_rows.iloc[0]
    PERTURB_PCTS = [-20, -10, -5, 5, 10, 20]

    def perturbed_row(row, pct):
        r = row.copy()
        r["MeanDistToCrack"] = row["MeanDistToCrack"] * (1 + pct / 100.0)
        r["MeanFlatBrightness"] = row["MeanFlatBrightness"] * (1 + pct / 100.0)
        return r

    # baseline LOO model (already fit above as sc_loo/clf_loo for last negative in loop --
    # refit explicitly here for negative #1 specifically, unperturbed, for clarity)
    sc_loo0, clf_loo0 = loo_refit_excluding_rows([(neg1["SourceImage"], neg1["Label"])])
    # best jitter LOO model for negative #1 specifically
    if best_jitter:
        rng = np.random.default_rng(1000)
        synth_all_bj = make_jitter_synthetic(neg_rows, pool_std, best_jitter["noise_frac"],
                                              best_jitter["n_per_neg"], rng)
        children_mask = ~((synth_all_bj["parent_source"] == neg1["SourceImage"]) &
                           (synth_all_bj["parent_label"] == neg1["Label"]))
        sc_loo_j, clf_loo_j = loo_refit_excluding_rows(
            [(neg1["SourceImage"], neg1["Label"])], extra_synth=synth_all_bj[children_mask],
            synth_weight=best_jitter["synth_weight"])
        floor_j = best_jitter["best"]["floor"]

    print(f"{'pct':>5s}  {'baseline_proba':>14s} {'baseline_ok?':>12s}   {'jitter_proba':>13s} {'jitter_ok?':>11s}")
    flip_base, flip_jit = None, None
    for pct in sorted(PERTURB_PCTS, key=abs):
        r = perturbed_row(neg1, pct)
        p_base = clf_loo0.predict_proba(sc_loo0.transform(r[FEATURES].values.reshape(1, -1)))[0, 1]
        ok_base = p_base < BASELINE_FLOOR  # rejected = good
        if best_jitter:
            p_jit = clf_loo_j.predict_proba(sc_loo_j.transform(r[FEATURES].values.reshape(1, -1)))[0, 1]
            ok_jit = p_jit < floor_j
        else:
            p_jit, ok_jit = float("nan"), None
        print(f"{pct:+5d}%  {p_base:14.3f} {'REJECT' if ok_base else 'ACCEPT!':>12s}   "
              f"{p_jit:13.3f} {'REJECT' if ok_jit else 'ACCEPT!':>11s}")
        if not ok_base and flip_base is None:
            flip_base = pct
        if best_jitter and not ok_jit and flip_jit is None:
            flip_jit = pct
    print(f"\nSmallest |perturbation| that flips negative #1 to ACCEPTED: "
          f"baseline={'none up to +/-20%' if flip_base is None else str(flip_base)+'%'}, "
          f"jitter={'none up to +/-20%' if flip_jit is None else str(flip_jit)+'%'}")
    print("If jitter's flip point is NOT meaningfully further out than baseline's, the apparent")
    print("relative-margin improvement in section 3 is an artifact of dividing by a smaller floor,")
    print("not a real increase in how much the feature values could drift before misclassification.")


if __name__ == "__main__":
    main()
