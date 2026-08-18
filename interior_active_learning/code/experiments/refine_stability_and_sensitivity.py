"""
Refinement experiment: stability_and_sensitivity

Two robustness checks on the CURRENT production interior_fill acceptance
rule (accept iff ML_proba >= 0.60 AND MeanDistToCrack <= 12.90px (70th pct)
AND MeanFlatBrightness <= 135.84 (90th pct)), as calibrated by
calibrate_interior_fill_rule() in ../train_interior_model.py. Does NOT
propose a different rule -- it only asks "how fragile is the CURRENT one?"

(1) LEAVE-ONE-IMAGE-OUT STABILITY: redo the entire calibration procedure
    (pooled model refit + calibrate_interior_fill_rule's grid search/LOO
    constraint) using only labeled data from a random ~80% subset of the 22
    labeled images, for 5 different random subsets (seeds 0-4). Report how
    much the resulting (N_pct, M_pct, floor, dist_thr, bri_thr) and the
    resulting recall/pool-acceptance vary, AND -- more importantly --
    whether the subset-calibrated rule, when checked against a proper
    full-data LOO-refit probability for the 2 known negatives, would still
    reject both of them.

(2) SENSITIVITY ANALYSIS: perturb each known negative's MeanDistToCrack and
    MeanFlatBrightness by +/-5%, +/-10%, +/-20% (one at a time and both
    together), re-score with the LOO-refit model (excluding that negative,
    unperturbed, from training -- consistent with the original LOO
    methodology) on the PERTURBED feature vector, and check whether the
    CURRENT rule's fixed cutoffs (dist<=12.90, bri<=135.84, floor>=0.60)
    still reject it. Report the smallest perturbation magnitude that flips
    either negative from rejected to accepted.
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
from train_interior_model import (
    load_labeled_interior, load_full_interior_fill_pool, calibrate_interior_fill_rule,
)

RNG_SEEDS = [0, 1, 2, 3, 4]
SUBSET_FRAC = 0.8
PERTURB_PCTS = [-20, -10, -5, 5, 10, 20]

# The exact production rule (from train_interior_model.py's calibrated output).
PROD_DIST_THR = 12.901030955813528
PROD_BRI_THR = 135.8445481106745
PROD_FLOOR = 0.60


def fit_pooled(df):
    X = df[INTERIOR_FEATURE_COLUMNS].values
    y = df["IsCrack"].astype(bool).values
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X), y)
    return scaler, clf


def score(sub_df, sc, model):
    return model.predict_proba(sc.transform(sub_df[INTERIOR_FEATURE_COLUMNS].values))[:, 1]


def loo_refit_excluding(df_full, source_image, label):
    """Refit pooled scaler+clf excluding exactly one (SourceImage, Label) row --
    same LOO methodology calibrate_interior_fill_rule() uses internally."""
    keep = ~((df_full["SourceImage"] == source_image) & (df_full["Label"].values == label))
    X_loo = df_full.loc[keep, INTERIOR_FEATURE_COLUMNS].values
    y_loo = df_full.loc[keep, "IsCrack"].astype(bool).values
    sc = StandardScaler().fit(X_loo)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X_loo), y_loo)
    return sc, clf


def main():
    df_full = load_labeled_interior()
    pool = load_full_interior_fill_pool()
    interior_lab_full = df_full[df_full["CandidateType"] == "interior_fill"].reset_index(drop=True)
    y_full = interior_lab_full["IsCrack"].astype(bool).values
    neg_rows_full = interior_lab_full[~y_full].reset_index(drop=True)
    pos_rows_full = interior_lab_full[y_full].reset_index(drop=True)

    all_images = sorted(df_full["SourceImage"].unique())
    n_images = len(all_images)
    print("=" * 90)
    print(f"Loaded {len(df_full)} labeled rows across {n_images} images.")
    print(f"interior_fill: {len(pos_rows_full)} positive, {len(neg_rows_full)} negative.")
    print("Known negatives:")
    for _, r in neg_rows_full.iterrows():
        print(f"  {r['SourceImage']}  Label={r['Label']}  "
              f"MeanDistToCrack={r['MeanDistToCrack']:.3f}  MeanFlatBrightness={r['MeanFlatBrightness']:.3f}")

    # Reproduce the production baseline exactly, as a sanity check that this
    # script's methodology matches train_interior_model.py apples-to-apples.
    scaler_full, clf_full = fit_pooled(df_full)
    baseline_rule = calibrate_interior_fill_rule(df_full, scaler_full, clf_full)
    print("\nBaseline (full-data) calibration reproduced from train_interior_model.py:")
    print(f"  {baseline_rule}")
    assert baseline_rule is not None
    assert abs(baseline_rule["dist_thr"] - PROD_DIST_THR) < 1e-6
    assert abs(baseline_rule["bri_thr"] - PROD_BRI_THR) < 1e-6
    assert abs(baseline_rule["floor"] - PROD_FLOOR) < 1e-9
    print("  -> matches documented production numbers exactly. Good, proceeding.")

    # Full-data LOO-refit probability for each known negative (used as the
    # reference "would a proper generalization test reject this?" signal in
    # both parts of this experiment).
    loo_models_full = {}
    loo_proba_neg_full = np.zeros(len(neg_rows_full))
    for i, r in neg_rows_full.iterrows():
        sc, clf = loo_refit_excluding(df_full, r["SourceImage"], r["Label"])
        loo_models_full[i] = (sc, clf)
        loo_proba_neg_full[i] = clf.predict_proba(
            sc.transform(neg_rows_full.iloc[[i]][INTERIOR_FEATURE_COLUMNS].values))[0, 0 + 1]
    print("\nFull-data LOO-refit probabilities for the 2 known negatives (unperturbed):")
    for i, r in neg_rows_full.iterrows():
        d, b = r["MeanDistToCrack"], r["MeanFlatBrightness"]
        dist_pass = d <= PROD_DIST_THR
        bri_pass = b <= PROD_BRI_THR
        floor_pass = loo_proba_neg_full[i] >= PROD_FLOOR
        which_leg_rejects = []
        if not dist_pass:
            which_leg_rejects.append("distance")
        if not bri_pass:
            which_leg_rejects.append("brightness")
        if not floor_pass:
            which_leg_rejects.append("floor")
        margin_dist_pct = (PROD_DIST_THR - d) / PROD_DIST_THR * 100
        margin_bri_pct = (PROD_BRI_THR - b) / PROD_BRI_THR * 100
        margin_floor = PROD_FLOOR - loo_proba_neg_full[i]
        print(f"  {r['SourceImage']}: LOO_proba={loo_proba_neg_full[i]:.4f}  "
              f"dist_pass={dist_pass} (margin {margin_dist_pct:+.1f}% of thr)  "
              f"bri_pass={bri_pass} (margin {margin_bri_pct:+.1f}% of thr)  "
              f"floor_pass={floor_pass} (margin {margin_floor:+.3f})  "
              f"-> rejected via leg(s): {which_leg_rejects}")

    # ======================================================================
    # PART 1: leave-one-image-out stability across 5 random 80% subsets
    # ======================================================================
    print("\n" + "=" * 90)
    print("PART 1: Leave-one-image-out stability (5 random ~80% subsets of 22 labeled images)")
    print("=" * 90)

    n_keep = round(SUBSET_FRAC * n_images)
    print(f"Each subset keeps {n_keep} of {n_images} images (drops {n_images - n_keep}).\n")

    subset_results = []
    for seed in RNG_SEEDS:
        rng = np.random.RandomState(seed)
        subset_images = set(rng.choice(all_images, size=n_keep, replace=False))
        df_sub = df_full[df_full["SourceImage"].isin(subset_images)].reset_index(drop=True)

        interior_sub = df_sub[df_sub["CandidateType"] == "interior_fill"]
        y_sub = interior_sub["IsCrack"].astype(bool)
        n_neg_sub = int((~y_sub).sum())
        n_pos_sub = int(y_sub.sum())
        neg_images_present = [r["SourceImage"] for _, r in neg_rows_full.iterrows()
                               if r["SourceImage"] in subset_images]

        print(f"--- seed {seed}: {len(subset_images)} images kept, "
              f"interior_fill labeled: {n_pos_sub} pos / {n_neg_sub} neg. "
              f"Known-negative images present: {neg_images_present or 'NONE'}")

        if n_neg_sub < 2:
            print(f"    -> DEGENERATE: fewer than 2 interior_fill negatives survive in this subset "
                  f"(calibrate_interior_fill_rule() returns None / falls back to no rule -- "
                  f"production would silently drop to a plain floor here). This IS itself a "
                  f"stability finding: losing either negative's image collapses the calibration.")
            subset_results.append(dict(seed=seed, degenerate=True, n_neg_sub=n_neg_sub,
                                        n_pos_sub=n_pos_sub))
            continue

        scaler_sub, clf_sub = fit_pooled(df_sub)
        rule_sub = calibrate_interior_fill_rule(df_sub, scaler_sub, clf_sub)
        if rule_sub is None:
            print("    -> Grid search found NO config rejecting both subset negatives (rule=None).")
            subset_results.append(dict(seed=seed, degenerate=True, n_neg_sub=n_neg_sub,
                                        n_pos_sub=n_pos_sub, grid_empty=True))
            continue

        # Evaluate the subset-derived rule against the FULL dataset (all 39
        # positives, full pool) using the SUBSET's own scaler/clf, to see how
        # a calibration born from a different 80% of images would perform
        # if deployed on the real full population.
        full_pos_proba_sub_model = score(pos_rows_full, scaler_sub, clf_sub)
        full_pos_rule_ok = (pos_rows_full["MeanDistToCrack"].values <= rule_sub["dist_thr"]) & \
                            (pos_rows_full["MeanFlatBrightness"].values <= rule_sub["bri_thr"])
        recall_on_full_39 = float((full_pos_rule_ok & (full_pos_proba_sub_model >= rule_sub["floor"])).mean())

        full_pool_proba_sub_model = score(pool, scaler_sub, clf_sub)
        full_pool_accept_sub_model = float(((pool["MeanDistToCrack"].values <= rule_sub["dist_thr"]) &
                                             (pool["MeanFlatBrightness"].values <= rule_sub["bri_thr"]) &
                                             (full_pool_proba_sub_model >= rule_sub["floor"])).mean())

        # The critical safety check: apply the subset-derived FIXED cutoffs
        # (dist_thr, bri_thr, floor) to the 2 known negatives, using the
        # proper FULL-DATA LOO-refit probability (i.e. "if this calibration
        # had been the one shipped, would it still reject both real known
        # negatives under a legitimate generalization test?").
        would_reject_both = True
        per_neg_detail = []
        for i, r in neg_rows_full.iterrows():
            d, b = r["MeanDistToCrack"], r["MeanFlatBrightness"]
            dist_pass = d <= rule_sub["dist_thr"]
            bri_pass = b <= rule_sub["bri_thr"]
            floor_pass = loo_proba_neg_full[i] >= rule_sub["floor"]
            accepted = dist_pass and bri_pass and floor_pass
            if accepted:
                would_reject_both = False
            per_neg_detail.append((r["SourceImage"], accepted))

        print(f"    -> rule: N={rule_sub['N_pct']} M={rule_sub['M_pct']} floor={rule_sub['floor']:.2f}  "
              f"dist_thr={rule_sub['dist_thr']:.2f}px  bri_thr={rule_sub['bri_thr']:.2f}")
        print(f"    -> in-subset recall={rule_sub['recall_on_known_positives']*100:.1f}%  "
              f"in-subset pool_accept={rule_sub['full_pool_accept_rate']*100:.1f}%")
        print(f"    -> applied to FULL data (39 pos / full pool) with subset's own model: "
              f"recall={recall_on_full_39*100:.1f}%  pool_accept={full_pool_accept_sub_model*100:.1f}%")
        print(f"    -> would this subset's rule reject BOTH known negatives (full-data LOO test)? "
              f"{would_reject_both}  detail={per_neg_detail}")

        subset_results.append(dict(
            seed=seed, degenerate=False, n_neg_sub=n_neg_sub, n_pos_sub=n_pos_sub,
            N_pct=rule_sub["N_pct"], M_pct=rule_sub["M_pct"], floor=rule_sub["floor"],
            dist_thr=rule_sub["dist_thr"], bri_thr=rule_sub["bri_thr"],
            in_subset_recall=rule_sub["recall_on_known_positives"],
            in_subset_pool_accept=rule_sub["full_pool_accept_rate"],
            recall_on_full_39=recall_on_full_39,
            full_pool_accept_sub_model=full_pool_accept_sub_model,
            would_reject_both_negatives=would_reject_both,
        ))

    ok_results = [r for r in subset_results if not r.get("degenerate")]
    n_degenerate = sum(1 for r in subset_results if r.get("degenerate"))
    print(f"\nSummary across {len(RNG_SEEDS)} subsets: {n_degenerate} degenerate "
          f"(lost calibrating power), {len(ok_results)} produced a rule.")
    if ok_results:
        for key in ["N_pct", "M_pct", "floor", "dist_thr", "bri_thr", "recall_on_full_39",
                    "full_pool_accept_sub_model"]:
            vals = [r[key] for r in ok_results]
            print(f"  {key:28s}: values={[round(v, 3) if isinstance(v, float) else v for v in vals]}  "
                  f"mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  range=[{min(vals):.3f}, {max(vals):.3f}]")
        n_safe = sum(1 for r in ok_results if r["would_reject_both_negatives"])
        print(f"  subsets whose rule rejects BOTH known negatives under full-data LOO: "
              f"{n_safe}/{len(ok_results)}")

    # ======================================================================
    # PART 2: sensitivity analysis via feature perturbation
    # ======================================================================
    print("\n" + "=" * 90)
    print("PART 2: Sensitivity analysis -- perturb MeanDistToCrack / MeanFlatBrightness "
          "of each known negative")
    print("=" * 90)
    print("(MeanDistToCrack and MeanFlatBrightness are also ML input features, so a perturbation "
          "changes both the corresponding rule leg AND the LOO-refit ML probability.)\n")

    sensitivity_rows = []
    for i, r in neg_rows_full.iterrows():
        sc, clf = loo_models_full[i]
        base_row = interior_lab_full[(interior_lab_full["SourceImage"] == r["SourceImage"]) &
                                      (interior_lab_full["Label"] == r["Label"])].iloc[0]
        base_feat = base_row[INTERIOR_FEATURE_COLUMNS].values.astype(float).copy()
        dist_idx = INTERIOR_FEATURE_COLUMNS.index("MeanDistToCrack")
        bri_idx = INTERIOR_FEATURE_COLUMNS.index("MeanFlatBrightness")

        print(f"--- Negative: {r['SourceImage']} (Label={r['Label']}), "
              f"baseline dist={base_feat[dist_idx]:.3f}, bri={base_feat[bri_idx]:.3f}, "
              f"baseline LOO_proba={loo_proba_neg_full[i]:.4f}")

        def evaluate(perturbed_feat, tag):
            proba = clf.predict_proba(sc.transform(perturbed_feat.reshape(1, -1)))[0, 1]
            d, b = perturbed_feat[dist_idx], perturbed_feat[bri_idx]
            dist_pass = d <= PROD_DIST_THR
            bri_pass = b <= PROD_BRI_THR
            floor_pass = proba >= PROD_FLOOR
            accepted = dist_pass and bri_pass and floor_pass  # True = rule FAILS to reject (bad)
            row = dict(negative=r["SourceImage"], perturbation=tag, dist=d, bri=b, proba=proba,
                       dist_pass=dist_pass, bri_pass=bri_pass, floor_pass=floor_pass,
                       still_rejected=not accepted)
            sensitivity_rows.append(row)
            flag = "" if not accepted else "  <<< RULE WOULD NOW ACCEPT (FAILS TO REJECT)"
            print(f"    {tag:28s} dist={d:8.3f} (pass={dist_pass})  bri={b:8.3f} (pass={bri_pass})  "
                  f"proba={proba:.4f} (floor_pass={floor_pass}){flag}")
            return not accepted

        # baseline (0% perturbation) for reference
        evaluate(base_feat.copy(), "baseline (0%)")

        for pct in PERTURB_PCTS:
            factor = 1 + pct / 100.0
            # dist only
            f = base_feat.copy(); f[dist_idx] = base_feat[dist_idx] * factor
            evaluate(f, f"dist {pct:+d}% only")
            # bri only
            f = base_feat.copy(); f[bri_idx] = base_feat[bri_idx] * factor
            evaluate(f, f"bri {pct:+d}% only")
            # both together
            f = base_feat.copy()
            f[dist_idx] = base_feat[dist_idx] * factor
            f[bri_idx] = base_feat[bri_idx] * factor
            evaluate(f, f"both {pct:+d}% together")
        print()

    sens_df = pd.DataFrame(sensitivity_rows)
    failures = sens_df[~sens_df["still_rejected"]]
    print("Sensitivity summary: perturbations that flip a known negative from rejected to accepted:")
    if len(failures) == 0:
        print("  NONE across the tested +/-5/10/20% grid (one-at-a-time and combined).")
    else:
        print(failures[["negative", "perturbation", "dist", "bri", "proba"]].to_string(index=False))

    # Find the smallest-magnitude perturbation (in the risky direction) that
    # flips each negative, by testing a finer grid between the coarse points
    # that bracket the flip, for reporting an exact breakeven number.
    print("\nBreak-even search (finer grid) for the negative(s) that flip within the tested range:")
    for i, r in neg_rows_full.iterrows():
        sc, clf = loo_models_full[i]
        base_row = interior_lab_full[(interior_lab_full["SourceImage"] == r["SourceImage"]) &
                                      (interior_lab_full["Label"] == r["Label"])].iloc[0]
        base_feat = base_row[INTERIOR_FEATURE_COLUMNS].values.astype(float).copy()
        dist_idx = INTERIOR_FEATURE_COLUMNS.index("MeanDistToCrack")
        bri_idx = INTERIOR_FEATURE_COLUMNS.index("MeanFlatBrightness")

        for feat_name, idx, direction_desc in [("MeanDistToCrack", dist_idx, "decrease (closer)"),
                                                 ("MeanFlatBrightness", bri_idx, "increase (brighter)")]:
            # risky direction: whichever direction pushes the currently-failing
            # leg (if any) toward passing, or -- if both legs already pass --
            # test both directions defensively.
            for pct in np.arange(0, 30.01, 0.5):
                for sign in (-1, 1):
                    factor = 1 + sign * pct / 100.0
                    f = base_feat.copy()
                    f[idx] = base_feat[idx] * factor
                    proba = clf.predict_proba(sc.transform(f.reshape(1, -1)))[0, 1]
                    dist_pass = f[dist_idx] <= PROD_DIST_THR
                    bri_pass = f[bri_idx] <= PROD_BRI_THR
                    floor_pass = proba >= PROD_FLOOR
                    if dist_pass and bri_pass and floor_pass:
                        print(f"  {r['SourceImage']}: perturbing {feat_name} by {sign*pct:+.1f}% "
                              f"({base_feat[idx]:.3f} -> {f[idx]:.3f}) is the first point (scanning "
                              f"outward from 0%) where the rule flips to ACCEPT.")
                        break
                else:
                    continue
                break
            else:
                print(f"  {r['SourceImage']}: perturbing {feat_name} alone by up to +/-30% never "
                      f"flips the rule to accept.")

    print("\nDone.")


if __name__ == "__main__":
    main()
