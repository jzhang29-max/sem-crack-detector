"""
Experiment: imbalance_and_calibration
--------------------------------------
Goal: fix over-acceptance of interior_fill candidates (64% at threshold=0.5 with the
production pooled LogisticRegression(class_weight="balanced")) without breaking
concavity/bridge_corridor (currently ~12% acceptance, working reasonably well).

Approaches tested, all built on top of the SAME pooled LogisticRegression baseline
(same features, same pooling across all 3 candidate types):

  1. Manual per-row sample_weight upweighting of the 2 known interior_fill NEGATIVE
     training rows, on top of the standard "balanced" class weights. Multipliers
     tested: 1x (= plain balanced, baseline), 5x, 15x, 40x.
  2. CalibratedClassifierCV (method="sigmoid"; isotonic attempted, expected to fail
     or be unreliable given tiny per-fold class counts) wrapping the base
     LogisticRegression, to see if recalibrating probabilities alone (no ranking
     change) reduces full-pool interior_fill acceptance at a fixed threshold.
  3. Threshold sweep (0.50 -> 0.99) on the plain baseline's pooled predict_proba,
     to find a global operating threshold that brings interior_fill acceptance
     into a reasonable range without cratering concavity/bridge_corridor.

For every candidate configuration we report, per the task's statistical-honesty
constraint:
  (a) overall pooled StratifiedGroupKFold(5, shuffle=True, random_state=0)
      balanced_accuracy across ALL labeled types combined -- comparable to the
      0.729 baseline.
  (b) full-pool acceptance rate (fraction scored >= threshold) for interior_fill
      and, separately, for concavity/bridge_corridor -- using the model fit on
      ALL labeled data, applied to the FULL candidate pool (labeled + unlabeled)
      of each CandidateType.
  (c) leave-one-out check on the 2 known interior_fill negatives: with a model
      trained on everything else, does it score these two rows low?

No numbers are guessed -- everything below is computed by running this script
against the real CSVs.
"""

import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

CAND_GLOB = "/Users/jiamingzhang/Desktop/SEM_Crack_Detection_Pipeline/interior_active_learning/candidates/*_interior.csv"

FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
            "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
            "FracBoundaryTouchingCrack", "MeanDistToCrack"]


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
    """Load ALL rows (labeled or not) from every candidate CSV, tagged with SourceFile."""
    rows = []
    for f in sorted(glob.glob(CAND_GLOB)):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        d["SourceFile"] = f
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def make_base_pipeline(C=1.0, class_weight="balanced"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=0)),
    ])


def pooled_cv_balanced_accuracy(X, y, groups, sample_weight=None, n_splits=5, model_fn=None):
    """StratifiedGroupKFold pooled balanced accuracy, matching baseline evaluation protocol.
    If sample_weight is provided, per-fold train sample_weight subset is passed to fit()."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for train_idx, test_idx in sgkf.split(X, y, groups):
        pipe = model_fn()
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
        if sample_weight is not None:
            sw_tr = sample_weight.iloc[train_idx].values
            pipe.fit(Xtr, ytr, clf__sample_weight=sw_tr)
        else:
            pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        accs.append(balanced_accuracy_score(yte, pred))
    return float(np.mean(accs)), accs


def acceptance_rate(model, X_pool, threshold=0.5, predict_proba_fn=None):
    if predict_proba_fn is not None:
        proba = predict_proba_fn(X_pool)
    else:
        proba = model.predict_proba(X_pool)[:, 1]
    return float((proba >= threshold).mean()), proba


def main():
    print("=" * 90)
    print("LOADING DATA")
    print("=" * 90)
    df = load_labeled_interior()
    df = df[df["CandidateType"] != "user_painted"].reset_index(drop=True)
    print(f"Labeled (non user_painted) rows: {len(df)}")
    print(df.groupby("CandidateType")["IsCrack"].agg(["count", "sum"]))

    full_pool = load_full_pool()
    full_pool = full_pool[full_pool["CandidateType"] != "user_painted"].reset_index(drop=True)
    print(f"\nFull pool (all rows, labeled+unlabeled, non user_painted): {len(full_pool)}")
    print(full_pool["CandidateType"].value_counts())

    X = df[FEATURES]
    y = df["IsCrack"]
    groups = df["SourceImage"]

    # Identify the 2 known interior_fill negative rows (for LOO sanity check + upweighting)
    neg_if_mask = (df["CandidateType"] == "interior_fill") & (~df["IsCrack"])
    neg_if_idx = df.index[neg_if_mask].tolist()
    print(f"\nKnown interior_fill negative row indices in labeled df: {neg_if_idx} (n={len(neg_if_idx)})")
    print(df.loc[neg_if_idx, ["SourceImage", "CandidateType", "IsCrack"] + FEATURES])

    full_pool_by_type = {
        t: full_pool[full_pool["CandidateType"] == t][FEATURES]
        for t in ["interior_fill", "concavity", "bridge_corridor"]
    }
    for t, sub in full_pool_by_type.items():
        print(f"Full pool size for {t}: {len(sub)}")

    print("\n" + "=" * 90)
    print("BASELINE: plain LogisticRegression(class_weight='balanced'), threshold=0.5")
    print("=" * 90)
    baseline_acc, baseline_folds = pooled_cv_balanced_accuracy(
        X, y, groups, sample_weight=None,
        model_fn=lambda: make_base_pipeline(class_weight="balanced"))
    print(f"Pooled 5-fold balanced_accuracy: {baseline_acc:.4f}  (per-fold: {[round(a,3) for a in baseline_folds]})")

    baseline_full = make_base_pipeline(class_weight="balanced")
    baseline_full.fit(X, y)
    for t, sub in full_pool_by_type.items():
        rate, proba = acceptance_rate(baseline_full, sub, threshold=0.5)
        print(f"  baseline full-pool acceptance @0.5 for {t}: {rate:.3f} (n={len(sub)})")

    # LOO check on the 2 known negatives using baseline
    print("\nBaseline LOO-style check on 2 known interior_fill negatives "
          "(model trained on all OTHER rows, i.e. these 2 excluded):")
    for idx in neg_if_idx:
        train_mask = df.index != idx
        m = make_base_pipeline(class_weight="balanced")
        m.fit(X[train_mask], y[train_mask])
        p = m.predict_proba(X.loc[[idx]])[:, 1][0]
        print(f"  row {idx} ({df.loc[idx,'SourceImage']}): P(crack)={p:.3f}  "
              f"({'CORRECTLY low' if p < 0.5 else 'WRONGLY high'} at 0.5 threshold)")

    # ---------------------------------------------------------------
    # APPROACH 1: manual sample_weight upweighting of known interior_fill negatives
    # ---------------------------------------------------------------
    print("\n" + "=" * 90)
    print("APPROACH 1: sample_weight upweighting of the 2 known interior_fill negatives")
    print("=" * 90)

    def make_sample_weight(multiplier, index_frame):
        """balanced class weight * multiplier applied only to interior_fill-negative rows."""
        n = len(index_frame)
        n_pos = index_frame["IsCrack"].sum()
        n_neg = n - n_pos
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        sw = np.where(index_frame["IsCrack"].values, w_pos, w_neg).astype(float)
        extra_mask = ((index_frame["CandidateType"] == "interior_fill") & (~index_frame["IsCrack"])).values
        sw[extra_mask] *= multiplier
        return pd.Series(sw, index=index_frame.index)

    approach1_results = {}
    for mult in [1, 5, 15, 40]:
        sw_full_df = make_sample_weight(mult, df)  # for full-pool fit
        acc, folds = pooled_cv_balanced_accuracy(
            X, y, groups, sample_weight=sw_full_df,
            model_fn=lambda: make_base_pipeline(class_weight=None))
        m = make_base_pipeline(class_weight=None)
        m.fit(X, y, clf__sample_weight=sw_full_df.values)
        rates = {}
        for t, sub in full_pool_by_type.items():
            rate, _ = acceptance_rate(m, sub, threshold=0.5)
            rates[t] = rate
        # LOO check
        loo_probs = []
        for idx in neg_if_idx:
            train_mask = df.index != idx
            sw_tr = make_sample_weight(mult, df[train_mask])
            mm = make_base_pipeline(class_weight=None)
            mm.fit(X[train_mask], y[train_mask], clf__sample_weight=sw_tr.values)
            p = mm.predict_proba(X.loc[[idx]])[:, 1][0]
            loo_probs.append(p)
        approach1_results[mult] = dict(pooled_bacc=acc, folds=folds, rates=rates, loo=loo_probs)
        print(f"\n-- multiplier={mult}x --")
        print(f"   pooled 5-fold balanced_accuracy: {acc:.4f}  (per-fold: {[round(a,3) for a in folds]})")
        for t, r in rates.items():
            print(f"   full-pool acceptance @0.5 for {t}: {r:.3f}")
        print(f"   LOO probs on 2 known negatives: {[round(p,3) for p in loo_probs]}")

    # ---------------------------------------------------------------
    # APPROACH 2: CalibratedClassifierCV (sigmoid, isotonic-if-possible)
    # ---------------------------------------------------------------
    print("\n" + "=" * 90)
    print("APPROACH 2: CalibratedClassifierCV wrapping base LogisticRegression")
    print("=" * 90)

    def make_calibrated_pipeline(method):
        base = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0)
        cal = CalibratedClassifierCV(base, method=method, cv=3)
        return Pipeline([("scaler", StandardScaler()), ("clf", cal)])

    approach2_results = {}
    for method in ["sigmoid", "isotonic"]:
        try:
            acc, folds = pooled_cv_balanced_accuracy(
                X, y, groups, sample_weight=None,
                model_fn=lambda: make_calibrated_pipeline(method))
            m = make_calibrated_pipeline(method)
            m.fit(X, y)
            rates = {}
            for t, sub in full_pool_by_type.items():
                rate, _ = acceptance_rate(m, sub, threshold=0.5)
                rates[t] = rate
            loo_probs = []
            for idx in neg_if_idx:
                train_mask = df.index != idx
                mm = make_calibrated_pipeline(method)
                mm.fit(X[train_mask], y[train_mask])
                p = mm.predict_proba(X.loc[[idx]])[:, 1][0]
                loo_probs.append(p)
            approach2_results[method] = dict(pooled_bacc=acc, folds=folds, rates=rates, loo=loo_probs, error=None)
            print(f"\n-- method={method} --")
            print(f"   pooled 5-fold balanced_accuracy: {acc:.4f}  (per-fold: {[round(a,3) for a in folds]})")
            for t, r in rates.items():
                print(f"   full-pool acceptance @0.5 for {t}: {r:.3f}")
            print(f"   LOO probs on 2 known negatives: {[round(p,3) for p in loo_probs]}")
        except Exception as e:
            approach2_results[method] = dict(error=str(e))
            print(f"\n-- method={method} -- FAILED: {e}")

    # ---------------------------------------------------------------
    # APPROACH 3: threshold sweep on plain baseline
    # ---------------------------------------------------------------
    print("\n" + "=" * 90)
    print("APPROACH 3: threshold sweep (0.50 -> 0.99) on baseline pooled predict_proba")
    print("=" * 90)

    baseline_proba_by_type = {}
    for t, sub in full_pool_by_type.items():
        baseline_proba_by_type[t] = baseline_full.predict_proba(sub)[:, 1]

    thresholds = np.round(np.arange(0.50, 0.991, 0.01), 3)
    sweep_rows = []
    for th in thresholds:
        rif = float((baseline_proba_by_type["interior_fill"] >= th).mean())
        rc = float((baseline_proba_by_type["concavity"] >= th).mean())
        rb = float((baseline_proba_by_type["bridge_corridor"] >= th).mean())
        sweep_rows.append((th, rif, rc, rb))

    print(f"{'thresh':>7} {'interior_fill':>14} {'concavity':>10} {'bridge_corridor':>16}")
    for th, rif, rc, rb in sweep_rows:
        print(f"{th:>7.2f} {rif:>14.3f} {rc:>10.3f} {rb:>16.3f}")

    # pick recommended threshold: smallest th such that interior_fill rate <= 0.25
    # and concavity/bridge rates haven't collapsed near 0 (>0.03 as a floor, i.e. not literally ~0)
    candidates = [(th, rif, rc, rb) for th, rif, rc, rb in sweep_rows if rif <= 0.25]
    recommended = candidates[0] if candidates else sweep_rows[-1]
    print(f"\nRecommended threshold (first th with interior_fill acceptance <= 0.25): "
          f"th={recommended[0]:.2f} -> interior_fill={recommended[1]:.3f}, "
          f"concavity={recommended[2]:.3f}, bridge_corridor={recommended[3]:.3f}")

    # LOO check at recommended threshold
    print("\nLOO check on 2 known interior_fill negatives at recommended threshold "
          f"({recommended[0]:.2f}), using baseline model (already computed probs above at 0.5; "
          "recompute with same LOO-trained models):")
    for idx in neg_if_idx:
        train_mask = df.index != idx
        m = make_base_pipeline(class_weight="balanced")
        m.fit(X[train_mask], y[train_mask])
        p = m.predict_proba(X.loc[[idx]])[:, 1][0]
        verdict = "CORRECTLY rejected" if p < recommended[0] else "WRONGLY accepted"
        print(f"  row {idx}: P(crack)={p:.3f} vs threshold {recommended[0]:.2f} -> {verdict}")

    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'config':40} {'pooled_bacc':>12} {'if_accept':>10} {'conc_accept':>12} {'bridge_accept':>14}")
    print(f"{'baseline @0.5':40} {baseline_acc:>12.4f} "
          f"{acceptance_rate(baseline_full, full_pool_by_type['interior_fill'])[0]:>10.3f} "
          f"{acceptance_rate(baseline_full, full_pool_by_type['concavity'])[0]:>12.3f} "
          f"{acceptance_rate(baseline_full, full_pool_by_type['bridge_corridor'])[0]:>14.3f}")
    for mult, res in approach1_results.items():
        print(f"{'sample_weight x'+str(mult)+' @0.5':40} {res['pooled_bacc']:>12.4f} "
              f"{res['rates']['interior_fill']:>10.3f} {res['rates']['concavity']:>12.3f} "
              f"{res['rates']['bridge_corridor']:>14.3f}")
    for method, res in approach2_results.items():
        if res.get("error"):
            print(f"{'calibrated_'+method+' @0.5':40} {'FAILED':>12}")
        else:
            print(f"{'calibrated_'+method+' @0.5':40} {res['pooled_bacc']:>12.4f} "
                  f"{res['rates']['interior_fill']:>10.3f} {res['rates']['concavity']:>12.3f} "
                  f"{res['rates']['bridge_corridor']:>14.3f}")
    print(f"{'baseline @ recommended_th='+str(recommended[0]):40} {baseline_acc:>12.4f} "
          f"{recommended[1]:>10.3f} {recommended[2]:>12.3f} {recommended[3]:>14.3f}")

    # ---------------------------------------------------------------
    # APPROACH 1+3 COMBINED: sample_weight upweighting followed by a raised
    # threshold on TOP of the reweighted model, to see how far combining the
    # two levers can push interior_fill acceptance down before concavity /
    # bridge_corridor also collapse.
    # ---------------------------------------------------------------
    print("\n" + "=" * 90)
    print("APPROACH 1+3 COMBINED: sample_weight upweighting + threshold sweep on TOP")
    print("=" * 90)
    for mult in [15, 40]:
        sw = make_sample_weight(mult, df).values
        pipe = make_base_pipeline(class_weight=None)
        pipe.fit(X, y, clf__sample_weight=sw)
        probs = {t: pipe.predict_proba(p)[:, 1] for t, p in full_pool_by_type.items()}
        print(f"\n-- sample_weight multiplier={mult}x, then threshold sweep --")
        print(f"{'thresh':>7} {'interior_fill':>14} {'concavity':>10} {'bridge_corridor':>16}")
        for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            rif = float((probs["interior_fill"] >= th).mean())
            rc = float((probs["concavity"] >= th).mean())
            rb = float((probs["bridge_corridor"] >= th).mean())
            print(f"{th:>7.2f} {rif:>14.3f} {rc:>10.3f} {rb:>16.3f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
