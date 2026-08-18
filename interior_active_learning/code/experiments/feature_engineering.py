"""
Feature-engineering experiment for the interior-candidate crack classifier.

Goal: fix the interior_fill over-acceptance problem (64% accepted at threshold=0.5
with the baseline pooled LogisticRegression) by engineering new features from the
EXISTING columns that might better separate "genuine crack extension" from
"gradual brightness fade-out that shouldn't count."

This script:
  1. Loads labeled data (concavity/bridge_corridor/interior_fill only, user_painted dropped).
  2. Builds an expanded feature set = 11 baseline features + engineered features.
  3. Refits pooled LogisticRegression(class_weight="balanced") with StandardScaler,
     evaluates via StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
     grouped by SourceImage -- directly comparable to the 0.729 baseline.
  4. Reports standardized coefficients to see which engineered features matter.
  5. Loads the FULL (unlabeled + labeled) interior_fill candidate pool and reports
     acceptance rate at threshold=0.5 for baseline features vs expanded features,
     plus the same check for concavity/bridge_corridor (should stay ~12%).
  6. Runs a leave-one-out sanity check on the 2 known interior_fill negatives.
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

# Project root derived from this file's location, not hardcoded: this script
# shipped with an absolute /Users/... path and could only run on the machine that
# wrote it, while archive/README.md advertises these as rerunnable.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CANDIDATES_GLOB = os.path.join(
    _ROOT, "interior_active_learning", "candidates", "*_interior.csv")

BASE_FEATURES = [
    "LogArea", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
    "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
    "FracBoundaryTouchingCrack", "MeanDistToCrack",
]

EPS = 1e-6


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


def load_all_interior():
    """Load ALL rows (labeled or not) from all CSVs, tagging source file."""
    rows = []
    for f in sorted(glob.glob(CANDIDATES_GLOB)):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        d["_source_file"] = f
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def engineer_features(df):
    """Add new candidate features derived from existing columns. Returns a copy."""
    df = df.copy()
    df["BrightnessRatio"] = df["MeanFlatBrightness"] / (df["MeanRawBrightness"] + EPS)
    df["LogDistToCrack"] = np.log(df["MeanDistToCrack"] + 1.0)
    df["DistToCrackSq"] = df["MeanDistToCrack"] ** 2
    df["FlatBrightness_x_Dist"] = df["MeanFlatBrightness"] * df["MeanDistToCrack"]
    df["Vesselness_over_Dist"] = df["MeanVesselness"] / (df["MeanDistToCrack"] + 1.0)
    df["BoundaryTouch_x_Vesselness"] = df["FracBoundaryTouchingCrack"] * df["MeanVesselness"]
    # A couple more targeted at "gradual fade-out": vesselness relative to raw
    # brightness (a real crack extension should have high vesselness even if dim;
    # a fade-out region has low vesselness relative to its brightness), and a
    # "boundary contact per unit distance" term meant to reward candidates that
    # are close to AND touching the confirmed crack (true extensions) versus far
    # AND barely touching (fade-outs / stray dim regions).
    df["Vesselness_over_RawBrightness"] = df["MeanVesselness"] / (df["MeanRawBrightness"] + EPS)
    df["BoundaryTouch_over_Dist"] = df["FracBoundaryTouchingCrack"] / (df["MeanDistToCrack"] + 1.0)
    return df


ENGINEERED_FEATURES = [
    "BrightnessRatio", "LogDistToCrack", "DistToCrackSq",
    "FlatBrightness_x_Dist", "Vesselness_over_Dist", "BoundaryTouch_x_Vesselness",
    "Vesselness_over_RawBrightness", "BoundaryTouch_over_Dist",
]

EXPANDED_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


def pooled_cv_balanced_accuracy(df, feature_cols, n_splits=5, seed=0):
    X = df[feature_cols].values
    y = df["IsCrack"].astype(int).values
    groups = df["SourceImage"].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in sgkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        clf.fit(Xtr, y[train_idx])
        pred = clf.predict(Xte)
        scores.append(balanced_accuracy_score(y[test_idx], pred))
    return np.array(scores)


def fit_full_model(df, feature_cols):
    X = df[feature_cols].values
    y = df["IsCrack"].astype(int).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=5000)
    clf.fit(Xs, y)
    return scaler, clf


def acceptance_rate(full_pool_df, feature_cols, scaler, clf, candidate_type, threshold=0.5):
    sub = full_pool_df[full_pool_df["CandidateType"] == candidate_type].copy()
    sub = sub.dropna(subset=feature_cols)
    if len(sub) == 0:
        return np.nan, 0
    X = sub[feature_cols].values
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)[:, 1]
    accept = (proba >= threshold).mean()
    return accept, len(sub)


def main():
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    labeled_all = load_labeled_interior()
    labeled_all = labeled_all[labeled_all["CandidateType"] != "user_painted"].copy()
    labeled_all = engineer_features(labeled_all)

    print(f"Total labeled (non-user_painted) rows: {len(labeled_all)}")
    print(labeled_all.groupby("CandidateType")["IsCrack"].agg(["count", "sum"]))

    full_pool = load_all_interior()
    full_pool = engineer_features(full_pool)
    print(f"\nTotal full pool rows (all CandidateTypes, labeled+unlabeled): {len(full_pool)}")
    print(full_pool["CandidateType"].value_counts())

    # ------------------------------------------------------------------
    # 1) Pooled CV balanced accuracy: baseline features vs expanded features
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("POOLED StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) CV")
    print("=" * 80)

    base_scores = pooled_cv_balanced_accuracy(labeled_all, BASE_FEATURES)
    print(f"\nBASELINE (11 features) fold balanced_accuracy: {np.round(base_scores, 4)}")
    print(f"BASELINE mean balanced_accuracy: {base_scores.mean():.4f} (reference target ~0.729)")

    expanded_scores = pooled_cv_balanced_accuracy(labeled_all, EXPANDED_FEATURES)
    print(f"\nEXPANDED ({len(EXPANDED_FEATURES)} features) fold balanced_accuracy: {np.round(expanded_scores, 4)}")
    print(f"EXPANDED mean balanced_accuracy: {expanded_scores.mean():.4f}")

    delta = expanded_scores.mean() - base_scores.mean()
    print(f"\nDelta (expanded - baseline): {delta:+.4f}")

    # ------------------------------------------------------------------
    # 2) Fit final model on ALL labeled data (expanded features), inspect coefficients
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL MODEL FIT ON ALL LABELED DATA (expanded features)")
    print("=" * 80)

    scaler_exp, clf_exp = fit_full_model(labeled_all, EXPANDED_FEATURES)
    coefs = clf_exp.coef_[0]
    coef_df = pd.DataFrame({
        "feature": EXPANDED_FEATURES,
        "standardized_coef": coefs,
        "abs_coef": np.abs(coefs),
        "is_engineered": [f in ENGINEERED_FEATURES for f in EXPANDED_FEATURES],
    }).sort_values("abs_coef", ascending=False)
    print(coef_df.to_string(index=False))

    top_engineered = coef_df[coef_df["is_engineered"]].sort_values("abs_coef", ascending=False)
    print("\nTop engineered features by |standardized coef|:")
    print(top_engineered.to_string(index=False))

    # Also fit baseline-features-only model on all labeled data, for comparison in acceptance rates
    scaler_base, clf_base = fit_full_model(labeled_all, BASE_FEATURES)

    # ------------------------------------------------------------------
    # 3) Full-pool acceptance rates at threshold=0.5
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FULL-POOL ACCEPTANCE RATES AT THRESHOLD=0.5")
    print("=" * 80)

    for ctype in ["interior_fill", "concavity", "bridge_corridor"]:
        acc_base, n_base = acceptance_rate(full_pool, BASE_FEATURES, scaler_base, clf_base, ctype)
        acc_exp, n_exp = acceptance_rate(full_pool, EXPANDED_FEATURES, scaler_exp, clf_exp, ctype)
        print(f"\n{ctype} (n={n_base} in full pool):")
        print(f"  baseline (11 feats)   acceptance rate @0.5: {acc_base:.3f}")
        print(f"  expanded ({len(EXPANDED_FEATURES)} feats)  acceptance rate @0.5: {acc_exp:.3f}")

    # ------------------------------------------------------------------
    # 4) Leave-one-out sanity check on the 2 known interior_fill negatives
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LEAVE-ONE-OUT CHECK ON KNOWN interior_fill NEGATIVES (expanded features)")
    print("=" * 80)

    neg_mask = (labeled_all["CandidateType"] == "interior_fill") & (~labeled_all["IsCrack"])
    neg_rows = labeled_all[neg_mask]
    print(f"\nFound {len(neg_rows)} known interior_fill negative examples.")

    for idx, row in neg_rows.iterrows():
        train_df = labeled_all.drop(index=idx)
        scaler_loo, clf_loo = fit_full_model(train_df, EXPANDED_FEATURES)
        x = row[EXPANDED_FEATURES].values.reshape(1, -1).astype(float)
        xs = scaler_loo.transform(x)
        proba = clf_loo.predict_proba(xs)[0, 1]

        # Also with baseline features, for comparison
        train_df_b = labeled_all.drop(index=idx)
        scaler_loo_b, clf_loo_b = fit_full_model(train_df_b, BASE_FEATURES)
        xb = row[BASE_FEATURES].values.reshape(1, -1).astype(float)
        xbs = scaler_loo_b.transform(xb)
        proba_b = clf_loo_b.predict_proba(xbs)[0, 1]

        print(f"\nRow index {idx} (SourceImage={row['SourceImage']}):")
        print(f"  P(crack) with BASELINE features (LOO):  {proba_b:.4f}  -> {'ACCEPT' if proba_b>=0.5 else 'reject'} @0.5")
        print(f"  P(crack) with EXPANDED features (LOO):  {proba:.4f}  -> {'ACCEPT' if proba>=0.5 else 'reject'} @0.5")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
