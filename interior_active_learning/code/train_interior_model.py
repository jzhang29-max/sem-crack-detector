"""
Train a classifier on whatever interior candidates have been manually
labeled so far (candidates/*_interior.csv, IsCrack column filled in by
ingest_labels.py). Same model family (Logistic Regression) as the main
project's region classifier, for the same reason it was preferred there:
robust on a modest amount of labeled data, doesn't quietly overfit
small/noisy feature interactions the way a deeper RandomForest can.

Also calibrates a SEPARATE, stricter acceptance rule for interior_fill
candidates specifically -- see calibrate_interior_fill_rule()'s docstring.
Tested against 4 alternatives (per-type models, RandomForest/GradientBoosting/
SVM, aggressive reweighting+calibration, hand-engineered features) via a
multi-agent experiment; this hybrid rule was the only one that passed a
leave-one-out check on both known interior_fill negatives without degrading
concavity/bridge_corridor's own (already-reasonable) acceptance behavior.
"""
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CANDIDATES_DIR, MODELS_DIR
from active_learning_select import INTERIOR_FEATURE_COLUMNS


def load_labeled_interior():
    rows = []
    for f in sorted(glob.glob(os.path.join(CANDIDATES_DIR, "*_interior.csv"))):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        # IsCrack is meant to hold Python True/False, but an all-unlabeled
        # column round-trips through CSV as float64 NaN, and concatenating
        # real bools onto that (apply_paint_annotations.py's ingest())
        # silently downcasts them to 1.0/0.0 -- confirmed this actually
        # happened and hid 20 real labels (9 of them negative) from every
        # training run with no error. Accept numeric 1/0 as well as the
        # string form so a recurrence of that (or a similarly-corrupted
        # file elsewhere) can't silently vanish from training again.
        as_str = d["IsCrack"].astype(str).str.strip().str.upper()
        is_true = as_str.isin(["TRUE", "1", "1.0"])
        is_false = as_str.isin(["FALSE", "0", "0.0"])
        mask = is_true | is_false
        if not mask.any():
            continue
        labeled = d[mask].copy()
        labeled["IsCrack"] = is_true[mask].values
        rows.append(labeled)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_full_interior_fill_pool():
    """ALL interior_fill rows (labeled or not) across every image -- used to
    define the percentile cutoffs in calibrate_interior_fill_rule(), so the
    rule adapts to whatever the candidate-generation code is currently
    proposing rather than being frozen to magic numbers from one snapshot."""
    rows = []
    for f in sorted(glob.glob(os.path.join(CANDIDATES_DIR, "*_interior.csv"))):
        d = pd.read_csv(f)
        if len(d):
            rows.append(d[d["CandidateType"] == "interior_fill"])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calibrate_interior_fill_rule(df_labeled, scaler, clf):
    """interior_fill candidates propose area by following a brightness
    gradient outward from a crack with NO sharp edge (see
    interior_fill_candidates()'s own docstring in interior_candidates.py) --
    the ambiguous boundary this creates means interior_fill needs far more
    negative examples than concavity/bridge_corridor to train a reliable
    plain ML decision boundary. As of this writing there are only 2 negative
    interior_fill examples in the whole dataset (vs 39 positive), so the
    pooled classifier alone accepts ~95% of all interior_fill candidates --
    confirmed as the dominant source of visibly over-extended crack regions.

    Rather than trust the ML boundary alone for this one type, require ALL
    of: (1) ML probability >= a floor, (2) MeanDistToCrack under the Nth
    percentile of the full interior_fill candidate pool, (3)
    MeanFlatBrightness under the Mth percentile of that same pool -- i.e.
    genuinely close to the crack AND still relatively dark, not just one or
    the other (close-but-bright is uninteresting background right next to
    the crack; dark-but-far is probably some unrelated dark structure).
    N/M/floor are grid-searched to maximize recall on the known positives
    subject to rejecting BOTH known negatives under LEAVE-ONE-OUT-refit
    probability (not in-sample probability, which an earlier version of this
    search used and which let a config through that only worked because the
    model had memorized that exact negative during training).

    Returns None (caller should skip the rule entirely, falling back to the
    plain floor-only threshold) if there are fewer than 2 negative examples
    to validate against, or if no grid point rejects them both."""
    interior_lab = df_labeled[df_labeled["CandidateType"] == "interior_fill"].reset_index(drop=True)
    y_interior = interior_lab["IsCrack"].astype(bool).values
    pos_mask, neg_mask = y_interior, ~y_interior
    if neg_mask.sum() < 2:
        return None

    pool = load_full_interior_fill_pool()
    if len(pool) == 0:
        return None
    dist_pool = pool["MeanDistToCrack"].values
    bri_pool = pool["MeanFlatBrightness"].values

    def score(sub_df, sc, model):
        return model.predict_proba(sc.transform(sub_df[INTERIOR_FEATURE_COLUMNS].values))[:, 1]

    full_fit_proba = score(interior_lab, scaler, clf)

    # Leave-one-out probability for each known negative -- refit excluding
    # just that row, so the grid search is constrained by "would this
    # generalize" rather than "did the model memorize this."
    neg_rows = interior_lab[neg_mask].reset_index(drop=True)
    loo_proba_neg = np.zeros(len(neg_rows))
    for i in range(len(neg_rows)):
        neg_row = neg_rows.iloc[[i]]
        keep = ~((df_labeled["SourceImage"] == neg_row["SourceImage"].values[0]) &
                 (df_labeled["Label"].values == neg_row["Label"].values[0]))
        X_loo = df_labeled.loc[keep, INTERIOR_FEATURE_COLUMNS].values
        y_loo = df_labeled.loc[keep, "IsCrack"].astype(bool).values
        sc_loo = StandardScaler().fit(X_loo)
        clf_loo = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc_loo.transform(X_loo), y_loo)
        loo_proba_neg[i] = clf_loo.predict_proba(sc_loo.transform(neg_row[INTERIOR_FEATURE_COLUMNS].values))[0, 1]

    grid = []
    for N in range(5, 71, 5):
        dist_thr = np.percentile(dist_pool, N)
        for M in range(20, 91, 5):
            bri_thr = np.percentile(bri_pool, M)  # capped below 100 so the AND is always genuinely binding
            neg_rule_ok = (neg_rows["MeanDistToCrack"].values <= dist_thr) & \
                          (neg_rows["MeanFlatBrightness"].values <= bri_thr)
            pos_rule_ok = (interior_lab["MeanDistToCrack"].values[pos_mask] <= dist_thr) & \
                          (interior_lab["MeanFlatBrightness"].values[pos_mask] <= bri_thr)
            for floor in np.arange(0.10, 0.86, 0.05):
                if (neg_rule_ok & (loo_proba_neg >= floor)).any():
                    continue  # would accept a known negative -- reject this config
                recall_pos = (pos_rule_ok & (full_fit_proba[pos_mask] >= floor)).mean()
                full_pool_accept = ((dist_pool <= dist_thr) & (bri_pool <= bri_thr) &
                                     (score(pool, scaler, clf) >= floor)).mean()
                grid.append((N, M, round(float(floor), 2), recall_pos, full_pool_accept))

    if not grid:
        return None
    # highest recall first, then lowest full-pool acceptance, then prefer a
    # genuinely-binding (non-vacuous) brightness cutoff, then simpler/looser
    grid.sort(key=lambda t: (-t[3], t[4], -t[0], -t[1], t[2]))
    N_best, M_best, floor_best, recall_best, poolacc_best = grid[0]
    return {
        "dist_thr": float(np.percentile(dist_pool, N_best)),
        "bri_thr": float(np.percentile(bri_pool, M_best)),
        "floor": floor_best,
        "N_pct": N_best, "M_pct": M_best,
        "recall_on_known_positives": float(recall_best),
        "full_pool_accept_rate": float(poolacc_best),
    }


def main(min_per_class=8):
    df = load_labeled_interior()
    if len(df) == 0:
        print("No labeled interior candidates yet. Run active_learning_select.py, label the "
              "review sheets, then ingest_labels.py before training.")
        return

    n_pos, n_neg = int(df["IsCrack"].sum()), int((~df["IsCrack"]).sum())
    print(f"Loaded {len(df)} labeled interior candidates from {df['SourceImage'].nunique()} images "
          f"({n_pos} True, {n_neg} False)")
    if n_pos < min_per_class or n_neg < min_per_class:
        print(f"Need at least {min_per_class} of each class to train something meaningful -- "
              f"label more candidates first (see active_learning_select.py).")
        return

    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    X = df[INTERIOR_FEATURE_COLUMNS].values
    y = df["IsCrack"].astype(bool).values
    groups = df["SourceImage"].values

    n_groups = df["SourceImage"].nunique()
    n_folds = min(5, n_groups) if n_groups >= 2 else None

    if n_folds and n_folds >= 2:
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=0)
        accs, baccs = [], []
        for train_idx, test_idx in sgkf.split(Xs, y, groups):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Xs[train_idx], y[train_idx])
            pred = clf.predict(Xs[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))
            baccs.append(balanced_accuracy_score(y[test_idx], pred))
        print(f"{n_folds}-fold grouped CV: acc mean={np.mean(accs):.3f} std={np.std(accs):.3f}, "
              f"balanced_acc mean={np.mean(baccs):.3f}")
    else:
        print("Too few distinct source images for grouped cross-validation yet -- "
              "training on everything without a held-out estimate. Label candidates "
              "from more images before trusting the model's accuracy.")

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X), y)

    coefs = sorted(zip(INTERIOR_FEATURE_COLUMNS, clf.coef_[0]), key=lambda t: -abs(t[1]))
    print("\nStandardized coefficients:")
    for name, coef in coefs:
        print(f"  {name:28s} {coef:+.3f}")

    print("\nCalibrating interior_fill-specific acceptance rule "
          "(plain threshold isn't trustworthy for this type -- see docstring)...")
    interior_fill_rule = calibrate_interior_fill_rule(df, scaler, clf)
    if interior_fill_rule is None:
        print("  Skipped -- fewer than 2 negative interior_fill examples labeled so far, or no "
              "candidate pool found. apply_interior_model.py will fall back to a plain 0.5 floor "
              "for interior_fill until more negatives are labeled (expect it to stay over-permissive).")
    else:
        print(f"  accept iff ML_proba >= {interior_fill_rule['floor']}  AND  "
              f"MeanDistToCrack <= {interior_fill_rule['dist_thr']:.2f}px "
              f"({interior_fill_rule['N_pct']}th pct of the full pool)  AND  "
              f"MeanFlatBrightness <= {interior_fill_rule['bri_thr']:.2f} "
              f"({interior_fill_rule['M_pct']}th pct of the full pool)")
        print(f"  recall on known positives: {interior_fill_rule['recall_on_known_positives']*100:.1f}%, "
              f"full interior_fill pool acceptance: {interior_fill_rule['full_pool_accept_rate']*100:.1f}%")

    out_path = os.path.join(MODELS_DIR, "interior_model.joblib")
    joblib.dump({
        "scaler": scaler, "clf": clf, "feature_names": INTERIOR_FEATURE_COLUMNS,
        "interior_fill_rule": interior_fill_rule,
    }, out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    # --help USED TO DO THE WORK. main() ignored argv, so asking for help ran a fit
    # and could write a bundle over the deployed one. Refuse before any of that.
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__ or "")
        print("usage: train_interior_model.py\n"
              "  no arguments. Fits the Pass 2 interior model and writes models/interior_model.joblib.")
        sys.exit(0)
    if len(sys.argv) > 1:
        print(f"unknown option {sys.argv[1]!r}. This script takes no arguments. "
              f"Use --help.")
        sys.exit(2)
    main()
