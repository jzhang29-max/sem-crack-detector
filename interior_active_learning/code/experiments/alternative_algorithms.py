"""
Experiment: alternative_algorithms
-----------------------------------
Test RandomForestClassifier, GradientBoostingClassifier, and SVC (rbf) as drop-in
replacements for the production pooled LogisticRegression(class_weight="balanced")
baseline used to decide whether an "interior candidate" region should count as crack.

Protocol (kept identical to production baseline for comparability):
  - Pool all 3 candidate types (concavity, bridge_corridor, interior_fill) together.
  - StandardScaler -> classifier.
  - StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) grouped by SourceImage.
  - Report mean balanced_accuracy across the 5 folds (comparable to baseline's 0.729).

Then, using each classifier refit on ALL labeled data:
  - Score the FULL interior_fill candidate pool (labeled + unlabeled rows of that type)
    and report the fraction accepted (predicted probability >= 0.5), compared against the
    baseline's reported 64%.
  - Same full-pool acceptance-rate check for concavity and bridge_corridor (should stay
    near baseline's ~12%, not collapse to ~0%).
  - Leave-one-out-style sanity check: refit each model on all labeled data EXCEPT one of
    the 2 known interior_fill negatives, and check whether it scores the held-out negative
    below 0.5. Repeat for both negatives. This is a concrete, falsifiable check (not proof
    of generalization) given there are only 2 known bad interior_fill examples in existence.
  - Also report training-set (in-sample) accuracy per classifier as an overfitting signal:
    a big gap between near-perfect training accuracy and unstable/mediocre CV accuracy is a
    red flag for RandomForest/GradientBoosting given the tiny interior_fill negative class.
"""
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# Project root derived from this file's location, not hardcoded: these scripts
# shipped with an absolute /Users/... path and could only run on the machine that
# wrote them, while archive/README.md advertises them as rerunnable.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CANDIDATES_GLOB = os.path.join(_ROOT, "interior_active_learning", "candidates", "*_interior.csv")

FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
            "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
            "FracBoundaryTouchingCrack", "MeanDistToCrack"]


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


def load_all_rows():
    """Load ALL rows (labeled or not) from all CSVs, tagging SourceImage from filename
    if not already present, for full-pool acceptance-rate computation."""
    rows = []
    for f in sorted(glob.glob(CANDIDATES_GLOB)):
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    df_all_labeled = load_labeled_interior()
    df_full_pool = load_all_rows()

    print(f"Total labeled rows (all types incl. user_painted): {len(df_all_labeled)}")
    print(f"CandidateType counts (labeled):\n{df_all_labeled['CandidateType'].value_counts()}\n")

    # Drop user_painted (always-true, no discriminative signal) for training/eval
    df = df_all_labeled[df_all_labeled["CandidateType"] != "user_painted"].copy()
    print(f"Rows after dropping user_painted: {len(df)}")
    print(df.groupby("CandidateType")["IsCrack"].agg(["count", "sum"]))
    print()

    X = df[FEATURES].values
    y = df["IsCrack"].values.astype(int)
    groups = df["SourceImage"].values
    ctype = df["CandidateType"].values

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)

    models = {
        "LogisticRegression_baseline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(class_weight="balanced", n_estimators=300,
                                            max_depth=4, random_state=0)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=0)),
        ]),
        "SVC_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=0)),
        ]),
    }

    results = {}

    for name, pipe in models.items():
        print("=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)

        fold_bal_accs = []
        fold_details = []
        for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            bal_acc = balanced_accuracy_score(y_test, preds)
            fold_bal_accs.append(bal_acc)

            # per-fold breakdown for transparency (esp. interior_fill instability)
            test_types = ctype[test_idx]
            detail = {}
            for t in np.unique(test_types):
                mask = test_types == t
                if mask.sum() == 0:
                    continue
                detail[t] = (int(mask.sum()), int(y_test[mask].sum()))
            fold_details.append((fold_i, bal_acc, detail))

        mean_bal_acc = float(np.mean(fold_bal_accs))
        std_bal_acc = float(np.std(fold_bal_accs))
        print(f"Per-fold balanced_accuracy: {[round(a, 3) for a in fold_bal_accs]}")
        for fold_i, bal_acc, detail in fold_details:
            print(f"  fold {fold_i}: bal_acc={bal_acc:.3f}  test-set composition (n, n_true) per type: {detail}")
        print(f"Mean balanced_accuracy across {n_splits} folds: {mean_bal_acc:.4f} (std {std_bal_acc:.4f})")

        # Refit on ALL labeled data for full-pool acceptance rate + training accuracy signal
        pipe.fit(X, y)
        train_preds = pipe.predict(X)
        train_acc = accuracy_score(y, train_preds)
        train_bal_acc = balanced_accuracy_score(y, train_preds)
        print(f"In-sample (train-on-all) accuracy: {train_acc:.4f}, balanced_accuracy: {train_bal_acc:.4f}")

        # Full-pool acceptance rate per candidate type
        acceptance = {}
        for t in ["interior_fill", "concavity", "bridge_corridor"]:
            pool_t = df_full_pool[df_full_pool["CandidateType"] == t].copy()
            if len(pool_t) == 0:
                continue
            Xt = pool_t[FEATURES].values
            probs = pipe.predict_proba(Xt)[:, 1]
            accept_rate = float((probs >= 0.5).mean())
            acceptance[t] = (accept_rate, len(pool_t))
            print(f"Full-pool acceptance rate @0.5 for {t}: {accept_rate:.3f} (n={len(pool_t)})")

        # Leave-one-out sanity check on the 2 known interior_fill negatives
        neg_mask = (df["CandidateType"] == "interior_fill") & (~df["IsCrack"])
        neg_idx = df.index[neg_mask].tolist()
        loo_results = []
        print(f"Known interior_fill negatives: {len(neg_idx)} (row indices: {neg_idx})")
        for hold_idx in neg_idx:
            train_mask = df.index != hold_idx
            X_loo_train = df.loc[train_mask, FEATURES].values
            y_loo_train = df.loc[train_mask, "IsCrack"].values.astype(int)
            X_held = df.loc[[hold_idx], FEATURES].values

            loo_pipe = Pipeline(pipe.steps)  # fresh clone with same hyperparams
            from sklearn.base import clone
            loo_pipe = clone(pipe)
            loo_pipe.fit(X_loo_train, y_loo_train)
            prob_held = loo_pipe.predict_proba(X_held)[:, 1][0]
            correct = prob_held < 0.5
            loo_results.append((hold_idx, float(prob_held), bool(correct)))
            print(f"  LOO holdout row {hold_idx}: predicted P(crack)={prob_held:.3f} -> "
                  f"{'CORRECTLY scored low (<0.5)' if correct else 'WRONG: scored >=0.5'}")

        results[name] = {
            "mean_bal_acc": mean_bal_acc,
            "std_bal_acc": std_bal_acc,
            "train_acc": train_acc,
            "acceptance": acceptance,
            "loo_results": loo_results,
        }
        print()

    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Model':28s} {'PooledBalAcc':>13s} {'std':>6s} {'TrainAcc':>9s} {'IF_accept':>10s} {'CC_accept':>10s} {'BC_accept':>10s}"
    print(header)
    for name, r in results.items():
        if_acc = r["acceptance"].get("interior_fill", (float("nan"), 0))[0]
        cc_acc = r["acceptance"].get("concavity", (float("nan"), 0))[0]
        bc_acc = r["acceptance"].get("bridge_corridor", (float("nan"), 0))[0]
        print(f"{name:28s} {r['mean_bal_acc']:13.4f} {r['std_bal_acc']:6.3f} {r['train_acc']:9.4f} "
              f"{if_acc:10.3f} {cc_acc:10.3f} {bc_acc:10.3f}")

    print()
    print("LOO known-negative results (prob < 0.5 desired):")
    for name, r in results.items():
        print(f"  {name}: {r['loo_results']}")


if __name__ == "__main__":
    main()
