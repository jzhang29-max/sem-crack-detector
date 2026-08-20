"""Measure and record the deployed model's OUT-OF-SAMPLE score, once.

WHY THIS IS NEEDED
The retrain promotion gate compares the candidate's out-of-sample leave-one-image-out AUC
against the incumbent's. Both sides must be out-of-sample or the comparison is rigged:

  * The candidate's number comes from fitting on every labelled region EXCEPT the held-out
    image and scoring on it. Honest.
  * The incumbent's number used to come from production_on_held_out.auc, which is the
    deployed model scored on the image it was TRAINED ON. train_v3_weighted.py prints
    "optimistic -- it was trained with this image's labels" beside that number, and
    "in-sample and optimistic. Compare LOIO AUC instead."

On disk the difference is 0.9153 (out-of-sample) against 0.9535 (in-sample). Grading the
first against the second biases the gate to refuse EVERY retrain, forever and silently:
the researcher labels for weeks, clicks Retrain, and is told their work made the model
worse when nothing about their labels was measured.

So bundles now record `loio_out_of_sample` when they are promoted, and the gate refuses
when it is missing rather than falling back to the inflated number. That is correct, but a
model installed from elsewhere -- like the archived bundle currently deployed, which
carries only scaler/clf/feature_names/sklearn_version -- has no such record and would be
un-retrainable forever. This script establishes it.

WHAT IT MEASURES, AND WHAT IT DOES NOT
It refits the deployed model's ESTIMATOR FAMILY, with its hyperparameters, on every
labelled region except the held-out image, then scores that refit on the held-out image.
That is the same procedure the candidate goes through, so the two numbers are comparable.

It is NOT the deployed pickle's own coefficients scored out-of-sample -- that quantity does
not exist for a model already trained on the held-out image, and no amount of arithmetic
recovers it. The recorded value is therefore labelled `loio_out_of_sample_source:
"refit-same-family"` so nobody later mistakes it for a direct measurement of the shipped
weights.

SAFETY
The deployed bundle is only ever ADDED to. Before writing, the script asserts that the
estimator's predict_proba on the full matrix is bit-identical to before, so establishing a
baseline cannot change a single prediction. The write is atomic and the previous bundle is
kept with a timestamp.

    python3 code/establish_baseline.py            # measure and record
    python3 code/establish_baseline.py --dry-run  # measure and print only
"""
import argparse
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "interior_active_learning", "code"))
sys.path.insert(0, HERE)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import PROD_MODEL_PATH

CSV = os.path.join(ROOT, "training_data", "labeled_regions.csv")
FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity",
            "Extent", "Circularity", "MeanDarkness", "MeanVesselness"]
HELD = "AS_24hr_BSE_Side_008"


def image_weights(src):
    """Same per-image weighting the trainer uses, so the refit matches its procedure."""
    counts = pd.Series(src).value_counts()
    return np.array([1.0 / counts[s] for s in src])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--held", default=HELD)
    args = ap.parse_args()

    if not os.path.exists(CSV):
        print(f"no training data at {CSV}; run build_training_data.py first")
        return 1
    bundle = joblib.load(PROD_MODEL_PATH)
    if "loio_out_of_sample" in bundle:
        print(f"already recorded: loio_out_of_sample = "
              f"{bundle['loio_out_of_sample']:.4f} "
              f"({bundle.get('loio_out_of_sample_source', 'unknown source')})")
        print("nothing to do; delete the key first if you want to re-measure")
        return 0

    df = pd.read_csv(CSV)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"training data is missing feature columns: {missing}")
        return 1
    X = df[FEATURES].values
    y = df["IsCrack"].astype(bool).values
    groups = df["SourceImage"].values
    te = groups == args.held

    if not te.any():
        print(f"the held-out image {args.held!r} has no rows in {CSV}. "
              f"Images present: {sorted(set(groups))[:6]}...")
        return 1
    if len(set(y[te])) < 2:
        # roc_auc_score raises on a single class, and a traceback out of the Retrain
        # button is a worse outcome than a clear refusal here.
        print(f"{args.held} has only one class among its labels "
              f"({int(y[te].sum())} crack / {int((~y[te]).sum())} not-crack), so an AUC "
              f"cannot be computed. Mark both classes on it, or pass --held with an image "
              f"that has both.")
        return 1

    W = image_weights(groups)
    clf = bundle["clf"]
    family = type(clf).__name__
    params = clf.get_params()

    # Refit the SAME family with the SAME hyperparameters, on everything but the held image.
    refit = type(clf)(**params)
    sc = StandardScaler().fit(X[~te])
    refit.fit(sc.transform(X[~te]), y[~te], sample_weight=W[~te])
    auc = float(roc_auc_score(y[te], refit.predict_proba(sc.transform(X[te]))[:, 1]))

    # For contrast, the in-sample figure the gate used to compare against.
    in_sample = float(roc_auc_score(
        y[te], clf.predict_proba(bundle["scaler"].transform(X[te]))[:, 1]))

    print(f"  deployed family      : {family}")
    print(f"  held-out image       : {args.held} "
          f"({int(te.sum())} rows, {int(y[te].sum())} crack / {int((~y[te]).sum())} not)")
    print(f"  OUT-OF-SAMPLE (refit): {auc:.4f}   <- recorded as the gate's baseline")
    # Do NOT assert the sign. In-sample is USUALLY higher -- that is the bias this whole
    # mechanism exists to remove -- but it is not higher for a model imported from
    # elsewhere that never saw this image, and printing "inflated by -0.053" in that case
    # is simply wrong. Describe what was measured and let the reader draw the conclusion.
    delta = in_sample - auc
    if delta > 0:
        note = (f"higher by {delta:.4f}, the in-sample advantage this gate no longer "
                f"grades against")
    else:
        note = (f"lower by {-delta:.4f}, so the deployed model was evidently NOT trained "
                f"on this image and its score here was already out-of-sample")
    print(f"  in-sample (deployed) : {in_sample:.4f}   <- {note}")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    before = clf.predict_proba(bundle["scaler"].transform(X))[:, 1]
    bundle["loio_out_of_sample"] = auc
    bundle["loio_out_of_sample_source"] = "refit-same-family"
    bundle["loio_out_of_sample_image"] = args.held
    bundle["loio_out_of_sample_set_at"] = time.strftime("%Y-%m-%dT%H:%M:%S",
                                                        time.localtime())
    bundle["loio_in_sample_for_reference"] = in_sample
    after = bundle["clf"].predict_proba(bundle["scaler"].transform(X))[:, 1]
    assert np.array_equal(before, after), (
        "establishing a baseline must not change a single prediction")

    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    shutil.copy2(PROD_MODEL_PATH,
                 PROD_MODEL_PATH.replace(".joblib", f".pre-baseline-{stamp}.joblib"))
    tmp = PROD_MODEL_PATH + ".tmp"
    joblib.dump(bundle, tmp)
    os.replace(tmp, PROD_MODEL_PATH)
    print(f"\n  recorded in {os.path.basename(PROD_MODEL_PATH)}; predictions verified "
          f"unchanged; previous bundle kept as "
          f"{os.path.basename(PROD_MODEL_PATH).replace('.joblib', f'.pre-baseline-{stamp}.joblib')}")
    print("  Retrain will now compare out-of-sample against out-of-sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
