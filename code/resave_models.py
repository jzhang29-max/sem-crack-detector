"""
Re-save the shipped models under the scikit-learn version you have installed.

Why this exists: a model here is a pickled scikit-learn estimator, and sklearn
records the version it was pickled under and compares it on every load, warning
"Trying to unpickle estimator ... from version X when using version Y. This might
lead to breaking code or invalid results." Every module in this project calls
warnings.filterwarnings("ignore"), so that warning was invisible -- the shipped
models were built with 1.7.2 while requirements.txt's `scikit-learn>=1.7` installs
whatever is newest, so a fresh clone silently ran a version mismatch on every
detection.

Running this makes the bundles match your environment and clears the warning. It
does NOT retrain anything: it loads the fitted estimator and writes the same
object back, so the coefficients are untouched.

    python3 code/resave_models.py            # check only, changes nothing
    python3 code/resave_models.py --write    # re-save the mismatched ones

Predictions are asserted identical before anything is written. If they ever are
not, the file is left alone and this exits non-zero -- a re-save that changed a
prediction would mean silently swapping the model out from under every number in
docs/MODEL_VALIDATION_BENCHMARK.md.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import joblib
import numpy as np
import sklearn

MODELS = [
    os.path.join(ROOT, "models", "crack_classifier.joblib"),
    os.path.join(ROOT, "interior_active_learning", "models", "unified_model.joblib"),
    os.path.join(ROOT, "interior_active_learning", "models", "interior_model.joblib"),
]


def probe(bundle, seed=1234, n=1000):
    """predict_proba on a fixed synthetic matrix, or None if this bundle has no
    classifier to probe.

    Each call builds its own RandomState from the same seed. An earlier version
    shared one generator across models, so the second and third comparisons ran on
    a different matrix than the first and reported a difference that was purely an
    artefact of the harness.
    """
    clf, sc, names = bundle.get("clf"), bundle.get("scaler"), bundle.get("feature_names")
    if clf is None or not names:
        return None
    X = np.random.RandomState(seed).normal(size=(n, len(names)))
    return clf.predict_proba(sc.transform(X) if sc is not None else X)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="actually re-save mismatched models")
    a = ap.parse_args()

    running = sklearn.__version__
    print(f"installed scikit-learn: {running}\n")
    n_mismatch = n_written = 0
    failed = False

    for path in MODELS:
        name = os.path.relpath(path, ROOT)
        if not os.path.exists(path):
            print(f"  {name}: absent, skipped")
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            b = joblib.load(path)
            live_warn = [w for w in caught if "Inconsistent" in w.category.__name__]
        built = b.get("sklearn_version")
        match = (built == running) and not live_warn
        state = "matches" if match else f"built with {built or 'unrecorded'}"
        print(f"  {name}: {state}" + (f", {len(live_warn)} sklearn warning(s)" if live_warn else ""))
        if match:
            continue
        n_mismatch += 1
        if not a.write:
            continue

        before = probe(b)
        b["sklearn_version"] = running
        tmp = path + ".part"
        joblib.dump(b, tmp)
        after = probe(joblib.load(tmp))
        same = (before is None and after is None) or (
            before is not None and after is not None and np.array_equal(before, after))
        if not same:
            os.remove(tmp)
            print(f"      REFUSED: predictions changed on re-save, left untouched")
            failed = True
            continue
        os.replace(tmp, path)
        n_written += 1
        print(f"      re-saved under {running}, predictions identical")

    print()
    if not a.write and n_mismatch:
        print(f"{n_mismatch} model(s) mismatch your environment. Re-run with --write to fix.")
    elif a.write:
        print(f"re-saved {n_written} of {n_mismatch} mismatched model(s)")
    else:
        print("all models match your installed scikit-learn")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
