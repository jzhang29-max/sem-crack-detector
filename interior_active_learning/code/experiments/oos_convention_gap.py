"""Is the headline convention gap an in-sample artefact?

THE PROBLEM
scoring_convention_bias.py neutralises human input at the PIXEL level: it patches
load_correction_mask and load_hard_overrides so the prediction cannot read the verdicts it is
scored against. That is necessary and it is not sufficient. The Pass-1 classifier doing the
predicting was FITTED on region labels derived from those same correction masks, and every one
of the ten scored frames contributed training rows -- 10 of 10, from 33 rows on one frame to
1,285 on another. So the prediction is in-sample against its own ground truth, and the
headline is measured on training data.

WHICH WAY THE BIAS RUNS, and why the answer is not obvious
A model that has seen a frame's labels makes fewer false positives on that frame. Fewer false
positives raises the ADJUDICATED specificity, tn/(tn+fp), because its negative pool is small
and fp-sensitive. The DENSE specificity is dominated by the vast unreviewed area where the
model predicts negative anyway, so it barely moves. Gap = dense - adjudicated, so circularity
should SHRINK the gap and the published +0.488 should be an underestimate.

That is a prediction, not a result, which is the whole reason to measure it.

METHOD
For each scored frame, refit the Pass-1 classifier on labeled_regions.csv with that frame's
WHOLE SPECIMEN excluded -- sibling frames from one session are near-duplicate leakage, a point
this repo already established for the promotion gate -- using the trainer's own estimator
factory and per-image weighting, so the refit is the procedure the project ships and not a
local invention. Then run the pipeline with that model and score both conventions.

    python3 oos_convention_gap.py
    python3 oos_convention_gap.py --shard 0 --nshard 4
    python3 oos_convention_gap.py --report
"""
import argparse
import contextlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "code")))

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

import unified_pipeline as up
from aggregate import specimen_key
from common import PROJECT_ROOT, load_correction_mask
from scoring_convention_bias import eligible
from train_v3_weighted import FEATURES, MODELS, image_weights

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(_HERE, "oos_convention_gap.json")
CSV = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")


@contextlib.contextmanager
def _without_human_input():
    a, b = up.load_correction_mask, up.load_hard_overrides
    up.load_correction_mask = lambda *x, **k: None
    up.load_hard_overrides = lambda *x, **k: None
    try:
        yield
    finally:
        up.load_correction_mask, up.load_hard_overrides = a, b


@contextlib.contextmanager
def _with_model(path):
    """Point Pass 1 at another bundle. PROD_MODEL_PATH is read from module globals at call
    time, so rebinding the attribute is enough and no environment games are needed."""
    old = up.PROD_MODEL_PATH
    up.PROD_MODEL_PATH = path
    try:
        yield
    finally:
        up.PROD_MODEL_PATH = old


def _score(pred, crack, neg):
    tp = int((pred & crack).sum())
    fn = int((~pred & crack).sum())
    fp = int((pred & neg).sum())
    tn = int((~pred & neg).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {"recall": rec, "precision": prec, "specificity": tn / max(tn + fp, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _refit_without(specimen, df, deployed):
    """Refit Pass 1 with a whole specimen held out, by the trainer's own procedure."""
    groups = df["SourceImage"].values
    held = np.array([specimen_key(g) == specimen for g in groups])
    if held.all() or not held.any():
        return None, f"holdout selects {int(held.sum())} of {len(held)} rows"
    X = df[FEATURES].values
    y = df["IsCrack"].astype(bool).values
    if len(set(y[~held])) < 2:
        return None, "training side has only one class"
    W = image_weights(groups)
    family = type(deployed["clf"]).__name__
    clf = MODELS[family]() if family in MODELS else type(deployed["clf"])(
        **deployed["clf"].get_params())
    sc = StandardScaler().fit(X[~held])
    clf.fit(sc.transform(X[~held]), y[~held], sample_weight=W[~held])
    return ({"clf": clf, "scaler": sc, "feature_names": list(FEATURES)},
            f"held out specimen {specimen}: {int(held.sum())} of {len(held)} rows, "
            f"{len(set(groups[held]))} frame(s)")


def run(shard=0, nshard=1):
    names = eligible()
    mine = names[shard::nshard]
    df = pd.read_csv(CSV)
    deployed = joblib.load(os.path.join(PROJECT_ROOT, "models", "crack_classifier.joblib"))
    print(f"{len(mine)} of {len(names)} frame(s)"
          f"{f'  [shard {shard}/{nshard}]' if nshard > 1 else ''}\n", flush=True)
    # RESUME. Each frame costs two full pipeline runs, so a killed shard must not redo the
    # frames it already finished. Only frames with BOTH arms recorded count as done: a frame
    # with one arm is useless, since the comparison is within-frame.
    out_path = OUT if nshard == 1 else OUT.replace(".json", f".shard{shard}.json")
    rows = []
    if os.path.exists(out_path):
        try:
            rows = json.load(open(out_path)).get("per_image", [])
        except (OSError, ValueError):
            rows = []
    have = {k for k, v in
            {r["image"]: None for r in rows}.items()
            if sum(1 for r in rows if r["image"] == k) >= 2}
    if have:
        print(f"  resuming: {len(have)} frame(s) already complete\n", flush=True)
    for n in mine:
        if n in have:
            continue
        spec = specimen_key(n)
        bundle, how = _refit_without(spec, df, deployed)
        if bundle is None:
            print(f"  {n[:38]:40s} SKIPPED: {how}", flush=True)
            continue
        tmp = os.path.join(_HERE, f".oos_model_{shard}.joblib")
        joblib.dump(bundle, tmp)
        try:
            for tag, path in (("in_sample", None), ("out_of_sample", tmp)):
                ctx = _with_model(path) if path else contextlib.nullcontext()
                with ctx, _without_human_input():
                    st = up.run_unified_pipeline(n)
                lab, d2 = st["labeled"], st["df"]
                pred = np.isin(lab, d2.loc[d2["IsCrack"], "Label"].tolist())
                m = load_correction_mask(n, lab.shape)
                if m is None:
                    break
                crack, neg = (m == 1), (m == 2)
                adj = _score(pred, crack, neg)
                dense = _score(pred, crack, ~crack)
                rec = {"image": n, "specimen": spec, "arm": tag, "holdout": how,
                       "adjudicated": adj, "dense": dense,
                       "gap_specificity": dense["specificity"] - adj["specificity"],
                       "gap_precision": dense["precision"] - adj["precision"]}
                rows.append(rec)
                print(f"  {n[:34]:36s} {tag:14s} adj spec {adj['specificity']:.4f}  "
                      f"gap {rec['gap_specificity']:+.4f}", flush=True)
        except Exception as e:
            print(f"  {n[:38]:40s} FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        out = out_path
        t = out + ".tmp"
        with open(t, "w") as fh:
            json.dump({"per_image": rows}, fh, indent=1)
        os.replace(t, out)
    return rows


def report():
    import glob as _g
    paths = sorted(_g.glob(OUT.replace(".json", ".shard*.json"))) or [OUT]
    rows = []
    for p in paths:
        try:
            rows += json.load(open(p))["per_image"]
        except (OSError, ValueError, KeyError):
            continue
    if not rows:
        print("no results")
        return None
    # Only frames measured under BOTH arms may enter, or the two means describe two samples.
    byimg = {}
    for r in rows:
        byimg.setdefault(r["image"], {})[r["arm"]] = r
    both = {k: v for k, v in byimg.items() if "in_sample" in v and "out_of_sample" in v}
    print(f"{len(both)} frame(s) measured under both arms "
          f"({len(byimg) - len(both)} incomplete, excluded)\n")
    if not both:
        return None
    print(f"  {'image':34s} {'adj in':>8s} {'adj oos':>8s} {'gap in':>9s} {'gap oos':>9s}")
    for k, v in sorted(both.items()):
        print(f"  {k[:34]:34s} {v['in_sample']['adjudicated']['specificity']:8.4f} "
              f"{v['out_of_sample']['adjudicated']['specificity']:8.4f} "
              f"{v['in_sample']['gap_specificity']:+9.4f} "
              f"{v['out_of_sample']['gap_specificity']:+9.4f}")
    mean = lambda arm, f: float(np.mean([f(v[arm]) for v in both.values()]))
    a_in = mean("in_sample", lambda r: r["adjudicated"]["specificity"])
    a_oos = mean("out_of_sample", lambda r: r["adjudicated"]["specificity"])
    g_in = mean("in_sample", lambda r: r["gap_specificity"])
    g_oos = mean("out_of_sample", lambda r: r["gap_specificity"])
    print(f"\n  macro adjudicated specificity  in-sample {a_in:.4f}  "
          f"out-of-sample {a_oos:.4f}  ({a_oos - a_in:+.4f})")
    print(f"  macro specificity GAP          in-sample {g_in:+.4f}  "
          f"out-of-sample {g_oos:+.4f}  ({g_oos - g_in:+.4f})")
    pred_held = g_oos > g_in
    print(f"\n  The prediction was that circularity SHRINKS the gap, so removing it should "
          f"make the gap larger. It {'HELD' if pred_held else 'DID NOT HOLD'}: "
          f"{g_in:+.4f} -> {g_oos:+.4f}.")
    print(f"  Either way the headline is now measured with the scored frame's whole specimen "
          f"held out of the Pass-1 fit, so it is no longer an in-sample number.")
    res = {"n_frames": len(both),
           "macro_adjudicated_specificity": {"in_sample": a_in, "out_of_sample": a_oos},
           "macro_gap_specificity": {"in_sample": g_in, "out_of_sample": g_oos},
           "prediction_that_circularity_shrinks_the_gap_held": bool(pred_held),
           "per_image": rows}
    json.dump(res, open(OUT.replace(".json", "_report.json"), "w"), indent=1)
    print(f"\n  -> {OUT.replace('.json', '_report.json')}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.shard, a.nshard)
