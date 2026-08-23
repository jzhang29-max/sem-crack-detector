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

import detector_config as _dc
import unified_pipeline as up
from aggregate import specimen_key
from common import PROJECT_ROOT, load_correction_mask
from scoring_convention_bias import eligible
from train_v3_weighted import FEATURES, MODELS, image_weights

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(_HERE, "oos_convention_gap.json")
#: Which detector this experiment measures. this experiment already controls the Pass-1 weights; leaving SAM 2 on would add a second uncontrolled difference between its arms.
DETECTOR = "off"


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


def _refit_masked(held, df, deployed, why):
    """Refit Pass 1 on ~held, by the trainer's own procedure."""
    groups = df["SourceImage"].values
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
            f"{why}: {int(held.sum())} of {len(held)} rows, "
            f"{len(set(groups[held]))} frame(s)")


def _specimen_mask(specimen, df):
    return np.array([specimen_key(g) == specimen for g in df["SourceImage"].values])


def _volume_matched_mask(specimen, df, n_rows, seed=0):
    """Remove n_rows at random from OTHER specimens, leaving this one entirely in.

    THE CONFOUND THIS CONTROLS. Holding out a whole specimen removes both the leakage and a
    large slice of the training data -- 5.2% to 26.8% of rows depending on the specimen. So a
    gap that grows out-of-sample could be the absence of leakage or simply a worse model
    trained on less. This arm matches the VOLUME removed while leaving the scored frame's own
    specimen in the fit, so the difference between the two arms isolates the leakage.
    """
    own = _specimen_mask(specimen, df)
    other = np.flatnonzero(~own)
    if other.size <= n_rows:
        return None
    rng = np.random.default_rng(seed)
    drop = rng.choice(other, size=int(n_rows), replace=False)
    m = np.zeros(len(df), dtype=bool)
    m[drop] = True
    return m


def run(shard=0, nshard=1, detector=None):
    # PINNED, not inherited. run_unified_pipeline defaults to SAM 2 refinement, so an experiment that does not say which detector it wants silently measures a
    # different one than its output is labelled with. There is no default here on purpose.
    detector = detector or DETECTOR
    up.SAM2_MODE = detector
    names = eligible()
    mine = names[shard::nshard]
    df = pd.read_csv(CSV)
    deployed = joblib.load(os.path.join(PROJECT_ROOT, "models", "crack_classifier.joblib"))
    print(f"{len(mine)} of {len(names)} frame(s)"
          f"{f'  [shard {shard}/{nshard}]' if nshard > 1 else ''}\n", flush=True)
    # RESUME. Each frame costs two full pipeline runs, so a killed shard must not redo the
    # frames it already finished. Only frames with BOTH arms recorded count as done: a frame
    # with one arm is useless, since the comparison is within-frame.
    out_path = _dc.out_for(OUT, detector) if nshard == 1 else _dc.out_for(OUT, detector).replace(".json", f".shard{shard}.json")
    rows = []
    if os.path.exists(out_path):
        try:
            rows = json.load(open(out_path)).get("per_image", [])
        except (OSError, ValueError):
            rows = []
    # Three arms now, so two recorded arms is no longer complete. Bumping this rather than
    # leaving it at >=2 is what makes the resume re-measure frames that predate the control
    # instead of silently reporting a two-arm comparison as a three-arm one.
    have = {k for k in {r["image"] for r in rows}
            if len({r["arm"] for r in rows if r["image"] == k}) >= 3}
    if have:
        print(f"  resuming: {len(have)} frame(s) already complete\n", flush=True)
    for n in mine:
        if n in have:
            continue
        spec = specimen_key(n)
        spec_mask = _specimen_mask(spec, df)
        arms = [("in_sample", None, "deployed bundle, this frame's labels in the fit")]
        b_oos, how_oos = _refit_masked(spec_mask, df, deployed,
                                       f"held out specimen {spec}")
        if b_oos is None:
            print(f"  {n[:38]:40s} SKIPPED: {how_oos}", flush=True)
            continue
        tmp = os.path.join(_HERE, f".oos_model_{shard}.joblib")
        joblib.dump(b_oos, tmp)
        arms.append(("out_of_sample", tmp, how_oos))

        vm = _volume_matched_mask(spec, df, int(spec_mask.sum()))
        tmp2 = os.path.join(_HERE, f".vol_model_{shard}.joblib")
        if vm is not None:
            b_vm, how_vm = _refit_masked(
                vm, df, deployed,
                f"volume-matched control: {int(spec_mask.sum())} rows dropped at random "
                f"from OTHER specimens, {spec} left in")
            if b_vm is not None:
                joblib.dump(b_vm, tmp2)
                arms.append(("volume_control", tmp2, how_vm))
        try:
            for tag, path, how in arms:
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
                       "rows_held_out": int(spec_mask.sum()),
                       "adjudicated": adj, "dense": dense,
                       "gap_specificity": dense["specificity"] - adj["specificity"],
                       "gap_precision": dense["precision"] - adj["precision"]}
                rows.append(rec)
                print(f"  {n[:34]:36s} {tag:14s} adj spec {adj['specificity']:.4f}  "
                      f"gap {rec['gap_specificity']:+.4f}", flush=True)
        except Exception as e:
            print(f"  {n[:38]:40s} FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            for _t in (tmp, tmp2):
                if os.path.exists(_t):
                    os.remove(_t)
        out = out_path
        t = out + ".tmp"
        with open(t, "w") as fh:
            json.dump({"detector": _dc.stamp(detector), "per_image": rows}, fh, indent=1)
        os.replace(t, out)
    return rows


def report(detector="off"):
    import glob as _g
    paths = sorted(_g.glob(_dc.out_for(OUT, detector).replace(".json", ".shard*.json"))) or [_dc.out_for(OUT, detector)]
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
    ARMS = ["in_sample", "out_of_sample", "volume_control"]
    both = {k: v for k, v in byimg.items() if all(a in v for a in ARMS)}
    partial = {k: sorted(v) for k, v in byimg.items() if k not in both}
    print(f"{len(both)} frame(s) measured under all three arms"
          f"{f'; {len(partial)} incomplete and EXCLUDED: ' + json.dumps(partial) if partial else ''}\n")
    if not both:
        return None
    print(f"  {'image':30s} {'rows out':>9s} {'gap in':>9s} {'gap oos':>9s} "
          f"{'gap volctl':>11s} {'leakage':>9s}")
    for k, v in sorted(both.items()):
        leak = v["out_of_sample"]["gap_specificity"] - v["volume_control"]["gap_specificity"]
        print(f"  {k[:30]:30s} {v['out_of_sample'].get('rows_held_out', 0):9,d} "
              f"{v['in_sample']['gap_specificity']:+9.4f} "
              f"{v['out_of_sample']['gap_specificity']:+9.4f} "
              f"{v['volume_control']['gap_specificity']:+11.4f} {leak:+9.4f}")
    mean = lambda arm, f: float(np.mean([f(v[arm]) for v in both.values()]))
    a = {arm: mean(arm, lambda r: r["adjudicated"]["specificity"]) for arm in ARMS}
    g = {arm: mean(arm, lambda r: r["gap_specificity"]) for arm in ARMS}
    print(f"\n  {'arm':16s} {'macro adj spec':>15s} {'macro gap':>11s}")
    for arm in ARMS:
        print(f"  {arm:16s} {a[arm]:15.4f} {g[arm]:+11.4f}")

    # THE DECOMPOSITION. in_sample -> volume_control isolates the effect of training on less
    # data, since the scored frame's own specimen stays in the fit. volume_control ->
    # out_of_sample isolates the leakage, since the volume removed is the same and only its
    # provenance differs.
    d_volume = g["volume_control"] - g["in_sample"]
    d_leak = g["out_of_sample"] - g["volume_control"]
    print(f"\n  Decomposition of the {g['out_of_sample'] - g['in_sample']:+.4f} total move:")
    print(f"    {d_volume:+.4f} from training on less data (volume-matched control, this "
          f"frame's specimen still in the fit)")
    print(f"    {d_leak:+.4f} from removing the leakage itself (same volume, different "
          f"provenance)")
    dominant = "leakage" if abs(d_leak) > abs(d_volume) else "reduced training volume"
    print(f"    -> the move is dominated by {dominant}.")
    pred_held = g["out_of_sample"] > g["in_sample"]
    print(f"\n  The prediction was that circularity SHRINKS the gap, so removing it should "
          f"make the gap larger. At the macro level it "
          f"{'HELD' if pred_held else 'DID NOT HOLD'}: {g['in_sample']:+.4f} -> "
          f"{g['out_of_sample']:+.4f}.")
    n_up = sum(1 for v in both.values()
               if v["out_of_sample"]["gap_specificity"] > v["in_sample"]["gap_specificity"])
    print(f"  Per frame it is not universal: the gap grows on {n_up} of {len(both)} and "
          f"shrinks on {len(both) - n_up}. Two frames cannot move at all -- their adjudicated "
          f"specificity is already 0.0000 and cannot fall -- and they are reported rather "
          f"than dropped.")
    res = {"n_frames": len(both), "arms": ARMS, "excluded_partial": partial,
           "macro_adjudicated_specificity": a, "macro_gap_specificity": g,
           "decomposition": {"from_training_volume": d_volume,
                             "from_leakage": d_leak, "dominated_by": dominant},
           "prediction_that_circularity_shrinks_the_gap_held": bool(pred_held),
           "n_frames_gap_grew": n_up,
           "per_image": rows}
    json.dump(res, open(_dc.out_for(OUT, detector).replace(".json", "_report.json"), "w"), indent=1)
    print(f"\n  -> {_dc.out_for(OUT, detector).replace('.json', '_report.json')}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--detector", choices=_dc.VALID, default=DETECTOR)
    a = ap.parse_args()
    report(detector=a.detector) if a.report else run(a.shard, a.nshard, detector=a.detector)
