"""Three more silent failure modes in quantitative micrograph metrology, with magnitudes.

Companion to scoring_convention_bias.py. Each failure mode below was OBSERVED in this
project, not hypothesised -- every one was a real defect that shipped, was caught, and was
fixed, which is why the magnitudes are measured rather than estimated. That provenance is
uncomfortable and it is also what makes them credible: these are the errors a careful
person makes anyway.

The common shape: a piece of software chooses a plausible default where the honest answer
is "I cannot tell", and nothing downstream can distinguish the default from a measurement.

  A. CALIBRATION ACCEPTED WITHOUT CROSS-CHECK
     A scale bar can be read wrongly in ways that look completely reasonable, and a single
     reading has nothing to disagree with. Measured: three plausible automatic readings of
     one frame's bar, against the independent field-width route.

  B. PROMOTION GATED ON AN IN-SAMPLE BASELINE
     Comparing an out-of-sample candidate against an in-sample incumbent biases a retrain
     loop toward one verdict, permanently and silently. Measured on this corpus.

  C. TRAIN/SERVE PREPROCESSING SKEW
     When the training builder and the inference pipeline run different preprocessing, some
     of the human's labels cannot reach the model at all -- and nothing reports it.
     Measured: rows and images recovered when the skew was removed.

    python3 failure_mode_magnitudes.py
"""
import contextlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "code")))

import joblib
import detector_config as _dc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import PROD_MODEL_PATH, PROJECT_ROOT

OUT = os.path.join(_HERE, "failure_mode_magnitudes.json")


# ---------------------------------------------------------------- A. calibration
def calibration_disagreement():
    """What a plausible-but-wrong scale-bar reading does to every exported length.

    These three readings of MAR_Amb_Cast_CBS_0002's burned-in bar were each produced by a
    detector that looked sound in isolation. The fourth column is the independent route:
    field width divided by image width. Only one reading agrees with it.
    """
    label_um, hfw_um, width_px = 400.0, 1040.0, 6144
    truth = hfw_um / width_px
    readings = [
        ("longest contiguous bright run in the panel", 1000),
        ("widest bright extent in the panel's right half", 2816),
        ("tick-to-tick (correct)", 2379),
    ]
    rows = []
    for how, span in readings:
        umpx = label_um / span
        rel = (umpx - truth) / truth
        rows.append({
            "method": how, "span_px": span, "um_per_px": umpx,
            "rel_error_vs_hfw": rel,
            # A length scales linearly, an area as the square. A 16% scale error is a 35%
            # area error, which is how a plausible reading becomes an implausible result.
            "length_error_pct": 100 * rel,
            "area_error_pct": 100 * ((1 + rel) ** 2 - 1),
        })
    return {"hfw_um_per_px": truth, "readings": rows,
            "worst_length_error_pct": max(abs(r["length_error_pct"]) for r in rows),
            "worst_area_error_pct": max(abs(r["area_error_pct"]) for r in rows),
            "note": ("A single reading has nothing to disagree with, so none of these is "
                     "detectable from inside the program. Requiring two independent "
                     "readings to agree within 5% rejects the first two.")}


# ------------------------------------------------------------------- B. the gate
def in_sample_gate_bias():
    """The gap between grading a candidate out-of-sample and its incumbent in-sample."""
    csv = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")
    if not os.path.exists(csv):
        return {"error": "no training data"}
    from train_v3_weighted import FEATURES, HELD, MODELS, image_weights
    df = pd.read_csv(csv)
    X, y = df[FEATURES].values, df["IsCrack"].astype(bool).values
    groups = df["SourceImage"].values
    te = groups == HELD
    if not te.any() or len(set(y[te])) < 2:
        return {"error": f"{HELD} lacks both classes"}
    W = image_weights(groups)

    bundle = joblib.load(PROD_MODEL_PATH)
    fam = type(bundle["clf"]).__name__

    # Out-of-sample: fit without the held image, score on it. What a candidate gets.
    sc = StandardScaler().fit(X[~te])
    m_out = MODELS[fam]() if fam in MODELS else type(bundle["clf"])(**bundle["clf"].get_params())
    m_out.fit(sc.transform(X[~te]), y[~te], sample_weight=W[~te])
    auc_out = float(roc_auc_score(y[te], m_out.predict_proba(sc.transform(X[te]))[:, 1]))

    # In-sample: fit on EVERYTHING including the held image, then score on it. What an
    # incumbent's own recorded number looks like, and what the gate used to compare against.
    sc_all = StandardScaler().fit(X)
    m_in = MODELS[fam]() if fam in MODELS else type(bundle["clf"])(**bundle["clf"].get_params())
    m_in.fit(sc_all.transform(X), y, sample_weight=W)
    auc_in = float(roc_auc_score(y[te], m_in.predict_proba(sc_all.transform(X[te]))[:, 1]))

    return {"family": fam, "held_image": HELD, "held_rows": int(te.sum()),
            "auc_out_of_sample": auc_out, "auc_in_sample": auc_in,
            "in_sample_advantage": auc_in - auc_out,
            "note": ("A gate comparing an out-of-sample candidate against an in-sample "
                     "incumbent must clear this gap before it can promote anything. The "
                     "loop then reports the user's new labels as harmful when nothing "
                     "about their labels was measured -- and because both numbers are "
                     "individually plausible, nothing on screen reveals it.")}


# ------------------------------------------------------- C. train/serve skew
def train_serve_skew():
    """Labels that could not reach the model while the two paths preprocessed differently.

    Recorded from the observed before/after: the training builder ran a background-exclusion
    step that inference had dropped, so regions deleted before the label vote produced no
    rows at all.
    """
    csv = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")
    now_rows = now_imgs = None
    if os.path.exists(csv):
        d = pd.read_csv(csv)
        now_rows, now_imgs = int(len(d)), int(d["SourceImage"].nunique())
    # A FROZEN SNAPSHOT, and it has to be labelled as one. These two numbers were measured
    # once, before the exclusion step was removed; everything below compares them against a
    # LIVE read of the same CSV. So any later change to labeled_regions.csv -- more marking,
    # a re-ingest, a different vote rule -- is silently credited to the train/serve skew, and
    # the percentage drifts without the code changing. That is the opposite of a measurement.
    before_rows, before_imgs = 4128, 32
    SNAPSHOT = ("measured once from training_data/labeled_regions.csv while "
                "build_training_data.py still called exclude_border_background(); it is a "
                "historical constant, not a re-measurement, so every figure derived from it "
                "is a comparison against that snapshot and will drift if the CSV changes for "
                "any other reason")
    out = {"rows_before": before_rows, "images_before": before_imgs,
           "rows_before_provenance": SNAPSHOT,
           "rows_after": now_rows, "images_after": now_imgs}

    # LIVE, and by ENUMERATION rather than subtraction. `images_recovered` was
    # now_imgs - before_imgs: a net difference between two distinct-image counts taken at
    # different times, which never identifies an image and never checks that one had labels
    # but produced no rows. An image with a correction mask and zero rows is identifiable
    # right now, so the current state can be established instead of inferred.
    try:
        import glob as _glob
        from common import PAINT_DIR as _PD
        # A mask FILE is not a mask. The app writes one when an image is opened, so an
        # untouched image has a full-size all-zero (all-UNREVIEWED) file on disk. Counting
        # those as labelled made the first version of this enumeration report one image
        # "contributing no rows" when that image has no marks to contribute -- a denominator
        # error of exactly the kind this experiment is about, committed inside it.
        import numpy as _np
        from PIL import Image as _PI
        _PI.MAX_IMAGE_PIXELS = None
        _masked, _empty = set(), []
        for q in _glob.glob(os.path.join(_PD, "*_correction_mask.png")):
            nm = os.path.basename(q).replace("_correction_mask.png", "")
            try:
                _a = _np.asarray(_PI.open(q))
                if _a.ndim > 2:
                    _a = _a[..., 0]
                if bool((_a != 0).any()):
                    _masked.add(nm)
                else:
                    _empty.append(nm)
            except Exception:
                continue
        _with_rows = set(d["SourceImage"].unique()) if now_rows else set()
        _silent = sorted(_masked - _with_rows)
        out["images_with_an_empty_mask_file"] = sorted(_empty)
        out["images_with_a_mask"] = len(_masked)
        out["images_contributing_no_rows_now"] = _silent
        out["n_images_contributing_no_rows_now"] = len(_silent)
        out["enumeration_note"] = (
            "images_contributing_no_rows_now counts images whose mask carries at least one "
            "MARKED pixel yet produce no training rows, measured by enumeration against the "
            "current files -- the claim this script can actually support. Empty mask files "
            "(written when an image is opened and never painted) are excluded and listed "
            "separately, because counting them as labelled would manufacture the very "
            "failure being looked for. images_recovered below is a subtraction against the "
            "frozen snapshot, identifies no image, and describes a historical change; it "
            "must not be quoted as a current count.")
    except Exception as _e:
        out["enumeration_error"] = f"{type(_e).__name__}: {_e}"

    if now_rows:
        out.update({
            "rows_recovered": now_rows - before_rows,
            "images_recovered": now_imgs - before_imgs,
            "images_recovered_is_a_subtraction": True,
            "pct_labels_unreachable": 100 * (now_rows - before_rows) / now_rows,
            "pct_labels_unreachable_basis": ("(rows_after - rows_before_SNAPSHOT) / "
                                             "rows_after -- see rows_before_provenance"),
        })
    out["note"] = ("Every path reported success throughout: the per-image worker caught its "
                   "own exception and returned None, the merge carried prior rows for any "
                   "image whose source was absent, and the process exited 0. Hand-drawn "
                   "labels were discarded by a run that looked clean.")
    return out


#: These three results do not read the detector's predicted mask, so unlike the other
#: experiments there is nothing to pin -- and that independence is structural at two levels,
#: which is worth writing down because it was checked rather than assumed:
#:
#:   1. Nothing here calls run_unified_pipeline. Result 2 is arithmetic on scale-bar readings,
#:      result 3 refits the classifier on stored training rows, result 4 counts rows and
#:      enumerates mask files. The `IsCrack` column read here is the CSV's human-derived
#:      label, not a prediction.
#:   2. Their input, training_data/labeled_regions.csv, is built by build_training_data.py
#:      calling extract_candidates DIRECTLY -- upstream of where SAM 2 refinement happens --
#:      so even the training data is untouched by the detector default.
#:
#: MEASURED, not asserted: run under --sam2 off and --sam2 refine, all 25 scalar fields across
#: the three results are identical. And the skew this could have introduced was checked too --
#: overlays are refined while training labels are voted onto unrefined candidates, so a human
#: painting on a refined boundary might have missed the candidate that gets the verdict. On
#: 260708_316_H_b2_front_CBS_002, 100% of the human's 141,662 crack pixels lie inside an
#: accepted region under BOTH configurations and 0 px are covered by only one, because
#: sam2_refine.refine_labeled restores painted verdicts after refinement.
DETECTOR_INDEPENDENT = True


def main():
    res = {"calibration_disagreement": calibration_disagreement(),
           "in_sample_gate_bias": in_sample_gate_bias(),
           "train_serve_skew": train_serve_skew()}

    a = res["calibration_disagreement"]
    print("A. CALIBRATION ACCEPTED WITHOUT A CROSS-CHECK")
    print(f"   independent route (field width / image width): {a['hfw_um_per_px']:.5f} um/px")
    for r in a["readings"]:
        print(f"   {r['method']:46s} {r['span_px']:5d} px -> {r['um_per_px']:.5f} um/px  "
              f"length {r['length_error_pct']:+7.1f}%  area {r['area_error_pct']:+7.1f}%")
    print(f"   worst case: a length off by {a['worst_length_error_pct']:.0f}% and an area "
          f"off by {a['worst_area_error_pct']:.0f}%, from a reading that looked sound\n")

    b = res["in_sample_gate_bias"]
    print("B. PROMOTION GATED ON AN IN-SAMPLE BASELINE")
    if "error" in b:
        print(f"   skipped: {b['error']}\n")
    else:
        print(f"   {b['family']} on {b['held_image']} ({b['held_rows']} rows)")
        print(f"   out-of-sample AUC {b['auc_out_of_sample']:.4f}")
        print(f"   in-sample     AUC {b['auc_in_sample']:.4f}")
        print(f"   in-sample advantage: {b['in_sample_advantage']:+.4f} -- the bar every "
              f"honest candidate must clear\n")

    c = res["train_serve_skew"]
    print("C. TRAIN/SERVE PREPROCESSING SKEW")
    if c.get("rows_after"):
        print(f"   rows   {c['rows_before']:,} -> {c['rows_after']:,} "
              f"({c['rows_recovered']:+,})")
        print(f"   images {c['images_before']} -> {c['images_after']} "
              f"({c['images_recovered']:+d})")
        if c.get("n_images_contributing_no_rows_now") is not None:
            _n = c["n_images_contributing_no_rows_now"]
            print(f"   live, by enumeration: {c.get('images_with_a_mask')} image(s) have a "
                  f"correction mask and {_n} of them contribute no training rows"
                  + (f" ({', '.join(c['images_contributing_no_rows_now'])})" if _n else ""))
            print(f"   the +{c.get('images_recovered')} above is a subtraction against a "
                  f"frozen snapshot and names no image; it describes a historical change, "
                  f"not a current count")
        print(f"   {c['pct_labels_unreachable']:.1f}% of the labelled regions now in the "
              f"corpus could not reach the model, with every path reporting success\n")

    # Record that these results are detector-independent AND that it was verified, so a
    # reader does not have to take the structural argument on trust.
    res["detector"] = dict(
        _dc.stamp("off"), detector_independent=True,
        independence_verified_by=(
            "run under both --sam2 off and --sam2 refine: all 25 scalar fields across "
            "results 2, 3 and 4 identical. These results never read the predicted mask, and "
            "their input CSV is built by build_training_data calling extract_candidates "
            "directly, upstream of where refinement happens."))
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")
    return res


if __name__ == "__main__":
    main()
