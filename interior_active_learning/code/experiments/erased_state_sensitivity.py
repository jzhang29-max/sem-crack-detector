"""Does the headline gap depend on a convention choice INSIDE the adjudicated convention?

THE QUESTION
scoring_convention_bias.py defines the adjudicated negative pool as `m == 2` -- pixels the
human painted "not crack". The correction mask has a fourth state, 3 = erased, and on the ten
both-class frames erased pixels are 435,058 against 208,763 marked not-crack: 2.08x more
numerous, and on one frame 24.9x.

The detector does not distinguish them. unified_pipeline.py groups both under one comment --
"Pixels the human took off the table: 3 = erased from candidacy, 2 = marked not-crack" -- and
protects `np.isin(_cm, (2, 3))` from being re-proposed as crack. So the pipeline treats an
erasure as a human "no", while the experiment that scores the pipeline does not.

WHY THIS IS NOT A ONE-LINE FIX
"Erased" is ambiguous in this corpus and the ambiguity is not resolvable from the pixels:

  reading A  the human erased a region because it is not a crack -> it is an adjudicated
             NEGATIVE, and excluding it discards 2.08x the evidence
  reading B  the human erased a region to take it out of consideration -- off-specimen
             background, a charging artefact, something outside the area of interest -> it is
             closer to UNREVIEWED than to not-crack, and scoring on it charges the detector
             for regions nobody claimed were specimen

THE PREDICTION IN THIS DOCSTRING WAS WRONG, AND THE REASON IS THE INTERESTING PART.
It read: counting erasures as negatives "adds easy negatives and inflates specificity, which
is the dense convention's error in miniature". Measured on all ten frames, it does the
opposite -- macro specificity 0.4599 -> 0.3490, a fall of 0.111.

The mechanism is obvious in hindsight and was not obvious in advance. An erased region is
one the DETECTOR PROPOSED; that is why a human was looking at it with the erase tool. So
those pixels are disproportionately predicted-positive, and adding them to the negative pool
adds false positives, not easy true negatives. The direction is frame-dependent for the same
reason: three frames have no erasures and do not move, two rise (0.0000 -> 0.6016 on one),
and five fall, one of them 0.9312 -> 0.0782.

CONSEQUENCE FOR THE PAPER. The adjudicated-vs-dense specificity gap is +0.4879 as shipped
and +0.5988 under the other reading, so the headline is 23% LARGER if an erasure counts as a
human "no". The result does not depend on the choice for its direction, only its size, which
is worth stating plainly: the finding survives either reading and the interval is the honest
form of it.

    python3 erased_state_sensitivity.py
    python3 erased_state_sensitivity.py --shard 0 --nshard 4
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

import numpy as np
from PIL import Image

import detector_config as _dc
import unified_pipeline as up
from common import load_correction_mask
from scoring_convention_bias import eligible

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(_HERE, "erased_state_sensitivity.json")
#: Which detector this experiment measures. result 1c was measured on the bare detector.
DETECTOR = "off"




@contextlib.contextmanager
def _without_human_input():
    a, b = up.load_correction_mask, up.load_hard_overrides
    up.load_correction_mask = lambda *x, **k: None
    up.load_hard_overrides = lambda *x, **k: None
    try:
        yield
    finally:
        up.load_correction_mask, up.load_hard_overrides = a, b


def _score(pred, crack, neg):
    tp = int((pred & crack).sum())
    fn = int((~pred & crack).sum())
    fp = int((pred & neg).sum())
    tn = int((~pred & neg).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {"recall": rec, "precision": prec,
            "specificity": tn / max(tn + fp, 1),
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "negative_pool_px": int(neg.sum())}


def run(shard=0, nshard=1, detector=None):
    # PINNED, not inherited. The default inside run_unified_pipeline has already changed
    # once -- SAM 2 refinement was briefly the default and was reverted to off after it
    # was found to fragment the mask -- so an experiment that does not say which detector
    # it wants measures whatever the default happens to be on the day it is re-run, and
    # its number silently stops matching the one in the writeup. Say it explicitly.
    detector = detector or DETECTOR
    up.SAM2_MODE = detector
    names = eligible()
    mine = names[shard::nshard]
    print(f"{len(mine)} of {len(names)} both-class frame(s)"
          f"{f'  [shard {shard}/{nshard}]' if nshard > 1 else ''}\n", flush=True)
    rows = []
    for n in mine:
        try:
            with _without_human_input():
                st = up.run_unified_pipeline(n)
            lab, df = st["labeled"], st["df"]
            pred = np.isin(lab, df.loc[df["IsCrack"], "Label"].tolist())
            m = load_correction_mask(n, lab.shape)
            if m is None:
                continue
            crack = (m == 1)
            neg_strict = (m == 2)
            neg_incl = (m == 2) | (m == 3)
            if not (crack.any() and neg_strict.any()):
                continue
            rec = {"image": n,
                   "px_crack": int(crack.sum()),
                   "px_not_crack": int(neg_strict.sum()),
                   "px_erased": int((m == 3).sum()),
                   # reading A: an erasure is a human "not crack"
                   "erased_as_negative": _score(pred, crack, neg_incl),
                   # reading B: an erasure is closer to unreviewed, so exclude it
                   "erased_excluded": _score(pred, crack, neg_strict),
                   # the dense convention, for scale
                   "unlabelled_as_background": _score(pred, crack, ~crack)}
            rows.append(rec)
            print(f"  {n[:38]:40s} spec: excl {rec['erased_excluded']['specificity']:.4f}  "
                  f"incl {rec['erased_as_negative']['specificity']:.4f}  "
                  f"dense {rec['unlabelled_as_background']['specificity']:.4f}", flush=True)
        except Exception as e:
            print(f"  {n[:38]:40s} FAILED {type(e).__name__}: {e}", flush=True)
        out = _dc.out_for(OUT, detector) if nshard == 1 else _dc.out_for(OUT, detector).replace(".json", f".shard{shard}.json")
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"detector": _dc.stamp(detector), "per_image": rows}, fh, indent=1)
        os.replace(tmp, out)
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
    print(f"{len(rows)} frame(s)\n")
    print(f"  {'convention':26s} {'specificity':>12s} {'precision':>11s} "
          f"{'neg pool px':>13s}")
    summary = {}
    for key, label in (("erased_excluded", "erased EXCLUDED (as shipped)"),
                       ("erased_as_negative", "erased counted NEGATIVE"),
                       ("unlabelled_as_background", "unlabelled = background")):
        spec = float(np.mean([r[key]["specificity"] for r in rows]))
        prec = float(np.mean([r[key]["precision"] for r in rows]))
        pool = int(sum(r[key]["negative_pool_px"] for r in rows))
        summary[key] = {"macro_specificity": spec, "macro_precision": prec,
                        "pooled_negative_px": pool}
        print(f"  {label:26s} {spec:12.4f} {prec:11.4f} {pool:13,d}")

    a = summary["erased_excluded"]["macro_specificity"]
    b = summary["erased_as_negative"]["macro_specificity"]
    d = summary["unlabelled_as_background"]["macro_specificity"]
    print(f"\n  Reading the human's ERASURES as a 'not crack' verdict moves macro "
          f"specificity {a:.4f} -> {b:.4f} ({b - a:+.4f}).")
    print(f"  The dense convention gives {d:.4f}. So the adjudicated-vs-dense gap this "
          f"paper reports as {d - a:+.4f} is {d - b:+.4f} under the other reading of the "
          f"same marks: {100 * abs((d - b) - (d - a)) / max(abs(d - a), 1e-9):.0f}% of the "
          f"headline moves on a convention choice INSIDE the adjudicated convention.")
    print(f"\n  Neither number is 'the' answer and this script does not pick one. An erasure "
          f"is either the human saying 'not a crack' or the human saying 'do not consider "
          f"this' -- off-specimen background, an artefact, something outside the area of "
          f"interest -- and nothing on disk distinguishes them. Closing it needs the "
          f"annotator, or an interface that records WHY a region was erased. Until then the "
          f"honest report is the interval.")
    json.dump({"per_image": rows, "summary": summary,
               "interval_note": (
                   "Macro specificity under the adjudicated convention lies between "
                   f"{min(a, b):.4f} and {max(a, b):.4f} depending on whether an erasure is "
                   "read as a not-crack verdict. The corpus cannot decide it.")},
              open(_dc.out_for(OUT, detector).replace(".json", "_report.json"), "w"), indent=1)
    print(f"\n  -> {_dc.out_for(OUT, detector).replace('.json', '_report.json')}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--detector", choices=_dc.VALID, default=DETECTOR,
                    help="which detector to measure; results land in a per-detector file")
    a = ap.parse_args()
    if a.report:
        report(detector=a.detector)
    else:
        run(a.shard, a.nshard, detector=a.detector)
