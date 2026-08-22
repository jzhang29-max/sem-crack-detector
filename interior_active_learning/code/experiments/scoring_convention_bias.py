"""How much does "unlabelled = background" distort the metrics a paper reports?

THE QUESTION
Sparse annotation is universal in micrograph segmentation: a human marks some crack and
some not-crack and leaves the rest alone. What happens to the unmarked pixels at scoring
time is a choice, and it is almost never stated. Two conventions exist:

  adjudicated       score only on pixels a human actually marked 1 (crack) or 2 (not-crack)
  unlabelled=bg     treat every pixel not marked crack as not-crack

The second is what a binary mask forces on you, and every tool surveyed here emits one
(Fiji, CVAT, micro-sam, the commercial suites; ilastik sidesteps it by reporting no metrics
at all). This project refuses it -- correction masks carry 0 = UNREVIEWED as a distinct
state -- and that refusal was, until this script, an unquantified design opinion.

WHY IT MATTERS BEYOND THIS REPO
Across the 38 masks in this corpus, by pixel: crack 6.727%, not-crack 0.034%,
UNREVIEWED 92.901%, erased 0.338% -- summing to 100%. So 6.761% is adjudicated and treating
the unreviewed remainder as negative inflates the negative class by roughly 2721x, and by up
to 2.5e7x on a single frame. Any specificity or precision computed that way is not
describing the annotator's judgement, it is describing the annotator's STAMINA.

Note the honest fragility in those same numbers: adjudicated NEGATIVES are 0.034% of pixels,
about three parts in ten thousand. Specificity under the exclusion convention is estimated
from that pool, so it is a high-variance quantity and must be reported as one. The dense
convention's real attraction is that it always returns a number; that is why it persists,
and saying so makes this a diagnosis rather than an accusation.

THE DENOMINATORS DIFFER, DELIBERATELY
Three numbers in this file use three denominators: 62 images in the corpus, 38 masks with
any verdict, and 10 masks carrying BOTH classes (the only ones where specificity exists at
all). They are not interchangeable and each is stated with its own.

WHAT IS HELD CONSTANT
One prediction, one ground truth, two scoring conventions. The prediction is the pipeline's
own output with human corrections and the override ledger NEUTRALISED, so the model cannot
see the answer -- without that the prediction contains the labels and everything scores
~1.000, the circularity that once made an unrelated sweep report +0.0000 for nine methods.

THE CONTROL THAT PROVES THE MECHANISM
Recall must be IDENTICAL under both conventions: it is tp/(tp+fn), and neither term
involves a negative. If recall moves, this script has a bug, not a finding. It is asserted
below rather than merely reported.

    python3 scoring_convention_bias.py            # every image with both classes marked
    python3 scoring_convention_bias.py IMG1 IMG2  # named images
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

import numpy as np
from PIL import Image

import detector_config as _dc
import unified_pipeline as up
from common import ORIGINAL_DIR, PAINT_DIR, load_correction_mask

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(_HERE, "scoring_convention_bias.json")
#: Which detector this experiment measures. every published figure from this experiment was measured on the bare detector.
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
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"recall": rec, "specificity": spec, "precision": prec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def eligible():
    """Images whose mask carries BOTH classes -- the only ones where specificity exists."""
    out = []
    for f in sorted(os.listdir(PAINT_DIR)):
        if not f.endswith("_correction_mask.png"):
            continue
        n = f[:-len("_correction_mask.png")]
        if not os.path.exists(os.path.join(ORIGINAL_DIR, f"{n}.tif")):
            continue
        a = np.asarray(Image.open(os.path.join(PAINT_DIR, f)))
        if a.ndim > 2:
            a = a[..., 0]
        if (a == 1).any() and (a == 2).any():
            out.append(n)
    out.sort(key=lambda n: os.path.getsize(os.path.join(ORIGINAL_DIR, f"{n}.tif")))
    return out


def run(names=None, detector=None):
    # PINNED, not inherited. run_unified_pipeline defaults to SAM 2 refinement since commit
    # 4d97602, so an experiment that does not say which detector it wants silently measures a
    # different one than its output is labelled with. There is no default here on purpose.
    detector = detector or DETECTOR
    up.SAM2_MODE = detector
    names = names or eligible()
    print(f"{len(names)} image(s) carry both a crack and a not-crack verdict\n")
    print(f"{'image':32s} {'spec adj':>9s} {'spec bg':>8s} {'prec adj':>9s} "
          f"{'prec bg':>8s} {'adj %':>7s}")
    rows = []
    for n in names:
        try:
            with _without_human_input():
                st = up.run_unified_pipeline(n)
            lab, df = st["labeled"], st["df"]
            pred = np.isin(lab, df.loc[df["IsCrack"], "Label"].tolist())
            m = load_correction_mask(n, lab.shape)
            if m is None:
                print(f"{n:32s} no usable mask")
                continue
            crack, neg = (m == 1), (m == 2)
            if not (crack.any() and neg.any()):
                continue
            adj = _score(pred, crack, neg)          # this project's convention
            bg = _score(pred, crack, ~crack)        # unlabelled = background
            # THE CONTROL. Recall cannot depend on how negatives are defined.
            assert abs(adj["recall"] - bg["recall"]) < 1e-12, (
                f"{n}: recall moved between conventions ({adj['recall']} vs "
                f"{bg['recall']}) -- that is impossible, so this script is wrong")
            frac = float((crack | neg).sum()) / crack.size
            rows.append({"image": n, "adjudicated_fraction": frac,
                         "adjudicated": adj, "unlabelled_as_background": bg})
            print(f"{n:32s} {adj['specificity']:9.3f} {bg['specificity']:8.3f} "
                  f"{adj['precision']:9.3f} {bg['precision']:8.3f} {100*frac:6.2f}%",
                  flush=True)
        except AssertionError:
            # THE CONTROL MUST HALT. It was inside this try, so a fired assertion was caught
            # by the `except Exception` below, printed as one image's failure, and the run
            # carried on publishing means from the rest -- a control that cannot stop the
            # experiment certifies nothing. Recall is identical under both conventions by
            # construction; if it ever moves, every number this script prints is suspect and
            # the right outcome is a crash, not a footnote.
            raise
        except Exception as e:
            print(f"{n:32s} FAILED {type(e).__name__}: {e}", flush=True)

    if not rows:
        print("\nnothing scored")
        return None

    def mean(conv, k):
        return float(np.mean([r[conv][k] for r in rows]))

    def micro(conv):
        """Pooled four-cell metrics. Reported ALONGSIDE the macro means, never instead.

        A reader who assumes micro-averaging will try to recompute f1 from the mean
        precision and recall, find it does not reconcile, and conclude there is an
        arithmetic error. There is not -- macro means of a ratio do not obey the ratio's
        algebra -- but the only defence is to label the averaging and publish both. The
        two also differ a lot here (macro f1 0.638 vs micro 0.334), because the images
        differ enormously in how much was adjudicated.
        """
        tp = sum(r[conv]["tp"] for r in rows)
        fp = sum(r[conv]["fp"] for r in rows)
        fn = sum(r[conv]["fn"] for r in rows)
        tn = sum(r[conv]["tn"] for r in rows)
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        spec = tn / max(tn + fp, 1)
        return {"recall": rec, "precision": prec, "specificity": spec,
                "f1": 2 * prec * rec / max(prec + rec, 1e-9),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    print(f"\nMACRO means over {len(rows)} images (mean of per-image metrics)")
    print(f"  {'metric':12s} {'adjudicated':>12s} {'unlabelled=bg':>14s} {'delta':>9s}")
    summary = {}
    for k in ("recall", "specificity", "precision", "f1"):
        a, b = mean("adjudicated", k), mean("unlabelled_as_background", k)
        summary[k] = {"adjudicated": a, "unlabelled_as_background": b, "delta": b - a}
        print(f"  {k:12s} {a:12.4f} {b:14.4f} {b - a:+9.4f}")

    ma, mb = micro("adjudicated"), micro("unlabelled_as_background")
    print(f"\nMICRO (pooled four cells across the same {len(rows)} images)")
    print(f"  {'metric':12s} {'adjudicated':>12s} {'unlabelled=bg':>14s} {'delta':>9s}")
    micro_summary = {}
    for k in ("recall", "specificity", "precision", "f1"):
        micro_summary[k] = {"adjudicated": ma[k], "unlabelled_as_background": mb[k],
                            "delta": mb[k] - ma[k]}
        print(f"  {k:12s} {ma[k]:12.4f} {mb[k]:14.4f} {mb[k] - ma[k]:+9.4f}")
    print(f"\n  four-cell counts, adjudicated : tp {ma['tp']:,} fp {ma['fp']:,} "
          f"fn {ma['fn']:,} tn {ma['tn']:,}")
    print(f"  four-cell counts, unlabelled=bg: tp {mb['tp']:,} fp {mb['fp']:,} "
          f"fn {mb['fn']:,} tn {mb['tn']:,}")
    print(f"  The adjudicated true-negative pool is {ma['tn']:,} px against "
          f"{mb['tn']:,} under the dense convention -- a factor of "
          f"{mb['tn'] / max(ma['tn'], 1):.0f}. Specificity is estimated from that small "
          f"pool, which is the honest fragility of this result and belongs beside it.")

    print(f"\n  Recall delta is {summary['recall']['delta']:+.1e} -- identical by "
          f"construction, which is the control.")
    print(f"  Specificity is inflated by {summary['specificity']['delta']:+.3f} and "
          f"precision deflated by {summary['precision']['delta']:+.3f}: the convention "
          f"moves the two in OPPOSITE directions, so a paper reporting specificity this "
          f"way flatters itself while one reporting precision punishes itself, and neither "
          f"states which convention was used.")
    # HOW CONCENTRATED IS THE MACRO GAP? A macro mean weights every image equally while
    # their adjudicated-negative pools differ by 500x, so a frame with 256 negative pixels
    # counts as much as one with 134,039. Two frames here have an adjudicated specificity of
    # exactly 0.000 -- no true negatives at all -- and they are the two largest contributors.
    # The micro figures below are the counterweight and are already reported; this states the
    # concentration outright so nobody has to derive it.
    _g = sorted(((r["unlabelled_as_background"]["specificity"]
                  - r["adjudicated"]["specificity"]), r["image"],
                 r["adjudicated"]["tn"] + r["adjudicated"]["fp"],
                 r["adjudicated"]["specificity"]) for r in rows)
    _tot = sum(x[0] for x in _g)
    _degen = [x for x in _g if x[3] == 0.0]
    concentration = None
    if _tot > 0 and _degen:
        _rest = [x for x in _g if x[3] != 0.0]
        concentration = {
            "n_frames_with_zero_adjudicated_specificity": len(_degen),
            "their_images": [x[1] for x in _degen],
            "their_negative_pools_px": [x[2] for x in _degen],
            "their_share_of_the_summed_gap": _tot and sum(x[0] for x in _degen) / _tot,
            "macro_gap_all_frames": _tot / len(_g),
            "macro_gap_excluding_them": (sum(x[0] for x in _rest) / len(_rest)
                                         if _rest else None)}
        print(f"\n  CONCENTRATION. {len(_degen)} of {len(_g)} frames have an adjudicated "
              f"specificity of exactly 0.000, estimated from "
              f"{' and '.join(f'{x[2]:,}' for x in _degen)} negative pixels, and they supply "
              f"{100 * concentration['their_share_of_the_summed_gap']:.0f}% of the summed "
              f"gap. Dropping them moves the macro gap "
              f"{concentration['macro_gap_all_frames']:+.4f} -> "
              f"{concentration['macro_gap_excluding_them']:+.4f}. Both figures are real; the "
              f"macro mean simply weights a 256-pixel estimate like a 134,039-pixel one, "
              f"which is why the micro figures are reported beside it.")

    frac = float(np.mean([r["adjudicated_fraction"] for r in rows]))
    print(f"  Mean adjudicated fraction: {100 * frac:.2f}% -- the other "
          f"{100 * (1 - frac):.2f}% is what the second convention silently converts into "
          f"negatives.")

    json.dump({"detector": _dc.stamp(detector),
               "n_images": len(rows), "per_image": rows,
               "macro_gap_concentration": concentration,
               "macro_means": summary, "micro": micro_summary,
               "averaging_note": ("macro_means are means of per-image metrics; micro pools "
                                  "the four cells. They are not interconvertible and both "
                                  "are reported so neither can be mistaken for the other."),
               "mean_adjudicated_fraction": frac}, open(_dc.out_for(OUT, detector), "w"), indent=1)
    print(f"\n  -> {_dc.out_for(OUT, detector)}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=_dc.VALID, default=DETECTOR)
    ap.add_argument("images", nargs="*")
    a = ap.parse_args()
    run(a.images or None, detector=a.detector)
