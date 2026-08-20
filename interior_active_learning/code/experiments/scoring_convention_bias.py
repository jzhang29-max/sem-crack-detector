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
In this corpus only 6.83% of pixels are adjudicated; 92.83% are UNREVIEWED. Treating them
as negatives inflates the negative class by ~2721x in aggregate, and by up to 2.5e7x on a
single frame. Any specificity or precision computed that way is not describing the
annotator's judgement, it is describing the annotator's STAMINA.

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

import unified_pipeline as up
from common import ORIGINAL_DIR, PAINT_DIR, load_correction_mask

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(_HERE, "scoring_convention_bias.json")


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


def run(names=None):
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
        except Exception as e:
            print(f"{n:32s} FAILED {type(e).__name__}: {e}", flush=True)

    if not rows:
        print("\nnothing scored")
        return None

    def mean(conv, k):
        return float(np.mean([r[conv][k] for r in rows]))

    print(f"\nMEANS over {len(rows)} images")
    print(f"  {'metric':12s} {'adjudicated':>12s} {'unlabelled=bg':>14s} {'delta':>9s}")
    summary = {}
    for k in ("recall", "specificity", "precision", "f1"):
        a, b = mean("adjudicated", k), mean("unlabelled_as_background", k)
        summary[k] = {"adjudicated": a, "unlabelled_as_background": b, "delta": b - a}
        print(f"  {k:12s} {a:12.4f} {b:14.4f} {b - a:+9.4f}")

    print(f"\n  Recall delta is {summary['recall']['delta']:+.1e} -- identical by "
          f"construction, which is the control.")
    print(f"  Specificity is inflated by {summary['specificity']['delta']:+.3f} and "
          f"precision deflated by {summary['precision']['delta']:+.3f}: the convention "
          f"moves the two in OPPOSITE directions, so a paper reporting specificity this "
          f"way flatters itself while one reporting precision punishes itself, and neither "
          f"states which convention was used.")
    frac = float(np.mean([r["adjudicated_fraction"] for r in rows]))
    print(f"  Mean adjudicated fraction: {100 * frac:.2f}% -- the other "
          f"{100 * (1 - frac):.2f}% is what the second convention silently converts into "
          f"negatives.")

    json.dump({"n_images": len(rows), "per_image": rows, "means": summary,
               "mean_adjudicated_fraction": frac}, open(OUT, "w"), indent=1)
    print(f"\n  -> {OUT}")
    return summary


if __name__ == "__main__":
    run(sys.argv[1:] or None)
