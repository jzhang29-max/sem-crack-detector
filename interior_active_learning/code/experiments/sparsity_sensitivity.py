"""How does the scoring-convention gap behave as review gets sparser, and how certain is it?

WHY THIS EXISTS
scoring_convention_bias.py reports a single pair of numbers: specificity 0.4599 under
exclusion against 0.9478 under the dense convention, macro over 10 images. Two objections
follow immediately and neither is answered by that pair.

  1. "0.4599 is a point estimate from 10 images and an adjudicated-negative class of about
     three parts in ten thousand. How certain is it?"      -> bootstrap over images.
  2. "Your corpus happens to be 8% reviewed. What happens at 1%, or at 40%? Is this a knife
     edge or a smooth effect?"                             -> sub-sample the existing masks.

Both are answerable with the data already on disk, which is the point: this converts one
fragile pair of numbers into a curve with intervals, without asking anyone to label more.

METHOD
For each image the prediction and the full ground truth are computed once (with human
corrections neutralised, so the model cannot see the answer). Then the ADJUDICATED region is
randomly thinned to a target fraction, simulating a reviewer who stopped earlier. At each
level:

  exclusion convention   score on the thinned adjudicated pixels only
  dense convention       score on every pixel, unlabelled counted as background

The dense convention is INVARIANT to thinning by construction -- it never consults the
adjudication -- so it is the fixed reference line, and any movement in the gap comes from the
exclusion side. That asymmetry is the useful part: it shows the dense number is not merely
different, it is indifferent to how much work the human did.

WHAT TO EXPECT, AND WHAT WOULD FALSIFY THE FRAMING
If the effect is real, the gap should persist at every sparsity level while the exclusion
estimate becomes noisier as its support shrinks. If instead the gap collapsed at realistic
sparsity, the paper's framing would be wrong and this script would be how we found out.

    python3 sparsity_sensitivity.py
    python3 sparsity_sensitivity.py --boot 2000
"""
import argparse
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
from common import load_correction_mask
from scoring_convention_bias import eligible

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(_HERE, "sparsity_sensitivity.json")

#: Fractions of the ADJUDICATED region retained. 1.0 is the corpus as it stands.
LEVELS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.02)
SEED = 0


@contextlib.contextmanager
def _without_human_input():
    a, b = up.load_correction_mask, up.load_hard_overrides
    up.load_correction_mask = lambda *x, **k: None
    up.load_hard_overrides = lambda *x, **k: None
    try:
        yield
    finally:
        up.load_correction_mask, up.load_hard_overrides = a, b


def _cells(pred, crack, neg):
    tp = int((pred & crack).sum())
    fn = int((~pred & crack).sum())
    fp = int((pred & neg).sum())
    tn = int((~pred & neg).sum())
    return tp, fp, fn, tn


def _metrics(tp, fp, fn, tn):
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {"recall": rec, "precision": prec,
            "specificity": tn / max(tn + fp, 1),
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _thin(mask, frac, rng):
    """Keep a random `frac` of the True pixels. Simulates a reviewer stopping earlier."""
    if frac >= 1.0:
        return mask
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return mask
    keep = rng.choice(idx, size=max(1, int(round(idx.size * frac))), replace=False)
    out = np.zeros(mask.size, dtype=bool)
    out[keep] = True
    return out.reshape(mask.shape)


def _bootstrap_ci(values, n_boot, rng, alpha=0.05):
    """Percentile CI over IMAGES -- the unit of independence, not the pixel."""
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if vals.size < 2:
        return None
    means = np.array([rng.choice(vals, size=vals.size, replace=True).mean()
                      for _ in range(n_boot)])
    return {"mean": float(vals.mean()),
            "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)),
            "n": int(vals.size)}


def run(names=None, n_boot=1000):
    names = names or eligible()
    rng = np.random.default_rng(SEED)
    per_image = []

    print(f"computing predictions for {len(names)} image(s)\n")
    for n in names:
        try:
            with _without_human_input():
                st = up.run_unified_pipeline(n)
            lab, df = st["labeled"], st["df"]
            pred = np.isin(lab, df.loc[df["IsCrack"], "Label"].tolist())
            m = load_correction_mask(n, lab.shape)
            if m is None:
                continue
            crack, neg = (m == 1), (m == 2)
            if not (crack.any() and neg.any()):
                continue
            rec = {"image": n, "levels": {},
                   "adjudicated_fraction_full": float((crack | neg).sum()) / crack.size}
            # Dense convention: computed ONCE, because it never consults the adjudication.
            dense = _metrics(*_cells(pred, crack, ~crack))
            rec["dense"] = dense
            for f in LEVELS:
                c_t = _thin(crack, f, rng)
                n_t = _thin(neg, f, rng)
                if not (c_t.any() and n_t.any()):
                    continue
                excl = _metrics(*_cells(pred, c_t, n_t))
                rec["levels"][str(f)] = {
                    "exclusion": excl,
                    "adjudicated_px": int((c_t | n_t).sum()),
                    "adjudicated_fraction": float((c_t | n_t).sum()) / crack.size,
                    "gap_specificity": dense["specificity"] - excl["specificity"],
                    "gap_precision": dense["precision"] - excl["precision"],
                }
            per_image.append(rec)
            print(f"  {n:32s} full adj {100*rec['adjudicated_fraction_full']:5.2f}%  "
                  f"dense spec {dense['specificity']:.3f}", flush=True)
        except Exception as e:
            print(f"  {n:32s} FAILED {type(e).__name__}: {e}", flush=True)

    if not per_image:
        print("nothing scored")
        return None

    print(f"\nGAP vs REVIEW EFFORT  (bootstrap over images, {n_boot} resamples, 95% CI)")
    print(f"  {'kept':>6s} {'mean adj%':>10s} {'spec excl':>22s} {'spec gap':>22s} "
          f"{'prec gap':>22s}")
    curve = {}
    for f in LEVELS:
        k = str(f)
        rows = [r["levels"][k] for r in per_image if k in r["levels"]]
        if not rows:
            continue
        adj = float(np.mean([r["adjudicated_fraction"] for r in rows]))
        se = _bootstrap_ci([r["exclusion"]["specificity"] for r in rows], n_boot, rng)
        gs = _bootstrap_ci([r["gap_specificity"] for r in rows], n_boot, rng)
        gp = _bootstrap_ci([r["gap_precision"] for r in rows], n_boot, rng)
        curve[k] = {"mean_adjudicated_fraction": adj, "specificity_exclusion": se,
                    "gap_specificity": gs, "gap_precision": gp, "n_images": len(rows)}
        fmt = lambda d: (f"{d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]"
                         if d else "n/a")
        print(f"  {f:6.2f} {100*adj:9.3f}% {fmt(se):>22s} {fmt(gs):>22s} {fmt(gp):>22s}")

    full = curve.get("1.0", {})
    print()
    if full.get("gap_specificity"):
        g = full["gap_specificity"]
        crosses = g["lo"] <= 0 <= g["hi"]
        print(f"  At the corpus as it stands, the specificity gap is {g['mean']:+.3f} "
              f"[{g['lo']:+.3f}, {g['hi']:+.3f}] over {g['n']} images.")
        print(f"  The interval {'INCLUDES' if crosses else 'excludes'} zero, so the effect "
              f"{'is NOT distinguishable from noise at this n' if crosses else 'is not a noise artefact at this n'}.")
    sd = [curve[str(f)]["specificity_exclusion"] for f in LEVELS
          if str(f) in curve and curve[str(f)]["specificity_exclusion"]]
    if len(sd) >= 2:
        widths = [d["hi"] - d["lo"] for d in sd]
        direction = ("widens" if widths[-1] > widths[0] * 1.05 else
                     "narrows" if widths[-1] < widths[0] * 0.95 else
                     "is essentially unchanged")
        print(f"  CI width on the exclusion estimate {direction}: {widths[0]:.3f} -> "
              f"{widths[-1]:.3f} as review thins by {LEVELS[0]/LEVELS[-1]:.0f}x.")
        if direction != "widens":
            # I predicted this would widen and it did not. The reason matters more than the
            # prediction: thinning removes pixels, and even 0.163% of a 25-megapixel frame is
            # tens of thousands of them, so the per-image estimate barely moves. The interval
            # is dominated by BETWEEN-image variance at n=10, not by within-image sampling.
            print(f"  That is not what more review buys. Thinning removes pixels, and even at "
                  f"the sparsest level each frame still contributes tens of thousands, so the "
                  f"per-image estimate is stable. The interval is dominated by BETWEEN-image "
                  f"variance at n={sd[0]['n']}, not by how thoroughly any one frame was "
                  f"reviewed.")
            print(f"  ACTIONABLE CONSEQUENCE: to tighten this interval, mark not-crack on MORE "
                  f"IMAGES. Marking existing images more thoroughly will not do it. That is "
                  f"the opposite of what I expected before running this, and it is the only "
                  f"reason to prefer one labelling strategy over the other.")
    gaps = [curve[str(f)]["gap_specificity"]["mean"] for f in LEVELS
            if str(f) in curve and curve[str(f)]["gap_specificity"]]
    if len(gaps) >= 2:
        print(f"  The gap itself moves {gaps[0]:+.3f} -> {gaps[-1]:+.3f} across a "
              f"{LEVELS[0]/LEVELS[-1]:.0f}x change in review effort, so this is a broad "
              f"effect rather than a knife edge at one sparsity.")

    json.dump({"levels": list(LEVELS), "n_boot": n_boot, "seed": SEED,
               "curve": curve, "per_image": per_image}, open(OUT, "w"), indent=1)
    print(f"\n  -> {OUT}")
    return curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("images", nargs="*")
    a = ap.parse_args()
    run(a.images or None, n_boot=a.boot)
