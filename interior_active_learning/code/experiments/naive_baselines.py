"""Does the two-pass pipeline actually beat a naive segmentation script?

WHY THIS EXISTS
docs/MODEL_VALIDATION_BENCHMARK.md compares six classifiers -- but all six run on the
SAME 8 hand-built features, over the same darkness-thresholded candidates. That answers
"which classifier is best on my features", not "are the features and the machinery worth
anything". A reviewer's first question about any segmentation pipeline is whether it beats
a two-line baseline, and until now this project had no answer.

So this scores, on identical pixels under an identical protocol:

  otsu            global Otsu threshold on the flattened image, nothing else
  otsu_clean      Otsu plus the same small-object removal the pipeline uses
  frangi          Frangi ridge filter thresholded at a percentile, size-filtered
  frangi_dark     Frangi ridges intersected with "darker than the median"
  pipeline        the deployed two-pass detector (darkness + LogisticRegression, then
                  interior/concavity/bridge candidates)

PROTOCOL, AND WHY IT IS THE ONLY HONEST ONE HERE
  * Scored ONLY on pixels a human actually adjudicated (mask 1 or 2). Mask 0 means
    UNREVIEWED, and counting it as background is how this project previously produced
    flattering specificity numbers.
  * The pipeline runs with human corrections and the override ledger NEUTRALISED, via
    proposal_harness._without_human_input(). Without that the pipeline's own output
    contains the ground truth and scores ~1.000 -- which is exactly the circularity that
    made an earlier sweep report +0.0000 for nine different proposers.
  * Every method sees the same flattened image and the same held pixels. No method gets a
    threshold tuned on the image it is scored on.

WHAT A RESULT HERE MEANS
If the pipeline does not clearly beat frangi_dark, that is worth knowing BEFORE publication
rather than after. It would not make the tool useless -- the calibration, unit discipline,
provenance and gating are independent of the detector -- but it would mean the detector
should be swapped for something better (or fed from ilastik / micro-sam) rather than
defended.

    python3 naive_baselines.py            # all benchmark cases
    python3 naive_baselines.py IMG1 IMG2  # named images
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
from skimage import filters, morphology

from proposal_harness import CASES, _score, _without_human_input, load_correction_mask

sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "code")))

from detect_cracks import (clean_mask, compute_vesselness, find_field_of_view,
                           flatten_background, load_as_uint8)
from common import contrast_kwargs_for
import unified_pipeline as up

MIN_AREA = 40


def _stage(name):
    """Flattened image, vesselness, and the pipeline's own crack mask -- corrections OFF."""
    img8 = load_as_uint8(os.path.join(up.ORIGINAL_DIR, f"{name}.tif"),
                         **contrast_kwargs_for(name))
    x0, y0, x1, y1 = find_field_of_view(img8)
    img8 = img8[y0:y1, x0:x1]
    flat = flatten_background(img8)
    ves = compute_vesselness(flat)
    with _without_human_input():
        st = up.run_unified_pipeline(name)
    lab, df = st["labeled"], st["df"]
    pipe = np.isin(lab, df.loc[df["IsCrack"], "Label"].tolist())
    return flat, ves, pipe


# ------------------------------------------------------------------ baselines
def otsu(flat, ves):
    return flat < filters.threshold_otsu(flat)


def otsu_clean(flat, ves):
    return clean_mask(otsu(flat, ves), min_area_px=MIN_AREA)


def frangi(flat, ves, q=98.0):
    """Ridge response over a percentile. Vesselness is already computed by the pipeline,
    so this baseline costs nothing extra -- which is rather the point."""
    m = ves > np.percentile(ves, q)
    return morphology.remove_small_objects(m, min_size=MIN_AREA)


def frangi_dark(flat, ves, q=97.0, k=1.0):
    """Ridges that are also darker than the frame's median. Two lines, no model."""
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat.astype(np.float32) - med))) or 1.0
    m = (ves > np.percentile(ves, q)) & (flat < med - k * mad)
    return morphology.remove_small_objects(m, min_size=MIN_AREA)


BASELINES = {"otsu": otsu, "otsu_clean": otsu_clean,
             "frangi": frangi, "frangi_dark": frangi_dark}


def run(names=None):
    names = names or CASES
    rows = {k: [] for k in list(BASELINES) + ["pipeline"]}
    for name in names:
        try:
            flat, ves, pipe = _stage(name)
        except Exception as e:
            print(f"  {name}: skipped ({type(e).__name__}: {e})", flush=True)
            continue
        m = load_correction_mask(name, flat.shape)
        if m is None:
            print(f"  {name}: no usable correction mask", flush=True)
            continue

        class _C:
            human_crack = (m == 1)
            human_not = (m == 2)

            @property
            def adjudicated(self):
                return self.human_crack | self.human_not
        case = _C()
        if not case.adjudicated.any():
            print(f"  {name}: nothing adjudicated", flush=True)
            continue

        print(f"  {name}  ({int(case.adjudicated.sum()):,} adjudicated px)", flush=True)
        for label, fn in list(BASELINES.items()) + [("pipeline", None)]:
            t = time.time()
            pred = pipe if fn is None else fn(flat, ves)
            s = _score(pred, case)
            s["seconds"] = time.time() - t
            rows[label].append(s)
            print(f"     {label:12s} f1 {s['f1']:.3f}  recall {s['recall']:.3f}  "
                  f"spec {s['specificity']:.3f}  prec {s['precision']:.3f}", flush=True)

    print("\n  MEANS over the images scored")
    print(f"  {'method':14s} {'f1':>7s} {'recall':>7s} {'spec':>7s} {'prec':>7s} {'n':>3s}")
    means = {}
    for label, rs in rows.items():
        if not rs:
            continue
        means[label] = {k: float(np.mean([r[k] for r in rs]))
                        for k in ("f1", "recall", "specificity", "precision")}
        print(f"  {label:14s} {means[label]['f1']:7.3f} {means[label]['recall']:7.3f} "
              f"{means[label]['specificity']:7.3f} {means[label]['precision']:7.3f} "
              f"{len(rs):3d}")

    if "pipeline" in means:
        best_naive = max((k for k in means if k != "pipeline"),
                         key=lambda k: means[k]["f1"], default=None)
        if best_naive:
            d = means["pipeline"]["f1"] - means[best_naive]["f1"]
            print(f"\n  pipeline minus best naive ({best_naive}): {d:+.3f} f1")
            if d <= 0:
                print("  THE PIPELINE DOES NOT BEAT THE NAIVE BASELINE on these images. "
                      "That is a result to act on, not to bury: the detector is the part "
                      "to replace, and the calibration/provenance/gating layer is "
                      "independent of it.")
            elif d < 0.05:
                print("  The margin is small. Worth stating explicitly in any write-up "
                      "rather than letting a reader assume the machinery is doing more "
                      "than it is.")
    return means


if __name__ == "__main__":
    run(sys.argv[1:] or None)
