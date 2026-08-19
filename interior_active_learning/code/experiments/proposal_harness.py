"""
Shared measurement harness for "can a cheap proposer replace SAM's contribution?".

WHY THIS EXISTS
SAM raises measured f1 from 0.715 to 0.776, and the gain is entirely recall (+8.1
points for -8.1 specificity). But SAM is not part of the model: the classifier is a
LogisticRegression over 8 features and is byte-identical whether SAM is installed.
SAM's only job is to PROPOSE regions the darkness threshold never found; the same
classifier then scores them. So the useful question is not "how do we shrink SAM"
but "what else can propose those regions for ~0 cost".

WHAT IS MEASURED, AND AGAINST WHAT
Not against SAM's output -- SAM is a means, not the target. Against the human
correction masks, which are the actual ground truth this project has. For each image
with adjudicated pixels:

    missed   = human_crack & ~pipeline_crack      what we want to recover
    forbidden= human_not_crack                    what we must not light up

A proposer returns a boolean mask of EXTRA candidate pixels. Those regions are then
scored by the production classifier through the same features and threshold the
pipeline uses, so a proposal only survives if the model already agrees with it. That
keeps any comparison honest: the proposer changes recall by finding candidates, not
by relaxing the decision.

Scored only on pixels a human actually adjudicated, because unreviewed pixels are
not evidence either way -- treating mask value 0 as "not crack" is the mistake that
produced this project's earlier bogus specificity numbers.

USAGE

    from proposal_harness import load_case, evaluate, CASES
    case = load_case("AS_24hr_BSE_Side_008")
    extra = my_proposer(case)              # boolean mask, same shape as case.img8
    print(evaluate(case, extra))           # dict of before/after recall, spec, f1

Baseline (extra = all False) reproduces the pipeline-only numbers, so any claimed
gain is a delta against a baseline you can see in the same run.
"""
import os
import sys
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "interior_active_learning", "code"))
sys.path.insert(0, os.path.join(ROOT, "code"))

import joblib
import numpy as np
from PIL import Image
from skimage import measure

from common import PAINT_DIR, PROD_MODEL_PATH, load_correction_mask
from detect_cracks import region_features_from_labeled
from unified_pipeline import run_unified_pipeline

Image.MAX_IMAGE_PIXELS = None
MIN_AREA = 40

#: Images with enough hand-marked not-crack pixels for specificity to mean anything.
#: The same five the benchmark uses, so numbers here are comparable to the README's.
CASES = ["AS_24hr_BSE_Side_008", "260708_316_H_b2_front_CBS_012",
         "260708_316_H_b2_front_CBS_015", "MAR_Amb_HIP_CBS_0006",
         "Cast_24hr_SE_Side_006"]


@dataclass
class Case:
    name: str
    img8: np.ndarray
    flat: np.ndarray
    vesselness: np.ndarray
    labeled: np.ndarray
    pipeline_crack: np.ndarray
    human_crack: np.ndarray
    human_not: np.ndarray

    @property
    def adjudicated(self):
        return self.human_crack | self.human_not

    @property
    def missed(self):
        """Human-marked crack the pipeline did not find: the recoverable ground."""
        return self.human_crack & ~self.pipeline_crack


def load_case(name):
    stage = run_unified_pipeline(name)
    labeled, df = stage["labeled"], stage["df"]
    pipe = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
    m = load_correction_mask(name, labeled.shape)
    if m is None:
        raise ValueError(f"{name}: no usable correction mask")
    return Case(name=name, img8=stage["img8"], flat=stage["flat"],
                vesselness=stage["vesselness"], labeled=labeled,
                pipeline_crack=pipe, human_crack=(m == 1), human_not=(m == 2))


def _bundle():
    if not hasattr(_bundle, "b"):
        _bundle.b = joblib.load(PROD_MODEL_PATH)
    return _bundle.b


def classify_extra(case, extra):
    """Keep only proposed regions the PRODUCTION classifier accepts.

    Same features, same scaler, same threshold as the pipeline. A proposer therefore
    cannot buy recall by lowering the bar -- it can only buy it by finding candidates
    the model already likes.
    """
    extra = np.asarray(extra, dtype=bool) & (case.labeled == 0)
    if not extra.any():
        return np.zeros_like(extra)
    b = _bundle()
    thr = float(b.get("threshold", 0.5))
    keep = np.zeros_like(extra)
    comp = measure.label(extra, connectivity=2)
    for pr in measure.regionprops(comp):
        if pr.area < MIN_AREA:
            continue
        sl = pr.slice
        _, fd = region_features_from_labeled(pr.image.astype(np.int32),
                                             case.flat[sl], case.vesselness[sl],
                                             min_area_px=MIN_AREA)
        if not len(fd):
            continue
        X = b["scaler"].transform(fd[b["feature_names"]].values[:1])
        if b["clf"].predict_proba(X)[0, 1] >= thr:
            keep[sl][pr.image] = True
    return keep


def _score(pred, case):
    adj = case.adjudicated
    tp = int((pred & case.human_crack).sum())
    fn = int((~pred & case.human_crack).sum())
    fp = int((pred & case.human_not).sum())
    tn = int((~pred & case.human_not).sum())
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(recall=rec, specificity=spec, precision=prec, f1=f1,
                adjudicated_px=int(adj.sum()))


def evaluate(case, extra=None, already_classified=False):
    """Before/after on this image. `extra` is a boolean mask of proposed pixels."""
    base = _score(case.pipeline_crack, case)
    if extra is None:
        return {"name": case.name, "baseline": base, "after": base,
                "delta_f1": 0.0, "delta_recall": 0.0, "delta_spec": 0.0}
    kept = np.asarray(extra, bool) if already_classified else classify_extra(case, extra)
    after = _score(case.pipeline_crack | kept, case)
    return {"name": case.name, "baseline": base, "after": after,
            "kept_px": int(kept.sum()),
            "delta_f1": after["f1"] - base["f1"],
            "delta_recall": after["recall"] - base["recall"],
            "delta_spec": after["specificity"] - base["specificity"]}


def run_all(proposer, names=None, verbose=True):
    """Evaluate a proposer across the benchmark images. Returns per-image + mean."""
    names = names or CASES
    rows = []
    for n in names:
        try:
            case = load_case(n)
        except Exception as e:
            if verbose:
                print(f"  {n}: skipped ({type(e).__name__}: {e})")
            continue
        t = time.time()
        extra = proposer(case)
        dt = time.time() - t
        r = evaluate(case, extra)
        r["seconds"] = dt
        rows.append(r)
        if verbose:
            print(f"  {n:32s} f1 {r['baseline']['f1']:.3f} -> {r['after']['f1']:.3f} "
                  f"({r['delta_f1']:+.3f})  recall {r['delta_recall']:+.3f}  "
                  f"spec {r['delta_spec']:+.3f}  {dt:.1f}s", flush=True)
    if rows and verbose:
        print(f"\n  MEAN over {len(rows)} images: f1 {np.mean([r['delta_f1'] for r in rows]):+.4f}  "
              f"recall {np.mean([r['delta_recall'] for r in rows]):+.4f}  "
              f"spec {np.mean([r['delta_spec'] for r in rows]):+.4f}  "
              f"{np.mean([r['seconds'] for r in rows]):.1f}s/image")
        print(f"  SAM, for comparison: f1 +0.061, recall +0.081, spec -0.081, ~480s/image")
    return rows


if __name__ == "__main__":
    print("baseline sanity check: a proposer that proposes nothing must change nothing")
    run_all(lambda case: np.zeros_like(case.pipeline_crack))
