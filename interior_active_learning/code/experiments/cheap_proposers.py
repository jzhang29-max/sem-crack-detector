"""
Candidate proposers that cost ~nothing, measured against SAM's contribution.

THE IDEA
SAM raises measured f1 from 0.715 to 0.776, entirely through recall. But it is not
part of the model: the classifier is a LogisticRegression over 8 features and is
byte-identical whether SAM is installed. SAM's only job is to PROPOSE regions the
single global darkness threshold never found; the same classifier then judges them.

Characterised on a restored SAM overlay, its finds are: elongation median 3.5
(quartiles 2.8-5.5), mean brightness 0.34x the image median, area median 289 px
(145-1026). Dark, elongated, small-to-medium. Nothing about that needs a 2.4 GB
transformer -- it needs a proposer that is not one global threshold.

So each proposer here returns extra candidate pixels, and proposal_harness scores
them through the PRODUCTION classifier at its real threshold, against the human
correction masks, on adjudicated pixels only. A proposer cannot buy recall by
relaxing the decision; it can only buy it by finding candidates the model already
likes.

    python3 cheap_proposers.py            # cache the cases, then sweep every proposer
    python3 cheap_proposers.py --list     # names only
"""
import argparse
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from proposal_harness import CASES, Case, evaluate, load_case

# v2: v1 pickles held correction-contaminated pipeline_crack (baseline recall 1.000),
# which is what made every proposer score exactly +0.0000. Bumping the version
# rather than reusing them, since a stale cache would reproduce the same fake result.
CACHE = os.path.join(_HERE, ".proposal_cache", "v2")


# --------------------------------------------------------------------------- cases
def cached_case(name):
    """Cases are cached because building one runs the whole pipeline (up to ~90 s on
    a 25-megapixel frame). Without this, sweeping 8 proposers over 5 images would
    spend an hour re-deriving identical inputs."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{name}.pkl")
    if os.path.exists(p):
        with open(p, "rb") as fh:
            return pickle.load(fh)
    case = load_case(name)
    with open(p, "wb") as fh:
        pickle.dump(case, fh, protocol=4)
    return case


# ---------------------------------------------------------------------- proposers
def looser_threshold(case, k=2.0):
    """The pipeline's own idea, one step less strict: darker than median - k*MAD."""
    f = case.flat.astype(np.float32)
    med = float(np.median(f))
    mad = float(np.median(np.abs(f - med))) or 1.0
    return f < (med - k * mad)


def hysteresis_dark(case, hi=3.0, lo=1.5):
    """Seed on confidently dark pixels, then grow through merely-dim ones.

    A crack fades at its tips, which is exactly where a single global threshold cuts
    it short -- and a truncated region reads as small and round to the classifier
    rather than long and thin.
    """
    f = case.flat.astype(np.float32)
    med = float(np.median(f))
    mad = float(np.median(np.abs(f - med))) or 1.0
    strong = f < (med - hi * mad)
    weak = f < (med - lo * mad)
    lab = measure.label(weak, connectivity=2)
    keep = np.zeros_like(weak)
    hit = np.unique(lab[strong])
    keep[np.isin(lab, hit[hit != 0])] = True
    return keep


def vesselness_ridges(case, q=99.0):
    """Top-percentile ridge response. Vesselness is already computed by the pipeline,
    so this proposal is free."""
    v = case.vesselness.astype(np.float32)
    thr = float(np.percentile(v, q))
    return v > thr


def black_tophat(case, sizes=(5, 11), q=99.0):
    """Black top-hat: structures darker than their own surroundings, which finds
    cracks lying on a locally dark background that a global threshold treats as all
    one blob."""
    out = np.zeros(case.flat.shape, bool)
    f = case.flat
    for s in sizes:
        th = morphology.black_tophat(f, morphology.disk(s))
        out |= th > float(np.percentile(th, q))
    return out


def local_adaptive(case, block=101, offset=6):
    """Locally adaptive threshold: the same crack contrast counts everywhere, rather
    than being judged against the whole frame's median."""
    f = case.flat.astype(np.float32)
    try:
        loc = filters.threshold_local(f, block_size=block, offset=offset)
    except Exception:
        return np.zeros(f.shape, bool)
    return f < loc


def h_minima(case, h=8):
    """Shallow dark basins the global cut misses entirely."""
    f = case.flat.astype(np.float32)
    try:
        ext = morphology.h_minima(f, h)
    except Exception:
        return np.zeros(f.shape, bool)
    return ext.astype(bool)


def mser_like(case, deltas=(4, 8), q=98.5):
    """Cheap stand-in for MSER: regions stable across two darkness levels."""
    f = case.flat.astype(np.float32)
    med = float(np.median(f))
    mad = float(np.median(np.abs(f - med))) or 1.0
    out = np.zeros(f.shape, bool)
    for d in deltas:
        a = f < (med - d * mad * 0.5)
        b = f < (med - (d + 1) * mad * 0.5)
        # "stable" = present at both levels without collapsing
        out |= a & ndi.binary_dilation(b, morphology.disk(1))
    return out


def elongation_gated_dark(case, k=2.0, min_elong=2.5):
    """Loose darkness, then keep only elongated components BEFORE the classifier.

    Not a shortcut past the model: it removes obviously-round proposals so the
    classifier spends its budget on plausible crack shapes, which is the shape prior
    SAM effectively supplies.
    """
    cand = looser_threshold(case, k) & (case.labeled == 0)
    if not cand.any():
        return cand
    lab = measure.label(cand, connectivity=2)
    keep = np.zeros_like(cand)
    for pr in measure.regionprops(lab):
        if pr.area < 40:
            continue
        minor = max(float(pr.axis_minor_length), 1e-6)
        if float(pr.axis_major_length) / minor >= min_elong:
            keep[pr.slice][pr.image] = True
    return keep


def union_best(case):
    """Hysteresis plus ridge response: the two mechanisms that address different
    failures -- faded crack tips, and thin ridges below the global cut."""
    return hysteresis_dark(case) | vesselness_ridges(case)


def sam_proposer(case):
    """SAM itself, measured through THIS harness so the comparison is apples-to-apples.

    The +0.0610 f1 figure quoted elsewhere in this project comes from the hybrid
    benchmark, a different measurement path with a different baseline. Quoting it as
    the bar for these proposers would be comparing two numbers that were never
    computed the same way. sam_crack_mask() returns masks the production classifier
    has already accepted, so this is passed through with already_classified=True.
    """
    from hybrid_detect import sam_crack_mask
    from proposal_harness import _bundle
    return sam_crack_mask(case.img8, case.flat, case.vesselness, _bundle())


#: SAM is excluded from the default sweep: ~480 s/image against ~5 s for the rest.
#: Measure it with `--only sam` when a same-harness reference is wanted.
PROPOSERS = {
    "looser_threshold": looser_threshold,
    "hysteresis_dark": hysteresis_dark,
    "vesselness_ridges": vesselness_ridges,
    "black_tophat": black_tophat,
    "local_adaptive": local_adaptive,
    "h_minima": h_minima,
    "mser_like": mser_like,
    "elongation_gated": elongation_gated_dark,
    "union_best": union_best,
    "sam": sam_proposer,
}


def sweep(names=None, cases=None):
    cases = cases or CASES
    loaded = []
    for n in cases:
        try:
            loaded.append(cached_case(n))
            print(f"  case ready: {n}", flush=True)
        except Exception as e:
            print(f"  case skipped: {n} ({type(e).__name__}: {e})", flush=True)
    print()
    results = {}
    for pname, fn in PROPOSERS.items():
        if names and pname not in names:
            continue
        if not names and pname == "sam":
            continue   # opt in with --only sam
        rows, t0 = [], time.time()
        for case in loaded:
            try:
                rows.append(evaluate(case, fn(case), already_classified=(pname == "sam")))
            except Exception as e:
                print(f"    {pname} failed on {case.name}: {type(e).__name__}: {e}", flush=True)
        if not rows:
            continue
        results[pname] = dict(
            f1=float(np.mean([r["delta_f1"] for r in rows])),
            rec=float(np.mean([r["delta_recall"] for r in rows])),
            spec=float(np.mean([r["delta_spec"] for r in rows])),
            secs=(time.time() - t0) / max(len(rows), 1),
            n=len(rows))
        r = results[pname]
        print(f"  {pname:18s} f1 {r['f1']:+.4f}  recall {r['rec']:+.4f}  "
              f"spec {r['spec']:+.4f}  {r['secs']:5.1f}s/img  (n={r['n']})", flush=True)
    print(f"\n  {'SAM (for reference)':18s} f1 +0.0610  recall +0.0810  spec -0.0810  ~480.0s/img")
    if results:
        best = max(results.items(), key=lambda kv: kv[1]["f1"])
        print(f"\n  best cheap proposer: {best[0]}  f1 {best[1]['f1']:+.4f} "
              f"at {best[1]['secs']:.1f}s/image")
        print(f"  fraction of SAM's gain recovered: {100*best[1]['f1']/0.061:.0f}%")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.list:
        print("\n".join(PROPOSERS))
    else:
        sweep(names=a.only)
