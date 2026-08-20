"""Does the decision threshold -- a number no paper states -- move the published quantity?

THE QUESTION
Every reported crack count, crack length and crack density rests on a decision threshold
that turns a probability into a verdict. In this pipeline that number is 0.5, and 0.5 is not
a physical constant: it is what `bundle.get("threshold", 0.5)` returns when a model bundle
carries no calibrated threshold, which the deployed one does not. So the operating point is
a LIBRARY DEFAULT, inherited rather than chosen, and it is invisible in every figure it
produces.

This is the same failure family as the other results here -- software substituting a
plausible default for a stated choice -- but it bites harder, because unlike the scoring
convention it changes the SEGMENTATION, and therefore every downstream number at once.

WHAT WOULD MAKE IT SERIOUS, AND WHAT WOULD MAKE IT MERELY UNTIDY
Magnitude alone is not damning: if crack count doubles between 0.3 and 0.7 but the ORDERING
of the conditions being compared never changes, then a paper's qualitative conclusion
survives its undocumented default, and the honest verdict is "report the threshold, but no
published comparison was wrong". If instead the ordering flips inside a plausible range,
then the comparison a paper exists to make is decided by a number nobody wrote down.

So this script measures both, and reports the ordering result whichever way it falls. The
falsifiable prediction is the second one: an ordering flip somewhere in 0.3-0.7.

DESIGN
The superalloy trio (AS / Cast / HIP, one specimen each, same alloy and ambient condition)
is exactly the comparison a materials paper makes, so it is the test case. Frames are
sampled evenly by sorted name -- not smallest-first, which would bias toward sparse frames.

The threshold-independent half of the pipeline (load, field of view, flatten, dark segment,
clean, vesselness, candidate extraction) is computed ONCE per frame and memoised; only
classification, corrections, merging, Pass 2 and measurement repeat. Without that this
sweep costs one full pipeline run per (frame, threshold) and is not worth running.

Measurement calls the production `measure_stage`, not a copy of it. A sweep carrying its own
merge rule or its own censoring test would report the sensitivity of a measurement nobody
ships.

HUMAN INPUT IS NEUTRALISED, deliberately. Corrections and the override ledger are
authoritative in production and must be; here they would pin regions regardless of
threshold and mask the very sensitivity being measured.

    python3 threshold_sensitivity.py                  # 4 frames/specimen, 5 thresholds
    python3 threshold_sensitivity.py --per-spec 2
    python3 threshold_sensitivity.py --shard 0 --nshard 4     # one worker of four
"""
import argparse
import contextlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "code")))

import numpy as np
from PIL import Image

import aggregate as ag
import crack_measurements as cm
import unified_pipeline as up

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(_HERE, "threshold_sensitivity.json")

#: Production sits at PRODUCTION. The span is what a reasonable person might have picked
#: had they known there was a choice; it is not a search for an extreme.
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
PRODUCTION = 0.5

#: The comparison under test: three processing routes on one alloy at one condition.
TRIO = ("MAR_Amb_AS", "MAR_Amb_Cast", "MAR_Amb_HIP")

#: Functions of the image alone, not of the threshold. Memoising these is what makes the
#: sweep affordable; if any of them ever becomes threshold-dependent, this list is the bug.
MEMOISE = ("load_as_uint8", "find_field_of_view", "flatten_background",
           "segment_dark_regions", "clean_mask", "compute_vesselness",
           "extract_candidates")

_CACHE = {}
_CURRENT = [None]


def _install_memo():
    for fname in MEMOISE:
        orig = getattr(up, fname)
        if getattr(orig, "_memoised", False):
            continue

        def wrapped(*a, _orig=orig, _f=fname, **k):
            # The scalar kwargs are part of the key. extract_candidates is called with
            # min_area_px=40 for built-in candidates and min_area_px=1 for an imported
            # mask; keying on the name alone would serve one call's regions to the other.
            key = (_f, _CURRENT[0],
                   tuple(sorted((kk, vv) for kk, vv in k.items()
                                if isinstance(vv, (int, float, str, bool, type(None))))))
            if key not in _CACHE:
                _CACHE[key] = _orig(*a, **k)
            v = _CACHE[key]
            # extract_candidates hands back a label image and a DataFrame that the
            # pipeline then mutates in place (corrections relabel, merging rewrites
            # IsCrack). Returning the cached objects themselves would let threshold 0.4
            # inherit 0.3's edits, which is not a subtle error -- it is a monotone
            # pipeline silently accumulating.
            if _f == "extract_candidates":
                return v[0].copy(), v[1].copy()
            return v
        wrapped._memoised = True
        setattr(up, fname, wrapped)


@contextlib.contextmanager
def _at_threshold(t):
    """Force BOTH passes to t: Pass 1 through classify_with_model, Pass 2 through the
    bundle's threshold_default. Patching only one would report the sensitivity of half
    the detector."""
    real_cls, real_bundle = up.classify_with_model, up._load_unified_bundle
    real_corr, real_over = up.load_correction_mask, up.load_hard_overrides

    def cls(df, model_path, proba_threshold=None, **k):
        return real_cls(df, model_path, proba_threshold=t, **k)

    def bundle():
        b = real_bundle()
        if isinstance(b, dict):
            b = dict(b)
            b["threshold_default"] = t
        return b

    up.classify_with_model = cls
    up._load_unified_bundle = bundle
    up.load_correction_mask = lambda *a, **k: None
    up.load_hard_overrides = lambda *a, **k: None
    try:
        yield
    finally:
        up.classify_with_model, up._load_unified_bundle = real_cls, real_bundle
        up.load_correction_mask, up.load_hard_overrides = real_corr, real_over


def _summarise(rows, n_px):
    """Frame-level quantities, each one something a paper actually prints."""
    if not rows:
        return {"n_cracks": 0, "total_length_px": 0.0, "mean_length_px": None,
                "max_length_px": None, "area_fraction": 0.0, "n_censored": 0,
                "max_length_is_censored": None}
    L = np.array([r["SkeletonLength_px"] for r in rows], float)
    A = np.array([r["Area_px"] for r in rows], float)
    cens = [bool(r["LengthIsCensored"]) for r in rows]
    imax = int(np.argmax(L))
    return {"n_cracks": len(rows),
            "total_length_px": float(L.sum()),
            "mean_length_px": float(L.mean()),
            "max_length_px": float(L.max()),
            "area_fraction": float(A.sum()) / n_px,
            "n_censored": int(sum(cens)),
            # The longest crack is the most likely to run off the frame, so the headline
            # number is the one most often a lower bound rather than a length.
            "max_length_is_censored": cens[imax]}


def frames(per_spec):
    """Evenly spaced by sorted name, so the sample is not biased toward small frames."""
    out = {}
    for spec in TRIO:
        names = sorted(n for n in cm.all_images() if ag.specimen_key(n) == spec)
        if not names:
            continue
        k = min(per_spec, len(names))
        idx = np.linspace(0, len(names) - 1, k).round().astype(int)
        out[spec] = [names[i] for i in sorted(set(idx.tolist()))]
    return out


def run(per_spec=4, shard=0, nshard=1):
    _install_memo()
    plan = frames(per_spec)
    todo = [(s, n) for s in TRIO for n in plan.get(s, [])]
    mine = todo[shard::nshard]
    print(f"{len(mine)} frame(s) of {len(todo)} x {len(THRESHOLDS)} thresholds"
          f"{f'  [shard {shard}/{nshard}]' if nshard > 1 else ''}\n", flush=True)

    per_frame = []
    for spec, name in mine:
        _CURRENT[0] = name
        rec = {"specimen": spec, "image": name, "levels": {}}
        for t in THRESHOLDS:
            try:
                with _at_threshold(t):
                    st = up.run_unified_pipeline(name)
                rows, _ = cm.measure_stage(name, st)
                s = _summarise(rows, st["labeled"].size)
                rec["levels"][str(t)] = s
                print(f"  {name:34s} t={t:.2f}  n={s['n_cracks']:5d} "
                      f"totlen={s['total_length_px']:11.0f} "
                      f"area={100*s['area_fraction']:6.3f}%", flush=True)
            except Exception as e:
                print(f"  {name:34s} t={t:.2f}  FAILED {type(e).__name__}: {e}", flush=True)
        per_frame.append(rec)
        # 25-megapixel float arrays; holding two frames' worth is how this gets OOM-killed.
        for k in list(_CACHE):
            if k[1] == name:
                del _CACHE[k]

    out = OUT if nshard == 1 else OUT.replace(".json", f".shard{shard}.json")
    json.dump({"thresholds": list(THRESHOLDS), "production": PRODUCTION,
               "per_spec": per_spec, "per_frame": per_frame}, open(out, "w"), indent=1)
    print(f"\n  -> {out}")
    return per_frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-spec", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    run(a.per_spec, a.shard, a.nshard)
