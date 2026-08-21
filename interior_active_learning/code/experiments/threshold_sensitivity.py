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


def out_path(shard, nshard):
    return OUT if nshard == 1 else OUT.replace(".json", f".shard{shard}.json")


def _dump(path, thresholds, per_spec, per_frame):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"thresholds": list(thresholds), "production": PRODUCTION,
                   "per_spec": per_spec, "n_frames_done": len(per_frame),
                   "per_frame": per_frame}, fh, indent=1)
    os.replace(tmp, path)


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
        # Checkpoint after EVERY frame, not at the end. A frame costs minutes and this
        # competes for the machine, so a shard that gets killed at frame 3 of 4 used to
        # lose all three. Write to a temp file and replace, so a kill mid-write cannot
        # leave a half-parsed JSON that --report silently skips.
        _dump(out_path(shard, nshard), THRESHOLDS, per_spec, per_frame)

    out = out_path(shard, nshard)
    _dump(out, THRESHOLDS, per_spec, per_frame)
    print(f"\n  -> {out}")
    return per_frame


def report(paths=None):
    """Merge shard files and answer the two questions: how big, and does the order flip?"""
    import glob as _glob
    paths = paths or sorted(_glob.glob(OUT.replace(".json", ".shard*.json"))) or [OUT]
    per_frame = []
    for p in paths:
        try:
            per_frame += json.load(open(p))["per_frame"]
        except (OSError, ValueError, KeyError):
            continue
    if not per_frame:
        print("no shard results found")
        return None

    # Only frames measured at EVERY threshold may enter the comparison. A frame present at
    # 0.3 but missing at 0.7 would shift a condition's mean between columns for reasons
    # that have nothing to do with the threshold.
    keys = [str(t) for t in THRESHOLDS]
    complete = [r for r in per_frame if all(k in r["levels"] for k in keys)]
    dropped = len(per_frame) - len(complete)
    print(f"{len(complete)} frame(s) measured at all {len(keys)} thresholds"
          f"{f'; {dropped} incomplete frame(s) EXCLUDED' if dropped else ''}\n")
    if not complete:
        return None

    METRICS = [("n_cracks", "crack count", 0),
               ("total_length_px", "total length px", 0),
               ("mean_length_px", "mean length px", 1),
               ("area_fraction", "area fraction %", 3)]

    def spec_mean(spec, key, metric):
        vals = [r["levels"][key][metric] for r in complete
                if r["specimen"] == spec and r["levels"][key].get(metric) is not None]
        return float(np.mean(vals)) if vals else None

    specs = [s for s in TRIO if any(r["specimen"] == s for r in complete)]
    out = {"per_metric": {}, "n_frames": len(complete), "excluded_frames": dropped}

    for metric, label, nd in METRICS:
        print(f"{label}  (mean over frames, per specimen)")
        head = "  " + " ".join(f"{('t=' + k):>13s}" for k in keys)
        print(f"  {'specimen':16s}" + head[2:] + f" {'max/min':>9s}")
        rows, orderings = {}, {}
        for spec in specs:
            v = [spec_mean(spec, k, metric) for k in keys]
            if any(x is None for x in v):
                continue
            scale = 100.0 if metric == "area_fraction" else 1.0
            rows[spec] = v
            span = (max(v) / min(v)) if min(v) > 0 else float("inf")
            cells = " ".join(f"{x * scale:13.{nd}f}" for x in v)
            print(f"  {spec:16s}{cells} {span:9.3f}x")
        for i, k in enumerate(keys):
            orderings[k] = tuple(sorted(rows, key=lambda s: rows[s][i], reverse=True))
        distinct = {v for v in orderings.values()}
        stable = len(distinct) == 1
        worst = max((max(v) / min(v)) if min(v) > 0 else float("inf")
                    for v in rows.values()) if rows else None
        print(f"  ranking across thresholds: "
              f"{'STABLE  ' + ' > '.join(next(iter(distinct))) if stable else 'FLIPS'}")
        if not stable:
            for k in keys:
                print(f"      t={k}: {' > '.join(orderings[k])}")
        print()
        out["per_metric"][metric] = {
            "per_specimen": rows, "orderings": {k: list(v) for k, v in orderings.items()},
            "ranking_stable": stable, "worst_span_ratio": worst}

    flips = [m for m, d in out["per_metric"].items() if not d["ranking_stable"]]
    spans = {m: d["worst_span_ratio"] for m, d in out["per_metric"].items()}

    # WITHIN one specimen, how much more sensitive is the count than the area? Comparing
    # worst-case spans across metrics is apples to apples but mixes specimens, and the
    # sharper comparison is the one a single paper would actually make about a single
    # material: the same frames, the same threshold move, two quantities.
    ratios = {}
    for spec in specs:
        c = out["per_metric"].get("n_cracks", {}).get("per_specimen", {}).get(spec)
        a = out["per_metric"].get("area_fraction", {}).get("per_specimen", {}).get(spec)
        if not c or not a or min(c) <= 0 or min(a) <= 0:
            continue
        cs, as_ = max(c) / min(c), max(a) / min(a)
        # Excess movement over unity: a span of 1.28 is 28% of movement, and 1.018 is 1.8%.
        # Dividing the spans themselves (1.28/1.018 = 1.26) badly understates a 15x
        # difference in how much each quantity actually moved.
        ratios[spec] = {"count_span": cs, "area_span": as_,
                        "count_pct": 100 * (cs - 1), "area_pct": 100 * (as_ - 1),
                        "times_more_sensitive": ((cs - 1) / (as_ - 1)) if as_ > 1 else None}
    out["count_vs_area_sensitivity"] = ratios

    # Where in the range does the movement happen? A quantity that slides gently is a
    # different problem from one with a cliff: the second means an operating point can sit
    # right beside a discontinuity, and averaging the range hides it.
    steps = {}
    for metric, d in out["per_metric"].items():
        worst = None
        for spec, vals in d["per_specimen"].items():
            for i in range(len(vals) - 1):
                if vals[i] <= 0:
                    continue
                ch = abs(vals[i + 1] - vals[i]) / vals[i]
                if worst is None or ch > worst["rel_change"]:
                    worst = {"specimen": spec, "from_threshold": keys[i],
                             "to_threshold": keys[i + 1], "rel_change": ch}
        if worst:
            steps[metric] = worst
    out["largest_single_step"] = steps
    print("WHAT THIS MEANS")
    biggest = max(spans, key=lambda m: spans[m] or 0)
    smallest = min(spans, key=lambda m: spans[m] if spans[m] else 9e9)
    print(f"  Largest movement in any published quantity: {spans[biggest]:.2f}x "
          f"({biggest}) across a threshold range of {min(THRESHOLDS)}-{max(THRESHOLDS)}, "
          f"a number that appears in no figure.")
    # WHICH quantity is sensitive matters more than that some quantity is. Moving the
    # threshold mostly adds and removes small marginal regions: that changes how many
    # objects there are a great deal and how much dark area there is very little. So the
    # robust quantity and the fragile one are not interchangeable, and "crack density" --
    # a count per unit area, the thing a materials paper reports -- is the fragile one.
    if biggest != smallest and spans[smallest]:
        print(f"  The spread across quantities is the useful part: {biggest} moves "
              f"{spans[biggest]:.2f}x while {smallest} moves only {spans[smallest]:.2f}x. "
              f"Raising the threshold removes marginal regions, which changes HOW MANY "
              f"objects there are far more than HOW MUCH dark area there is. So a paper "
              f"reporting crack AREA FRACTION is nearly immune to this choice, and a paper "
              f"reporting CRACK DENSITY -- a count, and the more common figure -- is not. "
              f"Those two are not interchangeable and the threshold has to be stated "
              f"alongside the second one.")
    if flips:
        print(f"  The ordering of the conditions FLIPS for: {', '.join(flips)}. A "
              f"comparison a paper exists to make is therefore decided by an undocumented "
              f"library default, not by the specimens.")
    else:
        print(f"  The ordering of the three conditions is STABLE for every quantity "
              f"measured. This is the weaker of the two possible findings and it is "
              f"reported as such: an undocumented threshold moves the MAGNITUDE of every "
              f"published number by up to {spans[biggest]:.2f}x, but it did not reverse a "
              f"qualitative conclusion in this corpus. The prediction going in was that it "
              f"would; it did not.")
        print(f"  So the claim is 'state your threshold and report sensitivity', not "
              f"'published comparisons are wrong'. Overstating this would be the same "
              f"error the paper is about.")
    if ratios:
        best = max(ratios, key=lambda k: ratios[k]["times_more_sensitive"] or 0)
        r = ratios[best]
        if r["times_more_sensitive"]:
            print(f"  Within a SINGLE specimen -- same frames, same threshold move -- the "
                  f"contrast is starker than the worst-case comparison above. On {best}, "
                  f"crack count moves {r['count_pct']:.1f}% while area fraction moves "
                  f"{r['area_pct']:.1f}%: the count is "
                  f"{r['times_more_sensitive']:.0f}x more sensitive to a choice nobody "
                  f"records.")
    if steps:
        worst_metric = max(steps, key=lambda m: steps[m]["rel_change"])
        w = steps[worst_metric]
        print(f"  The movement is also not evenly spread. The largest single step is "
              f"{100 * w['rel_change']:.1f}% in {worst_metric} on {w['specimen']}, between "
              f"t={w['from_threshold']} and t={w['to_threshold']} alone. An operating point "
              f"can therefore sit next to a step change, which a range-averaged sensitivity "
              f"figure would hide -- so report the curve, not one ratio.")
    print(f"  n = {len(specs)} specimen(s), one per condition, so the ordering itself is "
          f"NOT a material finding either way -- it is the object whose stability is under "
          f"test, not a result about the alloy.")

    json.dump(out, open(OUT.replace(".json", "_report.json"), "w"), indent=1)
    print(f"\n  -> {OUT.replace('.json', '_report.json')}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-spec", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--report", action="store_true",
                    help="merge shard files and print the sensitivity tables")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run(a.per_spec, a.shard, a.nshard)
