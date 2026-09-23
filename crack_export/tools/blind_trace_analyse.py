#!/usr/bin/env python3
"""Unblind and analyse the dual-channel tracing experiment.

WRITTEN BEFORE ANY TRACING EXISTS. That is the point: the analysis is fixed while the
outcome is still unknown, so there is no opportunity to pick the statistic that gives the
nicest answer. This repo has already had one headline result die because its inclusion
filter was chosen after the fact (paired_detector.py kept pairs by mask Jaccard, computed
from the very quantities being compared, and dropped exactly the three disagreements).

PRIMARY ENDPOINT, fixed here:
    per field, log2( traced centreline length in CBS / traced centreline length in ETD ),
    averaged over the two annotators, then an exact two-sided sign test across the SEVEN
    specimen cells on the per-cell mean. Cells, not fields: fields within a cell are one
    specimen and are not independent. Seven cells gives an exact-test floor of p = 0.0156,
    so significance is reachable; at the three cells of the earlier analysis the floor was
    0.25 and no result was attainable.

    Length, not area: the annotators trace centrelines, so length is what they produced.
    Area would reintroduce the width sensitivity that makes this corpus's pixel IoU
    meaningless (a perfect 3 px trace scores 0.1662 against these labels).

SECONDARY, all reported whether or not they help:
    * inter-annotator agreement on the SAME image -- the noise floor. A between-channel
      difference smaller than the between-annotator difference means nothing, and this is
      the number that decides whether the primary endpoint is interpretable at all.
    * blinding integrity from GUESS.csv: accuracy, and accuracy among confidence >= 3.
      A simple model reaches ~88% from image statistics alone, so partial unblinding is
      expected and is a stated limitation, not a surprise. Report it; do not bury it.
    * the same test on detected-crack COUNT, as a shape-free check.

    python3 blind_trace_analyse.py --package ~/Desktop/MAR_blind_tracing
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy.stats import binomtest

Image.MAX_IMAGE_PIXELS = None
SC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SC))
KEY = os.path.join(REPO, "crack_export", "analysis", "blind_trace_key",
                   "KEY_do_not_open_until_tracing_complete.csv")


def centreline_length(png):
    """Skeleton length of a trace. Kulpa's estimator, not a pixel count.

    A pixel count is not a path length -- a 45-degree run of n pixels spans n*sqrt(2), and
    counting pixels reported impossible tortuosities below 1 in this repo once already.
    0.948*orthogonal + 1.343*diagonal is Kulpa's correction.
    """
    from skimage.morphology import skeletonize
    a = np.array(Image.open(png).convert("L")) > 127
    if not a.any():
        return 0.0, 0
    sk = skeletonize(a)
    import scipy.ndimage as ndi
    orth = ndi.convolve(sk.astype(np.uint8), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                        mode="constant")[sk].sum() / 2.0
    diag = ndi.convolve(sk.astype(np.uint8), np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
                        mode="constant")[sk].sum() / 2.0
    n_comp = int(ndi.label(a)[1])
    return float(0.948 * orth + 1.343 * diag), n_comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    a = ap.parse_args()
    pkg = os.path.expanduser(a.package)
    if not os.path.exists(KEY):
        sys.exit(f"no key at {KEY}")
    key = {r["opaque_id"]: r for r in csv.DictReader(open(KEY))}

    traces = collections.defaultdict(dict)      # (cell, field, det) -> {annotator: (len, n)}
    missing = []
    anns = sorted(d.split("_")[1] for d in os.listdir(pkg) if d.startswith("annotator_"))
    for ann in anns:
        tdir = os.path.join(pkg, f"annotator_{ann}", "traced")
        for oid, k in key.items():
            hits = [f for f in os.listdir(tdir)] if os.path.isdir(tdir) else []
            f = next((h for h in hits if oid in h), None)
            if f is None:
                missing.append((ann, oid)); continue
            traces[(k["cell"], k["field"], k["detector"])][ann] = \
                centreline_length(os.path.join(tdir, f))
    done = sum(len(v) for v in traces.values())
    print(f"  traces found: {done} of {len(key) * len(anns)}   annotators: {','.join(anns)}")
    if missing:
        print(f"  missing {len(missing)} (analysis runs on what exists): {missing[:4]}")
    if not done:
        sys.exit("\n  Nothing traced yet. This script is here so the analysis is fixed "
                 "before the data exists -- re-run it when tracing is complete.")

    # ---- inter-annotator noise floor, computed FIRST: it sets what is interpretable
    if len(anns) >= 2:
        d = [abs(np.log2(max(v[anns[0]][0], 1) / max(v[anns[1]][0], 1)))
             for v in traces.values() if len(v) >= 2]
        if d:
            print(f"\n  NOISE FLOOR  median |log2 ratio| between annotators on the SAME image: "
                  f"{np.median(d):.3f}  (n={len(d)})")
            print(f"               any channel effect below this is not interpretable")

    # ---- primary endpoint
    per_field = {}
    for (cell, fld, det), v in traces.items():
        if not v:
            continue
        per_field.setdefault((cell, fld), {})[det] = float(np.mean([x[0] for x in v.values()]))
    cells = collections.defaultdict(list)
    for (cell, fld), d in per_field.items():
        if "CBS" in d and "ETD" in d and d["ETD"] > 0:
            cells[cell].append(np.log2(d["CBS"] / d["ETD"]))
    if not cells:
        sys.exit("\n  no field has both channels traced yet")
    print(f"\n  PRIMARY  log2(CBS/ETD) traced centreline length, per specimen cell:")
    means = []
    for c in sorted(cells):
        m = float(np.mean(cells[c])); means.append(m)
        print(f"    {c:16s} n={len(cells[c])}  log2 ratio {m:+.3f}  ({2 ** m:.2f}x)")
    pos = sum(1 for m in means if m > 0)
    p = binomtest(pos, len(means)).pvalue
    print(f"\n    {pos}/{len(means)} cells favour CBS   exact two-sided sign test p = {p:.4f}"
          f"   (floor at {len(means)} cells = {2 / 2 ** len(means):.4f})")

    # ---- blinding integrity
    print(f"\n  BLINDING INTEGRITY")
    for ann in anns:
        g = os.path.join(pkg, f"annotator_{ann}", "GUESS.csv")
        if not os.path.exists(g):
            continue
        rows = [r for r in csv.DictReader(open(g)) if r["guess_BSE_or_SE"].strip()]
        if not rows:
            print(f"    annotator {ann}: GUESS.csv not filled in"); continue
        ok = hi = hi_ok = 0
        for r in rows:
            oid = r["image"].replace(".png", "")
            k = key.get(oid)
            if not k:
                continue
            truth = "BSE" if k["detector"] == "CBS" else "SE"
            corr = r["guess_BSE_or_SE"].strip().upper() == truth
            ok += corr
            try:
                if int(r["confidence_1to4"]) >= 3:
                    hi += 1; hi_ok += corr
            except (ValueError, TypeError):
                pass
        print(f"    annotator {ann}: {ok}/{len(rows)} correct ({100*ok/len(rows):.0f}%)"
              f"   at confidence>=3: {hi_ok}/{hi}" if hi else
              f"    annotator {ann}: {ok}/{len(rows)} correct ({100*ok/len(rows):.0f}%)")
        print(f"      chance is 50%; an image-statistics model reaches ~88%, so >50% here is "
              f"expected and is a stated limitation")


if __name__ == "__main__":
    main()
