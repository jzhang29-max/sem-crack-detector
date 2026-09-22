#!/usr/bin/env python3
"""Prove the model input does not encode the label. Run this before believing any score.

This guard exists because the first SAM 3 run on this corpus measured its own annotation. The
input was built from the overlay's green channel; the overlay burns opaque red (225,25,25) --
not pure red -- whose green channel is a constant 25, far below the 80 the threshold used, so
labelled pixels arrived at the model as black. (The value is 25, not 0: see the "third wrong
diagnosis" section of LEAK_POSTMORTEM.md, where the retraction itself got this number wrong.)

WHAT DISCRIMINATES A LEAK, AND WHAT DOES NOT

Two plausible-sounding tests were built here and REJECTED on measurement, because in this
corpus they fire on clean inputs:

  * "A threshold recovers the label" -- NOT evidence. Cracks are genuinely the darkest pixels,
    and these micrographs are clipped at acquisition: one frame holds 9.4% of its pixels at
    exactly 0, another 69.0% at exactly 65535, and one carries only 256 distinct values in a
    uint16 container. On one clean tile 97.8% of labelled pixels are exactly 0 in the RAW
    original, so a threshold at 0 reaches recall 0.978 with no leak of any kind.
  * "Image edges coincide with the label boundary" -- NOT evidence either. Measured on the
    known-contaminated input as a positive control, the ratio of boundary gradient at the true
    label to displaced labels was median 3.76; on the clean input it was 3.05, with fully
    overlapping ranges (0.82-8.80 vs 0.90-5.56). It does not separate. Cracks have edges where
    their labels are, which is the whole point.

What does discriminate is EXACTNESS. A burn-in writes one value, so the label region loses all
variance, including across places where the underlying micrograph was bright. Measured:
within-label std was exactly 0.00 and P(input == modal | label) exactly 1.0000 on every
burned tile. No micrograph does that. So the criterion is

    std_inside < 0.5  OR  P(input == modal value | label) > 0.999

reported per tile, with the best-single-threshold oracle alongside as the trivial baseline any
method must beat (that number is useful, it is simply not a leak test).

Limits worth stating: this catches a CONSTANT burn-in, which is what a rendered overlay
produces. It would not catch an alpha-blended annotation that preserves within-label variance,
and the boundary test that might have caught that one does not work here. The durable protection
is structural, not statistical: make_tiles.py reads the raw original and writes the overlay only
under a name no loader would pick up.
"""
import os, json, sys
import numpy as np
from PIL import Image

SC = os.path.dirname(os.path.abspath(__file__))

def oracle_threshold(g, gt):
    """Best IoU achievable by ANY single global threshold, either polarity, plus the highest
    recall reachable by a non-vacuous threshold. An upper bound, not a method."""
    best = {"iou": -1.0, "f1": 0.0}
    hi = {"recall": -1.0}
    for pol in ("dark", "bright"):
        for t in range(0, 256, 2):
            p = (g <= t) if pol == "dark" else (g >= t)
            ps = p.sum()
            if ps == 0:
                continue
            inter = np.logical_and(p, gt).sum()
            if inter:
                iou = inter / np.logical_or(p, gt).sum()
                if iou > best["iou"]:
                    rec, prec = inter / gt.sum(), inter / ps
                    best = {"iou": float(iou), "f1": float(2 * prec * rec / (prec + rec)),
                            "t": t, "pol": pol}
            if ps <= 0.5 * p.size:          # a threshold firing on half the tile is vacuous
                rec = inter / gt.sum()
                if rec > hi["recall"]:
                    hi = {"recall": float(rec), "prec": float(inter / ps), "t": t, "pol": pol}
    return best, hi

def check(tag, gray_of, note=""):
    rows = []
    for m in json.load(open(f"{SC}/tiles/meta.json")):
        tid = m["tile"]
        g = gray_of(tid)
        gt = np.array(Image.open(f"{SC}/tiles/{tid}_gt.png")) > 127
        vals, cnts = np.unique(g[gt], return_counts=True)
        modal_share = float(cnts.max() / cnts.sum())
        std_in = float(g[gt].std())
        best, hi = oracle_threshold(g, gt)
        leak = (std_in < 0.5) or (modal_share > 0.999)
        rows.append({"tile": tid, "std_inside": round(std_in, 3),
                     "modal_share_inside": round(modal_share, 4),
                     "oracle_iou": round(best["iou"], 4), "oracle_f1": round(best["f1"], 4),
                     "max_recall": round(hi["recall"], 4), "leak": bool(leak)})
    n = len(rows); nl = sum(r["leak"] for r in rows)
    med = lambda k: float(np.median([r[k] for r in rows]))
    print(f"\n{tag}")
    if note:
        print(f"  {note}")
    print(f"  median within-label std        : {med('std_inside'):8.3f}   (0.00 = a value was written)")
    print(f"  median modal share inside      : {med('modal_share_inside'):8.4f}   (1.0000 = point mass)")
    print(f"  median best-threshold IoU      : {med('oracle_iou'):8.4f}   <- trivial baseline, not a leak test")
    print(f"  median max reachable recall    : {med('max_recall'):8.4f}")
    print(f"  VERDICT: {nl}/{n} tiles carry a written-in label" + ("   <-- CONTAMINATED" if nl else "   <-- clean"))
    return rows, nl

new_gray = lambda t: np.array(Image.open(f"{SC}/tiles/{t}_gray.png").convert("L"))

def old_gray(t):
    """Reconstruct exactly what the invalid run fed: the overlay crop's green channel."""
    p = f"{SC}/tiles/{t}_overlay_REFERENCE_DO_NOT_FEED.png"
    return np.array(Image.open(p).convert("RGB"))[..., 1] if os.path.exists(p) else None

if __name__ == "__main__":
    first = json.load(open(f"{SC}/tiles/meta.json"))[0]["tile"]
    if old_gray(first) is not None:
        check("POSITIVE CONTROL -- the invalid input (overlay green channel)", old_gray,
              "expected to fail; if it ever passes, this guard has stopped working")
    rows, nl = check("THE CURRENT INPUT -- raw original .tif, registered", new_gray)
    json.dump(rows, open(f"{SC}/leak_check.json", "w"), indent=1)
    print("\nwrote leak_check.json")
    sys.exit(1 if nl else 0)
