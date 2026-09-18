#!/usr/bin/env python3
"""Threshold baselines at MATCHED oracle budgets, which is the comparison that was wrong.

The first write-up reported "an oracle-tuned single global grey threshold reaches median IoU
0.3843, the bar any method must beat" and concluded SAM 3 failed to clear it. That comparison
was rigged in the baseline's favour: the threshold was tuned PER TILE over 512 candidate
operating points chosen WITH the ground truth, while SAM 3's union arm was un-tuned at a single
fixed tau. Worse, per-tile oracle thresholding has a name -- it is OIS, defined in
Arbelaez/Maire/Fowlkes/Malik, IEEE TPAMI 33(5):898-916, 2011 (10.1109/TPAMI.2010.161), after
Martin/Fowlkes/Malik, IEEE TPAMI 26(5):530-549, 2004 (10.1109/TPAMI.2004.1273918) -- and with
ODS it has been the standard operating-point pair in crack segmentation for years (DeepCrack,
IEEE TIP 28(3):1498-1512, 2019, 10.1109/TIP.2018.2878966; OmniCrack30k, CVPRW 2024,
10.1109/CVPRW63382.2024.00392). It is not a baseline anyone forgot to state.

Four rungs, increasing label access:
  Otsu       -- no label access at all, per tile
  ODS        -- ONE threshold shared by every tile, fitted on this same set (still optimistic)
  OIS        -- per-tile oracle over all thresholds and both polarities (the original claim)
  const-area -- the information-free null: predict the label's own area fraction, arbitrarily
                placed. Scored on IoU, this is what "no information" actually looks like.
"""
import os, json
import numpy as np
from PIL import Image

SC = os.path.dirname(os.path.abspath(__file__))


def masks(tid):
    g = np.array(Image.open(f"{SC}/tiles/{tid}_gray.png").convert("L"))
    gt = np.array(Image.open(f"{SC}/tiles/{tid}_gt.png")) > 127
    return g, gt


def iou(p, gt):
    u = np.logical_or(p, gt).sum()
    return float(np.logical_and(p, gt).sum() / u) if u else 0.0


def otsu(g):
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    w = hist.cumsum(); m = (hist * np.arange(256)).cumsum()
    tot, mt = w[-1], m[-1]
    best, bt = -1, 0
    for t in range(1, 255):
        w0, w1 = w[t], tot - w[t]
        if w0 == 0 or w1 == 0:
            continue
        v = w0 * w1 * (m[t] / w0 - (mt - m[t]) / w1) ** 2
        if v > best:
            best, bt = v, t
    return bt


def main():
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}
    tiles = [m["tile"] for m in json.load(open(f"{SC}/tiles/meta.json"))]
    G = {t: masks(t) for t in tiles}

    ois, ots, const = [], [], []
    for t in tiles:
        g, gt = G[t]
        ois.append(max(max(iou(g <= th, gt) for th in range(0, 256, 2)),
                       max(iou(g >= th, gt) for th in range(0, 256, 2))))
        ots.append(iou(g <= otsu(g), gt))
        # information-free: a block of exactly the label's area, placed at the top-left
        k = int(gt.sum()); flat = np.zeros(gt.size, bool); flat[:k] = True
        const.append(iou(flat.reshape(gt.shape), gt))

    # ODS: one threshold + polarity for the whole set, fitted on the whole set
    best = (-1, None, None)
    for pol in ("dark", "bright"):
        for th in range(0, 256, 2):
            v = float(np.median([iou((G[t][0] <= th) if pol == "dark" else (G[t][0] >= th), G[t][1])
                                 for t in tiles]))
            if v > best[0]:
                best = (v, th, pol)
    _, ods_th, ods_pol = best
    ods = [iou((G[t][0] <= ods_th) if ods_pol == "dark" else (G[t][0] >= ods_th), G[t][1])
           for t in tiles]

    sam_u = [res.get((t, "crack"), {}).get("union_IoU", 0.0) for t in tiles]
    sam_o = [res.get((t, "crack"), {}).get("oracle_IoU", 0.0) for t in tiles]

    from scipy.stats import wilcoxon
    print(f"n = {len(tiles)} disjoint tiles\n")
    print(f"  {'arm':<34} {'label access':<22} {'median':>8} {'mean':>8}")
    for name, acc, v in (("Otsu, per tile", "none", ots),
                         ("ODS: one shared threshold", "set-level (optimistic)", ods),
                         ("OIS: per-tile oracle threshold", "per tile (512 points)", ois),
                         ("information-free const-area", "area only", const),
                         ("SAM 3 union, 'crack', tau=0.3", "none", sam_u),
                         ("SAM 3 oracle instance", "per tile (picks instance)", sam_o)):
        print(f"  {name:<34} {acc:<22} {np.median(v):>8.4f} {np.mean(v):>8.4f}")
    print(f"\n  ODS operating point: threshold {ods_th}, {ods_pol} polarity")
    print("\n  MATCHED comparisons (paired Wilcoxon, two-sided):")
    for name, v, ref, refname in (("ODS vs SAM union", ods, sam_u, "both un-tuned per tile"),
                                  ("OIS vs SAM union", ois, sam_u, "OIS has the oracle; unmatched"),
                                  ("OIS vs SAM oracle", ois, sam_o, "both per-tile oracle")):
        d = [a - b for a, b in zip(v, ref) if a != b]
        p = wilcoxon(d).pvalue if len(d) >= 6 else float("nan")
        wins = sum(1 for a, b in zip(v, ref) if a > b)
        print(f"    {name:<20} threshold wins {wins}/{len(tiles)}   p = {p:.4f}   ({refname})")
    print(f"\n  SAM 3 union beats the information-free null on "
          f"{sum(1 for a, b in zip(sam_u, const) if a > b)}/{len(tiles)} tiles "
          f"({np.median(sam_u):.4f} vs {np.median(const):.4f})")


if __name__ == "__main__":
    main()
