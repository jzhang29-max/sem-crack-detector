#!/usr/bin/env python3
"""Thin-structure metrics on the SAME matched-budget arms as baseline_matched.py.

Why: pixel IoU against a broad-brush region assertion (median stroke 59 px vs a ~3 px crack)
is bounded far below 1 by the brush, not by the model. The thin-structure literature has
metrics that are *less* coupled to stroke width. This script measures them here, and --
crucially -- runs the information-free null and two decoys through every metric, so we can
see which metrics still separate signal from nothing on THIS label type.

Metrics (all binary, CPU, no training):
  IoU
  clDice          Shit et al., CVPR 2021 (10.1109/CVPR46437.2021.01629) -- harmonic mean of
                  Tprec = |skel(P) & G|/|skel(P)| and Tsens = |skel(G) & P|/|skel(G)|
  Tprec           the half of clDice that is NOT penalised by an over-wide GT brush
  Tsens           the half that IS
  skelR(r)        Skeleton Recall, Kirchhoff et al., ECCV 2024 (arXiv:2404.03010) style:
                  |skel(G) & dilate(P,r)|/|skel(G)|
  tolF1(tau)      buffered / tolerance F1: prec = frac of P within tau px of G,
                  rec = frac of skel(G) within tau px of P. Tolerance framing follows
                  surface-DSC (Nikolov et al., arXiv:1809.04430) and Metrics Reloaded
                  (Nature Methods 2024, 10.1038/s41592-023-02151-z).
  dB0             |#cc(P) - #cc(G)|, Betti-0 error (Hu et al., arXiv:1906.05404 uses Betti
                  number error). Reported with #cc(P) because a count is not a mass.

Operating points are re-tuned PER METRIC. ODS = one threshold+polarity shared by all tiles,
chosen on this same set. OIS = per-tile oracle. Both are label-tuned and labelled as such.
"""
import os, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

SC = os.path.dirname(os.path.abspath(__file__))
THS = list(range(0, 256, 2))


def load(tid):
    g = np.array(Image.open(f"{SC}/tiles/{tid}_gray.png").convert("L"))
    gt = np.array(Image.open(f"{SC}/tiles/{tid}_gt.png")) > 127
    return g, gt


def otsu(g):
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    w = hist.cumsum(); m = (hist * np.arange(256)).cumsum()
    tot, mt = w[-1], m[-1]
    best, bt = -1.0, 0
    for t in range(1, 255):
        w0, w1 = w[t], tot - w[t]
        if w0 == 0 or w1 == 0:
            continue
        v = w0 * w1 * (m[t] / w0 - (mt - m[t]) / w1) ** 2
        if v > best:
            best, bt = v, t
    return bt


class GT:
    """Precompute everything that depends only on the label."""
    def __init__(self, gt):
        self.m = gt
        self.skel = skeletonize(gt)
        self.nskel = int(self.skel.sum())
        self.ncc = int(ndi.label(gt)[1])
        # distance from every pixel to the nearest GT pixel
        self.dist_to_gt = ndi.distance_transform_edt(~gt)


def metrics(P, G, taus=(2, 5), rs=(2,)):
    out = {}
    inter = np.logical_and(P, G.m).sum()
    union = np.logical_or(P, G.m).sum()
    out["IoU"] = float(inter / union) if union else 0.0
    if not P.any():
        out.update({"Tprec": 0.0, "Tsens": 0.0, "clDice": 0.0, "ncc_P": 0, "dB0": G.ncc})
        for r in rs:
            out[f"skelR{r}"] = 0.0
        for t in taus:
            out[f"tolP{t}"] = 0.0; out[f"tolR{t}"] = 0.0; out[f"tolF1_{t}"] = 0.0
        return out
    sp = skeletonize(P)
    nsp = int(sp.sum())
    tprec = float(np.logical_and(sp, G.m).sum() / nsp) if nsp else 0.0
    tsens = float(np.logical_and(G.skel, P).sum() / G.nskel) if G.nskel else 0.0
    out["Tprec"], out["Tsens"] = tprec, tsens
    out["clDice"] = 2 * tprec * tsens / (tprec + tsens) if (tprec + tsens) > 0 else 0.0
    ncc_P = int(ndi.label(P)[1])
    out["ncc_P"] = ncc_P
    out["dB0"] = abs(ncc_P - G.ncc)
    dist_to_P = ndi.distance_transform_edt(~P)
    for r in rs:
        out[f"skelR{r}"] = float((dist_to_P[G.skel] <= r).mean()) if G.nskel else 0.0
    for t in taus:
        p = float((G.dist_to_gt[P] <= t).mean())
        rr = float((dist_to_P[G.skel] <= t).mean()) if G.nskel else 0.0
        out[f"tolP{t}"] = p; out[f"tolR{t}"] = rr
        out[f"tolF1_{t}"] = 2 * p * rr / (p + rr) if (p + rr) > 0 else 0.0
    return out


def main():
    tiles = [m["tile"] for m in json.load(open(f"{SC}/tiles/meta.json"))]
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}
    grays, Gs = {}, {}
    for t in tiles:
        g, gt = load(t)
        grays[t] = g; Gs[t] = GT(gt)

    KEYS = ["IoU", "clDice", "Tprec", "Tsens", "skelR2", "tolF1_2", "tolF1_5"]

    # ---- per-tile, per-threshold cache for the grey-level arms
    cache = {}   # (tile, pol, th) -> metric dict
    for t in tiles:
        g, G = grays[t], Gs[t]
        for pol in ("dark", "bright"):
            for th in THS:
                P = (g <= th) if pol == "dark" else (g >= th)
                cache[(t, pol, th)] = metrics(P, G)

    arms = {}

    # Otsu -- no label access
    arms["Otsu per tile (no labels)"] = ("none", [
        metrics(grays[t] <= otsu(grays[t]), Gs[t]) for t in tiles])

    # ODS / OIS, re-tuned separately for EACH metric
    ods_pt, ois_rows = {}, {}
    for k in KEYS:
        best = (-1.0, None, None)
        for pol in ("dark", "bright"):
            for th in THS:
                v = float(np.median([cache[(t, pol, th)][k] for t in tiles]))
                if v > best[0]:
                    best = (v, pol, th)
        ods_pt[k] = best[1:]
        ois_rows[k] = [max(cache[(t, pol, th)][k] for pol in ("dark", "bright") for th in THS)
                       for t in tiles]

    # SAM 3
    def sam(field):
        rows = []
        for t in tiles:
            fn = f"{SC}/masks/{t}__crack.npz"
            if os.path.exists(fn):
                P = np.load(fn)[field]
            else:
                P = np.zeros_like(Gs[t].m)
            rows.append(metrics(P, Gs[t]))
        return rows
    arms["SAM 3 union tau=0.3 (no labels)"] = ("none", sam("union"))
    arms["SAM 3 oracle instance"] = ("per tile (picks instance)", sam("oracle"))

    # information-free null and two decoys
    null_rows, shift_rows, dil_rows = [], [], []
    rng = np.random.default_rng(0)
    for t in tiles:
        G = Gs[t]
        k = int(G.m.sum()); flat = np.zeros(G.m.size, bool); flat[:k] = True
        null_rows.append(metrics(flat.reshape(G.m.shape), G))
        shift_rows.append(metrics(np.roll(G.m, (40, 40), (0, 1)), G))
        dil_rows.append(metrics(ndi.binary_dilation(G.skel, iterations=1), G))
    arms["NULL: constant area, arbitrary place"] = ("area only", null_rows)
    arms["DECOY: label shifted 40 px"] = ("label itself", shift_rows)
    arms["DECOY: GT skeleton, 3 px wide"] = ("label itself", dil_rows)

    print(f"n = {len(tiles)} disjoint tiles; medians over tiles\n")
    hdr = f"  {'arm':<36} {'label access':<26}" + "".join(f"{k:>9}" for k in KEYS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, (acc, rows) in arms.items():
        vals = "".join(f"{np.median([r[k] for r in rows]):>9.4f}" for k in KEYS)
        print(f"  {name:<36} {acc:<26}{vals}")
    for k in KEYS:
        pol, th = ods_pt[k]
        row = [cache[(t, pol, th)][k] for t in tiles]
        arms.setdefault("_ods", {})[k] = row
    print(f"  {'ODS (one shared threshold)':<36} {'set-level, per metric':<26}"
          + "".join(f"{np.median(arms['_ods'][k]):>9.4f}" for k in KEYS))
    print(f"  {'OIS (per-tile oracle threshold)':<36} {'per tile, per metric':<26}"
          + "".join(f"{np.median(ois_rows[k]):>9.4f}" for k in KEYS))
    print("\n  ODS operating point per metric: "
          + ", ".join(f"{k}={ods_pt[k][0]}<={ods_pt[k][1]}" for k in KEYS))

    print("\n  Betti-0 (component counts, median): "
          f"GT={np.median([Gs[t].ncc for t in tiles]):.0f}")
    for name, (acc, rows) in arms.items():
        if name.startswith("_"):
            continue
        print(f"    {name:<36} #cc(P)={np.median([r['ncc_P'] for r in rows]):>8.0f}"
              f"   dB0={np.median([r['dB0'] for r in rows]):>8.0f}")

    # does each metric separate SAM 3 from the information-free null?
    from scipy.stats import wilcoxon
    print("\n  SEPARATION CHECK -- SAM 3 union vs the information-free null, per metric:")
    su = arms["SAM 3 union tau=0.3 (no labels)"][1]
    for k in KEYS:
        a = [r[k] for r in su]; b = [r[k] for r in null_rows]
        d = [x - y for x, y in zip(a, b) if x != y]
        p = wilcoxon(d).pvalue if len(d) >= 6 else float("nan")
        print(f"    {k:<10} SAM wins {sum(1 for x, y in zip(a, b) if x > y):>2}/{len(tiles)}"
              f"   median {np.median(a):.4f} vs {np.median(b):.4f}   p = {p:.4f}")

    print("\n  DECOY CHECK -- a metric a 3 px GT skeleton scores near 1.0 on is a metric that"
          "\n  is insensitive to the brush width of the region assertion (that is the point);"
          "\n  a metric the 40 px shift also scores high on is a metric with no localisation.")

    json.dump({name: {k: [r[k] for r in rows] for k in KEYS + ["ncc_P", "dB0"]}
               for name, (acc, rows) in arms.items() if not name.startswith("_")},
              open(f"{SC}/thin_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
