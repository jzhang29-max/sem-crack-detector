#!/usr/bin/env python3
"""Score the training-free methods on every labelled frame, with width-insensitive metrics.

The point of this file is statistical power, not a new method. power_analysis.py showed that
separating these methods needs ~40-60 source frames and the main evaluation has 9. This runs
the same arms over every frame that carries a hand label, using only metrics that survive a
broad brush.

METRICS, and why pixel IoU is absent as a headline. iou_ceiling.py measured that a perfect 3 px
trace down the label centreline scores median IoU 0.1662 on the fine frames; on the coarse
frames, whose brush is wider still, the ceiling is lower again. Ranking by a quantity whose
maximum is set by the annotator's brush is not measurement. So:

  clIoU_4      OmniCrack30k's tolerant IoU at tau = 4 px, both sides skeletonised
  containment  fraction of predicted pixels inside the human's asserted corridor
  PAR          predicted area / labelled area -- the bias diagnostic that exposes a method
               winning containment by painting everything (Otsu scored 0.779 at PAR 6.86)

Pixel IoU is still computed, and reported for the fine subset only, so the expanded result can
be checked against the 15-tile result on common ground.

Protocol is unchanged: OIS (per-tile oracle, an upper bound), ODS (one setting for all), and
LOFO (setting chosen on the other frames). One tile per frame, so the tile count IS the frame
count and leave-one-frame-out is leave-one-tile-out without the aliasing.
"""
import os, json
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, sato, meijering
from skimage.morphology import skeletonize, remove_small_objects, disk, binary_dilation
from scipy import ndimage as ndi

SC = os.path.dirname(os.path.abspath(__file__))
QS = [90, 94, 96, 97, 98, 99, 99.5]
rng = np.random.default_rng(20260919)


def cliou(p, g, tau=4):
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    sp, sg = skeletonize(p), skeletonize(g)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    k = disk(tau)
    tp = int((sg & binary_dilation(sp, k)).sum())
    fp = int((sp & ~(sp & binary_dilation(sg, k))).sum())
    fn = int((sg & ~(sg & binary_dilation(sp, k))).sum())
    d = tp + fp + fn
    return float(tp / d) if d else 0.0


def iou(p, g):
    u = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / u) if u else 0.0


def main():
    meta = json.load(open(f"{SC}/tiles_all/meta.json"))
    print(f"{len(meta)} frames ({sum(1 for m in meta if m['fine'])} fine, "
          f"{sum(1 for m in meta if not m['fine'])} coarse)\n")
    data = {}
    for m in meta:
        n = m["tile"]
        g = np.array(Image.open(f"{SC}/tiles_all/{n}_gray.png").convert("L"))
        gt = np.array(Image.open(f"{SC}/tiles_all/{n}_gt.png")) > 127
        cor = np.array(Image.open(f"{SC}/tiles_all/{n}_corridor.png")) > 127
        gf = g.astype(np.float32) / 255.0
        data[n] = {"g": g, "gt": gt, "cor": cor | gt, "fine": m["fine"],
                   "sato": sato(gf, sigmas=np.arange(1, 5.0), black_ridges=True),
                   "meij": meijering(gf, sigmas=np.arange(1, 5.0), black_ridges=True)}
    names = list(data)

    def predict(n, method, q):
        d = data[n]
        if method == "global threshold":
            r = 255 - d["g"].astype(np.float32)
        elif method == "Otsu":
            return remove_small_objects(d["g"] <= threshold_otsu(d["g"]), max_size=32)
        else:
            r = d["sato"] if method == "Sato ridge" else d["meij"]
        return remove_small_objects(r >= np.percentile(r, q), max_size=32)

    METHODS = ["global threshold", "Otsu", "Sato ridge", "Meijering ridge"]
    sc = {}
    for mth in METHODS:
        qs = [None] if mth == "Otsu" else QS
        for q in qs:
            key = (mth, q)
            sc[key] = {}
            for n in names:
                p = predict(n, mth, q)
                d = data[n]
                sc[key][n] = (cliou(p, d["gt"]), iou(p, d["gt"]),
                              float(p[d["cor"]].sum() / p.sum()) if p.sum() else 0.0,
                              float(p.sum() / max(d["gt"].sum(), 1)))
        print(f"  scored {mth}", flush=True)

    def boot(v):
        out = []
        for _ in range(4000):
            s = rng.choice(v, len(v), replace=True)
            out.append(np.median(s))
        return np.percentile(out, 2.5), np.percentile(out, 97.5)

    print(f"\n{'method':<20} {'metric':<12} {'OIS':>7} {'ODS':>7} {'LOFO':>7}   {'LOFO 95% CI':>18}")
    out = {}
    for mth in METHODS:
        keys = [k for k in sc if k[0] == mth]
        for idx, metric in ((0, "clIoU4"), (2, "containment"), (3, "PAR"), (1, "IoU")):
            ois = [max(sc[k][n][idx] for k in keys) for n in names]
            odsk = max(keys, key=lambda k: np.median([sc[k][n][idx] for n in names]))
            lofo = []
            for n in names:
                tr = [u for u in names if u != n]
                bk = max(keys, key=lambda k: np.median([sc[k][u][idx] for u in tr]))
                lofo.append(sc[bk][n][idx])
            lo, hi = boot(lofo)
            print(f"{mth:<20} {metric:<12} {np.median(ois):>7.4f} "
                  f"{np.median([sc[odsk][n][idx] for n in names]):>7.4f} "
                  f"{np.median(lofo):>7.4f}   [{lo:.4f}, {hi:.4f}]")
            out[f"{mth}|{metric}"] = {"OIS": float(np.median(ois)),
                                      "LOFO": float(np.median(lofo)),
                                      "ci": [float(lo), float(hi)],
                                      "per_frame": lofo}
    out["_frames"] = names
    out["_fine"] = {n: data[n]["fine"] for n in names}
    json.dump(out, open(f"{SC}/expanded_bench.json", "w"), indent=1)

    from scipy.stats import wilcoxon
    print(f"\nDOES THE EXPANDED n SEPARATE THEM?  (paired over {len(names)} frames, clIoU4)")
    for a in METHODS:
        for b in METHODS:
            if a >= b:
                continue
            d = [x - y for x, y in zip(out[f"{a}|clIoU4"]["per_frame"],
                                       out[f"{b}|clIoU4"]["per_frame"])]
            nz = [x for x in d if x != 0]
            if len(nz) < 6:
                continue
            p = wilcoxon(nz, method="asymptotic").pvalue
            star = "  <-- SEPARATES" if p < 0.05 else ""
            print(f"  {a:<20} vs {b:<20} median d {np.median(d):>+8.4f}  p = {p:.4g}{star}")
    print("\nwrote expanded_bench.json")


if __name__ == "__main__":
    main()
