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

  clIoU_tau    OmniCrack30k's tolerant IoU, both sides skeletonised. TAU IS SET PER FRAME to
               half that frame's median brush width, NOT to the fixed 4 px of the fine-subset
               analysis. The first run of this script used tau = 4 everywhere and clIoU
               collapsed to 0.03-0.05: against a 59 px median brush (max 413 px) the label's
               skeleton can lie 206 px from a correct thin crack, so a 4 px tolerance cannot
               match them and every method scores near zero for the same reason. A fixed tau is
               only meaningful when the brush is roughly constant, which is exactly what this
               corpus is not.
  cont_lift    containment DIVIDED BY the corridor's own area fraction, i.e. by what an
               all-ones prediction would score on that frame. Raw containment is unusable here:
               the corridor covers a median 42.2% of a coarse tile against 5.5% of a fine one,
               so 0.99 containment on a coarse frame is nearly unavoidable and means almost
               nothing. Lift of 1.0 is chance; the ceiling is 1/corridor_fraction.
  PAR          predicted area / labelled area -- the bias diagnostic that exposes a method
               winning containment by painting everything (Otsu scored 0.779 at PAR 6.86)

Pixel IoU is still computed and reported, but OVER ALL 44 FRAMES, not the fine subset -- the
aggregation loops over `names`, which is every frame, and expanded_bench.json's per_frame
arrays are length 44 for IoU exactly as for the other metrics. This docstring claimed
"reported for the fine subset only ... can be checked against the 15-tile result on common
ground", which is the one thing it cannot be used for: the populations differ. The JSON does
record `_fine` per frame, so a fine-only median can be taken from the committed artefact if
that comparison is wanted; it is not what the printed table shows.

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
    """clIoU_tau via the Euclidean distance transform rather than a disk dilation.

    dilate(X, disk(tau)) is exactly {pixels within tau of X}, i.e. EDT(~X) <= tau. Both give
    identical results, but the EDT is computed ONCE per skeleton for any tau, whereas
    binary_dilation costs more as the radius grows. With tau scaled per frame to half the brush
    width, radii here reach 206 px and the dilation form was too slow to finish. This is an
    exact reformulation, not an approximation.
    """
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    sp, sg = skeletonize(p), skeletonize(g)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    near_p = ndi.distance_transform_edt(~sp) <= tau      # within tau of the prediction skeleton
    near_g = ndi.distance_transform_edt(~sg) <= tau      # within tau of the label skeleton
    tp = int((sg & near_p).sum())
    fp = int((sp & ~(sp & near_g)).sum())
    fn = int((sg & ~(sg & near_p)).sum())
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
        tau = max(2, int(round(m["median_thick_px"] / 2)))   # tolerance scaled to THIS brush
        data[n] = {"g": g, "gt": gt, "cor": cor | gt, "fine": m["fine"], "tau": tau,
                   "corfrac": float((cor | gt).mean()),
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
                cont = float(p[d["cor"]].sum() / p.sum()) if p.sum() else 0.0
                sc[key][n] = (cliou(p, d["gt"], d["tau"]), iou(p, d["gt"]),
                              cont / max(d["corfrac"], 1e-9),          # lift over the all-ones null
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
        for idx, metric in ((0, "clIoU_adapt"), (2, "cont_lift"), (3, "PAR"), (1, "IoU")):
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
    tt = [data[n]["tau"] for n in names]
    print(f"\n  per-frame tau: median {int(np.median(tt))} px, range {min(tt)}-{max(tt)} px")
    cfr = [data[n]["corfrac"] for n in names]
    print(f"  all-ones containment null: median {np.median(cfr):.4f} "
          f"(fine {np.median([data[n]['corfrac'] for n in names if data[n]['fine']]):.4f}, "
          f"coarse {np.median([data[n]['corfrac'] for n in names if not data[n]['fine']]):.4f})")
    print(f"\nDOES THE EXPANDED n SEPARATE THEM?  (paired over {len(names)} frames, clIoU_adapt)")
    for a in METHODS:
        for b in METHODS:
            if a >= b:
                continue
            d = [x - y for x, y in zip(out[f"{a}|clIoU_adapt"]["per_frame"],
                                       out[f"{b}|clIoU_adapt"]["per_frame"])]
            nz = [x for x in d if x != 0]
            if len(nz) < 6:
                continue
            p = wilcoxon(nz, method="asymptotic").pvalue
            star = "  <-- SEPARATES" if p < 0.05 else ""
            print(f"  {a:<20} vs {b:<20} median d {np.median(d):>+8.4f}  p = {p:.4g}{star}")
    print("\nwrote expanded_bench.json")


if __name__ == "__main__":
    main()
