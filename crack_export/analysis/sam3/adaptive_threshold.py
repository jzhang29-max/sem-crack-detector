#!/usr/bin/env python3
"""Replace the fixed top-2% cut with a per-frame threshold chosen from the image alone.

WHY. On whole frames the detector's PAR (predicted area / labelled area) runs 0.03 to 48.16,
and the penalty is DIRECTIONAL: signed log PAR against clIoU_adapt is rho -0.529 (p = 0.0006),
over-predictors alone rho -0.800 (p = 0.0096), under-predictors n.s. (p = 0.42). Median
clIoU_adapt is 0.0209 for the 9 frames that over-predict and 0.1729 for the 29 that under-
predict -- an 8x gap. The mechanism is the metric: clIoU matches skeleton to skeleton, so a thin
prediction along the crack still matches the label's centreline while a bloated one grows extra
skeleton branches that count against it.

So the rule should adapt per frame AND lean thin. Erring thin is nearly free; erring fat is not.

EVERY CANDIDATE IS LABEL-FREE. A threshold picked with the ground truth is an oracle, and this
project has already mistaken one for a result once. Each rule sees only the ridge response.

  fixed q98 / q99      the current baseline and a thinner variant
  Otsu / Triangle /    classical histogram thresholds computed on the response
  Yen / Li
  mean + k*std         k = 3, 4
  knee                 point of maximum curvature on the sorted response
  thin-of(q98, Otsu)   whichever of the two predicts LESS area -- the directional-penalty rule

Selection is leave-one-frame-out over the rules, so the rule itself is never chosen on the
frame it is scored on.
"""
import os, sys, csv, json, time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.filters import meijering, threshold_otsu, threshold_triangle, threshold_yen, threshold_li
from skimage.morphology import remove_small_objects, skeletonize
from scipy import ndimage as ndi

SC = os.path.dirname(os.path.abspath(__file__))
_CE = os.path.dirname(os.path.dirname(SC))
_REPO = os.path.dirname(_CE)
ORIG, PAINT = f"{_REPO}/original", f"{_REPO}/interior_active_learning/paint"
rng = np.random.default_rng(20260920)


def to8(a):
    a = a.astype(np.float64)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def image_area(a):
    h, w = a.shape[:2]
    return a[:min(int(round(w * 2 / 3)), h)]


def cliou(p, g, tau):
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    sp, sg = skeletonize(p), skeletonize(g)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    a = ndi.distance_transform_edt(~sp) <= tau
    b = ndi.distance_transform_edt(~sg) <= tau
    tp = int((sg & a).sum()); fp = int((sp & ~(sp & b)).sum()); fn = int((sg & ~(sg & a)).sum())
    return float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0


def knee(r):
    """Threshold at the point of maximum curvature of the sorted response tail."""
    q = np.percentile(r, np.linspace(90, 99.99, 60))
    x = np.arange(len(q), dtype=float)
    y = (q - q.min()) / max(np.ptp(q), 1e-12)
    x = x / x.max()
    d = np.abs(y - (y[0] + (y[-1] - y[0]) * x))      # distance from the chord
    return float(q[int(np.argmax(d))])


def rules(r):
    """label-free threshold candidates on the ridge response r"""
    out = {}
    out["fixed q98"] = np.percentile(r, 98)
    out["fixed q99"] = np.percentile(r, 99)
    samp = r[::7, ::7].ravel()                        # subsample: histogram methods are global
    for nm, fn in (("Otsu", threshold_otsu), ("Triangle", threshold_triangle),
                   ("Yen", threshold_yen), ("Li", threshold_li)):
        try:
            out[nm] = float(fn(samp))
        except Exception:
            out[nm] = np.percentile(r, 98)
    mu, sd = float(samp.mean()), float(samp.std())
    out["mean+3sd"] = mu + 3 * sd
    out["mean+4sd"] = mu + 4 * sd
    out["knee"] = knee(r)
    out["thin-of(q98,Otsu)"] = max(out["fixed q98"], out["Otsu"])   # higher thr => LESS area
    return out


def main():
    gran = {r["frame"]: float(r["median_thick_px"] or 0)
            for r in csv.DictReader(open(f"{_CE}/analysis/label_granularity.csv"))}
    frames = []
    for n in sorted(gran):
        if os.path.exists(f"{ORIG}/{n}.tif") and os.path.exists(f"{PAINT}/{n}_correction_mask.png"):
            frames.append(n)
    print(f"{len(frames)} labelled frames\n", flush=True)
    res, t0 = {}, time.time()
    for i, n in enumerate(frames):
        g = image_area(to8(np.array(Image.open(f"{ORIG}/{n}.tif"))))
        cm = np.array(Image.open(f"{PAINT}/{n}_correction_mask.png"))
        while cm.ndim > 2:
            cm = cm[..., 0]
        gt = cm == 1
        if gt.shape != g.shape or gt.sum() < 2000:
            continue
        r = meijering(g.astype(np.float32) / 255.0, sigmas=np.arange(1, 5.0), black_ridges=True)
        tau = max(2, int(round(gran[n] / 2)))
        res[n] = {}
        for nm, thr in rules(r).items():
            p = remove_small_objects(r >= thr, max_size=32)
            res[n][nm] = {"clIoU": cliou(p, gt, tau),
                          "PAR": float(p.sum() / max(gt.sum(), 1)),
                          "frac": float(p.mean())}
        b = max(res[n], key=lambda k: res[n][k]["clIoU"])
        print(f"  [{i+1}/{len(frames)}] {n[:36]:<38} best {b:<18} "
              f"{res[n][b]['clIoU']:.3f}   ({time.time()-t0:.0f}s)", flush=True)
    names = list(res)
    RL = list(next(iter(res.values())))
    print(f"\n{'rule':<20} {'median clIoU':>13} {'median PAR':>11} {'over-pred frames':>17}")
    summ = {}
    for nm in RL:
        c = [res[n][nm]["clIoU"] for n in names]
        pa = [res[n][nm]["PAR"] for n in names]
        summ[nm] = {"clIoU": float(np.median(c)), "PAR": float(np.median(pa)),
                    "over": int(sum(1 for x in pa if x > 1)), "per_frame": c}
        print(f"{nm:<20} {np.median(c):>13.4f} {np.median(pa):>11.2f} {summ[nm]['over']:>13}/{len(pa)}")
    # LOFO over the RULE
    lofo = []
    for n in names:
        tr = [u for u in names if u != n]
        best = max(RL, key=lambda k: np.median([res[u][k]["clIoU"] for u in tr]))
        lofo.append(res[n][best]["clIoU"])
    print(f"\n  LOFO over rules: median clIoU_adapt {np.median(lofo):.4f}")
    print(f"  fixed q98 baseline (whole frame): {summ['fixed q98']['clIoU']:.4f}")
    from scipy.stats import wilcoxon
    d = [a - b for a, b in zip(lofo, summ["fixed q98"]["per_frame"])]
    nz = [x for x in d if x != 0]
    if len(nz) >= 6:
        print(f"  paired vs fixed q98: median delta {np.median(d):+.4f}, "
              f"wins {sum(1 for x in d if x>0)}/{len(d)}, p = {wilcoxon(nz, method='asymptotic').pvalue:.4g}")
    json.dump({"summary": summ, "lofo_median": float(np.median(lofo)), "frames": names},
              open(f"{SC}/adaptive_threshold.json", "w"), indent=1)
    print("\nwrote adaptive_threshold.json")


if __name__ == "__main__":
    main()
