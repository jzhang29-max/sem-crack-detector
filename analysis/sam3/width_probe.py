#!/usr/bin/env python3
"""Test whether IoU on this corpus measures crack detection or LABEL THICKNESS.

Label strokes in the fine subset are pooled median 14.6 px wide (p90 40.8, max 125.9) while
the crack itself is of order 3 px. If IoU is really scoring detection, thickening a correct
thin prediction should not help. If IoU is scoring how well the prediction matches the BRUSH,
dilation should help, and the best dilation should track the label half-width.

Prediction being tested: best dilation ~ (label width)/2 - (prediction width)/2, i.e. several
pixels, and IoU should rise materially. A null result -- no gain from dilation -- would mean
the metric is not dominated by stroke width and the earlier concern was wrong.
"""
import os, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

SC = os.path.dirname(os.path.abspath(__file__))


def iou(p, gt):
    u = np.logical_or(p, gt).sum()
    return float(np.logical_and(p, gt).sum() / u) if u else 0.0


def main():
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}
    print(f"  {'tile':<32} {'labelW':>7} " + " ".join(f"{'dil'+str(d):>7}" for d in (0, 2, 4, 6, 8, 12)) + f" {'best':>5}")
    bests, widths, gains = [], [], []
    for m in meta:
        t = m["tile"]
        gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
        f = f"{SC}/masks/{t}__crack.npz"
        if not os.path.exists(f) or gt.sum() < 50:
            continue
        pred = np.load(f)["union"]
        if pred.sum() == 0:
            continue
        w = float(np.median(2 * ndi.distance_transform_edt(gt)[skeletonize(gt)]))
        pw = float(np.median(2 * ndi.distance_transform_edt(pred)[skeletonize(pred)])) if pred.sum() > 50 else 0.0
        vals = []
        for d in (0, 2, 4, 6, 8, 12):
            p = ndi.binary_dilation(pred, iterations=d) if d else pred
            vals.append(iou(p, gt))
        bd = (0, 2, 4, 6, 8, 12)[int(np.argmax(vals))]
        bests.append(bd); widths.append(w); gains.append(max(vals) - vals[0])
        print(f"  {t[:30]:<32} {w:>7.1f} " + " ".join(f"{v:>7.3f}" for v in vals) + f" {bd:>5}")
    print(f"\n  median IoU gain from dilation alone: {np.median(gains):+.4f}")
    print(f"  median best dilation: {np.median(bests):.0f} px  (label half-width median "
          f"{np.median(widths)/2:.1f} px)")
    from scipy.stats import spearmanr
    rho, p = spearmanr(widths, bests)
    print(f"  Spearman(label width, best dilation) = {rho:+.3f}, p = {p:.4f}  n={len(bests)}")
    print("\n  If the gain is material and the correlation positive, IoU here is scoring how")
    print("  well a prediction matches the BRUSH WIDTH, not how well it finds the crack.")
    json.dump({"best_dilation": bests, "label_width": widths, "gain": gains,
               "rho": float(rho), "p": float(p)}, open(f"{SC}/width_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
