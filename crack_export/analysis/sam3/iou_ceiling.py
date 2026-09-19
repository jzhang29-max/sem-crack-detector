#!/usr/bin/env python3
"""The ceiling pixel IoU can reach on this corpus for a PHYSICALLY CORRECT crack trace.

Take the ground truth's own skeleton, dilate it to ~3 px -- the actual width of a crack in
these micrographs -- and score it against the ground truth. This is the best a method could
possibly do while still predicting a crack rather than a paintbrush stroke.

Result: median IoU 0.1662, median clDice 0.9997.

Consequences, and they apply to every IoU in this project:
  * pixel IoU here has a CEILING of about 0.17 for a correct answer;
  * SAM 3's measured 0.1914 is ABOVE that ceiling, which means it is scoring by being thicker
    than a crack, not by being more accurate;
  * the global threshold's 0.2579 "win" is a statement that it imitates a 16 px brush better;
  * clDice gives the perfect thin answer 0.9997, so it is the metric that can actually see a
    correct result.

This does NOT mean the IoU numbers were computed wrongly. It means the quantity they measure is
'agreement with the annotator's brush width', and on this corpus that quantity is nearly
orthogonal to 'found the crack'.
"""
import os, json
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from scipy import ndimage as ndi

SC = os.path.dirname(os.path.abspath(__file__))


def main():
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}
    ious, clds, rows = [], [], []
    for m in meta:
        gt = np.array(Image.open(f"{SC}/tiles/{m['tile']}_gt.png")) > 127
        if gt.sum() < 50:
            continue
        sk = skeletonize(gt)
        p = ndi.binary_dilation(sk, iterations=1)
        i = float((p & gt).sum() / np.logical_or(p, gt).sum())
        sp = skeletonize(p)
        tp = (sp & gt).sum() / max(sp.sum(), 1)
        ts = (sk & p).sum() / max(sk.sum(), 1)
        c = float(2 * tp * ts / (tp + ts)) if (tp + ts) > 0 else 0.0
        ious.append(i); clds.append(c)
        rows.append({"tile": m["tile"], "IoU": i, "clDice": c})
    tiles = [m["tile"] for m in meta]
    sam = float(np.median([res.get((t, "crack"), {}).get("union_IoU", 0) for t in tiles]))
    ceil = float(np.median(ious))
    print(f"perfect 3 px trace down the label centreline, {len(ious)} tiles:")
    print(f"  median IoU    {ceil:.4f}   <- the ceiling for a correct answer")
    print(f"  median clDice {np.median(clds):.4f}   <- clDice can see a correct answer")
    print(f"\n  SAM 3 measured IoU {sam:.4f}  ->  "
          f"{'ABOVE' if sam > ceil else 'below'} the ceiling")
    print(f"  Any method scoring IoU > {ceil:.2f} here is being thicker than a crack.")
    json.dump({"ceiling_IoU": ceil, "ceiling_clDice": float(np.median(clds)),
               "sam3_IoU": sam, "per_tile": rows},
              open(f"{SC}/iou_ceiling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
