#!/usr/bin/env python3
"""Where do the model's predicted pixels land, relative to the two human annotation layers?

This decides how to read the low pixel IoU. Every predicted pixel is in exactly one of:
  A  inside the FINE correction mask   -- the layer IoU is scored against
  B  inside the overlay's PAINTED region but not the fine mask -- a human did mark this as
     crack, but not in the layer being scored, so IoU counts it as a false positive
  C  outside every human assertion     -- the only unambiguously wrong part

Presentation note: A, B and C are per-tile fractions that sum to 1 WITHIN a tile. Their three
medians come from different tiles and do NOT sum to 1, so quoting the three medians side by
side implies a decomposition that does not exist. Pooled fractions (all predicted pixels in
the corpus) are reported as the headline, with the per-tile medians shown separately and
labelled as such.
"""
import os, json
import numpy as np
from PIL import Image

SC = os.path.dirname(os.path.abspath(__file__))


def main():
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    tot = np.zeros(3, dtype=np.int64)
    per = []
    for m in meta:
        t = m["tile"]
        f = f"{SC}/masks/{t}__crack.npz"
        if not os.path.exists(f):
            continue
        p = np.load(f)["union"]
        if p.sum() == 0:
            continue
        gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
        ov = np.array(Image.open(f"{SC}/tiles/{t}_overlay_REFERENCE_DO_NOT_FEED.png").convert("RGB"))
        red = (ov[..., 0] > 150) & (ov[..., 1] < 80) & (ov[..., 2] < 80)
        a = int((p & gt).sum()); b = int((p & red & ~gt).sum()); c = int((p & ~red & ~gt).sum())
        assert a + b + c == int(p.sum()), t
        tot += np.array([a, b, c])
        per.append((t, a / p.sum(), b / p.sum(), c / p.sum()))
    pooled = tot / tot.sum()
    print(f"POOLED over all {tot.sum():,} predicted pixels in {len(per)} tiles "
          f"(these three DO sum to 100%):")
    for nm, v in zip(("inside the fine correction mask (scored as TP)",
                      "inside the painted region only (scored as FP)",
                      "outside every human assertion (unambiguously FP)"), pooled):
        print(f"  {nm:<50} {100*v:>6.1f}%")
    print(f"\nPer-tile medians (from DIFFERENT tiles -- they do not sum to 100%, do not add them):")
    for i, nm in enumerate(("in fine mask", "painted only", "outside both"), start=1):
        print(f"  median {nm:<14} {100*np.median([x[i] for x in per]):>6.1f}%")
    ab = [x[1] + x[2] for x in per]
    print(f"\n  median fraction inside SOME human assertion (A+B, per tile): {100*np.median(ab):.1f}%")
    print(f"  pooled fraction inside some human assertion:                 {100*(pooled[0]+pooled[1]):.1f}%")
    json.dump({"pooled": pooled.tolist(), "per_tile": per}, open(f"{SC}/pred_decomposition.json", "w"), indent=1)


if __name__ == "__main__":
    main()
