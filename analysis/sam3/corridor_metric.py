#!/usr/bin/env python3
"""Score predictions the way SUPERSET labels should be scored, not with pixel IoU.

The labels in this corpus are region assertions: a brush of median 59 px (max 413) asserting
that a ~3 px crack lies somewhere inside. Pixel IoU against such a label punishes a CORRECT
thin prediction for being thin -- a perfect 3 px trace inside a 59 px stroke scores IoU ~0.05.
That is not a model failure, it is a metric failure, and it is why every IoU in this project
is labelled indicative.

The right scoring for a superset label is the TIGHTNESS PRIOR of Kervadec, Dolz, Wang, Granger
& Ben Ayed, MIDL 2020 (PMLR 121:365-381, arXiv:2004.06816): inside the annotated region the
prediction must be present, outside it must be absent. Restated as two measurable quantities:

  containment  = fraction of PREDICTED pixels that fall inside the annotated corridor.
                 Penalises hallucination outside the assertion. 1.0 is perfect.
  coverage     = fraction of connected corridor COMPONENTS containing at least one predicted
                 pixel. This is the tightness prior itself -- "this region contains crack" is
                 satisfied by one pixel, not by filling the region. 1.0 is perfect.
  crossing     = fraction of corridor SCANLINES that contain at least one predicted pixel.

MEASURED WARNING -- COVERAGE AND CROSSING DO NOT DISCRIMINATE, DO NOT QUOTE THEM ALONE.
They were built here as tightness-prior statistics and then tested against nulls, which is the
only reason this is known. At this corridor density (median 5.5% of the tile) an area-matched
RANDOM SCATTER reaches coverage 0.902 and crossing 0.937 against SAM 3's 0.923 and 0.937, and
simply predicting the entire tile reaches 1.000 on both. A corridor component is large enough
that almost anything touches it. The null panel is therefore printed on every run and the
numbers are meaningless without it.

CONTAINMENT is the only component that separates: SAM 3 0.707, area-matched random 0.055,
all-ones 0.055, random strokes 0.041. It is not gameable by predicting everything, because
predicting the tile gives containment = the corridor area fraction. But note that on this
corpus Otsu scores 0.779 -- HIGHER than SAM 3.

Reported alongside IoU, never instead of it, and all of it is indicative because the corridor
is itself a human's assertion.
"""
import os, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SC = os.path.dirname(os.path.abspath(__file__))


def corridor_scores(pred, corridor):
    """(containment, coverage, crossing) of `pred` against a superset `corridor`."""
    if corridor.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    containment = float(pred[corridor].sum() / pred.sum()) if pred.sum() else 0.0
    lab, n = ndi.label(corridor)
    if n == 0:
        return containment, float("nan"), float("nan")
    hit = ndi.sum(pred, lab, index=np.arange(1, n + 1))
    coverage = float((hit > 0).mean())
    # crossing: for each corridor component, scan along its longer axis
    tot = ok = 0
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = (lab[sl] == i)
        sp = pred[sl] & sub
        h, w = sub.shape
        axis = 1 if w >= h else 0          # scan across the short axis
        present = sub.any(axis=axis)
        got = sp.any(axis=axis)
        tot += int(present.sum()); ok += int((present & got).sum())
    crossing = float(ok / tot) if tot else float("nan")
    return containment, coverage, crossing


def nulls(meta, corridor_of):
    """Null predictors. Without these the corridor numbers cannot be interpreted."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    rng = np.random.default_rng(7)
    out = {}
    for name in ("Otsu", "random strokes", "all-ones", "area-matched random"):
        C, V, X = [], [], []
        for m in meta:
            t = m["tile"]
            cor, gt = corridor_of(t)
            g = np.array(Image.open(f"{SC}/tiles/{t}_gray.png").convert("L"))
            if name == "Otsu":
                p = remove_small_objects(g <= threshold_otsu(g), max_size=32)
            elif name == "all-ones":
                p = np.ones_like(gt)
            elif name == "random strokes":
                p = np.zeros_like(gt)
                for _ in range(40):
                    y0, x0 = rng.integers(0, 1024, 2)
                    a = rng.uniform(0, np.pi); L = rng.integers(100, 600)
                    ys = (y0 + np.arange(L) * np.sin(a)).astype(int)
                    xs = (x0 + np.arange(L) * np.cos(a)).astype(int)
                    ok = (ys >= 0) & (ys < 1024) & (xs >= 0) & (xs < 1024)
                    p[np.clip(ys[ok], 0, 1023), np.clip(xs[ok], 0, 1023)] = True
                p = ndi.binary_dilation(p, iterations=1)
            else:
                f = f"{SC}/masks/{t}__crack.npz"
                k = int(np.load(f)["union"].sum()) if os.path.exists(f) else 0
                flat = np.zeros(gt.size, bool)
                if k:
                    flat[rng.choice(gt.size, k, replace=False)] = True
                p = ndi.binary_dilation(flat.reshape(gt.shape), iterations=1)
            c, cov, cr = corridor_scores(p, cor)
            C.append(c); V.append(cov); X.append(cr)
        out[name] = (np.nanmedian(C), np.nanmedian(V), np.nanmedian(X))
    return out


def main():
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}
    print("Corridor scoring against the PAINTED region (the superset the human actually asserted)")
    print(f"  {'tile':<34} {'corridor %':>10} {'contain':>8} {'coverage':>9} {'crossing':>9} {'pixIoU':>7}")
    rows = []
    for m in meta:
        t = m["tile"]
        ov = np.array(Image.open(f"{SC}/tiles/{t}_overlay_REFERENCE_DO_NOT_FEED.png").convert("RGB"))
        corridor = (ov[..., 0] > 150) & (ov[..., 1] < 80) & (ov[..., 2] < 80)
        gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
        corridor = corridor | gt                       # the assertion is the union of both layers
        f = f"{SC}/masks/{t}__crack.npz"
        pred = np.load(f)["union"] if os.path.exists(f) else np.zeros_like(gt)
        c, cov, cr = corridor_scores(pred, corridor)
        pix = res.get((t, "crack"), {}).get("union_IoU", 0.0)
        rows.append({"tile": t, "frame": m["frame"], "corridor_frac": float(corridor.mean()),
                     "containment": c, "coverage": cov, "crossing": cr, "pixel_IoU": pix})
        print(f"  {t[:33]:<34} {100*corridor.mean():>9.2f}% {c:>8.3f} {cov:>9.3f} {cr:>9.3f} {pix:>7.3f}")
    f = lambda k: np.nanmedian([r[k] for r in rows])
    print(f"\n  median containment {f('containment'):.3f} | coverage {f('coverage'):.3f} | "
          f"crossing {f('crossing'):.3f} | pixel IoU {f('pixel_IoU'):.3f}")
    print(f"  median corridor area fraction {100*f('corridor_frac'):.2f}% -- which is also the")
    print(f"  containment a tile-filling prediction would score, so containment is not gameable.")
    def corridor_of(t):
        ov = np.array(Image.open(f"{SC}/tiles/{t}_overlay_REFERENCE_DO_NOT_FEED.png").convert("RGB"))
        gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
        return ((ov[..., 0] > 150) & (ov[..., 1] < 80) & (ov[..., 2] < 80)) | gt, gt

    print("\n  NULL PANEL -- these numbers are why coverage and crossing must not be quoted:")
    print(f"    {'predictor':<22} {'containment':>12} {'coverage':>9} {'crossing':>9}")
    print(f"    {'SAM 3 union':<22} {f('containment'):>12.3f} {f('coverage'):>9.3f} {f('crossing'):>9.3f}")
    nl = nulls(meta, corridor_of)
    for k, (a, b, c) in nl.items():
        print(f"    {k:<22} {a:>12.3f} {b:>9.3f} {c:>9.3f}")
    print("    -> an area-matched RANDOM SCATTER matches SAM 3 on coverage and crossing;")
    print("       all-ones scores 1.000 on both. Only containment separates, and Otsu wins it.")
    json.dump({"per_tile": rows, "nulls": {k: list(v) for k, v in nl.items()}},
              open(f"{SC}/corridor_scores.json", "w"), indent=1)
    print("\nwrote corridor_scores.json")


if __name__ == "__main__":
    main()
