#!/usr/bin/env python3
"""Run the best detector over every original frame and write viewable overlays.

Outputs, all regenerable and gitignored:
  detector_masks/<frame>.png     full-resolution binary mask (PNG, compresses well)
  detector_overlays/<frame>.png  1/3-scale RGB overlay for looking at
  detector_contact_sheet.png     every frame on one page
  detector_all.json              per-frame predicted area and, where a label exists, clIoU_adapt

THE DATABAR IS CROPPED. These SEM originals carry a printed information bar below the image
(6144x4376 where the image is 6144x4096; 3072x2188 where it is 3072x2048). A ridge filter fires
enthusiastically on printed text, so the image area is taken as width x 2/3 -- which reproduces
both observed geometries exactly -- and anything below it is discarded before detection rather
than after, so the response normalisation is not skewed by the bar either.

The greyscale is the raw .tif on its own 0.5/99.5 percentiles. Never a rendered overlay: feeding
one is what invalidated this project's first benchmark.
"""
import os, sys, json, glob, time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import best_detector as BD
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

SC = os.path.dirname(os.path.abspath(__file__))
_CE = os.path.dirname(os.path.dirname(SC))
_REPO = os.path.dirname(_CE)
ORIG = f"{_REPO}/original"
PAINT = f"{_REPO}/interior_active_learning/paint"


def to8(a):
    a = a.astype(np.float64)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def image_area(a):
    """Drop the printed databar: the image is width x 2/3 on every frame in this corpus."""
    h, w = a.shape[:2]
    ih = int(round(w * 2 / 3))
    return a[:min(ih, h)]


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


def main():
    import csv
    gran = {r["frame"]: float(r["median_thick_px"] or 0)
            for r in csv.DictReader(open(f"{_CE}/analysis/label_granularity.csv"))}
    os.makedirs(f"{SC}/detector_masks", exist_ok=True)
    os.makedirs(f"{SC}/detector_overlays", exist_ok=True)
    files = sorted(glob.glob(f"{ORIG}/*.tif"))
    rows, t0 = [], time.time()
    for i, f in enumerate(files):
        n = os.path.basename(f)[:-4]
        g = image_area(to8(np.array(Image.open(f))))
        m = BD.detect(g)
        Image.fromarray((m * 255).astype(np.uint8)).save(f"{SC}/detector_masks/{n}.png",
                                                         optimize=True)
        ov = np.stack([g] * 3, -1)
        ov[m] = (0.25 * ov[m] + 0.75 * np.array([235, 105, 50])).astype(np.uint8)
        Image.fromarray(ov).resize((ov.shape[1] // 3, ov.shape[0] // 3),
                                   Image.LANCZOS).save(f"{SC}/detector_overlays/{n}.png")
        rec = {"frame": n, "pred_frac": round(float(m.mean()), 5),
               "shape": list(g.shape), "brush_px": gran.get(n)}
        cmp_ = f"{PAINT}/{n}_correction_mask.png"
        if os.path.exists(cmp_):
            cm = np.array(Image.open(cmp_))
            while cm.ndim > 2:
                cm = cm[..., 0]
            gt = cm == 1
            if gt.shape == m.shape and gt.sum() > 2000:
                tau = max(2, int(round(gran.get(n, 8) / 2)))
                rec["clIoU_adapt"] = round(cliou(m, gt, tau), 4)
                rec["PAR"] = round(float(m.sum() / gt.sum()), 3)
        rows.append(rec)
        print(f"  [{i+1}/{len(files)}] {n[:40]:<42} pred {100*m.mean():>5.2f}%"
              + (f"  clIoU {rec['clIoU_adapt']:.3f}" if "clIoU_adapt" in rec else "")
              + f"   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(f"{SC}/detector_all.json", "w"), indent=1)
    sc = [r["clIoU_adapt"] for r in rows if "clIoU_adapt" in r]
    print(f"\n{len(rows)} frames. {len(sc)} scorable; median clIoU_adapt {np.median(sc):.4f}")
    print(f"median predicted area {100*np.median([r['pred_frac'] for r in rows]):.2f}%")
    print(f"wrote detector_masks/, detector_overlays/, detector_all.json  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
