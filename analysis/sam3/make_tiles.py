#!/usr/bin/env python3
"""Cut 1024x1024 test tiles centred on HAND-LABELLED crack, from the 9 fine-stroke frames.

Only the fine-stroke frames (median brush <=25 px) are used, because those are the only
labels in this corpus that approximate a crack outline rather than a region assertion
(see analysis/LABEL_GRANULARITY.md: 91% of marked pixels come from strokes up to 413 px).
Even these are not pixel-precise, so every score below is indicative, not a benchmark.

Tiles are cut at NATIVE resolution: SAM-family models ingest ~1024 px, and downscaling a
6144 px frame to 1024 would make a 3 px crack sub-pixel and guarantee failure for a reason
that has nothing to do with the model.
"""
import os, csv, json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy import ndimage as ndi

ROOT = "/Users/jiamingzhang/Desktop/crack_export"
PAINT = "/Users/jiamingzhang/Desktop/sem-crack-detector/interior_active_learning/paint"
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "tiles"), exist_ok=True)
S = 1024

fine = [r["frame"] for r in csv.DictReader(open(f"{ROOT}/analysis/label_granularity.csv"))
        if r["median_thick_px"] and float(r["median_thick_px"]) <= 25]

meta = []
for n in fine:
    cm = np.array(Image.open(f"{PAINT}/{n}_correction_mask.png"))
    while cm.ndim > 2:
        cm = cm[..., 0]
    gt = cm == 1
    if gt.sum() < 2000:
        continue
    ov = Image.open(f"{ROOT}/overlays/{n}_overlay.png").convert("RGB")
    H, W = gt.shape
    # two tiles per frame: the densest labelled window, and a second one far from it
    dens = ndi.uniform_filter(gt.astype(np.float32), size=256)
    picked = []
    d = dens.copy()
    for k in range(2):
        cy, cx = np.unravel_index(np.argmax(d), d.shape)
        y = int(np.clip(cy - S // 2, 0, max(H - S, 0)))
        x = int(np.clip(cx - S // 2, 0, max(W - S, 0)))
        sub = gt[y:y + S, x:x + S]
        if sub.shape != (S, S) or sub.sum() < 400:
            break
        tid = f"{n}__t{k}"
        Image.fromarray((sub * 255).astype(np.uint8)).save(f"{OUT}/tiles/{tid}_gt.png")
        ov.crop((x, y, x + S, y + S)).save(f"{OUT}/tiles/{tid}_rgb.png")
        # grayscale source without the red burn-in, for models that want the raw image
        a = np.array(ov.crop((x, y, x + S, y + S)))
        red = (a[..., 0] > 150) & (a[..., 1] < 80) & (a[..., 2] < 80)
        g = a[..., 1].copy()           # green channel is unaffected by the red overlay
        Image.fromarray(g).save(f"{OUT}/tiles/{tid}_gray.png")
        meta.append({"tile": tid, "frame": n, "y": y, "x": x,
                     "gt_px": int(sub.sum()), "gt_frac": round(float(sub.mean()), 5),
                     "red_frac_in_tile": round(float(red.mean()), 5)})
        picked.append((y, x))
        d[max(0, y - S):y + S, max(0, x - S):x + S] = 0   # suppress for the next pick
json.dump(meta, open(f"{OUT}/tiles/meta.json", "w"), indent=1)
print(f"{len(meta)} tiles from {len(set(m['frame'] for m in meta))} frames")
for m in meta:
    print(f"  {m['tile']:<46} gt {m['gt_px']:>7,} px ({100*m['gt_frac']:.2f}% of tile)")
