#!/usr/bin/env python3
"""Enclosed regions ("holes") of the crack network.

Why this is the sharpest available test of intergranular cracking: if a crack
network runs along grain boundaries, the islands it encloses ARE grains. Their
size and shape distribution should then match the material's grain structure --
equiaxed, narrow size distribution, solidity near that of a convex polygon.
A transgranular or blob-over-detection network encloses nothing systematic.

Sizes are reported in px AND normalised by the frame's own mean crack width,
because 26 of 62 frames are 2x-downsampled versions of the same optics
(verified: same field, linear ratio 2.000, crack-width ratio 2.023), so raw px
lengths are not comparable across frames.
"""
import csv, glob, json, os
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = "/Users/jiamingzhang/Desktop/crack_export"
os.chdir(ROOT)
N8 = np.ones((3, 3), np.uint8)
skel = {json.load(open(p))["frame"]["SourceImage"]: json.load(open(p))["frame"]
        for p in glob.glob("analysis/skeleton/*.json")}

out = []
for p in sorted(glob.glob("masks/*_mask.png")):
    n = os.path.basename(p)[:-len("_mask.png")]
    m = np.array(Image.open(p))
    while m.ndim > 2:
        m = m[..., 0]
    m = m < 128
    filled = ndi.binary_fill_holes(m)
    holes = filled & ~m
    lab, k = ndi.label(holes, structure=N8)
    if k == 0:
        out.append({"SourceImage": n, "Holes": 0}); continue
    areas = np.bincount(lab.ravel())[1:].astype(float)
    W = skel[n].get("MeanWidth_px") or np.nan
    eqd = 2.0 * np.sqrt(areas / np.pi)                 # equivalent circular diameter
    keep = areas >= 100                                # below this is threshold noise
    a, d = areas[keep], eqd[keep]
    rec = {"SourceImage": n, "H": skel[n]["H"], "W_px": skel[n]["W"],
           "MeanCrackWidth_px": W, "Holes": int(keep.sum()),
           "HoleAreaPct_ofFrame": round(100 * a.sum() / (skel[n]["H"] * skel[n]["W"]), 4),
           "HoleAreaPct_ofFilled": round(100 * a.sum() / float(filled.sum()), 4) if filled.sum() else "",
           "EqDiam_median_px": round(float(np.median(d)), 2) if d.size else "",
           "EqDiam_p25_px": round(float(np.percentile(d, 25)), 2) if d.size else "",
           "EqDiam_p75_px": round(float(np.percentile(d, 75)), 2) if d.size else "",
           "EqDiam_IQRoverMedian": round(float((np.percentile(d, 75) - np.percentile(d, 25))
                                               / np.median(d)), 3) if d.size else "",
           "EqDiam_median_overCrackWidth": round(float(np.median(d) / W), 3)
                                            if (d.size and W and W > 0) else "",
           }
    out.append(rec)
    print(f"{n}: {rec['Holes']} holes, median eqdiam {rec['EqDiam_median_px']} px, "
          f"/width {rec['EqDiam_median_overCrackWidth']}", flush=True)

cols = sorted({k for r in out for k in r})
cols = ["SourceImage"] + [c for c in cols if c != "SourceImage"]
with open("analysis/holes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)
print("\nwrote analysis/holes.csv")
