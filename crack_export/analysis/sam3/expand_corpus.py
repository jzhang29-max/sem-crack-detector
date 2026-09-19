#!/usr/bin/env python3
"""Extend the evaluation from 9 frames to every frame that carries a hand label.

Why this exists: power_analysis.py measured that separating these methods needs ~40-60 source
frames and the evaluation uses 9. The corpus has 47 frames with a correction mask; 38 were
excluded because their median brush is wider than 25 px, which makes pixel IoU meaningless
against them. It does NOT make clIoU_tau or containment meaningless -- those are
width-insensitive by construction and were built for exactly this label type. So the 38 are
usable, and using them crosses the power threshold for three of the four comparisons.

This script does the two prerequisites for all 47 frames:
  1. register each label frame to its RAW original (the grey a model sees must never come from
     a rendered overlay -- that mistake voided the first run entirely);
  2. cut ONE 1024x1024 tile per frame, centred on the densest labelled window.

One tile per frame, deliberately: the unit of independence is the frame, so a second tile adds
pixels without adding evidence, and the earlier 16-tile set had two pairs sharing >50% of their
area because of a coordinate bug. One per frame makes the tile count and the frame count the
same number and removes the whole class of error.

Writes tiles_all/ and alignment_all.json. Frames that fail to register are reported and skipped.
"""
import os, csv, json
import numpy as np
from PIL import Image
from skimage.feature import match_template
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None

_CE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO = os.path.dirname(_CE)
SC = os.path.dirname(os.path.abspath(__file__))
PAINT = f"{_REPO}/interior_active_learning/paint"
ORIG = f"{_REPO}/original"
S = 1024
P = 512


def to8(a):
    a = a.astype(np.float64)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def frame_ncc(A, keep, org, dy, dx, step=4):
    H, W = A.shape
    if dy < 0 or dx < 0 or dy + H > org.shape[0] or dx + W > org.shape[1]:
        return -1.0
    B = org[dy:dy + H:step, dx:dx + W:step]
    a = A[::step, ::step]; k = keep[::step, ::step]
    if B.shape != a.shape or k.sum() < 64:
        return -1.0
    x = a[k].astype(np.float64); y = B[k].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else -1.0


def register(A, keep, org):
    """Red-free patch matched at FULL resolution, then full-frame ncc arbitrates."""
    H, W = A.shape
    if H > org.shape[0] or W > org.shape[1]:
        return None
    seeds = [(0, 0)]
    best_patch = None
    ys = np.linspace(0, H - P, 10).astype(int) if H >= P else []
    xs = np.linspace(0, W - P, 10).astype(int) if W >= P else []
    for y in ys:
        for x in xs:
            if not keep[y:y + P, x:x + P].all():
                continue
            v = float(A[y:y + P, x:x + P].std())
            if best_patch is None or v > best_patch[0]:
                best_patch = (v, int(y), int(x))
    if best_patch:
        _, py, px = best_patch
        r = match_template(org.astype(np.float32), A[py:py + P, px:px + P].astype(np.float32))
        rc = r.copy()
        for _ in range(6):
            i = np.unravel_index(np.argmax(rc), rc.shape)
            seeds.append((int(i[0]) - py, int(i[1]) - px))
            rc[max(0, i[0] - 8):i[0] + 9, max(0, i[1] - 8):i[1] + 9] = -1
    best = None
    for (y0, x0) in seeds:
        for dy in range(y0 - 3, y0 + 4):
            for dx in range(x0 - 3, x0 + 4):
                c = frame_ncc(A, keep, org, dy, dx)
                if best is None or c > best[0]:
                    best = (c, dy, dx)
    return best


def main():
    gran = {r["frame"]: r for r in csv.DictReader(open(f"{_CE}/analysis/label_granularity.csv"))}
    os.makedirs(f"{SC}/tiles_all", exist_ok=True)
    align, meta, failed = {}, [], []
    for i, n in enumerate(sorted(gran)):
        cm_p = f"{PAINT}/{n}_correction_mask.png"
        ov_p = f"{_CE}/overlays/{n}_overlay.png"
        or_p = f"{ORIG}/{n}.tif"
        if not all(os.path.exists(p) for p in (cm_p, ov_p, or_p)):
            failed.append((n, "missing input")); continue
        cm = np.array(Image.open(cm_p))
        while cm.ndim > 2:
            cm = cm[..., 0]
        gt = cm == 1
        if gt.sum() < 2000:
            failed.append((n, f"only {int(gt.sum())} labelled px")); continue
        ov = np.array(Image.open(ov_p).convert("RGB"))
        if ov.shape[:2] != gt.shape:
            failed.append((n, "overlay/mask shape mismatch")); continue
        org = to8(np.array(Image.open(or_p)))
        red = (ov[..., 0] > 150) & (ov[..., 1] < 80) & (ov[..., 2] < 80)
        b = register(ov[..., 1], ~red, org)
        if b is None or b[0] < 0.99:
            failed.append((n, f"ncc {b[0]:.3f}" if b else "no registration")); continue
        c, dy, dx = b
        align[n] = {"ncc": round(c, 4), "dy": dy, "dx": dx}
        raw = org[dy:dy + gt.shape[0], dx:dx + gt.shape[1]]
        dens = ndi.uniform_filter(gt.astype(np.float32), size=256)
        cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
        y = int(np.clip(cy - S // 2, 0, max(gt.shape[0] - S, 0)))
        x = int(np.clip(cx - S // 2, 0, max(gt.shape[1] - S, 0)))
        sub = gt[y:y + S, x:x + S]
        if sub.shape != (S, S) or sub.sum() < 400:
            failed.append((n, "no usable 1024 window")); continue
        Image.fromarray((sub * 255).astype(np.uint8)).save(f"{SC}/tiles_all/{n}_gt.png")
        Image.fromarray(raw[y:y + S, x:x + S]).save(f"{SC}/tiles_all/{n}_gray.png")
        Image.fromarray((red[y:y + S, x:x + S] * 255).astype(np.uint8)).save(f"{SC}/tiles_all/{n}_corridor.png")
        thick = float(gran[n]["median_thick_px"] or 0)
        meta.append({"tile": n, "frame": n, "y": y, "x": x, "ncc": round(c, 4),
                     "gt_px": int(sub.sum()), "gt_frac": round(float(sub.mean()), 5),
                     "median_thick_px": thick, "fine": thick <= 25})
        print(f"  [{i+1}/{len(gran)}] {n[:40]:<42} ncc {c:.4f}  gt {sub.sum():>7,} px  "
              f"brush {thick:>5.1f} px", flush=True)
    json.dump(align, open(f"{SC}/alignment_all.json", "w"), indent=1)
    json.dump(meta, open(f"{SC}/tiles_all/meta.json", "w"), indent=1)
    print(f"\n{len(meta)} frames tiled ({sum(1 for m in meta if m['fine'])} fine, "
          f"{sum(1 for m in meta if not m['fine'])} coarse)")
    if failed:
        print(f"{len(failed)} skipped:")
        for n, why in failed:
            print(f"   {n[:44]:<46} {why}")


if __name__ == "__main__":
    main()
