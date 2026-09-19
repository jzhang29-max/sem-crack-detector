#!/usr/bin/env python3
"""Register each hand-label frame to its RAW original .tif, so model inputs never come from
a rendered overlay.

Why this file exists: the first version of make_tiles.py built the model input from the GREEN
channel of the annotated overlay, on the belief that green survived a red overlay. The overlay
burns opaque red (225,25,25), whose green channel is a constant 25, so the label was written into the input as
black pixels and a bare threshold scored recall 1.000 on 16/16 tiles. The only leak-free source
is the original micrograph, which means we must first establish, per frame, the transform from
label coordinates to original-image coordinates.

Two cases in this corpus:
  (a) label frame is the original minus its databar strip -> offset (0,0), verified not assumed
  (b) label frame is a square crop at an unknown offset   -> recovered here

METHOD, and the two wrong instruments it replaced:

  * A coarse GRID over offsets fails outright. At a 133 px step on SEM texture every candidate
    scores ~0, so the argmax is noise dressed as an answer.
  * FFT correlation of the whole frame DOWNSCALED to 1/4 also fails. Downscaling destroys the
    texture that distinguishes the true offset; on one frame the /4 peak sat at the edge of the
    valid map, 566 px from a true offset that scores exactly 1.0000.

What works is correlating at FULL resolution, which for a whole 2952^2 template would be a
multi-GB FFT. So instead a small RED-FREE, high-variance PATCH of the label frame is matched
against the full original at full resolution -- exact NCC over every shift, one cheap FFT --
and the frame offset follows by subtracting the patch's own position. The patch must be red-free
because the burn-in has no counterpart in the original and would only add mismatch.

Every candidate is then scored by full-frame ncc with red EXCLUDED from the support, and that
score, not the patch correlation, decides. Registration is accepted only at ncc >= 0.99.
"""
import os, csv, json
import numpy as np
from PIL import Image
from skimage.feature import match_template
Image.MAX_IMAGE_PIXELS = None

ROOT = "/Users/jiamingzhang/Desktop/crack_export"
ORIG = "/Users/jiamingzhang/Desktop/sem-crack-detector/original"
OUT  = os.path.dirname(os.path.abspath(__file__))
P     = 512      # patch side
NPEAK = 8        # patch-correlation peaks refined by full-frame ncc

def to8(a):
    """16-bit SEM -> 8-bit on its own 0.5/99.5 percentiles.

    This reproduces the overlay renderer exactly: on frames needing no registration the
    recovered ncc is 1.0000, which a different stretch could not produce.
    """
    a = a.astype(np.float64)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

def frame_ncc(A, keep, org, dy, dx, step=4):
    """Full-frame NCC of the label frame's grey against the original at offset (dy,dx)."""
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

def pick_patch(A, keep):
    """Highest-variance PxP window containing no red at all."""
    H, W = A.shape
    if H < P or W < P:
        return None
    best = None
    ys = np.linspace(0, H - P, 12).astype(int)
    xs = np.linspace(0, W - P, 12).astype(int)
    for y in ys:
        for x in xs:
            k = keep[y:y + P, x:x + P]
            if not k.all():
                continue
            v = float(A[y:y + P, x:x + P].std())
            if best is None or v > best[0]:
                best = (v, int(y), int(x))
    return best

fine = [r["frame"] for r in csv.DictReader(open(f"{ROOT}/analysis/label_granularity.csv"))
        if r["median_thick_px"] and float(r["median_thick_px"]) <= 25]

res = {}
for n in fine:
    ov  = np.array(Image.open(f"{ROOT}/overlays/{n}_overlay.png").convert("RGB"))
    org = to8(np.array(Image.open(f"{ORIG}/{n}.tif")))
    red  = (ov[..., 0] > 150) & (ov[..., 1] < 80) & (ov[..., 2] < 80)
    keep = ~red
    A = ov[..., 1]
    H, W = A.shape

    if H > org.shape[0] or W > org.shape[1]:
        res[n] = {"ok": False, "why": "label frame larger than original"}
    else:
        seeds = [(0, 0)]                       # identity is always a candidate
        pp = pick_patch(A, keep)
        if pp is not None:
            _, py, px = pp
            tmpl = A[py:py + P, px:px + P].astype(np.float32)
            r = match_template(org.astype(np.float32), tmpl)
            rc = r.copy()
            for _ in range(NPEAK):
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
        c, dy, dx = best
        res[n] = {"ok": bool(c >= 0.99), "ncc": round(c, 4), "dy": int(dy), "dx": int(dx),
                  "label_hw": [int(H), int(W)], "orig_hw": [int(org.shape[0]), int(org.shape[1])],
                  "red_frac": round(float(red.mean()), 5)}
    r = res[n]
    print(f"  {'OK ' if r.get('ok') else 'FAIL'} {n:<40} ncc={r.get('ncc')} offset=({r.get('dy')},{r.get('dx')})")

json.dump(res, open(f"{OUT}/alignment.json", "w"), indent=1)
ok = sum(1 for v in res.values() if v.get("ok"))
print(f"\n{ok}/{len(res)} frames registered to their raw original at ncc >= 0.99")
if ok < len(res):
    raise SystemExit(1)
