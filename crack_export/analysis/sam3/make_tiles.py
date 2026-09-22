#!/usr/bin/env python3
"""Cut 1024x1024 test tiles centred on HAND-LABELLED crack, from the 9 fine-stroke frames.

Only the fine-stroke frames (median brush <=25 px) are used, because those are the only
labels in this corpus that approximate a crack outline rather than a region assertion
(see analysis/LABEL_GRANULARITY.md: 91% of marked pixels come from strokes up to 413 px).
Even these are not pixel-precise, so every score below is indicative, not a benchmark.

Tiles are cut at NATIVE resolution: SAM-family models ingest ~1024 px, and downscaling a
6144 px frame to 1024 would make a 3 px crack sub-pixel and guarantee failure for a reason
that has nothing to do with the model.

THE MODEL INPUT COMES FROM THE RAW ORIGINAL .tif, NEVER FROM THE OVERLAY.
The first version of this script did the opposite. It built the input from the overlay's GREEN
channel, with the comment "green channel is unaffected by the red overlay". The overlay burns
opaque red (225, 25, 25), whose green channel is a constant 25, so every labelled pixel was written into the
input as black. A bare threshold `green < 80` -- no model -- then recovered the ground truth at
recall 1.000 on 16 of 16 tiles, median IoU 0.106, against SAM 3's median union IoU of 0.119 over the 14 tiles that returned anything (0.0775 over all 16).
The whole experiment measured its own annotation.

So: the grey is read from sem-crack-detector/original/<frame>.tif at the offset established by
align_originals.py (9/9 frames at ncc >= 0.99, FIVE of them exactly 1.0000 -- this said four;
alignment.json lists CBS_01, b4_CBS_02, CBS_001, AS_CBS_0001 and Cast_ETD_0003, and
verify_claims.py registers the count as 5), and the overlay is
written out only under a name no model loader would reach for. leak_check.py then proves the
input does not encode the label -- run it, and do not trust any score produced without it.
"""
import os, csv, json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy import ndimage as ndi

# --- portable roots -------------------------------------------------------------
# Resolved from this file's own location so the analysis runs from a fresh clone.
# crack_export used to be a separate repo beside sem-crack-detector, and every script
# hard-coded /Users/jiamingzhang/Desktop/... Now that it lives inside the repo, those
# literals would have made a clone unrunnable for anyone but this laptop.
import os as _os
_CE = _os.path.dirname(_os.path.abspath(__file__))
while _os.path.basename(_CE) != "crack_export" and _os.path.dirname(_CE) != _CE:
    _CE = _os.path.dirname(_CE)
_REPO = _os.path.dirname(_CE)
# --------------------------------------------------------------------------------
ROOT = _CE
PAINT = f"{_REPO}/interior_active_learning/paint"
ORIG = f"{_REPO}/original"
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "tiles"), exist_ok=True)
S = 1024

ALIGN = json.load(open(f"{OUT}/alignment.json"))    # run align_originals.py first


def to8(a):
    """16-bit SEM -> 8-bit on its own 0.5/99.5 percentiles (verified against the renderer)."""
    a = a.astype(np.float64)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


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
    al = ALIGN.get(n)
    if not al or not al.get("ok"):
        print(f"  SKIP {n}: not registered to its original (ncc {al.get('ncc') if al else None})")
        continue
    ov = Image.open(f"{ROOT}/overlays/{n}_overlay.png").convert("RGB")
    # the leak-free grey: raw original, cropped to the label frame's own coordinate system
    raw = to8(np.array(Image.open(f"{ORIG}/{n}.tif")))
    dy, dx = al["dy"], al["dx"]
    H, W = gt.shape
    raw = raw[dy:dy + H, dx:dx + W]
    assert raw.shape == gt.shape, f"{n}: registered crop {raw.shape} != label {gt.shape}"
    # two tiles per frame: the densest labelled window, and a second NON-OVERLAPPING one.
    # The suppression below was wrong until 2026-09-18: argmax returns CENTRE coordinates but
    # the zeroing used the tile ORIGIN (cy - S//2), shifting the exclusion window by 512 px.
    # AS_24hr t0/t1 came out 512 px apart -- 50.0% overlap -- and 260708 t0/t1 at 56.6%, while
    # the comment claimed "far from it". Two tiles sharing half their pixels are one
    # observation reported as two.
    #
    # Suppressing in centre space is still not enough: on a 1490x1490 frame two non-overlapping
    # 1024 tiles do not FIT, and the clip to [0, H-S] collapses distinct centres onto nearly the
    # same origin (that frame came out at 80% overlap). So the guarantee is enforced where it
    # matters -- an explicit rejection in ORIGIN space against every tile already taken from this
    # frame. A frame that cannot yield a second disjoint tile contributes one.
    dens = ndi.uniform_filter(gt.astype(np.float32), size=256)
    picked = []
    d = dens.copy()
    k = 0
    while k < 2 and d.max() > 0:
        cy, cx = np.unravel_index(np.argmax(d), d.shape)
        y = int(np.clip(cy - S // 2, 0, max(H - S, 0)))
        x = int(np.clip(cx - S // 2, 0, max(W - S, 0)))
        d[max(0, cy - S + 1):cy + S, max(0, cx - S + 1):cx + S] = 0   # never revisit this peak
        sub = gt[y:y + S, x:x + S]
        if sub.shape != (S, S) or sub.sum() < 400:
            break
        # reject any candidate that shares a single pixel with a tile already taken
        if any(min(y + S, py + S) > max(y, py) and min(x + S, px + S) > max(x, px)
               for py, px in picked):
            continue
        tid = f"{n}__t{k}"
        Image.fromarray((sub * 255).astype(np.uint8)).save(f"{OUT}/tiles/{tid}_gt.png")
        # THE MODEL INPUT: raw original grey, no annotation anywhere in it
        Image.fromarray(raw[y:y + S, x:x + S]).save(f"{OUT}/tiles/{tid}_gray.png")
        # the overlay, for human eyes only. Named so that no loader picks it up by accident;
        # feeding this file is precisely the bug that invalidated the first run.
        a = np.array(ov.crop((x, y, x + S, y + S)))
        ov.crop((x, y, x + S, y + S)).save(f"{OUT}/tiles/{tid}_overlay_REFERENCE_DO_NOT_FEED.png")
        red = (a[..., 0] > 150) & (a[..., 1] < 80) & (a[..., 2] < 80)
        meta.append({"tile": tid, "frame": n, "y": y, "x": x,
                     "gt_px": int(sub.sum()), "gt_frac": round(float(sub.mean()), 5),
                     "red_frac_in_tile": round(float(red.mean()), 5),
                     "grey_source": f"original/{n}.tif+({dy},{dx})", "align_ncc": al["ncc"]})
        picked.append((y, x))
        k += 1
json.dump(meta, open(f"{OUT}/tiles/meta.json", "w"), indent=1)
print(f"{len(meta)} tiles from {len(set(m['frame'] for m in meta))} frames")
for m in meta:
    print(f"  {m['tile']:<46} gt {m['gt_px']:>7,} px ({100*m['gt_frac']:.2f}% of tile)")
