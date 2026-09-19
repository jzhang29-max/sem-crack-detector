#!/usr/bin/env python3
"""How much of each frame a human actually reviewed. READ-ONLY.

Correction-mask codes (common.load_correction_mask): 0 = unreviewed,
1 = forced CRACK, 2 = forced NOT-CRACK, 3 = erased from candidacy.

Why this belongs in the crack analysis: the exported masks are detector output
with corrections merged in. A frame at 51% crack area is either a genuinely
shattered specimen or an over-detection nobody has checked yet, and the area
number alone cannot tell those apart. Reviewed fraction can.
"""
import csv, os, glob
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

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
PAINT = f"{_REPO}/interior_active_learning/paint"
ROOT = _CE
OUT = os.path.join(ROOT, "analysis", "review_coverage.csv")

names = sorted(n[:-len("_mask.png")] for n in os.listdir(os.path.join(ROOT, "masks"))
               if n.endswith("_mask.png"))
rows = []
for n in names:
    p = os.path.join(PAINT, n + "_correction_mask.png")
    rec = {"SourceImage": n, "HasCorrections": os.path.exists(p),
           "CrackMarked_px": 0, "NotCrackMarked_px": 0, "Erased_px": 0,
           "ReviewedPct": 0.0, "MaskH": "", "MaskW": ""}
    if os.path.exists(p):
        m = np.array(Image.open(p))
        while m.ndim > 2:
            m = m[..., 0]
        rec["MaskH"], rec["MaskW"] = m.shape[0], m.shape[1]
        c1 = int((m == 1).sum()); c2 = int((m == 2).sum()); c3 = int((m == 3).sum())
        rec["CrackMarked_px"], rec["NotCrackMarked_px"], rec["Erased_px"] = c1, c2, c3
        rec["ReviewedPct"] = 100.0 * (c1 + c2 + c3) / m.size
    rows.append(rec)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

tot = sum(r["CrackMarked_px"] + r["NotCrackMarked_px"] + r["Erased_px"] for r in rows)
have = sum(1 for r in rows if r["HasCorrections"])
print(f"{have}/{len(rows)} frames have a correction mask; {tot:,} hand-marked pixels total")
print(f"frames with ZERO human review: {sum(1 for r in rows if r['ReviewedPct'] == 0)}")
print(f"\n{'frame':<40}{'reviewed%':>10}{'crack px':>12}{'not-crack px':>14}")
for r in sorted(rows, key=lambda r: -r["ReviewedPct"])[:8]:
    print(f"{r['SourceImage']:<40}{r['ReviewedPct']:>10.2f}{r['CrackMarked_px']:>12,}{r['NotCrackMarked_px']:>14,}")
print("  ... lowest:")
for r in sorted(rows, key=lambda r: r["ReviewedPct"])[:6]:
    print(f"{r['SourceImage']:<40}{r['ReviewedPct']:>10.2f}{r['CrackMarked_px']:>12,}{r['NotCrackMarked_px']:>14,}")
print(f"\nwrote {OUT}")
