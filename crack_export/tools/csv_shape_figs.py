#!/usr/bin/env python3
"""CSV-only shape metrics per set, with per-frame dimensions read from summary.csv.

The dimensions matter: 26 of 62 frames are not 4096x6144 (they run down to
1490x1490), so hardcoding the frame size makes the edge-censoring test silently
pass every region in those frames -- no region can ever reach y1 >= 4096 in a
2045-row image. All bounds here come from summary.csv per frame.

These metrics describe each region's best-fit ELLIPSE and CONVEX HULL. They do not
describe the crack path; that is what the skeleton analysis is for.
"""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
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
sys.path.insert(0, f"{_REPO}/interior_active_learning/code")
from aggregate import parse_name, specimen_key

os.chdir(_CE)
CB = {"steel": "#4C72B0", "superalloy": "#DD8452", "exposure": "#55A868"}
DIM = {r["SourceImage"]: (int(r["ImageH_px"]), int(r["ImageW_px"]))
       for r in csv.DictReader(open("summary.csv"))}

rows = defaultdict(list)
fam = {}
nc = 0
for p in glob.glob("regions/*_regions.csv"):
    n = os.path.basename(p)[:-len("_regions.csv")]
    H, W = DIM[n]
    k = specimen_key(n)
    fam[k] = parse_name(n).get("family")
    for r in csv.DictReader(open(p)):
        A = float(r["Area_px"])
        if A < 500:
            continue
        if (int(r["BBoxY0"]) <= 0 or int(r["BBoxX0"]) <= 0
                or int(r["BBoxY1"]) >= H or int(r["BBoxX1"]) >= W):
            nc += 1
            continue
        L = float(r["Length_px"]); Wd = float(r["Width_px"]); P = float(r["Perimeter_px"])
        if L <= 0 or Wd <= 0 or P <= 0:
            continue
        rows[k].append({
            "Aspect": float(r["AspectRatio"]),
            "Solidity": float(r["Solidity"]),
            "Ecc": float(r["Eccentricity"]),
            "Ribbon": (P / 2.0) / L,
            "Circ": 4 * np.pi * A / (P * P),
        })
print(f"excluded {nc:,} edge-touching regions using per-frame bounds")

keys = sorted(rows, key=lambda k: ({"steel": 0, "superalloy": 1, "exposure": 2}[fam[k]], k))
METRICS = [("Aspect", "aspect ratio (ellipse major/minor)", True),
           ("Solidity", "solidity (area / convex area)\n1 = convex, low = ragged or branched", False),
           ("Ribbon", "(perimeter/2) / ellipse major axis\n1 = straight ribbon, >1 wavy or branched", False),
           ("Circ", "circularity 4πA/P²", False)]
fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
for ax, (m, lab, logy) in zip(axes.ravel(), METRICS):
    for i, k in enumerate(keys):
        v = np.array([x[m] for x in rows[k]])
        if not v.size:
            continue
        q = np.percentile(v, [25, 50, 75])
        ax.plot([i, i], [q[0], q[2]], color=CB[fam[k]], lw=6, alpha=.45, solid_capstyle="butt")
        ax.plot(i, q[1], "o", color=CB[fam[k]], ms=9, mec="k", mew=.7, zorder=4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}  (n={len(rows[k])})" for k in keys], rotation=38, ha="right", fontsize=7)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(lab, fontsize=9)
    ax.grid(axis="y", alpha=.3)
fig.suptitle("Region shape from the CSVs alone — networks ≥500 px², edge-touching regions excluded\n"
             "dot = median, bar = interquartile range. These describe the fitted ellipse and convex "
             "hull, NOT the crack path.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("analysis/figures/fig12_csv_shape.png", dpi=150)
print("wrote fig12")

print(f"\n{'set':<20}{'n':>6}{'aspect':>8}{'solidity':>10}{'ribbon':>8}{'circ':>8}")
for k in keys:
    g = rows[k]
    f = lambda c: np.median([x[c] for x in g])
    print(f"{k:<20}{len(g):>6}{f('Aspect'):>8.2f}{f('Solidity'):>10.3f}"
          f"{f('Ribbon'):>8.2f}{f('Circ'):>8.4f}")
