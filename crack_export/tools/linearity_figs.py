#!/usr/bin/env python3
"""Crack linearity / path-morphology figures.

Every metric here is resolution-invariant or explicitly controlled, because 26 of
62 frames are 2x-downsampled versions of the same optics. Verified on a known
same-field pair (260708_..._CBS_004 at 2045x3072 vs CBS_005 at 4091x6139):
  raw skeleton px per Mpx  ratio 0.59  <- NOT comparable
  raw mean crack width     ratio 2.02  <- NOT comparable
  LineDensity = skel_px/W  ratio 1.19  <- invariant
  RelWidth = width/W       ratio 1.01  <- invariant
  CrackAreaPct             ratio 0.96  <- invariant
  FractalDim               ratio 1.02  <- invariant
"""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
sys.path.insert(0, "/Users/jiamingzhang/Desktop/sem-crack-detector/interior_active_learning/code")
from aggregate import parse_name, specimen_key

os.chdir("/Users/jiamingzhang/Desktop/crack_export")
FIG = "analysis/figures"
os.makedirs(FIG, exist_ok=True)
CB = {"steel": "#4C72B0", "superalloy": "#DD8452", "exposure": "#55A868"}

def fnum(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d

# ---------------------------------------------------------------- frame level
FR = list(csv.DictReader(open("analysis/skeleton_frames.csv")))
for r in FR:
    for c in ("H", "W", "SkeletonPx", "CrackAreaPct", "MeanWidth_px", "FractalDim",
              "SkelPerMpx", "SkeletonNodes", "Branches"):
        r[c] = fnum(r[c])
    r["LineDensity"] = r["SkeletonPx"] / r["W"]
    r["RelWidth"] = r["MeanWidth_px"] / r["W"]

def sp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

# FIG 7: the decomposition -- more cracking, or just wider detections?
fig, ax = plt.subplots(figsize=(8.6, 6.2))
for r in FR:
    c = CB[r["Family"]]
    ax.scatter(r["LineDensity"], r["RelWidth"], s=26 + 5.5 * r["CrackAreaPct"],
               facecolor=c, edgecolor="k", lw=.5, alpha=.72, zorder=3)
lab = {}
for r in FR:
    lab.setdefault(r["Set"], []).append(r)
for k, g in lab.items():
    x = np.median([r["LineDensity"] for r in g]); y = np.median([r["RelWidth"] for r in g])
    ax.plot(x, y, "k+", ms=14, mew=2.2, zorder=5)
    ax.annotate(k, (x, y), fontsize=8, xytext=(5, 5), textcoords="offset points", zorder=6)
ax.set_yscale("log")
ax.set_xlabel("crack line density  (skeleton length / frame width)  — resolution-invariant")
ax.set_ylabel("relative feature width  (mean width / frame width, log)")
ax.set_title("Is a set more cracked, or just more thickly detected?\n"
             "bubble area ∝ crack area fraction; black + = set median", fontsize=11)
ax.grid(alpha=.3, which="both")
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=c, mec="k", label=f)
                   for f, c in CB.items()], frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig7_density_vs_width.png", dpi=150); plt.close(fig)

# FIG 8: what area fraction actually measures
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, xc, xl in ((axes[0], "RelWidth", "relative feature width (log)"),
                   (axes[1], "LineDensity", "crack line density"),
                   (axes[2], "FractalDim", "box-counting fractal dimension")):
    for r in FR:
        ax.plot(r[xc], r["CrackAreaPct"], "o", color=CB[r["Family"]], ms=7, alpha=.8)
    rho = sp([r[xc] for r in FR], [r["CrackAreaPct"] for r in FR])
    if xc == "RelWidth":
        ax.set_xscale("log")
    ax.set_xlabel(xl); ax.set_ylabel("crack area fraction (%)")
    ax.set_title(f"Spearman ρ = {rho:+.3f}", fontsize=11); ax.grid(alpha=.3)
fig.suptitle("Crack area fraction tracks feature WIDTH more than crack line density — "
             "so it is partly a segmentation-thickness metric", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig8_what_area_measures.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- branch level
B = [b for b in csv.DictReader(open("analysis/branches.csv")) if b["Censored"] == "False"]
for b in B:
    for c in ("Tortuosity", "PathLen_px", "Width_mean_px", "TurnMedian_deg", "Straightness"):
        b[c] = fnum(b[c])

# FIG 9: the two confounds, then the controlled comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
bands = [(12, 25), (25, 50), (50, 100), (100, 250), (250, 1000)]
xs, ys = [], []
for lo, hi in bands:
    g = [b["Tortuosity"] for b in B if lo <= b["PathLen_px"] < hi and np.isfinite(b["Tortuosity"])]
    xs.append(f"{lo}-{hi}"); ys.append(np.median(g))
axes[0].plot(range(len(xs)), ys, "o-", color="#C44E52", lw=2, ms=8)
axes[0].set_xticks(range(len(xs))); axes[0].set_xticklabels(xs, fontsize=8)
axes[0].set_xlabel("branch path length (px)"); axes[0].set_ylabel("median tortuosity")
axes[0].set_title("Confound 1: length\npeaks at 50-100 px, then falls sharply", fontsize=10)
axes[0].grid(alpha=.3)

wb = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]
g0 = [b for b in B if 100 <= b["PathLen_px"] < 250]
xs, ys = [], []
for lo, hi in wb:
    g = [b["Tortuosity"] for b in g0 if lo <= b["Width_mean_px"] < hi and np.isfinite(b["Tortuosity"])]
    if len(g) < 15:
        continue
    xs.append(f"{lo}-{hi:.0f}" if hi < 1e8 else ">128"); ys.append(np.median(g))
axes[1].plot(range(len(xs)), ys, "o-", color="#8172B2", lw=2, ms=8)
axes[1].set_xticks(range(len(xs))); axes[1].set_xticklabels(xs, fontsize=8)
axes[1].set_xlabel("branch mean width (px), length held at 100-250 px")
axes[1].set_ylabel("median tortuosity")
axes[1].set_title(f"Confound 2: width (ρ={sp([b['Tortuosity'] for b in g0],[b['Width_mean_px'] for b in g0]):+.2f})\n"
                  "flat to ~64 px, then collapses", fontsize=10)
axes[1].grid(alpha=.3)

thin = defaultdict(list)
for b in B:
    if 100 <= b["PathLen_px"] < 250 and b["Width_mean_px"] <= 16:
        thin[b["Set"]].append(b["Tortuosity"])
keys = [k for k in sorted(thin) if len(thin[k]) >= 10]
pos = range(len(keys))
fam = {r["Set"]: r["Family"] for r in FR}
for i, k in enumerate(keys):
    v = np.array([x for x in thin[k] if np.isfinite(x)])
    axes[2].plot(np.full(v.size, i) + np.random.default_rng(i).normal(0, .06, v.size),
                 v, "o", color=CB[fam[k]], ms=3, alpha=.35)
    axes[2].hlines(np.median(v), i - .3, i + .3, color="k", lw=2.5, zorder=5)
axes[2].set_xticks(list(pos)); axes[2].set_xticklabels(keys, rotation=35, ha="right", fontsize=7)
axes[2].set_ylabel("tortuosity"); axes[2].set_ylim(0.98, 1.45)
axes[2].set_title("Controlled: thin (≤16 px) branches, 100-250 px long\n"
                  "black bar = median. Differences vanish.", fontsize=10)
axes[2].grid(axis="y", alpha=.3)
fig.suptitle("Crack path tortuosity: two confounds, and what survives controlling for both",
             fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig9_tortuosity_controlled.png", dpi=150); plt.close(fig)

# FIG 10: turn angles -- is there a faceting signature?
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
ta = np.array([b["TurnMedian_deg"] for b in B if b["PathLen_px"] >= 100
               and np.isfinite(b["TurnMedian_deg"])])
axes[0].hist(ta, bins=np.arange(0, 121, 5), color="#4C72B0", edgecolor="k", lw=.5)
axes[0].axvspan(55, 70, color="red", alpha=.13)
axes[0].annotate("where a grain-boundary\nfaceting signature would sit\n(~60° deviation at a\n120° triple junction)",
                 (62, axes[0].get_ylim()[1] * .62), fontsize=8, ha="center", color="#8B0000")
axes[0].set_xlabel("median turn angle per branch (deg; 0 = straight continuation)")
axes[0].set_ylabel("branches")
axes[0].set_title(f"Turn angles, branches ≥100 px (n={len(ta)})", fontsize=10)
for k in sorted(thin):
    pass
for k, c in (("thin ≤16 px", "#4C72B0"), ("wide >128 px", "#C44E52")):
    if k.startswith("thin"):
        v = [b["TurnMedian_deg"] for b in B if b["PathLen_px"] >= 100 and b["Width_mean_px"] <= 16]
    else:
        v = [b["TurnMedian_deg"] for b in B if b["PathLen_px"] >= 100 and b["Width_mean_px"] > 128]
    v = np.array([x for x in v if np.isfinite(x)])
    if v.size:
        axes[1].hist(v, bins=np.arange(0, 121, 5), histtype="step", lw=2, color=c,
                     density=True, label=f"{k} (n={v.size})")
axes[1].set_xlabel("median turn angle per branch (deg)"); axes[1].set_ylabel("density")
axes[1].set_title("Thin vs wide features", fontsize=10); axes[1].legend(frameon=False, fontsize=9)
for a in axes:
    a.grid(alpha=.3)
fig.suptitle("No high-angle faceting signature: turns peak at 20-30° and essentially vanish above 60°",
             fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig10_turn_angles.png", dpi=150); plt.close(fig)

# FIG 11: the intergranular test -- enclosed islands vs visible grain size
HO = list(csv.DictReader(open("analysis/holes.csv")))
fig, ax = plt.subplots(figsize=(8.4, 5.4))
by = defaultdict(list)
for r in HO:
    k = specimen_key(r["SourceImage"])
    d = fnum(r.get("EqDiam_median_px"))
    if np.isfinite(d):
        by[k].append(d)
keys = sorted(by)
for i, k in enumerate(keys):
    v = np.array(by[k])
    ax.plot(np.full(v.size, i), v, "o", color=CB[fam.get(k, "steel")], ms=7, alpha=.8)
    ax.hlines(np.median(v), i - .3, i + .3, color="k", lw=2.2)
ax.axhspan(300, 500, color="green", alpha=.12)
ax.annotate("grain size visible in the 316 steel CBS frames\n(~300-500 px at full resolution)",
            (len(keys) * .5, 390), fontsize=9, ha="center", color="darkgreen")
ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("median equivalent diameter of enclosed island (px)")
ax.set_yscale("log")
ax.set_title("Intergranular test: if cracks ran along grain boundaries, the islands they\n"
             "enclose would BE grains. They are 10-40 px — an order of magnitude too small.",
             fontsize=11)
ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/fig11_intergranular_test.png", dpi=150); plt.close(fig)

print("wrote fig7..fig11")
