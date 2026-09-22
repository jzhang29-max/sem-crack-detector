#!/usr/bin/env python3
"""Final per-set crack analysis: merged table, figures, markdown report.

Reads analysis/per_frame_metrics.csv + analysis/review_coverage.csv.

Conventions established by measurement, not assumption:
  * Orientation_deg = np.degrees(skimage regionprops.orientation), measured from
    the ROW axis. Verified with synthetic bars: a horizontal bar gives +90.00,
    a vertical bar gives 0.00. So |angle| near 90 means the crack runs ACROSS
    the frame (horizontal); near 0 means it runs DOWN it.
  * Orientation is axial (period 180 deg): -89 and +89 are nearly the same
    direction, so all orientation pooling is circular on the doubled angle.
  * Set central tendency is reported as MEDIAN as well as mean, because every
    set is right-skewed by a few very-high-area frames (MAR_Amb_AS: mean
    14.25%, median 2.90% -- the mean describes no frame in the set).
"""
import csv, os, math
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
AN = os.path.join(ROOT, "analysis")
FIG = os.path.join(AN, "figures")
os.makedirs(FIG, exist_ok=True)

FAMORDER = {"steel": 0, "superalloy": 1, "exposure": 2}

def fnum(v):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return float("nan")

frames = list(csv.DictReader(open(os.path.join(AN, "per_frame_metrics.csv"))))
rev = {r["SourceImage"]: r for r in csv.DictReader(open(os.path.join(AN, "review_coverage.csv")))}
for m in frames:
    r = rev.get(m["SourceImage"], {})
    m["ReviewedPct"] = fnum(r.get("ReviewedPct", 0))
    m["CrackMarked_px"] = fnum(r.get("CrackMarked_px", 0))
    m["NotCrackMarked_px"] = fnum(r.get("NotCrackMarked_px", 0))
    for k in ("CrackAreaPct", "NetworkAreaPct", "SpeckAreaPct", "CensoredAreaSharePct",
              "LargestAreaSharePct", "AreaGini", "SpeckDensity_perMpx", "NetAwMeanAspect",
              "NetAwMeanWidth_px", "NetAwMeanSolidity", "OrientMean_deg", "Anisotropy_R",
              "LengthDensity_px_perMpx", "NetMedianArea_px", "H", "W"):
        m[k] = fnum(m[k])
    # what fraction of the model's crack area did a human explicitly confirm?
    frame_px = m["H"] * m["W"]
    crack_px = m["CrackAreaPct"] / 100.0 * frame_px
    m["ConfirmedShareOfCrackPct"] = (100.0 * m["CrackMarked_px"] / crack_px) if crack_px > 0 else float("nan")

with open(os.path.join(AN, "per_frame_metrics_with_review.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(frames[0].keys())); w.writeheader(); w.writerows(frames)

def axial(theta, weights):
    if len(theta) == 0 or np.sum(weights) <= 0:
        return float("nan"), float("nan")
    phi = 2 * np.radians(np.asarray(theta, float)); w = np.asarray(weights, float)
    C = np.sum(w * np.cos(phi)) / w.sum(); S = np.sum(w * np.sin(phi)) / w.sum()
    R = math.hypot(C, S); mean = math.degrees(0.5 * math.atan2(S, C))
    if mean <= -90: mean += 180
    if mean > 90: mean -= 180
    return mean, R

bysets = defaultdict(list)
for m in frames:
    bysets[m["Set"]].append(m)
order = sorted(bysets, key=lambda k: (FAMORDER.get(bysets[k][0]["Family"], 9), k))

setrows = []
for k in order:
    g = bysets[k]
    def col(c): return np.array([m[c] for m in g], float)
    def stat(c):
        v = col(c); v = v[np.isfinite(v)]
        return (float(np.mean(v)) if v.size else float("nan"),
                float(np.median(v)) if v.size else float("nan"),
                float(np.std(v, ddof=1)) if v.size >= 3 else float("nan"))
    r = {"Set": k, "Family": g[0]["Family"], "Frames": len(g),
         "Detectors": ",".join(sorted({m["Detector"] for m in g})),
         "DispersionEstimable": len(g) >= 3}
    for c in ("CrackAreaPct", "NetworkAreaPct", "SpeckAreaPct", "CensoredAreaSharePct",
              "LargestAreaSharePct", "AreaGini", "SpeckDensity_perMpx", "NetAwMeanAspect",
              "NetAwMeanWidth_px", "NetAwMeanSolidity", "LengthDensity_px_perMpx",
              "NetMedianArea_px", "ReviewedPct", "ConfirmedShareOfCrackPct", "Anisotropy_R"):
        mu, md, sd = stat(c)
        r[f"{c}_mean"], r[f"{c}_median"], r[f"{c}_sd"] = mu, md, sd
    th = [m["OrientMean_deg"] for m in g if np.isfinite(m["OrientMean_deg"])]
    wt = [m["NetworkAreaPct"] for m in g if np.isfinite(m["OrientMean_deg"])]
    r["SetOrientMean_deg"], r["BetweenFrameConsistency_R"] = axial(th, wt)
    r["WithinFrameAnisotropy_R_mean"] = r["Anisotropy_R_mean"]
    r["FramesReviewed"] = sum(1 for m in g if m["ReviewedPct"] > 0)
    r["SpanningFrames"] = sum(1 for m in g if m["LargestSpansH"] == "True" or m["LargestSpansV"] == "True")
    setrows.append(r)

with open(os.path.join(AN, "per_set_metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(setrows[0].keys())); w.writeheader(); w.writerows(setrows)

# ------------------------------------------------------------------ console
print("=== per set: crack area %, mean vs median (skew matters) ===")
h = (f"{'set':<20}{'fam':<11}{'n':>3}{'rev':>5}{'mean':>8}{'median':>8}{'sd':>7}"
     f"{'cens%':>7}{'dom%':>7}{'|R|':>6}{'angle':>7}{'reads':>18}")
print(h); print("-" * len(h))
for r in setrows:
    ang = r["SetOrientMean_deg"]
    reads = "horizontal" if abs(ang) > 67.5 else ("vertical" if abs(ang) < 22.5 else "diagonal")
    sd = f"{r['CrackAreaPct_sd']:.2f}" if np.isfinite(r["CrackAreaPct_sd"]) else "  --"
    print(f"{r['Set']:<20}{r['Family']:<11}{r['Frames']:>3}{r['FramesReviewed']:>5}"
          f"{r['CrackAreaPct_mean']:>8.2f}{r['CrackAreaPct_median']:>8.2f}{sd:>7}"
          f"{r['CensoredAreaSharePct_mean']:>7.1f}{r['LargestAreaSharePct_mean']:>7.1f}"
          f"{r['WithinFrameAnisotropy_R_mean']:>6.2f}{ang:>7.1f}{reads:>18}")

print("\n=== the high-area frames: confirmed damage or unchecked over-detection? ===")
hi = sorted([m for m in frames if m["CrackAreaPct"] > 18], key=lambda m: -m["CrackAreaPct"])
print(f"{'frame':<38}{'area%':>7}{'reviewed%':>10}{'of crack confirmed%':>21}  verdict")
for m in hi:
    c = m["ConfirmedShareOfCrackPct"]
    verdict = ("human-confirmed" if c >= 80 else
               "partly confirmed" if c >= 20 else "UNVERIFIED")
    print(f"{m['SourceImage']:<38}{m['CrackAreaPct']:>7.2f}{m['ReviewedPct']:>10.2f}"
          f"{c:>21.1f}  {verdict}")

print("\n=== MAR_Amb: process x detector, medians (crossed design) ===")
import sys
sys.path.insert(0, f"{_REPO}/interior_active_learning/code")
from aggregate import parse_name
cell = defaultdict(list)
for m in frames:
    if m["Family"] == "superalloy":
        cell[(parse_name(m["SourceImage"])["process"], m["Detector"])].append(m["CrackAreaPct"])
print(f"{'process':<9}{'CBS median (n)':>20}{'ETD median (n)':>20}{'both':>10}")
for proc in ("AS", "Cast", "HIP"):
    line = f"{proc:<9}"
    allv = []
    for det in ("CBS", "ETD"):
        v = cell.get((proc, det), [])
        allv += v
        line += f"{np.median(v):>14.2f} (n={len(v)})" if v else f"{'--':>20}"
    line += f"{np.median(allv):>10.2f}"
    print(line)
for det in ("CBS", "ETD"):
    v = [x for (p, d), vv in cell.items() if d == det for x in vv]
    s = [m["SpeckDensity_perMpx"] for m in frames if m["Family"] == "superalloy" and m["Detector"] == det]
    print(f"  {det} marginal median area {np.median(v):.2f}% (n={len(v)}), specks {np.mean(s):.1f}/Mpx")

# ------------------------------------------------------------------ figures
CB = {"steel": "#4C72B0", "superalloy": "#DD8452", "exposure": "#55A868"}

# Fig 1: per-set strip plot, frames as points, marker encodes review
fig, ax = plt.subplots(figsize=(11, 5.6))
for i, r in enumerate(setrows):
    g = bysets[r["Set"]]
    xs = np.full(len(g), i, float) + np.linspace(-.18, .18, len(g))
    for x, m in zip(xs, g):
        conf = m["ConfirmedShareOfCrackPct"]
        filled = np.isfinite(conf) and conf >= 20
        ax.plot(x, m["CrackAreaPct"], "o", ms=7,
                mfc=CB[r["Family"]] if filled else "white",
                mec=CB[r["Family"]], mew=1.6, zorder=3)
    ax.hlines(r["CrackAreaPct_median"], i - .30, i + .30, color="k", lw=2.4, zorder=4)
ax.set_xticks(range(len(setrows)))
ax.set_xticklabels([r["Set"] for r in setrows], rotation=32, ha="right", fontsize=9)
ax.set_ylabel("crack area fraction of frame (%)")
ax.set_title("Crack area fraction per set — one point per frame, black bar = set median\n"
             "filled = human-confirmed crack, hollow = detector output nobody has checked",
             fontsize=11)
ax.grid(axis="y", alpha=.3)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=c, mec=c, label=f)
                   for f, c in CB.items()], frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_area_per_set.png"), dpi=150); plt.close(fig)

# Fig 2: MAR process x detector
fig, ax = plt.subplots(figsize=(7.4, 5))
W = 0.34
for j, det in enumerate(("CBS", "ETD")):
    for i, proc in enumerate(("AS", "Cast", "HIP")):
        v = np.array(cell.get((proc, det), []))
        if not v.size: continue
        x = i + (j - .5) * W
        ax.bar(x, np.median(v), W * .9, color="#4C72B0" if det == "CBS" else "#DD8452",
               alpha=.55, edgecolor="k", lw=.6,
               label=det if i == 0 else None, zorder=2)
        ax.plot(np.full(v.size, x) + np.linspace(-.05, .05, v.size), v, "ko", ms=5, zorder=3)
ax.set_xticks(range(3)); ax.set_xticklabels(("AS", "Cast", "HIP"))
ax.set_xlabel("processing route"); ax.set_ylabel("crack area fraction (%)")
# THE TITLE MUST MATCH THE BARS. It said "HIP is lowest under BOTH detectors"; the medians
# this figure plots are CBS AS 2.523 / Cast 25.508 / HIP 4.399 and ETD AS 3.457 / Cast 19.457
# / HIP 3.469, so AS is lowest under both and HIP under neither. A baked title is not checked
# by anything, so it outlived the data it describes.
ax.set_title("MAR_Amb: process is crossed with detector, so the two are separable\n"
             "bar = median, dots = frames. Cast is highest under both detectors;\n"
             "AS and HIP are close and their order is not stable across detectors.\n"
             "Routes were shot at different magnifications - see CORRECTION_scale_and_magnification.md",
             fontsize=10)
ax.legend(frameon=False); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_mar_process_detector.png"), dpi=150); plt.close(fig)

# Fig 3: region size distribution, area share by size class
edges = [1, 10, 50, 500, 5e3, 5e4, 5e5, 1e8]
lbl = ["1-10", "10-50", "50-500", "0.5k-5k", "5k-50k", "50k-500k", ">500k"]
import glob
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for fam in ("steel", "superalloy", "exposure"):
    areas = []
    for m in frames:
        if m["Family"] != fam: continue
        p = os.path.join(ROOT, "regions", m["SourceImage"] + "_regions.csv")
        areas += [float(x["Area_px"]) for x in csv.DictReader(open(p))]
    a = np.array(areas)
    cnt = np.array([((a >= lo) & (a < hi)).sum() for lo, hi in zip(edges[:-1], edges[1:])], float)
    sh = np.array([a[(a >= lo) & (a < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])], float)
    axes[0].plot(range(len(lbl)), 100 * cnt / cnt.sum(), "o-", color=CB[fam], label=fam)
    axes[1].plot(range(len(lbl)), 100 * sh / sh.sum(), "o-", color=CB[fam], label=fam)
for ax, t, yl in ((axes[0], "where the REGIONS are", "% of regions"),
                  (axes[1], "where the AREA is", "% of crack area")):
    ax.set_xticks(range(len(lbl))); ax.set_xticklabels(lbl, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("region area (px$^2$)"); ax.set_ylabel(yl); ax.set_title(t, fontsize=11)
    ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=9)
fig.suptitle("Two populations: thousands of specks carry almost no area; a handful of networks carry nearly all of it",
             fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_size_populations.png"), dpi=150); plt.close(fig)

# Fig 4: orientation rose, axial, area-weighted, per family
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), subplot_kw={"projection": "polar"})
NB = 18
for ax, fam in zip(axes, ("steel", "superalloy", "exposure")):
    th, wt = [], []
    for m in frames:
        if m["Family"] != fam: continue
        p = os.path.join(ROOT, "regions", m["SourceImage"] + "_regions.csv")
        for x in csv.DictReader(open(p)):
            A = float(x["Area_px"])
            if A < 500 or float(x["Eccentricity"]) < .90: continue
            if (int(x["BBoxY0"]) <= 0 or int(x["BBoxX0"]) <= 0
                    or int(x["BBoxY1"]) >= m["H"] or int(x["BBoxX1"]) >= m["W"]): continue
            th.append(float(x["Orientation_deg"])); wt.append(A)
    th = np.array(th); wt = np.array(wt)
    # axial: fold to [0,180) and mirror so the rose is symmetric
    t2 = np.radians(np.mod(th, 180.0))
    bins = np.linspace(0, np.pi, NB + 1)
    hist, _ = np.histogram(t2, bins=bins, weights=wt)
    hist = hist / hist.sum() if hist.sum() else hist
    ctr = (bins[:-1] + bins[1:]) / 2
    ax.bar(ctr, hist, width=np.pi / NB, color=CB[fam], alpha=.8, edgecolor="k", lw=.4)
    ax.bar(ctr + np.pi, hist, width=np.pi / NB, color=CB[fam], alpha=.8, edgecolor="k", lw=.4)
    _, R = axial(th, wt)
    ax.set_title(f"{fam}\nn={len(th)} networks, anisotropy |R|={R:.2f}", fontsize=10)
    ax.set_theta_zero_location("N"); ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0°\nvertical", "", "90°\nhorizontal", "", "", "", "", ""], fontsize=8)
fig.suptitle("Crack orientation (area-weighted, uncensored networks, axial so opposite lobes are identical)\n"
             "0° = crack runs down the frame, 90° = across it", fontsize=11, y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.subplots_adjust(top=0.72)
fig.savefig(os.path.join(FIG, "fig4_orientation_rose.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# Fig 5: the measurement limit -- censoring vs area
fig, ax = plt.subplots(figsize=(7.6, 5))
for fam in CB:
    g = [m for m in frames if m["Family"] == fam]
    ax.plot([m["CrackAreaPct"] for m in g], [m["CensoredAreaSharePct"] for m in g],
            "o", color=CB[fam], label=fam, ms=7, alpha=.85)
ax.set_xlabel("crack area fraction (%)"); ax.set_ylabel("% of crack area in edge-touching regions")
ax.set_title("Why no absolute crack LENGTH is quoted here\n"
             "the more cracked the frame, the more of its damage leaves the field of view", fontsize=11)
ax.grid(alpha=.3); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_censoring.png"), dpi=150); plt.close(fig)

print(f"\nwrote 5 figures to {FIG}")
