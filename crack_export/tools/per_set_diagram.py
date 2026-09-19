#!/usr/bin/env python3
"""One self-contained diagram per set. No cross-set comparison anywhere.

Each figure describes a single specimen set using only that set's own frames, and
is written both to sets/<set>/<set>_diagram.png (next to the data it describes)
and to analysis/figures/per_set/.

Conventions carried over, and the reason for each:
  * Orientation is AXIAL (period 180 deg) and measured from the ROW axis, verified
    with synthetic bars: horizontal bar -> +90.0, vertical bar -> 0.0. Pooling is
    circular on the doubled angle, area-weighted, uncensored networks only.
  * Edge-touching regions are right-censored (their extent continues outside the
    frame) and are excluded from size/shape panels, using PER-FRAME bounds from
    summary.csv -- 26 of 62 frames are not 4096x6144.
  * Tortuosity is shown only for THIN (<=16 px) branches of moderate length
    (100-250 px). Unfiltered, it measures feature width (rho=-0.56) and branch
    length rather than crack path straightness.
  * Panels that would be built on too little data say so instead of drawing.
"""
import csv, glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUTD = "analysis/figures/per_set"
os.makedirs(OUTD, exist_ok=True)
FAMCOL = {"steel": "#4C72B0", "superalloy": "#DD8452", "exposure": "#55A868"}
MINBR = 10          # below this many branches a distribution panel is not drawn

def fnum(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d

DIM = {r["SourceImage"]: (int(r["ImageH_px"]), int(r["ImageW_px"]))
       for r in csv.DictReader(open("summary.csv"))}
PFM = {r["SourceImage"]: r for r in csv.DictReader(open("analysis/per_frame_metrics_with_review.csv"))}
SKF = {r["SourceImage"]: r for r in csv.DictReader(open("analysis/skeleton_frames.csv"))}
HOL = {r["SourceImage"]: r for r in csv.DictReader(open("analysis/holes.csv"))}
BR = defaultdict(list)
for b in csv.DictReader(open("analysis/branches.csv")):
    if b["Censored"] == "False":
        BR[b["Set"]].append(b)

sets = defaultdict(list)
for n in DIM:
    sets[specimen_key(n)].append(n)


def axial(theta, w):
    if len(theta) == 0 or np.sum(w) <= 0:
        return float("nan"), float("nan")
    phi = 2 * np.radians(np.asarray(theta, float)); w = np.asarray(w, float)
    C = (w * np.cos(phi)).sum() / w.sum(); S = (w * np.sin(phi)).sum() / w.sum()
    R = math.hypot(C, S); m = math.degrees(0.5 * math.atan2(S, C))
    if m <= -90: m += 180
    if m > 90: m -= 180
    return m, R


def note(ax, msg):
    ax.text(.5, .5, msg, ha="center", va="center", fontsize=9, color="#666",
            transform=ax.transAxes, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def build(key, frames):
    frames = sorted(frames)
    tok = parse_name(frames[0])
    fam = tok.get("family", "?")
    col = FAMCOL.get(fam, "#777")
    dets = sorted({parse_name(n).get("detector", "?") for n in frames})

    # ---- gather this set's regions (uncensored networks, per-frame bounds)
    reg, reg_all = [], []
    for n in frames:
        H, W = DIM[n]
        for r in csv.DictReader(open(f"regions/{n}_regions.csv")):
            A = fnum(r["Area_px"])
            reg_all.append(A)
            if A < 500:
                continue
            if (int(r["BBoxY0"]) <= 0 or int(r["BBoxX0"]) <= 0
                    or int(r["BBoxY1"]) >= H or int(r["BBoxX1"]) >= W):
                continue
            reg.append({"A": A, "ecc": fnum(r["Eccentricity"]), "sol": fnum(r["Solidity"]),
                        "ar": fnum(r["AspectRatio"]), "ang": fnum(r["Orientation_deg"]),
                        "P": fnum(r["Perimeter_px"]), "L": fnum(r["Length_px"])})
    reg_all = np.array(reg_all, float)

    br = BR.get(key, [])
    for b in br:
        for c in ("Tortuosity", "PathLen_px", "Width_mean_px", "TurnMedian_deg"):
            b[c] = fnum(b[c])
    thin = [b for b in br if 100 <= b["PathLen_px"] < 250 and b["Width_mean_px"] <= 16]

    fig = plt.figure(figsize=(15.5, 12.6))
    gs = fig.add_gridspec(3, 3, hspace=.46, wspace=.28,
                          left=.06, right=.97, top=.875, bottom=.055)

    # (0,0) per-frame crack area fraction
    ax = fig.add_subplot(gs[0, 0])
    a = [fnum(PFM[n]["CrackAreaPct"]) for n in frames]
    conf = [fnum(PFM[n]["ConfirmedShareOfCrackPct"]) for n in frames]
    xs = np.arange(len(frames))
    bw = .72 if len(frames) >= 4 else .34
    for i, (v, c) in enumerate(zip(a, conf)):
        filled = np.isfinite(c) and c >= 20
        ax.bar(i, v, bw, color=col if filled else "white",
               edgecolor=col, lw=1.6, hatch="" if filled else "///")
    if len(frames) >= 3:
        ax.axhline(np.median(a), color="k", lw=1.6, ls="--", zorder=4)
        ax.annotate(f"median {np.median(a):.2f}%", (len(frames) - .5, np.median(a)),
                    fontsize=8, va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.85))
    ax.set_xlim(-.7, len(frames) - .3)
    short = [n.replace(key + "_", "").replace("_", " ")[-9:] for n in frames]
    ax.set_xticks(xs); ax.set_xticklabels(short, rotation=90, fontsize=6.5)
    ax.set_ylabel("crack area fraction (%)")
    ax.set_title("Crack area per frame\nsolid = overlaps a human crack mark;"
                 " hatched = unreviewed\n(marks are region assertions, median brush 59 px)",
                 fontsize=8.5)
    ax.grid(axis="y", alpha=.3)

    # (0,1) region size distribution
    ax = fig.add_subplot(gs[0, 1])
    if reg_all.size >= 20:
        v = reg_all[reg_all > 0]
        s = np.sort(v)
        ax.plot(s, 100 * np.arange(1, s.size + 1) / s.size, color=col, lw=2,
                label="% of regions")
        cum = 100 * np.cumsum(s) / s.sum()
        ax.plot(s, cum, color="#C44E52", lw=2, ls="--", label="% of crack area")
        ax.set_xscale("log")
        ax.axvline(500, color="k", lw=1, ls=":")
        ax.annotate("500 px²\nspeck / network split", (500, 88), fontsize=7,
                    ha="right", color="#444", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.85))
        ax.set_xlabel("region area (px², log)"); ax.set_ylabel("cumulative %")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_title(f"Where the regions are vs where the area is\n"
                     f"{v.size:,} regions in this set", fontsize=9.5)
        ax.grid(alpha=.3, which="both")
    else:
        note(ax, f"only {reg_all.size} regions in this set")

    # (0,2) orientation rose
    ax = fig.add_subplot(gs[0, 2], projection="polar")
    o = [r for r in reg if r["A"] >= 500 and r["ecc"] >= .90]
    if len(o) >= 15:
        t2 = np.radians(np.mod([r["ang"] for r in o], 180.0))
        wt = np.array([r["A"] for r in o])
        NB = 18
        bins = np.linspace(0, np.pi, NB + 1)
        h, _ = np.histogram(t2, bins=bins, weights=wt)
        h = h / h.sum() if h.sum() else h
        ctr = (bins[:-1] + bins[1:]) / 2
        for off in (0, np.pi):
            ax.bar(ctr + off, h, width=np.pi / NB, color=col, alpha=.82,
                   edgecolor="k", lw=.4)
        m, R = axial([r["ang"] for r in o], wt)
        rd = ("horizontal" if abs(m) > 67.5 else "vertical" if abs(m) < 22.5 else "diagonal")
        ax.set_title(f"Crack orientation (area-weighted, axial)\n"
                     f"mean {m:+.1f}° = {rd}, anisotropy |R|={R:.2f}, n={len(o)}", fontsize=9.5)
    else:
        note(ax, f"only {len(o)} elongated uncensored networks")
    ax.set_theta_zero_location("N"); ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0°\ndown frame", "", "90°\nacross", "", "", "", "", ""], fontsize=7)

    # (1,0) tortuosity, controlled
    ax = fig.add_subplot(gs[1, 0])
    tv = np.array([b["Tortuosity"] for b in thin if np.isfinite(b["Tortuosity"])])
    if tv.size >= MINBR:
        ax.hist(tv, bins=np.linspace(1.0, 1.5, 26), color=col, edgecolor="k", lw=.5)
        ax.axvline(np.median(tv), color="k", lw=2)
        ax.annotate(f"median {np.median(tv):.3f}", (np.median(tv), ax.get_ylim()[1] * .97),
                    fontsize=8, ha="left", va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.85))
        ax.set_xlabel("tortuosity  (path length / straight-line distance)")
        ax.set_ylabel("branches")
        ax.set_title(f"Crack path tortuosity\nthin (≤16 px), 100-250 px long — n={tv.size}",
                     fontsize=9.5)
        ax.grid(axis="y", alpha=.3)
    else:
        note(ax, f"only {tv.size} thin branches of controlled length\n"
                 f"(need ≥{MINBR}); this set's cracks are mostly\ntoo wide or too short "
                 f"for a path measurement")

    # (1,1) turn angles
    ax = fig.add_subplot(gs[1, 1])
    ta = np.array([b["TurnMedian_deg"] for b in br
                   if b["PathLen_px"] >= 100 and np.isfinite(b["TurnMedian_deg"])])
    if ta.size >= MINBR:
        ax.hist(ta, bins=np.arange(0, 121, 5), color=col, edgecolor="k", lw=.5)
        ax.axvspan(55, 70, color="red", alpha=.12)
        ax.annotate("a grain-boundary\nfaceting signature\nwould sit here",
                    (72, ax.get_ylim()[1] * .78), fontsize=7, ha="left", color="#8B0000",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.85))
        ax.set_xlabel("median turn angle per branch (deg; 0 = straight)")
        ax.set_ylabel("branches")
        ax.set_title(f"Direction changes along the crack\nbranches ≥100 px — n={ta.size}, "
                     f"median {np.median(ta):.1f}°", fontsize=9.5)
        ax.grid(axis="y", alpha=.3)
    else:
        note(ax, f"only {ta.size} branches ≥100 px")

    # (1,2) branch width
    ax = fig.add_subplot(gs[1, 2])
    wv = np.array([b["Width_mean_px"] for b in br if np.isfinite(b["Width_mean_px"])
                   and b["Width_mean_px"] > 0])
    if wv.size >= MINBR:
        ax.hist(wv, bins=np.logspace(np.log10(max(1, wv.min())), np.log10(wv.max() * 1.05), 30),
                color=col, edgecolor="k", lw=.5)
        ax.set_xscale("log")
        ax.axvline(np.median(wv), color="k", lw=2)
        ax.axvline(16, color="#C44E52", lw=1.4, ls="--")
        ax.annotate("16 px — above this the skeleton\nsmooths and tortuosity is unreliable",
                    (16, ax.get_ylim()[1] * .97), fontsize=7, ha="left", va="top",
                    color="#C44E52", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.85))
        ax.set_xlabel("branch mean width (px, log)"); ax.set_ylabel("branches")
        ax.set_title(f"Detected feature width\nn={wv.size}, median {np.median(wv):.1f} px",
                     fontsize=9.5)
        ax.grid(alpha=.3, which="both")
    else:
        note(ax, f"only {wv.size} branches")

    # (2,0) shape: aspect vs solidity
    ax = fig.add_subplot(gs[2, 0])
    if len(reg) >= 10:
        ar = np.array([r["ar"] for r in reg]); so = np.array([r["sol"] for r in reg])
        sz = np.array([r["A"] for r in reg])
        ax.scatter(ar, so, s=6 + 40 * (sz / sz.max()) ** .4, color=col,
                   edgecolor="k", lw=.3, alpha=.55)
        ax.set_xscale("log")
        ax.set_xlabel("aspect ratio (ellipse major/minor, log)")
        ax.set_ylabel("solidity (area / convex area)")
        ax.set_title(f"Region shape — n={len(reg)} uncensored networks\n"
                     f"median aspect {np.median(ar):.2f}, solidity {np.median(so):.3f}",
                     fontsize=9.5)
        ax.grid(alpha=.3, which="both")
    else:
        note(ax, f"only {len(reg)} uncensored networks ≥500 px²")

    # (2,1) this set's frames in density-vs-width space
    ax = fig.add_subplot(gs[2, 1])
    ld, rw = [], []
    for n in frames:
        s = SKF[n]
        ld.append(fnum(s["SkeletonPx"]) / fnum(s["W"]))
        rw.append(fnum(s["MeanWidth_px"]) / fnum(s["W"]))
    ax.scatter(ld, rw, s=[30 + 6 * x for x in a], color=col, edgecolor="k", lw=.6, alpha=.8)
    for i, n in enumerate(frames):
        if a[i] >= max(a) * .75:
            ax.annotate(short[i].strip(), (ld[i], rw[i]), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points")
    ax.set_yscale("log")
    if len(frames) < 3:
        ax.set_xlim(0, max(ld) * 1.6)
        ax.set_ylim(min(rw) / 3, max(rw) * 3)
    ax.set_xlabel("crack line density (skeleton length / frame width)")
    ax.set_ylabel("relative feature width (log)")
    ax.set_title("Each frame: how much cracking vs how thick\n"
                 "bubble ∝ area fraction. Both axes resolution-invariant.", fontsize=9.5)
    ax.grid(alpha=.3, which="both")

    # (2,2) numbers
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    cens = np.array([fnum(PFM[n]["CensoredAreaSharePct"]) for n in frames])
    dom = np.array([fnum(PFM[n]["LargestAreaSharePct"]) for n in frames])
    fd = np.array([fnum(SKF[n]["FractalDim"]) for n in frames])
    rev = np.array([fnum(PFM[n]["ReviewedPct"]) for n in frames])
    hol = np.array([fnum(HOL[n].get("Holes", 0), 0) for n in frames])
    m, R = axial([r["ang"] for r in o], [r["A"] for r in o]) if len(o) >= 15 else (float("nan"),) * 2
    L = []
    L.append(("family", fam))
    for k2 in ("date", "alloy", "condition", "block", "process", "exposure"):
        if tok.get(k2):
            L.append((k2, str(tok[k2])))
    L.append(("detectors", ", ".join(dets)))
    L.append(("frames", str(len(frames))))
    L.append(("", ""))
    L.append(("crack area %  median", f"{np.median(a):.2f}"))
    L.append(("               mean", f"{np.mean(a):.2f}"))
    L.append(("               range", f"{min(a):.2f} – {max(a):.2f}"))
    L.append(("line density  median", f"{np.median(ld):.2f}"))
    L.append(("rel. width    median", f"{np.median(rw):.4f}"))
    L.append(("fractal dim   median", f"{np.nanmedian(fd):.3f}"))
    L.append(("", ""))
    L.append(("area in edge-touching regions", f"{np.nanmedian(cens):.1f} %"))
    L.append(("area in the single largest region", f"{np.nanmedian(dom):.1f} %"))
    L.append(("enclosed islands per frame", f"{np.nanmedian(hol):.0f}"))
    L.append(("", ""))
    if np.isfinite(m):
        L.append(("orientation (axial mean)", f"{m:+.1f}°"))
        L.append(("anisotropy |R|", f"{R:.2f}"))
    if tv.size >= MINBR:
        L.append(("tortuosity (controlled)", f"{np.median(tv):.3f}  (n={tv.size})"))
    if ta.size >= MINBR:
        L.append(("median turn angle", f"{np.median(ta):.1f}°"))
    L.append(("", ""))
    L.append(("frames with any human review", f"{int((rev > 0).sum())} / {len(frames)}"))
    L.append(("reviewed % of frame  median", f"{np.median(rev):.2f}"))
    y = 1.0
    ax.text(0, y, key, fontsize=13, fontweight="bold", va="top", family="monospace")
    y -= .075
    for k2, v in L:
        if k2 == "":
            y -= .022; continue
        ax.text(0, y, k2, fontsize=8.2, va="top", color="#333")
        ax.text(1.0, y, v, fontsize=8.2, va="top", ha="right",
                family="monospace", fontweight="bold")
        y -= .0405

    sub = " · ".join(f"{k2}={tok[k2]}" for k2 in
                     ("condition", "block", "process", "exposure", "date") if tok.get(k2))
    fig.suptitle(f"{key}    —    {fam}{'   ·   ' + sub if sub else ''}   ·   "
                 f"{len(frames)} frame{'s' if len(frames) > 1 else ''}   ·   "
                 f"detector {', '.join(dets)}",
                 fontsize=13, y=.985)
    fig.text(.5, .945,
             "Panels use only this set's own frames and are in PIXELS. µm/px is recoverable "
             "per frame from the SEM databar — see analysis/scale_hfw.csv.",
             ha="center", fontsize=9.5, color="#333")
    fig.text(.5, .925,
             "CAVEAT: crack area fraction is magnification-dependent (corpus spans HFW "
             "10.4 µm – 2.59 mm); on heavily painted frames it reflects the brush, not the crack.",
             ha="center", fontsize=9.5, color="#8B0000")
    for p in (f"{OUTD}/{key}_diagram.png", f"sets/{key}/{key}_diagram.png"):
        fig.savefig(p, dpi=140)
    plt.close(fig)
    return len(frames), len(reg), len(br), tv.size


print(f"{'set':<20}{'frames':>7}{'networks':>10}{'branches':>10}{'thin ctrl':>11}")
for key in sorted(sets):
    nf, nr, nb, nt = build(key, sets[key])
    print(f"{key:<20}{nf:>7}{nr:>10}{nb:>10}{nt:>11}")
print(f"\nwrote 10 diagrams to {OUTD}/ and into each sets/<set>/ folder")
