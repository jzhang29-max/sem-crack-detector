#!/usr/bin/env python3
"""Per-set crack analysis from the region CSVs.

Design decisions, and why:

* UNITS ARE PIXELS. The export README says the scale bar is cropped before
  analysis, so no um/px factor is recoverable. Every length here is px and
  every area px^2. Densities are therefore per megapixel, not per mm^2.

* TWO POPULATIONS. Pooled over all 62 frames the top 1% of regions hold
  92.5% of crack area while 66% of regions are <=500 px and hold 0.73%.
  Reporting one "mean region size" mixes a few frame-spanning networks with
  thousands of specks and describes neither. Everything is split at
  SPECK_MAX and the two classes are reported separately.

* RIGHT-CENSORING. Regions whose bbox touches a frame edge continue outside
  the field of view, so their Length/Area are LOWER BOUNDS. On the first
  frame probed, 5 edge-touching regions held 88.7% of the crack area -- so a
  "mean crack length" over all regions would be an average of lower bounds.
  Size and morphology summaries use uncensored regions only; the censored
  area share is reported alongside so the reader knows how much was set aside.

* AREAL FRACTION IS THE ROBUST CROSS-SET METRIC. An area fraction measured
  inside the frame is a density: it does not care that a network leaves the
  frame. That makes CrackAreaPct comparable across sets in a way that
  max-length or region-count are not.

* ORIENTATION IS AXIAL, period 180 deg, so -89 deg and +89 deg are nearly
  the SAME orientation. Arithmetic means of Orientation_deg are meaningless
  (they average to ~0 for a perfectly vertical set). Circular statistics on
  the doubled angle are used instead, area-weighted, networks only.

* THE SPECIMEN IS THE STATISTICAL UNIT, not the frame. Sibling frames from
  one block are near-duplicates. Set-level sd is only printed at n>=3 frames
  and is a within-specimen spread, NOT an error bar for the material.

* THESE MASKS ARE DETECTOR OUTPUT with human corrections merged in, not
  validated ground truth. A between-set difference can be a contrast /
  detector-sensitivity difference rather than a material difference. The MAR
  family crosses process with detector, so that confound is estimated here
  instead of being assumed away.
"""
import csv, glob, os, sys, math, json
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/jiamingzhang/Desktop/sem-crack-detector/interior_active_learning/code")
from aggregate import parse_name, specimen_key

SETS = os.path.join(ROOT, "sets")
OUTD = os.path.join(ROOT, "analysis")
os.makedirs(OUTD, exist_ok=True)

SPECK_MAX = 500.0     # px^2; holds 0.73% of pooled area over 66% of regions
MIN_ORIENT_AREA = 500.0
MIN_ORIENT_ECC = 0.90  # a near-round blob has no meaningful major-axis angle


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.size == 0 or x.sum() <= 0:
        return float("nan")
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def axial_stats(theta_deg, weights):
    """Area-weighted mean orientation and anisotropy for axial (180-periodic) data.

    R is the resultant length on the doubled angle: 0 = isotropic (no preferred
    direction), 1 = every region on one axis.
    """
    if len(theta_deg) == 0 or np.sum(weights) <= 0:
        return float("nan"), float("nan")
    phi = 2 * np.radians(np.asarray(theta_deg, float))
    w = np.asarray(weights, float)
    C = np.sum(w * np.cos(phi)) / w.sum()
    S = np.sum(w * np.sin(phi)) / w.sum()
    R = math.hypot(C, S)
    mean = math.degrees(0.5 * math.atan2(S, C))
    if mean <= -90:
        mean += 180
    if mean > 90:
        mean -= 180
    return mean, R


def frame_metrics(name, rows, H, W):
    a  = np.array([float(r["Area_px"]) for r in rows])
    ln = np.array([float(r["Length_px"]) for r in rows])
    wd = np.array([float(r["Width_px"]) for r in rows])
    ar = np.array([float(r["AspectRatio"]) for r in rows])
    so = np.array([float(r["Solidity"]) for r in rows])
    ec = np.array([float(r["Eccentricity"]) for r in rows])
    th = np.array([float(r["Orientation_deg"]) for r in rows])
    y0 = np.array([int(r["BBoxY0"]) for r in rows]); x0 = np.array([int(r["BBoxX0"]) for r in rows])
    y1 = np.array([int(r["BBoxY1"]) for r in rows]); x1 = np.array([int(r["BBoxX1"]) for r in rows])

    frame_area = float(H) * float(W)
    cens = (y0 <= 0) | (x0 <= 0) | (y1 >= H) | (x1 >= W)
    net  = a >= SPECK_MAX
    spk  = ~net
    ok   = net & ~cens                      # uncensored networks: safe for size/shape

    m = {"SourceImage": name, "H": H, "W": W}
    t = parse_name(name)
    m["Family"] = t.get("family", "?")
    m["Detector"] = t.get("detector", "?")
    m["Set"] = specimen_key(name)

    m["Regions"] = len(rows)
    m["Networks"] = int(net.sum())
    m["Specks"] = int(spk.sum())
    m["SpeckDensity_perMpx"] = spk.sum() / (frame_area / 1e6)
    m["CrackAreaPct"] = 100 * a.sum() / frame_area
    m["NetworkAreaPct"] = 100 * a[net].sum() / frame_area
    m["SpeckAreaPct"] = 100 * a[spk].sum() / frame_area

    m["CensoredRegions"] = int(cens.sum())
    m["CensoredAreaSharePct"] = 100 * a[cens].sum() / a.sum() if a.sum() else float("nan")

    m["LargestRegion_px"] = float(a.max()) if a.size else 0.0
    m["LargestAreaSharePct"] = 100 * a.max() / a.sum() if a.sum() else float("nan")
    m["AreaGini"] = gini(a)

    # the largest network: does it span the field of view?
    if a.size:
        i = int(np.argmax(a))
        m["LargestSpansH"] = bool(x0[i] <= 0 and x1[i] >= W)
        m["LargestSpansV"] = bool(y0[i] <= 0 and y1[i] >= H)
        m["LargestCensored"] = bool(cens[i])
    else:
        m["LargestSpansH"] = m["LargestSpansV"] = m["LargestCensored"] = False

    # size + shape from uncensored networks only
    m["UncensoredNetworks"] = int(ok.sum())
    if ok.sum():
        m["NetMedianArea_px"] = float(np.median(a[ok]))
        m["NetP90Area_px"] = float(np.percentile(a[ok], 90))
        m["NetMedianLength_px"] = float(np.median(ln[ok]))
        m["NetMaxLength_px"] = float(ln[ok].max())
        wsum = a[ok].sum()
        m["NetAwMeanWidth_px"] = float((a[ok] * wd[ok]).sum() / wsum)
        m["NetAwMeanAspect"] = float((a[ok] * ar[ok]).sum() / wsum)
        m["NetAwMeanSolidity"] = float((a[ok] * so[ok]).sum() / wsum)
        m["NetMedianAspect"] = float(np.median(ar[ok]))
    else:
        for k in ("NetMedianArea_px", "NetP90Area_px", "NetMedianLength_px", "NetMaxLength_px",
                  "NetAwMeanWidth_px", "NetAwMeanAspect", "NetAwMeanSolidity", "NetMedianAspect"):
            m[k] = float("nan")

    m["TotalLength_px"] = float(ln.sum())
    m["LengthDensity_px_perMpx"] = ln.sum() / (frame_area / 1e6)

    orient = ok & (a >= MIN_ORIENT_AREA) & (ec >= MIN_ORIENT_ECC)
    mo, R = axial_stats(th[orient], a[orient])
    m["OrientMean_deg"], m["Anisotropy_R"] = mo, R
    m["OrientN"] = int(orient.sum())
    return m


def main():
    summary = {r["SourceImage"]: r for r in csv.DictReader(open(os.path.join(ROOT, "summary.csv")))}
    frames = []
    for setdir in sorted(os.listdir(SETS)):
        d = os.path.join(SETS, setdir)
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "regions", "*_regions.csv"))):
            name = os.path.basename(p)[:-len("_regions.csv")]
            rows = list(csv.DictReader(open(p)))
            s = summary[name]
            frames.append(frame_metrics(name, rows, int(s["ImageH_px"]), int(s["ImageW_px"])))

    cols = list(frames[0].keys())
    with open(os.path.join(OUTD, "per_frame_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(frames)

    # ---- set level -----------------------------------------------------------
    NUM = [c for c in cols if c not in ("SourceImage", "Family", "Detector", "Set",
                                        "LargestSpansH", "LargestSpansV", "LargestCensored")]
    bysets = defaultdict(list)
    for m in frames:
        bysets[m["Set"]].append(m)

    setrows = []
    for k in sorted(bysets):
        g = bysets[k]
        row = {"Set": k, "Family": g[0]["Family"], "Frames": len(g),
               "Detectors": ",".join(sorted({m["Detector"] for m in g})),
               "DispersionEstimable": len(g) >= 3,
               "SpanningFrames": sum(1 for m in g if m["LargestSpansH"] or m["LargestSpansV"])}
        for c in NUM:
            v = np.array([m[c] for m in g], float)
            v = v[np.isfinite(v)]
            row[f"{c}_mean"] = float(v.mean()) if v.size else float("nan")
            row[f"{c}_sd"] = float(v.std(ddof=1)) if (v.size >= 3) else float("nan")
        # orientation must be pooled circularly, not averaged
        th, wt = [], []
        for m in g:
            if np.isfinite(m["OrientMean_deg"]):
                th.append(m["OrientMean_deg"]); wt.append(m["NetworkAreaPct"])
        mo, R = axial_stats(th, wt)
        row["SetOrientMean_deg"], row["SetAnisotropy_R"] = mo, R
        setrows.append(row)

    with open(os.path.join(OUTD, "per_set_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(setrows[0].keys()))
        w.writeheader(); w.writerows(setrows)

    # ---- MAR: process x detector, the confound made explicit ----------------
    cell = defaultdict(list)
    for m in frames:
        if m["Family"] == "superalloy":
            cell[(parse_name(m["SourceImage"])["process"], m["Detector"])].append(m)
    print("\n=== MAR_Amb: crack area % by process x detector (crossed) ===")
    print(f"{'':<8}" + "".join(f"{d:>22}" for d in ("CBS", "ETD")))
    for proc in ("AS", "Cast", "HIP"):
        line = f"{proc:<8}"
        for det in ("CBS", "ETD"):
            g = cell.get((proc, det), [])
            if g:
                v = np.array([m["CrackAreaPct"] for m in g])
                line += f"{v.mean():>13.2f} (n={len(g)})"
            else:
                line += f"{'--':>22}"
        print(line)
    for det in ("CBS", "ETD"):
        v = np.array([m["CrackAreaPct"] for m in frames
                      if m["Family"] == "superalloy" and m["Detector"] == det])
        s = np.array([m["SpeckDensity_perMpx"] for m in frames
                      if m["Family"] == "superalloy" and m["Detector"] == det])
        print(f"  {det} marginal: area {v.mean():.2f}% (n={len(v)})   specks {s.mean():.1f}/Mpx")

    json.dump({"speck_max_px": SPECK_MAX, "frames": len(frames), "sets": len(bysets)},
              open(os.path.join(OUTD, "params.json"), "w"), indent=2)

    # ---- console table -------------------------------------------------------
    print("\n=== per set ===")
    hdr = (f"{'set':<20}{'n':>3}{'area%':>9}{'net%':>8}{'spk%':>7}{'cens%':>7}"
           f"{'dom%':>7}{'gini':>6}{'specks/Mpx':>11}{'aspect':>8}{'|R|':>6}{'ang':>7}{'span':>6}")
    print(hdr); print("-" * len(hdr))
    for r in setrows:
        print(f"{r['Set']:<20}{r['Frames']:>3}{r['CrackAreaPct_mean']:>9.2f}"
              f"{r['NetworkAreaPct_mean']:>8.2f}{r['SpeckAreaPct_mean']:>7.3f}"
              f"{r['CensoredAreaSharePct_mean']:>7.1f}{r['LargestAreaSharePct_mean']:>7.1f}"
              f"{r['AreaGini_mean']:>6.2f}{r['SpeckDensity_perMpx_mean']:>11.1f}"
              f"{r['NetAwMeanAspect_mean']:>8.2f}{r['SetAnisotropy_R']:>6.2f}"
              f"{r['SetOrientMean_deg']:>7.1f}{r['SpanningFrames']:>4}/{r['Frames']}")

    print("\n=== frames flagged for review (area% far above their set) ===")
    for k in sorted(bysets):
        g = bysets[k]
        if len(g) < 4:
            continue
        v = np.array([m["CrackAreaPct"] for m in g])
        med = np.median(v); mad = np.median(np.abs(v - med)) or 1e-9
        for m in g:
            z = 0.6745 * (m["CrackAreaPct"] - med) / mad
            if z > 3.5:
                print(f"  {m['SourceImage']:<38} {m['CrackAreaPct']:>6.2f}%  "
                      f"set median {med:.2f}%  robust z={z:.1f}")
    print(f"\nwrote {OUTD}/per_frame_metrics.csv, per_set_metrics.csv")

main()
