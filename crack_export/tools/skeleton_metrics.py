#!/usr/bin/env python3
"""Skeleton-branch path morphology from the crack masks. READ-ONLY on inputs.

WHY THE MASKS AND NOT JUST THE CSVs: the region CSVs carry Length_px/Width_px/
AspectRatio/Solidity/Perimeter_px, which describe a region's best-fit ELLIPSE and
convex hull. None of them describe the crack PATH. Tortuosity, turn angles,
branching and segment-length structure -- everything "linearity" actually means --
requires the skeleton, so it is computed here from the masks.

ANALYSIS UNIT = SKELETON BRANCH, not connected region. A single connected region
here can be 12 million px of branching network; its end-to-end "tortuosity" would
measure network topology, not how straight a crack runs. A branch (the arc between
two nodes/endpoints of the skeleton graph) is the closest thing in this data to
"one crack segment".

Per branch: path length, chord, tortuosity, Douglas-Peucker segment lengths and
turn angles, mean/sd local width from the distance transform, and an edge-censoring
flag. Per frame: branch-point density and box-counting fractal dimension.
"""
import csv, json, os, sys, math
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
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
ROOT = _CE
OUT = os.path.join(ROOT, "analysis", "skeleton")
os.makedirs(OUT, exist_ok=True)

DP_TOL = 3.0          # px; Douglas-Peucker tolerance for segment/turn extraction
MIN_BRANCH_PX = 12    # shorter arcs are skeletonization spurs, not crack segments
N8 = np.ones((3, 3), np.uint8)


def load_mask(name):
    a = np.array(Image.open(os.path.join(ROOT, "masks", name + "_mask.png")))
    while a.ndim > 2:
        a = a[..., 0]
    return a < 128                      # crack = BLACK on white


def order_path(coords):
    """Order a set of 8-connected path pixels end to end. Returns None for loops."""
    S = {(int(y), int(x)) for y, x in coords}
    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in S]
    ends = [p for p in S if len(nbrs(p)) <= 1]
    if not ends:
        return None                     # closed loop: no well-defined chord
    start = min(ends)
    path, seen, cur, prev = [start], {start}, start, None
    while True:
        nxt = [q for q in nbrs(cur) if q not in seen]
        if not nxt:
            break
        # prefer the 4-connected continuation to avoid diagonal shortcuts
        nxt.sort(key=lambda q: (abs(q[0] - cur[0]) + abs(q[1] - cur[1]), q))
        cur = nxt[0]
        path.append(cur); seen.add(cur)
    return np.array(path, float)


def polyline_length(p):
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*(np.diff(p, axis=0).T)).sum())


def rdp(p, tol):
    """Douglas-Peucker. Iterative to avoid recursion limits on long paths."""
    if len(p) < 3:
        return p
    keep = np.zeros(len(p), bool); keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        L = math.hypot(*ab)
        seg = p[i + 1:j]
        if L < 1e-9:
            d = np.hypot(*(seg - a).T)
        else:
            ap = seg - a          # explicit 2-D cross magnitude; np.cross on 2-vectors
            d = np.abs(ab[0] * ap[:, 1] - ab[1] * ap[:, 0]) / L   # is deprecated in numpy 2
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m)); stack.append((m, j))
    return p[keep]


def turn_angles(v):
    """Interior turn angles (deg) at each internal vertex of a polyline.
    0 = perfectly straight continuation; 90 = right-angle kink."""
    if len(v) < 3:
        return np.empty(0)
    d = np.diff(v, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    ok = n > 1e-9
    d, n = d[ok], n[ok]
    if len(d) < 2:
        return np.empty(0)
    u = d / n[:, None]
    cosang = np.clip((u[:-1] * u[1:]).sum(1), -1, 1)
    return np.degrees(np.arccos(cosang))


def fractal_dim(mask):
    """Box-counting dimension of the crack trace. 1 = line, 2 = area-filling."""
    m = mask
    sizes, counts = [], []
    for s in (2, 4, 8, 16, 32, 64, 128, 256):
        H, W = m.shape
        h, w = H // s, W // s
        if h < 4 or w < 4:
            break
        blk = m[:h * s, :w * s].reshape(h, s, w, s).any(axis=(1, 3))
        c = int(blk.sum())
        if c == 0:
            break
        sizes.append(s); counts.append(c)
    if len(sizes) < 4:
        return float("nan")
    a = np.polyfit(np.log(1.0 / np.array(sizes, float)), np.log(np.array(counts, float)), 1)
    return float(a[0])


def process(name):
    mask = load_mask(name)
    H, W = mask.shape
    skel = skeletonize(mask)
    dt = ndi.distance_transform_edt(mask)

    nc = ndi.convolve(skel.astype(np.uint8), N8, mode="constant") - skel.astype(np.uint8)
    nodes = skel & (nc >= 3)
    n_nodes = int(nodes.sum())
    n_ends = int((skel & (nc == 1)).sum())
    skel_px = int(skel.sum())

    seg = skel & ~nodes
    lab, nlab = ndi.label(seg, structure=N8)

    coords = np.argwhere(seg)
    if coords.size == 0:
        return {"SourceImage": name, "SkeletonPx": skel_px, "Branches": 0}, []
    labs = lab[coords[:, 0], coords[:, 1]]
    o = np.argsort(labs, kind="stable")
    coords, labs = coords[o], labs[o]
    bounds = np.searchsorted(labs, np.arange(1, nlab + 1), side="left")
    bounds = np.append(bounds, len(labs))

    rows = []
    for i in range(nlab):
        c = coords[bounds[i]:bounds[i + 1]]
        if len(c) < MIN_BRANCH_PX:
            continue
        p = order_path(c)
        if p is None or len(p) < 3:
            continue
        plen = polyline_length(p)
        chord = float(np.hypot(*(p[-1] - p[0])))
        if plen <= 0:
            continue
        v = rdp(p, DP_TOL)
        segs = np.hypot(*(np.diff(v, axis=0).T)) if len(v) > 1 else np.empty(0)
        ta = turn_angles(v)
        wid = 2.0 * dt[c[:, 0], c[:, 1]]
        cens = bool((c[:, 0] <= 0).any() or (c[:, 1] <= 0).any()
                    or (c[:, 0] >= H - 1).any() or (c[:, 1] >= W - 1).any())
        # net direction of the branch, axial, measured like Orientation_deg
        dy, dx = (p[-1] - p[0])
        ang = math.degrees(math.atan2(dx, dy)) if chord > 1e-9 else float("nan")
        if ang is not None and not math.isnan(ang):
            ang = ((ang + 90) % 180) - 90
        rows.append({
            "SourceImage": name,
            "PathLen_px": round(plen, 2),
            "Chord_px": round(chord, 2),
            "Tortuosity": round(plen / chord, 4) if chord > 1e-6 else "",
            "Vertices": len(v),
            "SegMean_px": round(float(segs.mean()), 2) if segs.size else "",
            "SegMax_px": round(float(segs.max()), 2) if segs.size else "",
            "Straightness": round(chord / plen, 4) if plen > 1e-6 else "",
            "TurnMean_deg": round(float(ta.mean()), 2) if ta.size else "",
            "TurnMedian_deg": round(float(np.median(ta)), 2) if ta.size else "",
            "TurnP90_deg": round(float(np.percentile(ta, 90)), 2) if ta.size else "",
            "TurnsPerLen_per100px": round(100.0 * ta.size / plen, 4) if plen > 0 else "",
            "Width_mean_px": round(float(wid.mean()), 2),
            "Width_sd_px": round(float(wid.std()), 2),
            "Width_cv": round(float(wid.std() / wid.mean()), 4) if wid.mean() > 0 else "",
            "NetAngle_deg": round(ang, 2) if not math.isnan(ang) else "",
            "Censored": cens,
        })

    frame = {
        "SourceImage": name, "H": H, "W": W,
        "SkeletonPx": skel_px, "SkeletonNodes": n_nodes, "SkeletonEnds": n_ends,
        "Branches": len(rows),
        "NodeDensity_per100px_skel": round(100.0 * n_nodes / skel_px, 4) if skel_px else "",
        "SkelPerMpx": round(skel_px / (H * W / 1e6), 1),
        "FractalDim": round(fractal_dim(mask), 4),
        "CrackAreaPct": round(100.0 * float(mask.mean()), 4),
        "MeanWidth_px": round(float(2.0 * dt[skel].mean()), 3) if skel_px else "",
    }
    return frame, rows


def main():
    names = sorted(n[:-len("_mask.png")] for n in os.listdir(os.path.join(ROOT, "masks"))
                   if n.endswith("_mask.png"))
    if len(sys.argv) > 1:
        lo, hi = int(sys.argv[1]), int(sys.argv[2])
        names = names[lo:hi]
    for name in names:
        fp = os.path.join(OUT, name + ".json")
        if os.path.exists(fp):
            print(f"skip {name}", flush=True); continue
        frame, rows = process(name)
        with open(fp, "w") as f:
            json.dump({"frame": frame, "branches": rows}, f)
        print(f"{name}: {frame['Branches']} branches, skel {frame['SkeletonPx']:,} px, "
              f"D={frame['FractalDim']}, nodes/100px={frame['NodeDensity_per100px_skel']}", flush=True)


main()
