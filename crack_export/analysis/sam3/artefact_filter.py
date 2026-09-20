#!/usr/bin/env python3
"""Reject the two structures the detector actually confuses for cracks.

Measured failure mode, not a guess: after fixing over-prediction of AREA (q98 -> q99 cut the
predicted fraction from 1.68% to 0.84% and raised median clIoU_adapt 0.1465 -> 0.1514), the
three worst frames still carry PAR 4.75, 12.04 and 14.63. A thinner cut removes area, not wrong
targets. Looking at the contact sheet, what survives on those frames is

  POLISHING SCRATCHES  long, near-straight, mutually parallel, uniform width
  CARBIDES / PITS      small, compact, roughly round

Both differ from a crack in SHAPE, so both are attackable per connected component without any
label. Three discriminants, each with its expected direction:

  eccentricity     pit  LOW (round)          crack HIGH (elongated)
  tortuosity       scratch ~1.0 (straight)   crack > 1 (wanders)
  axial alignment  scratch aligned with the frame's dominant direction; crack arbitrary

Orientation is AXIAL -- period 180 degrees, not 360 -- so the dominant direction is the circular
mean of DOUBLED angles, length-weighted. This corpus has been bitten by the orientation
convention before; regionprops.orientation is measured from the row axis, so +/-90 deg is
horizontal, and that was verified here with synthetic bars rather than assumed.

THE RULE IS DELIBERATELY CONSERVATIVE ON SCRATCHES. A component is only dropped as a scratch if
it is BOTH near-straight AND aligned with the frame's dominant direction. Straightness alone
would delete straight cracks, of which this corpus has several. Alignment alone would delete a
crack that happens to follow the rolling direction.

Every threshold is chosen leave-one-frame-out. Nothing is tuned on the frame it is scored on.
"""
import os, csv, json, time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import skeletonize
from scipy import ndimage as ndi

SC = os.path.dirname(os.path.abspath(__file__))
_CE = os.path.dirname(os.path.dirname(SC))
_REPO = os.path.dirname(_CE)
PAINT = f"{_REPO}/interior_active_learning/paint"


def gt_side(g, tau):
    """Cache the LABEL half of clIoU. It does not depend on the prediction, and recomputing a
    skeletonize plus a distance transform on 27 Mpx for every threshold setting was the real
    cost of this sweep -- not the shape features I first blamed."""
    sg = skeletonize(g)
    return sg, (ndi.distance_transform_edt(~sg) <= tau)


def image_area_grey(n):
    a = np.array(Image.open(f"{_REPO}/original/{n}.tif")).astype(np.float64)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    g = np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    return g[:int(round(g.shape[1] * 2 / 3))]


def cliou(p, g, tau, cache=None):
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    sp = skeletonize(p)
    sg, b = cache if cache is not None else gt_side(g, tau)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    a = ndi.distance_transform_edt(~sp) <= tau
    tp = int((sg & a).sum()); fp = int((sp & ~(sp & b)).sum()); fn = int((sg & ~(sg & a)).sum())
    return float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0


def metric_skeleton_length(sk):
    """True path length of a skeleton: 1 per orthogonal step, sqrt(2) per diagonal step.

    Counting skeleton PIXELS instead underestimates a diagonal path by up to sqrt(2), which
    makes tortuosity = length/chord come out BELOW 1 -- geometrically impossible, since the
    path can never be shorter than the straight line between its ends. A pure diagonal bar
    scored 0.7252 under the pixel-count version. This project already has a memory note titled
    "pixel count is not path length" from the first time it happened; this is the second.
    """
    ort = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], np.uint8)
    dia = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], np.uint8)
    s8 = sk.astype(np.uint8)
    n_ort = int((s8 * ndi.convolve(s8, ort, mode="constant")).sum()) \
          + int((s8 * ndi.convolve(s8, ort.T, mode="constant")).sum())
    n_dia = int((s8 * ndi.convolve(s8, dia, mode="constant")).sum()) \
          + int((s8 * ndi.convolve(s8, dia[:, ::-1], mode="constant")).sum())
    # Kulpa's estimator, the same constants tools/skeleton_metrics.py already uses. Plain
    # sqrt(2) weighting is exact at 0/45/90 degrees but over-counts intermediate angles: a
    # straight bar at 30 degrees measured 1.074, and the scratch threshold is 1.08, so a
    # straight line was within 0.006 of being called curved. Kulpa removes most of that.
    return 0.948 * n_ort + 1.343 * n_dia


def tortuosity_of(sub):
    """METRIC skeleton length / end-to-end chord. 1.0 for a straight bar at any angle."""
    sk = skeletonize(sub)
    if int(sk.sum()) < 4:
        return 1.0
    L = metric_skeleton_length(sk)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    ends = np.argwhere(sk & (nb == 2))
    if len(ends) < 2:
        ys, xs = np.nonzero(sk)
        ends = np.array([[ys.min(), xs[ys.argmin()]], [ys.max(), xs[ys.argmax()]]])
    if len(ends) > 12:
        ends = ends[np.linspace(0, len(ends) - 1, 12).astype(int)]
    d = np.hypot(ends[:, 0][:, None] - ends[:, 0][None, :],
                 ends[:, 1][:, None] - ends[:, 1][None, :])
    chord = float(d.max())
    return float(L / chord) if chord > 1 else 1.0


def scale_map(grey, sigmas=(1.0, 1.5, 2.0, 3.0, 4.0, 6.0)):
    """Local structure width taken from the GREY image, not from the binary mask.

    For each pixel, the sigma whose single-scale Meijering response is strongest. This is a
    width estimate that never touches the threshold. Measuring width as 2*EDT on the mask does
    NOT work here and was tried first: after a q99 cut both cracks and polishing striations come
    out 3-4 px wide because the cut, not the feature, sets the width, and their width-CV
    distributions were 0.271 vs 0.284 -- indistinguishable and in the wrong direction. Read off
    the grey instead, the same two populations give 0.269 vs 0.208, AUC 0.680, correct direction.
    """
    from skimage.filters import meijering as _mj
    g = grey.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    r = np.stack([_mj(g, sigmas=[s], black_ridges=True) for s in sigmas])
    return np.asarray(sigmas, dtype=np.float32)[np.argmax(r, axis=0)]


def describe(mask, scale=None):
    """Per-component shape features, plus the frame's dominant axial orientation."""
    lab = cc_label(mask)
    props = regionprops(lab)
    comps = []
    for p in props:
        if p.area < 8:
            continue
        sk = skeletonize(p.image)
        L = float(metric_skeleton_length(sk))
        wcv = np.nan
        if scale is not None:
            sl = scale[p.slice][sk]
            if len(sl) >= 5 and sl.mean() > 0:
                wcv = float(sl.std() / sl.mean())
        comps.append({"label": p.label, "area": int(p.area),
                      "ecc": float(p.eccentricity),
                      "orient": float(p.orientation),          # radians, from the ROW axis
                      "major": float(p.major_axis_length),
                      "L": L,
                      # elongation gate: geodesic diameter >= 8 * mean width, with L standing in
                      # for the geodesic diameter (equal for a simple curve) and w_bar = area/L.
                      # L >= 8*(area/L)  <=>  L^2 >= 8*area. Cheap, scale-free, and it is what
                      # protects SHORT cracks: a component too short to have a meaningful
                      # straightness must not be judged on its straightness.
                      "elong_ok": (L * L) >= 8.0 * float(p.area),
                      "wcv": wcv,
                      "tort": tortuosity_of(p.image)})
    if not comps:
        return lab, comps, 0.0, 0.0
    # length-weighted circular mean of DOUBLED angles (axial data, period pi)
    w = np.array([c["major"] for c in comps])
    a2 = 2 * np.array([c["orient"] for c in comps])
    C, S = float((w * np.cos(a2)).sum()), float((w * np.sin(a2)).sum())
    dom = 0.5 * np.arctan2(S, C)
    R = float(np.hypot(C, S) / max(w.sum(), 1e-9))              # 0 = isotropic, 1 = all parallel
    return lab, comps, dom, R


def apply_filter(mask, ecc_min, tort_min, align_deg, cached=None, r_min=0.40, wcv_max=None):
    """Drop round components, and straight ones aligned with the frame's dominant direction.

    `cached` is the (lab, comps, dom, R) tuple from describe(). It MUST be passed when sweeping
    thresholds: the component features do not depend on the thresholds, and recomputing
    regionprops plus a per-component tortuosity for all 49 settings made the first version of
    this sweep too slow to finish a single frame.
    """
    lab, comps, dom, R = cached if cached is not None else describe(mask)
    if not comps:
        return mask, {"dropped_round": 0, "dropped_scratch": 0, "R": R}
    drop, dr, ds = [], 0, 0
    thr = np.deg2rad(align_deg)
    # frame-level totals for the leave-one-out axis
    w_all = np.array([c["major"] for c in comps])
    C_all = float((w_all * np.cos(2 * np.array([c["orient"] for c in comps]))).sum())
    S_all = float((w_all * np.sin(2 * np.array([c["orient"] for c in comps]))).sum())
    W_all = float(w_all.sum())
    for c in comps:
        if ecc_min > 0 and c["ecc"] < ecc_min:                   # round -> pit / carbide
            drop.append(c["label"]); dr += 1; continue
        if tort_min <= 0:
            continue
        # ABSTAIN: too short to have a meaningful straightness -> keep unconditionally
        if not c["elong_ok"]:
            continue
        # LEAVE-ONE-OUT axis: without this the longest scratch sets the very axis that then
        # deletes it, and a lone dominant crack votes for its own removal.
        Cl = C_all - c["major"] * np.cos(2 * c["orient"])
        Sl = S_all - c["major"] * np.sin(2 * c["orient"])
        Wl = W_all - c["major"]
        if Wl <= 0:
            continue
        R_lofo = float(np.hypot(Cl, Sl) / Wl)
        if R_lofo < r_min:            # no scratch population -> the axis is noise
            continue
        mu = 0.5 * np.arctan2(Sl, Cl)
        da = abs(np.angle(np.exp(1j * 2 * (c["orient"] - mu)))) / 2   # signed, period pi
        # WIDTH GUARD: only delete if the width is also UNIFORM along the path. A crack tapers
        # and branches; a polishing mark does not. This can only make the rule stricter, which
        # is what the catastrophic frame needs -- it deleted 57% of real, parallel cracks.
        if wcv_max is not None and not (c["wcv"] == c["wcv"] and c["wcv"] < wcv_max):
            continue
        if c["tort"] < tort_min and da < thr:                    # straight AND aligned -> scratch
            drop.append(c["label"]); ds += 1
    if not drop:
        return mask, {"dropped_round": 0, "dropped_scratch": 0, "R": R}
    out = mask & ~np.isin(lab, drop)
    return out, {"dropped_round": dr, "dropped_scratch": ds, "R": R}


def main():
    gran = {r["frame"]: float(r["median_thick_px"] or 0)
            for r in csv.DictReader(open(f"{_CE}/analysis/label_granularity.csv"))}
    rows = [r for r in json.load(open(f"{SC}/detector_all.json")) if "clIoU_adapt" in r]
    # A focused grid, not a full sweep. Each setting costs a skeletonize plus a distance
    # transform on a 27 Mpx frame, so 49 settings x 38 frames was about four hours. These six
    # cover the corners: eccentricity alone, tortuosity+alignment alone, both together, and off.
    # No eccentricity branch: every setting containing one scored BELOW filter-off in the
    # previous sweep (0.1416, 0.1379, 0.1336, 0.1309 against 0.1514). Round components are not
    # simply carbides -- crack junctions and blobby crack mouths are round too.
    GRID = [(0.0, 0.0, 0),          # filter off -- must be able to win
            (0.0, 1.08, 30),        # best WITHOUT the width guard; destroys AS_24hr
            (0.0, 1.08, 20)]
    WCV = [None, 0.28, 0.24, 0.20]  # None = no width guard
    sc, t0 = {}, time.time()
    for i, r in enumerate(rows):
        n = r["frame"]
        m = np.array(Image.open(f"{SC}/detector_masks/{n}.png")) > 127
        cm = np.array(Image.open(f"{PAINT}/{n}_correction_mask.png"))
        while cm.ndim > 2:
            cm = cm[..., 0]
        gt = cm == 1
        if gt.shape != m.shape:
            continue
        tau = max(2, int(round(gran[n] / 2)))
        gcache = gt_side(gt, tau)
        sc[n] = {}
        gimg = image_area_grey(n)
        cached = describe(m, scale=scale_map(gimg))                      # once per frame, not once per setting
        print(f"      {len(cached[1])} components, orientation coherence R={cached[3]:.3f}",
              flush=True)
        for g in GRID:
            for wc in (WCV if g != (0.0, 0.0, 0) else [None]):
                key = g + (wc,)
                p, info = apply_filter(m, *g, cached=cached, wcv_max=wc) \
                    if g != (0.0, 0.0, 0) else (m, {})
                sc[n][key] = {"clIoU": cliou(p, gt, tau, cache=gcache),
                              "PAR": float(p.sum() / max(gt.sum(), 1)),
                              "kept": float(p.sum() / max(m.sum(), 1))}
        b = max(sc[n], key=lambda k: sc[n][k]["clIoU"])
        print(f"  [{i+1}/{len(rows)}] {n[:34]:<36} off {sc[n][(0.0,0.0,0,None)]['clIoU']:.3f} "
              f"-> best {sc[n][b]['clIoU']:.3f} @ {b}   ({time.time()-t0:.0f}s)", flush=True)
    names = list(sc)
    OFFK = (0.0, 0.0, 0, None)
    off = np.array([sc[n][OFFK]["clIoU"] for n in names])
    print(f"\n{'setting (ecc,tort,align)':<26} {'median clIoU':>13} {'median kept':>12}")
    best = []
    KEYS = list(next(iter(sc.values())))
    for g in KEYS:
        v = np.array([sc[n][g]["clIoU"] for n in names])
        k = np.median([sc[n][g]["kept"] for n in names])
        best.append((float(np.median(v)), g, k))
    for med, g, k in sorted(best, reverse=True)[:8]:
        print(f"  {str(g):<24} {med:>13.4f} {k:>12.3f}")
    print(f"  {'(filter off)':<24} {np.median(off):>13.4f} {1.000:>12.3f}")
    # LOFO over the grid
    lofo = []
    for n in names:
        tr = [u for u in names if u != n]
        bg = max(KEYS, key=lambda g: np.median([sc[u][g]["clIoU"] for u in tr]))
        lofo.append(sc[n][bg]["clIoU"])
    from scipy.stats import wilcoxon
    d = np.array(lofo) - off
    nz = [x for x in d if x != 0]
    p = wilcoxon(nz, method="asymptotic").pvalue if len(nz) >= 6 else float("nan")
    print(f"\n  LOFO over settings : {np.median(lofo):.4f}")
    print(f"  filter off         : {np.median(off):.4f}")
    print(f"  paired: median delta {np.median(d):+.4f}, wins {int((d>0).sum())}/{len(d)}, p = {p:.4g}")
    json.dump({"lofo": float(np.median(lofo)), "off": float(np.median(off)),
               "p": float(p), "per_frame_lofo": [float(x) for x in lofo],
               "grid": [[list(g), m, k] for m, g, k in sorted(best, reverse=True)]},
              open(f"{SC}/artefact_filter.json", "w"), indent=1)
    print("\nwrote artefact_filter.json")


if __name__ == "__main__":
    main()
