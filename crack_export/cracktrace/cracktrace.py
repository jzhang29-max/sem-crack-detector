#!/usr/bin/env python3
"""CrackTrace - scale-native, corridor-supervised, globally-solved centreline extraction.

=============================================================================
STATUS: PROTOTYPE. VALIDATED ON SYNTHETIC DATA ONLY. NOT RUN ON REAL FRAMES.
=============================================================================
What was actually tested (see cracktrace/README.md for the numbers):
  * scale-native features recover a crack at 0.05 and 0.20 um/px, and correctly
    DROP an unresolvable scale at coarse magnification;
  * global extraction covers 95.8% / 98.0% of two synthetic curves and ignores a
    bright decoy scratch (5 px);
  * the corridor argument is quantified: a 59-px brush over a 3-px curve would be
    96.8% false-negative as a PIXEL label, while being nearly exact as a path
    constraint;
  * the paired agreement metric behaves (identical -> F=1.000, perpendicular -> 0.405).

What was NOT done: no run on any real SEM frame, no fit of the weights to real
corridors, no comparison against the shipped pipeline. The intended comparison
(cross-detector centreline agreement on the 8 registration-confirmed CBS/ETD pairs
versus the raw baseline) was set up but the baseline dump failed and was not retried.
Treat every design claim below as a HYPOTHESIS with a synthetic feasibility check,
not as a result.

WHY NOT A CNN OR SAM, stated as testable design claims.
NOTE (2026-09-18): a 2026-09-15 note here claimed SAM 3 had been measured on this corpus and
was "NOT weak" (recall 0.969-1.000 on 10/16 tiles). THAT MEASUREMENT IS INVALID -- its input was
the green channel of the annotated overlay, and opaque red (225,25,25) has green 25, well under the 80 the threshold used, so the label was
written into the input as black pixels on 14 of 16 tiles. See analysis/sam3/LEAK_POSTMORTEM.md.
The claim is withdrawn in both directions: SAM 3 is neither shown strong nor shown weak here.
What IS established, by reading Meta's source rather than by any run, is that instance scores
are rescaled by one global presence scalar per (image, prompt) before a PER-QUERY threshold --
sam3_image_processor.py:195-200 -- so whether an image returns anything is decided by
s_i * max_j q_ij, and an empty result carries no per-instance evidence. Measured: that scalar
spans 105x across four prompts, but only 6.3x without the out-of-vocabulary noun "fracture"
and 3.0x between the two real synonyms. NOT a contribution -- the presence head is in Meta's
own abstract (arXiv:2511.16719). See analysis/sam3/PRIOR_ART_KILL.md. The bar to beat on the clean input is a
single oracle-tuned global threshold at median IoU 0.384.


1. A crack is a 1-D curve embedded in 2-D. Pixel-mask methods represent it as an AREA
   and recover connectivity afterwards. The two dominant pathologies measured in this
   corpus are exactly the artefacts of that choice: fragmentation (the repo needs a
   Dijkstra merge as CLEANUP) and width confusion (area fraction correlates with
   detected width at rho=+0.86; "crack" widths run 5-520 px). Predicting the CENTRELINE
   GRAPH directly makes connectivity a property of the output, not a post-hoc repair.

2. Global by construction. Cracks here span up to the frame diagonal (~7385 px). A
   tiled CNN cannot link fragments 5000 px apart; SAM's receptive field and prompt
   semantics are object-shaped, not curve-shaped. A minimum-spanning-forest over a
   geodesic cost field is solved on the WHOLE frame at once, so a single crack is a
   single connected object however far it runs.

3. Scale-native. The corpus spans HFW 10.4 um to 2.59 mm (249x). Every filter here is
   parameterised in MICRONS and converted to pixels per frame using um/px = HFW/width.
   A sigma of 0.4 um means the same physical thing at every magnification; a sigma of
   3 px does not. CNNs trained on mixed magnifications must learn scale invariance from
   data (CIMP, arXiv:2604.24909, shows they encode it implicitly) -- here it is imposed.

4. IT CONSUMES SUPERSET LABELS NATIVELY. (The claim that this is "the gap nothing in 2026
   occupies" is WITHDRAWN, 2026-09-18 -- box supervision IS superset supervision, the formal
   setting is Superset Label Learning, Liu & Dietterich ICML 2014, and a crack paper already
   ships a shrink module. See cracktrace/README.md item 4. The mechanism below is still what
   this prototype does; only the novelty claim is gone.)
   Measured: 91% of this corpus's 70M hand-marked pixels come from brush strokes with a
   median thickness of 59 px (max 413 px) asserting regions that CONTAIN a ~3 px crack.
   As a pixel label that is dense-but-wrong and unusable. As a CORRIDOR CONSTRAINT on a
   curve it is exactly right and nearly free of annotation noise: "a crack passes
   through this region" is a true statement about a path. Every 2026 weak-supervision
   method assumes the weak label is a SUBSET of the object (scribbles inside, points on,
   boxes around). Here it is a SUPERSET, and a path model is the natural consumer.

5. Interpretable and cheap: the cost field is a linear combination of a handful of
   physically-named terms, so the learned object is a short weight vector, auditable and
   trainable from very few corridors. No GPU.

EVALUATION WITHOUT GROUND TRUTH. There is no pixel-precise ground truth in this corpus
(see analysis/LABEL_GRANULARITY.md). So the primary metric here is LABEL-FREE:
cross-detector geometric agreement on the 8 registration-confirmed CBS/ETD pairs. The
same physical field imaged with two detectors must contain the same cracks; a method
whose extracted centrelines agree more across detectors is measuring the material rather
than the detector. This turns the corpus's unique asset into a validation signal and
sidesteps the missing-label blocker entirely.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


# ----------------------------------------------------------------- scale-native features

def um_per_px(hfw_um: float, width_px: int) -> float:
    return float(hfw_um) / float(width_px)


def _gaussian(img, sigma_px):
    return ndi.gaussian_filter(img, sigma_px, mode="nearest")


def ridge_response(img, sigma_px):
    """Dark-ridge (valley) strength from the Hessian at one scale.

    For a dark line on a brighter background the Hessian has one large POSITIVE
    eigenvalue across the line and a near-zero one along it. Returns
    max(lambda_major, 0), so bright ridges and blobs score ~0.
    """
    s = float(sigma_px)
    g = _gaussian(img, s)
    gyy = ndi.gaussian_filter(g, s, order=(2, 0), mode="nearest") * s * s
    gxx = ndi.gaussian_filter(g, s, order=(0, 2), mode="nearest") * s * s
    gxy = ndi.gaussian_filter(g, s, order=(1, 1), mode="nearest") * s * s
    tr = gxx + gyy
    det = gxx * gyy - gxy * gxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    lam_major = tr / 2.0 + disc          # algebraically larger eigenvalue
    return np.maximum(lam_major, 0.0)


def local_darkness(img, sigma_px):
    """How much darker than its own neighbourhood a pixel is.

    Background-flattened so it is invariant to the slow illumination gradients and
    detector-gain differences that make a global darkness threshold detector-dependent.
    """
    bg = _gaussian(img, sigma_px)
    return np.clip(bg - img, 0.0, None)


def feature_stack(img, upp, scales_um=(0.15, 0.35, 0.8, 2.0), bg_um=8.0):
    """Features at PHYSICAL scales. img float in [0,1]; upp = um per px.

    scales_um are crack half-widths to be sensitive to. A scale is skipped when it
    falls below ~0.8 px for this frame, because it is then unresolved -- the honest
    behaviour at coarse magnification, instead of silently filtering noise.
    """
    feats, names = [], []
    for s_um in scales_um:
        s_px = s_um / upp
        if s_px < 0.8:
            continue
        r = ridge_response(img, s_px)
        feats.append(r); names.append(f"ridge_{s_um}um")
    bg_px = max(bg_um / upp, 3.0)
    d = local_darkness(img, bg_px)
    feats.append(d); names.append(f"dark_{bg_um}um")
    if not feats:
        raise ValueError("no resolvable scale for this frame")
    F = np.stack(feats, 0)
    # per-feature robust normalisation so weights are comparable across frames
    for i in range(F.shape[0]):
        hi = np.percentile(F[i], 99.5)
        if hi > 0:
            F[i] = np.clip(F[i] / hi, 0, 1)
    return F, names


def cost_field(F, w, eps=1e-3):
    """Crackness -> traversal cost. High crackness = cheap to travel.

    cost = 1/(eps + crackness) so a path prefers to stay on crack-like pixels, and the
    cost of a route is a physically meaningful "how uncrack-like is this route" integral.
    """
    w = np.asarray(w, float)
    crack = np.tensordot(w, F, axes=(0, 0))
    crack = np.clip(crack, 0, None)
    hi = np.percentile(crack, 99.5)
    if hi > 0:
        crack = crack / hi
    return 1.0 / (eps + crack), crack


# ------------------------------------------------------- global minimum-spanning network

def _pool_max(a, k):
    """Max-pool by integer factor k, keeping ridges (mean-pooling erases thin lines)."""
    if k <= 1:
        return a
    h, w = a.shape
    H, W = (h // k) * k, (w // k) * k
    return a[:H, :W].reshape(H // k, k, W // k, k).max(axis=(1, 3))


def pick_terminals(crack, n_max=160, min_sep_px=12, q=99.0):
    """Seed points: strong, well-separated crackness maxima.

    Greedy farthest-first on descending crackness. These are anchors the network must
    connect, not a segmentation -- a missed terminal costs a branch, not a whole crack.
    """
    thr = np.percentile(crack, q)
    ys, xs = np.where(crack >= thr)
    if ys.size == 0:
        return np.empty((0, 2), int)
    order = np.argsort(-crack[ys, xs])
    ys, xs = ys[order], xs[order]
    keep = []
    for y, x in zip(ys, xs):
        ok = True
        for y2, x2 in keep:
            if (y - y2) ** 2 + (x - x2) ** 2 < min_sep_px ** 2:
                ok = False
                break
        if ok:
            keep.append((y, x))
            if len(keep) >= n_max:
                break
    return np.array(keep, int)


def extract_network(cost, terminals, max_edge_cost=np.inf):
    """Prim-style minimum spanning FOREST over the geodesic cost field, on the whole grid.

    Grows a tree from the strongest terminal: at each step the cheapest geodesic route
    from the CURRENT NETWORK to any unconnected terminal is traced and appended. Because
    every search runs over the full grid, a route may be thousands of pixels long -- the
    property a tiled network cannot have.

    max_edge_cost stops the forest from inventing a link between genuinely separate
    cracks: an edge whose cheapest route is too expensive is refused, leaving several
    trees. That refusal is the analogue of the repo's brightness gate on its bridges.
    """
    from skimage.graph import MCP_Geometric
    net = np.zeros(cost.shape, bool)
    if len(terminals) == 0:
        return net, []
    remaining = [tuple(t) for t in terminals]
    start = remaining.pop(0)
    net[start] = True
    edges = []
    while remaining:
        mcp = MCP_Geometric(cost, fully_connected=True)
        starts = np.argwhere(net)
        cum, _ = mcp.find_costs(starts.tolist())
        costs = np.array([cum[p] for p in remaining])
        finite = np.isfinite(costs)
        if not finite.any():
            break
        j = int(np.argmin(np.where(finite, costs, np.inf)))
        if costs[j] > max_edge_cost:
            # refuse this link; seed a NEW tree so separate cracks stay separate
            p = remaining.pop(j)
            net[p] = True
            edges.append({"cost": float(costs[j]), "n_px": 1, "refused": True})
            continue
        p = remaining.pop(j)
        try:
            path = mcp.traceback(p)
        except ValueError:
            net[p] = True
            continue
        for (yy, xx) in path:
            net[yy, xx] = True
        edges.append({"cost": float(costs[j]), "n_px": len(path), "refused": False})
    return net, edges


def network_length_um(net, upp_eff):
    """Path length of the extracted network in microns.

    Euclidean step summation on the skeleton graph -- NOT a pixel count. A pixel count
    under-reads a diagonal path by up to 1/sqrt2 (Dorst & Smeulders 1987), which is the
    bug this project already carries in extended_features.py.
    """
    from skimage.morphology import skeletonize
    sk = skeletonize(net)
    n_orth = 0
    n_diag = 0
    ys, xs = np.where(sk)
    S = set(zip(ys.tolist(), xs.tolist()))
    for (y, x) in S:
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if (y + dy, x + dx) in S:
                if dy and dx:
                    n_diag += 1
                else:
                    n_orth += 1
    # Kulpa's corrected estimator, not the naive sqrt(2)
    return (0.948 * n_orth + 1.343 * n_diag) * upp_eff


def trace(img, upp, pool=4, n_terminals=160, weights=None,
          scales_um=(0.15, 0.35, 0.8, 2.0), max_edge_cost=None):
    """Full pipeline for one frame. Returns dict with the network and diagnostics."""
    F, names = feature_stack(img, upp, scales_um=scales_um)
    if weights is None:
        weights = np.ones(F.shape[0]) / F.shape[0]
    cost, crack = cost_field(F, weights)
    crack_p = _pool_max(crack, pool)
    cost_p = 1.0 / (1e-3 + crack_p)
    upp_eff = upp * pool
    term = pick_terminals(crack_p, n_max=n_terminals,
                          min_sep_px=max(int(round(2.0 / upp_eff)), 6))
    if max_edge_cost is None:
        max_edge_cost = np.percentile(cost_p, 60) * (30.0 / upp_eff)
    net, edges = extract_network(cost_p, term, max_edge_cost=max_edge_cost)
    return {"net": net, "crack": crack_p, "cost": cost_p, "terminals": term,
            "edges": edges, "upp_eff": upp_eff, "features": names,
            "length_um": network_length_um(net, upp_eff),
            "n_refused": sum(1 for e in edges if e.get("refused"))}


# ------------------------------------------------- corridor supervision (superset labels)

def corridor_bags(corr_mask, min_area=200):
    """Connected components of a broad-brush crack region = positive BAGS.

    The semantics of a superset label is 'at least one pixel in this region is crack',
    which is the multiple-instance assumption exactly. Nothing here pretends the label
    tells us WHICH pixel, so a 413-px brush costs no accuracy.
    """
    from scipy import ndimage as _ndi
    lab, n = _ndi.label(corr_mask, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() >= min_area:
            out.append(m)
    return out


def mil_objective(F, bags, neg_mask, w, q=98.0):
    """Score to MAXIMISE: bag-level top-q crackness minus background crackness.

    Per bag we take the q-th percentile rather than the mean, because only a thin
    sub-part of a broad corridor is really crack -- the mean would punish the correct
    answer. This is the standard max-pooling MIL score, softened to a high quantile for
    robustness to a few saturated pixels.
    """
    crack = np.tensordot(np.asarray(w, float), F, axes=(0, 0))
    hi = np.percentile(crack, 99.5)
    if hi > 0:
        crack = crack / hi
    pos = np.mean([np.percentile(crack[b], q) for b in bags]) if bags else 0.0
    neg = crack[neg_mask].mean() if neg_mask.any() else 0.0
    return float(pos - neg)


def fit_weights(frames, q=98.0, seed=0):
    """Fit the cost-field weights from corridors only. frames: list of (F, bags, neg).

    The learned object is a handful of non-negative weights on named physical features,
    so it is auditable and needs very few corridors -- unlike a CNN, which needs dense
    masks this corpus does not have.
    """
    from scipy.optimize import minimize
    k = frames[0][0].shape[0]

    def negobj(z):
        w = np.abs(z)
        s = w.sum()
        if s <= 0:
            return 1e6
        w = w / s
        return -np.mean([mil_objective(F, bags, neg, w, q=q) for F, bags, neg in frames])

    best = None
    rng = np.random.default_rng(seed)
    for t in range(4):
        z0 = np.ones(k) / k if t == 0 else rng.random(k)
        r = minimize(negobj, z0, method="Nelder-Mead",
                     options={"maxiter": 220, "xatol": 1e-3, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    w = np.abs(best.x); w = w / w.sum()
    return w, -best.fun


# ------------------------------------------------------------- label-free paired metric

def centreline_agreement(net_a, upp_a, net_b, upp_b, tol_um=2.0):
    """Mutual centreline agreement between two extractions of the SAME field.

    Resamples both to a common physical grid, then asks, symmetrically, what fraction of
    one network's centreline lies within tol_um of the other's. Reported as precision /
    recall / F alongside each network's length, because a method that simply draws MORE
    network would otherwise score well on one direction alone.

    This needs no ground truth. It is the corpus's paired-detector design used as a
    validation signal: the same physical cracks must be found under both detectors.
    """
    from skimage.morphology import skeletonize
    from skimage.transform import resize
    upp = max(upp_a, upp_b)
    ext_y = min(net_a.shape[0] * upp_a, net_b.shape[0] * upp_b)
    ext_x = min(net_a.shape[1] * upp_a, net_b.shape[1] * upp_b)
    H, W = max(int(ext_y / upp), 8), max(int(ext_x / upp), 8)

    def to_common(net, u):
        ny, nx = int(ext_y / u), int(ext_x / u)
        crop = net[:max(ny, 1), :max(nx, 1)]
        r = resize(crop.astype(float), (H, W), order=0, preserve_range=True,
                   anti_aliasing=False) > 0.5
        return skeletonize(r)

    A, B = to_common(net_a, upp_a), to_common(net_b, upp_b)
    if A.sum() == 0 or B.sum() == 0:
        return {"precision": float("nan"), "recall": float("nan"), "F": float("nan"),
                "len_a_um": A.sum() * upp, "len_b_um": B.sum() * upp}
    tol_px = max(tol_um / upp, 1.0)
    dA = ndi.distance_transform_edt(~A)
    dB = ndi.distance_transform_edt(~B)
    prec = float((dB[A] <= tol_px).mean())     # of A's centreline, how much is near B
    rec = float((dA[B] <= tol_px).mean())      # of B's centreline, how much is near A
    F = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "F": F,
            "len_a_um": float(A.sum() * upp), "len_b_um": float(B.sum() * upp),
            "common_grid": (H, W), "tol_px": tol_px}
