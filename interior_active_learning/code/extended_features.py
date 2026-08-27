"""
Shape/topology measurements that depend on NOTHING dataset-specific -- no
absolute brightness scale, no contrast-stretch convention, no material
assumption. Two uses:

1. As candidate-level ML features (BoundaryRoughness, BranchPointDensity) --
   see experiments/benchmark_extended_features.py for the empirical test of
   whether adding these to the unified model actually helps.
2. As descriptive per-crack MEASUREMENTS for the output CSV (crack_measurements.py)
   -- skeleton length, mean/max width, tortuosity, branch-point count -- the
   kind of numbers a materials-science paper reports about detected cracks,
   independent of whether they're used for classification at all.

Kept in one module so both call sites can never define "branch point" or
"roughness" two different ways. Skeleton LENGTH is in that same category and
was the one that got away: three quantities here divided by the skeleton, and
all three divided by a pixel count rather than a path length. See
skeleton_path_length.
"""
import math

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

# 8-connected neighbor-count kernel for skeleton branch-point detection.
_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

_SQRT2 = math.sqrt(2.0)


def _local_crop(mask_bool, margin=2):
    """Crop a boolean mask to its bounding box (+ a small margin so
    skeletonize/perimeter operations near the crop edge aren't clipped) --
    same performance rationale as every other per-candidate feature function
    in this project: never run a full-frame op per candidate."""
    ys, xs = np.where(mask_bool)
    H, W = mask_bool.shape
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)
    return mask_bool[y0:y1, x0:x1]


def boundary_roughness(mask_bool):
    """Perimeter / convex-hull perimeter. A smooth blob (or a smooth,
    gently-curved crack) has a perimeter close to its convex hull's, so this
    sits near 1.0. A jagged, notched, or branching boundary -- the kind
    fracture surfaces actually have, physically, in most materials -- pushes
    this well above 1.0. Deliberately NOT the same information as Solidity
    (area(region)/area(hull)): a long thin region can be highly solid by
    area while still having a rough, wiggly edge that Solidity barely
    penalizes, because a thin sliver's area difference from its hull is
    small in absolute terms even when the boundary itself is very jagged."""
    local = _local_crop(mask_bool)
    props_list = measure.regionprops(local.astype(np.uint8))
    if not props_list:
        return 1.0
    props = props_list[0]
    perim = props.perimeter if props.perimeter > 0 else 1.0
    convex_img = props.convex_image
    convex_perim = measure.perimeter(convex_img) if convex_img.sum() > 0 else perim
    convex_perim = max(convex_perim, 1.0)
    return float(perim / convex_perim)


def skeleton_path_length(skel):
    """Skeleton length as a sum of EUCLIDEAN STEP LENGTHS, not a pixel count.

    A PIXEL COUNT IS NOT A PATH LENGTH. A diagonal step advances sqrt(2) in
    space but counts as one pixel, so counting pixels understates the length of
    any path that is not axis-aligned -- by up to 1/sqrt(2) for a 45-degree
    crack. That count used to be divided by a Euclidean endpoint distance to
    get Tortuosity, and 819 of the 1503 uncensored cracks in the exported
    measurement CSVs came out BELOW 1.0, which is geometrically impossible: a
    path cannot be shorter than the straight line between its own endpoints.
    The measured floor was 0.759, against the 0.707 the bug allows. The same
    count was the denominator of MeanWidth_px (width OVERstated by up to
    sqrt(2) for a diagonal crack) and of branch-point density, so all three
    quantities were partly reporting crack DIRECTION: Spearman between the old
    Tortuosity and the angular offset from the nearest pixel axis was -0.648.

    Summed over the skeleton's ADJACENCY EDGES rather than along one ordered
    path, so one measure serves a branching crack network and a single
    unbranched crack alike -- MeanWidth_px and branch density are emitted for
    both, and a length that changed definition with topology would put a
    discontinuity in the headline length column. For an unbranched skeleton the
    two agree exactly, not approximately: such a skeleton has exactly 2 pixels
    of degree 1 and the rest of degree 2, so it has n-1 edges over n pixels,
    so it is a tree, so it is a single path -- the edge set IS the path's step
    set. Checked against the ordered-path reference implementation
    (crack_export/tools/skeleton_metrics.py, order_path + polyline_length): bit
    for bit identical on 1052 skeleton branches, and agreeing to 3e-14 px on
    the straight-crack probes in the regression test, where the two sum the
    same steps in a different order.

    A DIAGONAL EDGE WHOSE TWO PIXELS SHARE AN ORTHOGONAL NEIGHBOUR IN THE
    SKELETON IS NOT COUNTED. That neighbour is one of the other two corners of
    the same 2x2 window, so the diagonal cuts across a staircase corner or a
    4-connected junction and runs parallel to a route already counted. These
    are not rare and are not a theoretical worry: measured, one per 20
    skeleton pixels on long wandering crack-like ridges and one per 7 on dense
    branching networks -- they cluster at junctions, which is exactly where a
    plain sum over every 8-neighbour pair would invent sqrt(2) of length per
    arm pair that no path traverses. A 4-armed junction reads 4.0 with the
    pruning and 9.66 without it. The pruning can never fire on an unbranched
    skeleton: it would require a 3-cycle, and a path has none. Enumerated
    rather than argued -- over all 65536 patterns in a 4x4 window plus 120000
    random 6x6 patterns, pruning never disconnected a set that 8-connectivity
    had joined, and of the 12411 patterns that satisfy the emit condition for
    Tortuosity not one produced a value below 1 (minimum exactly 1.0).

    RESIDUAL BIAS, LEFT IN PLACE AND BOUNDED. This is a chain-code length, and
    a digital straight line at angle t to the pixel axes is a staircase of
    (cos t - sin t) orthogonal and (sin t) diagonal steps per unit length, so
    the measure reads cos t + (sqrt(2) - 1) sin t. Exact at 0 and 45 degrees,
    +8.24% at worst (22.5 degrees). So a dead-straight crack at 20 degrees
    still reports a tortuosity near 1.08 rather than 1.00, and lengths at
    intermediate angles are overstated by the same factor. Two things make
    that a different class of defect from the one this replaces: it is bounded
    by a known constant instead of scaling to 29%, and it errs in the
    direction that KEEPS Tortuosity >= 1 rather than the direction that
    produces impossible values. Removing it needs either a corner-count
    estimator (Vossepoel-Smeulders and friends), which is unbiased to well
    under a percent but multiplies orthogonal runs by 0.98 and so gives a
    straight axis-aligned crack a tortuosity of 0.98 -- surrendering the
    invariant that motivated this fix -- or polyline simplification, which
    works (0.7% residual at 20 degrees, and >= 1 survives because a simplified
    polyline keeps the same two endpoints) but makes tortuosity a function of a
    chosen length scale. Neither is adopted without a decision on that
    trade-off; see the tortuosity regression test, which asserts the >= 1
    invariant unconditionally and the straight-line reading against the bound
    above rather than against 1.00.
    """
    S = np.asarray(skel, dtype=bool)
    if S.ndim != 2 or S.size == 0 or S.sum() < 2:
        return 0.0
    n_orth = int((S[:, :-1] & S[:, 1:]).sum()) + int((S[:-1, :] & S[1:, :]).sum())
    # Both diagonals of every 2x2 window, each kept only when neither of the
    # window's other two corners is skeleton -- i.e. only when the diagonal is
    # the sole route between its endpoints.
    main = S[:-1, :-1] & S[1:, 1:] & ~(S[:-1, 1:] | S[1:, :-1])
    anti = S[:-1, 1:] & S[1:, :-1] & ~(S[:-1, :-1] | S[1:, 1:])
    return float(n_orth + _SQRT2 * int(main.sum() + anti.sum()))


def skeleton_stats(mask_bool):
    """Skeletonize the region and return (skeleton_path_length, branch_point_count,
    endpoint_count, skeleton_mask_local, local_crop_mask, skeleton_pixel_count).

    The first element is a LENGTH in pixel units (see skeleton_path_length --
    diagonal steps carry sqrt(2)); the last is the raw pixel COUNT, kept only
    for the degeneracy guards that ask "are there enough skeleton pixels for
    this to mean anything", which is a question about pixels and not about
    length. Everything that divides BY the skeleton -- tortuosity, mean width,
    branch density -- must use the length. Branch points are
    skeleton pixels with >=3 skeleton neighbors (8-connectivity) -- a fork in
    the medial axis, which happens where a crack branches or where two
    candidates' shapes merge into one blob. Endpoints have exactly 1
    neighbor. A simple, unbranched crack skeletonizes to a single path with
    exactly 2 endpoints and 0 branch points; real branching crack networks
    (or two blobs fused into one candidate) show >=1 branch point."""
    local = _local_crop(mask_bool)
    skel = morphology.skeletonize(local)
    skel_px = int(skel.sum())
    if skel_px == 0:
        return 0.0, 0, 0, skel, local, 0
    skel_len = skeleton_path_length(skel)
    neighbor_count = ndi.convolve(skel.astype(int), _NEIGHBOR_KERNEL, mode="constant", cval=0)
    branch_points = int(((neighbor_count >= 3) & skel).sum())
    endpoints = int(((neighbor_count == 1) & skel).sum())
    return skel_len, branch_points, endpoints, skel, local, skel_px


def branch_point_density(mask_bool):
    """Branch points per 100px of skeleton length -- normalized so a big
    crack and a small crack with the same TOPOLOGY (e.g. both simple,
    unbranched paths) score the same, rather than the raw count scaling
    with size. A purely blob-shaped artifact typically skeletonizes to a
    short stub with 0 branch points (density 0); an interconnected crack
    network scores higher. General topological property -- doesn't depend
    on brightness, contrast, or which material/instrument produced the
    image, so it should transfer to other SEM crack-detection datasets
    better than the brightness-based features do.

    Per 100px of skeleton LENGTH, which is what the name says and what this
    now divides by. It used to divide by the skeleton PIXEL COUNT, so the same
    branching topology scored up to sqrt(2) higher when the crack happened to
    run diagonally -- an orientation term in a feature whose whole claim is
    that it is topological and therefore transferable."""
    skel_len, branch_points, _, _, _, skel_px = skeleton_stats(mask_bool)
    # Guarded on the PIXEL COUNT: this is the "too small to skeletonize into
    # anything meaningful" test the feature has always applied, and it is a
    # statement about how many pixels there are, not about how long they run.
    if skel_px < 3 or skel_len <= 0:
        return 0.0
    return float(branch_points / skel_len * 100)


def crack_shape_measurements(mask_bool):
    """Full descriptive measurement set for ONE final crack region (used by
    crack_measurements.py's per-image report, not the ML feature set) --
    physical numbers a materials-science reader actually wants: length,
    width, tortuosity, branching, orientation. Returns a dict; NaN for any
    quantity that isn't well-defined for this region's topology (e.g.
    tortuosity needs exactly 2 skeleton endpoints)."""
    local = _local_crop(mask_bool, margin=0)
    props_list = measure.regionprops(local.astype(np.uint8))
    props = props_list[0] if props_list else None
    area = int(mask_bool.sum())

    skel_len, branch_points, endpoints, skel, skel_local, skel_px = skeleton_stats(mask_bool)
    # area / LENGTH, not area / pixel count. Dividing by the count overstated the
    # width of a diagonal crack by up to sqrt(2), because the count is the shorter
    # of the two numbers for exactly the cracks whose skeleton runs off-axis.
    mean_width = (area / skel_len) if skel_len > 0 else float(np.sqrt(area))

    # Max width: 2x the largest distance-to-background value found ON the
    # skeleton (medial-axis radius) -- more robust to a jagged boundary than
    # measuring width at one arbitrary cross-section.
    max_width = float(np.sqrt(area))  # fallback for a to-degenerate-to-skeletonize region
    # PIXEL COUNT, not length: this reads the distance transform AT the skeleton
    # pixels, so what it needs is at least one pixel to read. A one-pixel
    # skeleton has a well-defined medial-axis radius and a path length of zero.
    if skel_px > 0:
        # The distance transform MUST be computed on the same crop the skeleton
        # was built from. skeleton_stats() uses _local_crop() with its DEFAULT
        # margin, so passing margin=0 here produced a differently-shaped array,
        # the shape guard below always failed, and max_width silently fell back
        # to sqrt(area) for every region ever measured -- meaningless for an
        # elongated crack (a 10x140 px crack reported 37.4 instead of 10).
        dist = ndi.distance_transform_edt(_local_crop(mask_bool))
        if dist.shape == skel.shape:
            on_skel = dist[skel]
            if len(on_skel):
                max_width = float(on_skel.max() * 2)

    tortuosity = float("nan")
    if endpoints == 2 and branch_points == 0 and skel_px > 1:
        ys, xs = np.where(skel)
        # endpoints are the 2 skeleton pixels with exactly 1 neighbor
        neighbor_count = ndi.convolve(skel.astype(int), _NEIGHBOR_KERNEL, mode="constant", cval=0)
        ey, ex = np.where((neighbor_count == 1) & skel)
        if len(ey) == 2:
            straight_dist = float(np.hypot(ey[0] - ey[1], ex[0] - ex[1]))
            if straight_dist > 0:
                # BOTH SIDES OF THIS RATIO ARE NOW EUCLIDEAN. The numerator was a
                # pixel count and the denominator a Euclidean distance, so the
                # units did not match and the result was biased down by up to
                # 1/sqrt(2) for a diagonal crack -- which is how 819 exported
                # values ended up below the geometric floor of 1. Here the
                # skeleton is unbranched (2 endpoints, 0 branch points), so
                # skel_len is the length of the single path between exactly these
                # two endpoints and the triangle inequality makes the ratio >= 1
                # by construction rather than by luck.
                tortuosity = float(skel_len / straight_dist)

    return {
        "Area_px": area,
        # A LENGTH, so it is no longer an integer. Was int(skel.sum()); a count of
        # pixels is not the distance travelled through them.
        "SkeletonLength_px": round(skel_len, 2),
        "MeanWidth_px": round(mean_width, 2),
        "MaxWidth_px": round(max_width, 2),
        "Tortuosity": round(tortuosity, 3) if tortuosity == tortuosity else "",  # "" not nan, for clean CSV
        "BranchPointCount": branch_points,
        # NAMED FOR WHAT THEY ARE. These are the axes of the ellipse with the same
        # second moments as the region -- NOT crack length. A reader who takes
        # "MajorAxisLength" as the crack's length gets a different quantity from the one
        # the literature reports, and for a curved crack a much smaller one:
        # SkeletonLength_px above is the actual path length, and Tortuosity is their
        # ratio. Renamed rather than documented, because a misleading column name in a
        # CSV outlives any docstring.
        "EllipseMajorAxis_px": round(float(props.axis_major_length), 2) if props else "",
        "EllipseMinorAxis_px": round(float(props.axis_minor_length), 2) if props else "",
        "Orientation_deg": round(float(np.degrees(props.orientation)), 2) if props else "",
        "BoundaryRoughness": round(boundary_roughness(mask_bool), 3),
    }
