"""
Automatic crack detection for SEM/TXM micrographs, with ML-based
artifact rejection.

Pipeline
--------
1. Load the (often 16-bit) grayscale micrograph and rescale it to 8-bit
   using a percentile stretch.
2. Flatten uneven illumination with a large-kernel background subtraction.
3. Segment "dark" pixels with Otsu thresholding on the flattened image,
   then clean with morphology. This stage is deliberately permissive
   (high recall) -- it will catch real cracks AND artifacts (pores,
   inclusions, grain-boundary pits).
4. Compute a Frangi "vesselness" map on the flattened image. This is a
   Hessian-eigenvalue-based ridge filter (the same math used to segment
   blood vessels in medical imaging): it responds strongly to thin,
   continuous, curvilinear structures like cracks, and weakly to round
   blobs like pores -- exactly the discrimination we need.
5. For every candidate region from step 3, compute a feature vector
   (elongation, solidity, eccentricity, extent, mean darkness, mean
   vesselness response, ...).
6. Classify each candidate as crack vs. artifact using one of:
     - "auto"  (default): unsupervised Gaussian-mixture clustering on the
       standardized features, with the more "crack-like" cluster
       (higher elongation/vesselness, lower solidity) auto-selected.
       No manual labeling needed.
     - "train": fit a RandomForestClassifier from a CSV where you've
       manually corrected the IsCrack column of a previous run, and
       save the model so it can be reused on the rest of your image
       series (see --model).
     - "apply": load a previously trained model and use it directly --
       no re-fitting, fully deterministic, good for batch-processing
       many images consistently.
7. Write a pure black-and-white image (crack = black, background =
   white) using only the *kept* regions, a QC overlay (kept regions in
   red & numbered, rejected candidates in cyan), and a CSV with every
   candidate's features + classification, so you can review / correct
   it and feed it back into --mode train.

Usage
-----
    python3 detect_cracks.py INPUT.tif                       # auto mode
    python3 detect_cracks.py INPUT.tif --mode train \\
        --train-labels reviewed_candidates.csv --model crack_clf.joblib
    python3 detect_cracks.py INPUT2.tif --mode apply --model crack_clf.joblib

Run with -h for all options.
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import tifffile
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage import exposure, filters, measure, morphology
from skimage.graph import route_through_array
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image, ImageDraw

FEATURE_COLUMNS = [
    "LogArea",
    "Elongation",
    "Solidity",
    "Eccentricity",
    "Extent",
    "Circularity",
    "MeanDarkness",
    "MeanVesselness",
]


def get_unique_filename(base_path):
    """Never clobber a previous run's output -- append _1, _2, ... instead."""
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    i = 1
    while os.path.exists(f"{root}_{i}{ext}"):
        i += 1
    return f"{root}_{i}{ext}"


def load_as_uint8(image_path, low_pct=1.0, high_pct=99.5):
    img = tifffile.imread(image_path)
    if img.ndim == 3:
        img = img[..., 0]
    lo, hi = np.percentile(img, [low_pct, high_pct])
    img8 = exposure.rescale_intensity(
        img.astype(np.float32), in_range=(lo, hi), out_range=(0, 255)
    ).astype(np.uint8)
    return img8


def _detect_databar_top(img8, search_frac=0.25, window=10, std_ratio=0.4, mean_ratio=0.85):
    """
    Some SEM exports (e.g. the raw instrument captures in this dataset) have
    an info bar burned in at the bottom (scale bar, detector, voltage, ...).
    Returns the first row of that bar, or h if there is none.

    Two independent signals, and the higher (earlier) one wins, because either
    alone leaves real bars in the frame.

    1. STATISTICAL. A databar background is close to solid, so both the row-wise
       mean brightness AND the row-wise standard deviation drop sharply at the
       transition. Comparing each row against the max over the preceding `window`
       rows keeps this stable against ordinary row-to-row noise. (An earlier
       version looked for a spike in horizontal gradient energy instead, which
       assumed the specimen surface is smooth -- false on busy, high-contrast
       captures where grain texture has edge energy as high as the bar.)

    2. GEOMETRIC. The panel is a RECTANGLE, so its top edge changes brightness
       across essentially the whole width at a single row -- unlike a jagged
       specimen boundary or a dark void, which change locally.

    Signal 1 alone missed five captures here (MAR_Amb_Cast_CBS_0002/0005,
    MAR_Amb_Cast_ETD_0002/0005, MAR_Amb_HIP_CBS_0003): their panel is a flat
    MID-GREY, not dark enough for mean_ratio to trip, so 240 rows of bar stayed in
    frame and the model detected the bar's own text as cracks -- 243,405 red pixels
    on one of them. Signal 2 catches all five. Measured across all 62 images, the
    60% width threshold moves the crop on exactly those five and nothing else.
    """
    h, w = img8.shape
    search_start = int(h * (1 - search_frac))
    rows = img8[search_start:].astype(np.float32)
    if rows.shape[0] < window + 5:
        return h

    bar_top_stat = h
    row_means = rows.mean(axis=1)
    row_stds = rows.std(axis=1)
    for i in range(window, len(row_means)):
        baseline_std = row_stds[i - window:i].max()
        baseline_mean = row_means[i - window:i].max()
        if (baseline_std > 3 and row_stds[i] < baseline_std * std_ratio
                and row_means[i] < baseline_mean * mean_ratio):
            bar_top_stat = search_start + i
            break

    bar_top_edge = h
    best_frac, best_row = 0.0, None
    for i in range(1, rows.shape[0]):
        frac = float((np.abs(rows[i] - rows[i - 1]) > 10).mean())
        if frac > best_frac:
            best_frac, best_row = frac, search_start + i
    # >=20 rows below it, so a near-bottom noise spike cannot masquerade as a panel
    if best_frac >= 0.60 and best_row is not None and h - best_row >= 20:
        bar_top_edge = best_row

    return min(bar_top_stat, bar_top_edge)


def find_field_of_view(img8, bright_thresh=12, shrink_frac=0.16, min_keep_frac=0.2):
    """
    Auto-detect the usable sample area, excluding any burned-in info bar and
    any circular aperture vignette (dark corners around a round field of
    view) -- both are common in raw SEM exports and would otherwise be
    misread as one giant 'crack' by the darkness threshold. Images that are
    already tightly cropped (no bar, no vignette) pass through unchanged.
    Returns (x0, y0, x1, y1).
    """
    h, w = img8.shape
    bar_top = _detect_databar_top(img8)
    workspace = img8[:bar_top].astype(np.float32)
    row_ok = np.where(workspace.mean(axis=1) > bright_thresh)[0]
    col_ok = np.where(workspace.mean(axis=0) > bright_thresh)[0]
    if len(row_ok) == 0 or len(col_ok) == 0:
        return 0, 0, w, bar_top

    y0, y1 = int(row_ok.min()), int(row_ok.max())
    x0, x1 = int(col_ok.min()), int(col_ok.max())
    fills_workspace = (x1 - x0) > 0.97 * w and (y1 - y0) > 0.97 * bar_top

    # A genuine circular aperture vignette darkens all FOUR corners
    # symmetrically (that's what "round field of view" means). A large but
    # ordinary dark REGION -- a sample edge, a void, a low-signal patch near
    # one side -- can just as easily pull the row/col brightness bounding
    # box in from one or two sides without being vignetting at all. Checked
    # directly on this dataset: of the images that failed the fills_workspace
    # check, none had all four corners dark in a way consistent with a round
    # cutoff (usually only one or two corners, or a band along one edge) --
    # confirmed by looking at actual corner brightness, not inferred. Require
    # that signature explicitly before committing to the inscribed-square
    # crop; otherwise this dataset's real dark regions get mistaken for
    # vignetting and lose 50-70% of a perfectly good frame.
    corner_size = max(20, int(0.03 * min(w, bar_top)))
    corners = [
        workspace[:corner_size, :corner_size].mean(),
        workspace[:corner_size, -corner_size:].mean(),
        workspace[-corner_size:, :corner_size].mean(),
        workspace[-corner_size:, -corner_size:].mean(),
    ]
    looks_like_round_vignette = all(c < bright_thresh for c in corners)

    if fills_workspace or not looks_like_round_vignette:
        # No vignette (or not the round kind this crop assumes) -- exclude
        # only the databar, keep the rest of the frame as-is.
        x0, y0, x1, y1 = 0, 0, w, bar_top
    else:
        # Treat the bright blob as a circle (its bounding box can be
        # asymmetric if the bar clipped one side) and crop to the largest
        # square inscribed in that circle. A per-axis shrink can still leave
        # a corner poking into the vignette when the box isn't square (e.g.
        # the bar ate into the height but not the width); an inscribed
        # square is geometrically guaranteed to clear every corner.
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        radius = max(y1 - y0, x1 - x0) / 2
        half_side = radius / math.sqrt(2) * (1 - shrink_frac)
        y0, y1 = int(cy - half_side), int(cy + half_side)
        x0, x1 = int(cx - half_side), int(cx + half_side)
        y0, x0 = max(y0, 0), max(x0, 0)
        y1, x1 = min(y1, bar_top), min(x1, w)

    if (x1 - x0) * (y1 - y0) < min_keep_frac * w * h:
        return 0, 0, w, bar_top  # detection looked wrong -- fall back safely
    return x0, y0, x1, y1


def flatten_background(img8, sigma=40, low_pct=0.5, high_pct=99.5):
    """
    Subtract a heavily blurred copy to remove slow illumination drift, then
    rescale to 8-bit using a PERCENTILE stretch rather than raw min/max.
    Min/max is fragile here: a handful of outlier pixels (a hot pixel, dust,
    or the crack itself sitting at 0) can dominate the range and compress
    all of the real background texture into a narrow band, which then
    breaks Otsu thresholding downstream (seen on low-contrast ETD captures).
    """
    background = filters.gaussian(img8, sigma=sigma, preserve_range=True)
    flat = img8.astype(np.float32) - background
    lo, hi = np.percentile(flat, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = flat.min(), flat.max()
    flat = exposure.rescale_intensity(flat, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)
    return flat


def segment_dark_regions(flat, denoise_sigma=1.0, mad_k=5.0, img8=None, absolute_dark_thresh=10):
    """
    Otsu assumes a roughly bimodal histogram (background vs. crack). That
    holds for high-contrast captures, but low-contrast/noisy images (seen on
    this dataset's ETD captures) have an essentially unimodal histogram --
    Otsu still forces a split, often near the median, flagging >50% of the
    image as "dark" and blowing up downstream. A robust median/MAD outlier
    threshold doesn't assume bimodality; taking whichever of the two
    thresholds is lower keeps Otsu's better localization on well-behaved
    images while capping the damage when it misfires.

    This RELATIVE (flattened-image) threshold has its own blind spot: a wide
    open crack/void, once background-flattening subtracts out its own
    surroundings, gets rescaled to a near-neutral value (measured: pixels
    that are genuinely pitch-black, raw value 0-4, land at ~122 out of 255
    after flattening -- indistinguishable from generic mid-gray background).
    Flattening is a high-pass filter; a void wider than the flattening blur
    radius (sigma=40, ~effective 120-160px) IS a low-frequency feature and
    gets removed along with the lighting drift it was meant to target. Thin
    cracks survive fine since they're far smaller than that blur radius.

    The fix is an OR with an ABSOLUTE darkness test on the un-flattened,
    percentile-stretched image (img8): true voids are unambiguously near-0
    there regardless of how wide they are, so this catches what the relative
    test structurally cannot, without needing the relative test to change.
    """
    smooth = filters.gaussian(flat, sigma=denoise_sigma, preserve_range=True)
    otsu_thresh = filters.threshold_otsu(smooth)
    median = np.median(smooth)
    mad = np.median(np.abs(smooth - median))
    robust_thresh = median - mad_k * mad * 1.4826
    thresh = min(otsu_thresh, robust_thresh)
    relative_mask = smooth < thresh
    if img8 is None:
        return relative_mask
    absolute_mask = img8 < absolute_dark_thresh
    return relative_mask | absolute_mask


def clean_mask(mask, open_radius=1, close_radius=3, min_area_px=15):
    mask = morphology.remove_small_objects(mask, max_size=min_area_px)
    mask = morphology.opening(mask, morphology.disk(open_radius))
    mask = morphology.closing(mask, morphology.disk(close_radius))
    mask = ndi.binary_fill_holes(mask)
    return mask


def compute_vesselness(flat, sigma_min=1, sigma_max=6):
    """Frangi ridge/vesselness filter: high on thin curvilinear structures
    (cracks), low on round blobs (pores/inclusions)."""
    ves = filters.frangi(
        flat.astype(np.float64) / 255.0,
        sigmas=range(sigma_min, sigma_max, 1),
        black_ridges=True,
    )
    if ves.max() > 0:
        ves = ves / ves.max()
    return ves


def exclude_border_background(clean, vesselness, min_area_frac=0.02, max_vesselness=0.0025, bridge_px=40):
    """
    Some captures are edge/fracture-surface views rather than flat polished
    cross-sections -- a real, physical sample boundary runs through the
    frame, with genuinely empty background beyond it. That background reads
    as "dark" to segment_dark_regions exactly like a real crack does, but
    it is NOT a circular aperture vignette (find_field_of_view already
    excludes that shape) -- it's an irregular, jagged boundary that no
    rectangular/circular crop can remove without also cutting away real
    sample area.

    Confirmed directly on real images from this dataset (not a theoretical
    concern): a background region like this is large, touches the frame's
    edge, and -- critically -- has essentially zero vesselness, since it
    has no material structure at all. A real crack, even a huge one, keeps
    a meaningfully higher vesselness signature because it has actual
    walls/edges for the Frangi filter to respond to (verified: a real
    695,894px crack-adjacent background region measured ~0.0002 mean
    vesselness on one test image, while confirmed real cracks of comparable
    size measured at least an order of magnitude higher). Both conditions
    are required together -- area/border-touching alone would also catch a
    real crack that happens to reach the frame's edge.

    A single connected-component pass isn't enough on its own: the jagged
    sample edge often pokes a thin bright ridge all the way to the frame
    border, which -- at the pixel level -- splits what is visually one
    continuous background region into several separate components. Checked
    directly: a same-background chunk cut off this way by one such ridge
    kept its neighbor's near-zero vesselness but no longer touched the
    border itself, and survived a border-touching-only version of this
    check untouched. Pass 1 finds the actual border-touching seed region(s);
    pass 2 dilates that seed to bridge exactly this kind of thin-ridge gap
    and absorbs any other low-vesselness component the dilated seed now
    overlaps, before removing the union from the mask.
    """
    labeled = measure.label(clean, connectivity=2)
    h, w = clean.shape
    min_area = min_area_frac * h * w

    # Single regionprops pass, cached -- a busy image can have several
    # thousand candidate regions, and re-deriving a full-image (labeled ==
    # region.label) boolean mask for every one of them on every round of
    # the absorption loop below turned out to dominate runtime completely
    # (measured: >10 minutes/image on this dataset's larger captures).
    # region.image + region.slice give the same per-region mask cropped to
    # its own bounding box -- every check below operates on that small crop
    # instead of the full multi-megapixel array.
    regions = []
    for region in measure.regionprops(labeled):
        touches_border = (region.bbox[0] == 0 or region.bbox[2] == h or
                           region.bbox[1] == 0 or region.bbox[3] == w)
        regions.append((region.label, region.area, region.slice, region.image, touches_border))

    seed = np.zeros_like(clean)
    for _, area, sl, img, touches_border in regions:
        if area < min_area or not touches_border:
            continue
        if vesselness[sl][img].mean() < max_vesselness:
            seed[sl][img] = True

    if not seed.any():
        return clean

    # Pass 2 deliberately does NOT reuse min_area here -- a satellite chunk
    # cut off from the seed by a thin ridge can be far smaller than the
    # seed itself (confirmed: a real 112,738px satellite, only 0.45% of the
    # image, sat right against a 695,894px seed on one test image and was
    # silently skipped the first time this reused the same 2%-of-image
    # threshold for both passes). Any size big enough to have been a
    # genuine candidate at all is eligible for absorption once it's next to
    # the dilated seed with the same near-zero vesselness.
    # Absorption is iterative, not a single dilate-and-check pass: a chain
    # of several separate satellite fragments (each cut off from its
    # neighbor by its own thin ridge) needs one round per link, since a
    # fragment 2 links away from the original seed doesn't overlap the
    # dilated seed until the fragment between them has already been
    # absorbed on a prior round. Capped at a handful of rounds -- a chain
    # long enough to need more than that is rare enough to leave for the
    # paint app's own erase tool rather than pay for many more full-image
    # dilations chasing an ever-longer tail.
    background = seed.copy()
    absorbed_labels = set()
    for _ in range(5):
        # Dilating by a disk of radius r is by definition every pixel within
        # Euclidean distance r of the set, so an EDT threshold computes the
        # identical mask -- in O(N) rather than O(N * r^2). With bridge_px=40
        # the disk is a ~5000-element structuring element applied to 25M
        # pixels, up to 5 rounds, and it dominated the whole pipeline.
        # Measured on MAR_Amb_AS_ETD_0002 (4096x6144, 695,894 px seed):
        #   one dilation      59.54s -> 0.78s   (77x)
        #   this function    784.86s -> 5.81s  (135x)
        # with 0 differing pixels and an identical 522,357 px kept, checked
        # against the disk version rather than assumed from the definitions.
        dilated = ndi.distance_transform_edt(~background) <= bridge_px
        absorbed_this_round = False
        for lbl, area, sl, img, touches_border in regions:
            if lbl in absorbed_labels:
                continue
            if dilated[sl][img].any() and vesselness[sl][img].mean() < max_vesselness:
                background[sl][img] = True
                absorbed_labels.add(lbl)
                absorbed_this_round = True
        if not absorbed_this_round:
            break

    out = clean.copy()
    out[background] = False
    return out


def extract_candidates(mask, flat, vesselness, min_area_px=40):
    # measure.label() numbers EVERY raw connected component in the mask,
    # including tiny noise specks that get discarded by the min-area check --
    # so a surviving candidate could keep an arbitrary, non-sequential ID
    # (e.g. "9") purely because 8 smaller regions before it in scan order
    # happened to get filtered out. Surviving candidates are renumbered
    # 1..N (in the same scan order), and `labeled` is remapped to match, so
    # the Label a user sees always reflects "the Nth real candidate," not a
    # leftover ID from regions that no longer exist.
    return region_features_from_labeled(
        measure.label(mask, connectivity=2), flat, vesselness, min_area_px)


def region_features_from_labeled(raw_labeled, flat, vesselness, min_area_px=40):
    """Feature extraction for an ALREADY-labelled region map.

    Split out of extract_candidates so that anything scoring regions this
    pipeline did not segment itself -- SAM masks, in particular -- goes
    through the exact same feature definitions instead of a reimplementation.
    Two of these are easy to get subtly wrong and both have bitten this
    project: MeanDarkness is the mean of (255 - flat), i.e. INVERTED so that
    larger means darker, and MeanVesselness is quantised to uint8 before
    averaging and then rescaled. Computing either the obvious-looking way
    silently shifts every score. One code path removes the possibility.
    """
    shape_props = measure.regionprops(raw_labeled)
    dark_props = {p.label: p for p in measure.regionprops(raw_labeled, intensity_image=255 - flat)}
    ves_props = {p.label: p for p in measure.regionprops(raw_labeled, intensity_image=(vesselness * 255).astype(np.uint8))}

    records = []
    remap = np.zeros(raw_labeled.max() + 1, dtype=np.int32)
    next_label = 1
    for p in shape_props:
        if p.area < min_area_px:
            continue
        new_label = next_label
        next_label += 1
        remap[p.label] = new_label
        minor = p.axis_minor_length if p.axis_minor_length > 0 else 0.5
        elongation = p.axis_major_length / minor
        perim = p.perimeter if p.perimeter > 0 else 1.0
        circularity = 4 * np.pi * p.area / (perim ** 2)
        y0, x0, y1, x1 = p.bbox
        records.append(
            {
                "Label": new_label,
                "Area": p.area,
                "Width": x1 - x0,
                "Height": y1 - y0,
                "X": p.centroid[1],
                "Y": p.centroid[0],
                "MajorAxisLength": p.axis_major_length,
                "MinorAxisLength": p.axis_minor_length,
                "Elongation": elongation,
                "Solidity": p.solidity,
                "Eccentricity": p.eccentricity,
                "Extent": p.extent,
                "Circularity": min(circularity, 1.0),
                "MeanDarkness": dark_props[p.label].intensity_mean,
                "MeanVesselness": ves_props[p.label].intensity_mean / 255.0,
                "LogArea": np.log10(p.area),
                "CrackLength_px": p.axis_major_length,
                "CrackHeight_px": y1 - y0,
                "AreaCoveragePct": 100.0 * p.area / (raw_labeled.shape[0] * raw_labeled.shape[1]),
                "HWRatio": (y1 - y0) / (x1 - x0),
            }
        )
    columns = ["Label", "Area", "Width", "Height", "X", "Y", "MajorAxisLength",
               "MinorAxisLength", "Elongation", "Solidity", "Eccentricity",
               "Extent", "Circularity", "MeanDarkness", "MeanVesselness", "LogArea",
               "CrackLength_px", "CrackHeight_px", "AreaCoveragePct", "HWRatio"]
    df = pd.DataFrame.from_records(records, columns=columns)
    labeled = remap[raw_labeled]  # discarded (sub-min-area) regions map to 0/background
    return labeled, df


def classify_auto(df, force_keep_length=150, force_keep_area=50000):
    """Unsupervised split into crack / artifact clusters via a 2-component
    Gaussian mixture on standardized shape+intensity+vesselness features.
    The cluster with higher elongation & vesselness and lower solidity is
    picked as the 'crack' cluster -- no manual labels required."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    if len(df) < 2:
        # Too few candidates to cluster -- fall back to the safety nets alone.
        df = df.copy()
        df["IsCrack"] = ((df["MajorAxisLength"] >= force_keep_length) |
                          (df["Area"] >= force_keep_area)) if len(df) else pd.Series(dtype=bool)
        df["CrackProbability"] = df["IsCrack"].astype(float)
        return df

    X = StandardScaler().fit_transform(df[FEATURE_COLUMNS].values)
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=5)
    cluster = gmm.fit_predict(X)
    proba = gmm.predict_proba(X)

    scores = []
    for c in (0, 1):
        sub = df[cluster == c]
        if len(sub) == 0:
            scores.append(-np.inf)
            continue
        score = sub["Elongation"].mean() + sub["MeanVesselness"].mean() * 5 - sub["Solidity"].mean()
        scores.append(score)
    crack_cluster = int(np.argmax(scores))

    df = df.copy()
    df["IsCrack"] = cluster == crack_cluster
    df["CrackProbability"] = proba[:, crack_cluster]

    # Safety nets: obviously long OR obviously huge candidates are kept
    # regardless of cluster assignment. Length catches a genuine large crack
    # that clustering noise might drop; area catches a wide-open void (now
    # detected via the absolute-darkness test) that's a solid, high-solidity
    # blob unlike any thin crack shape the clustering would naturally group
    # as "crack" -- artifacts (pores, inclusions) never get this big.
    df.loc[(df["MajorAxisLength"] >= force_keep_length) | (df["Area"] >= force_keep_area), "IsCrack"] = True
    return df


def classify_with_model(df, model_path, proba_threshold=None, force_keep_area=50000):
    """proba_threshold=None means "use whatever threshold this model was
    calibrated at", read from the bundle and falling back to 0.5.

    The threshold belongs to the model, not to the call site. A bundle that carries a
    calibrated threshold should be run at it: crack_classifier_v3_weighted.joblib is
    calibrated at 0.5615, chosen on a held-out image to match the previous model's recall
    while cutting its false positives, and running that bundle at a hardcoded 0.5 instead
    trades the gain away silently.

    WHAT THE DEPLOYED BUNDLE ACTUALLY DOES, because this docstring used to imply
    otherwise. models/crack_classifier.joblib has NO "threshold" key -- keys are
    clf, scaler, feature_names, sklearn_version and the loio_* baseline record -- so it
    falls back and the shipped operating point is exactly 0.5. That 0.5 is a library
    default reached by omission, not a calibration decision, and it is the number every
    crack count and crack length in this repo rests on. It is invisible in any figure it
    produces, which is why experiments/threshold_sensitivity.py measures how far the
    published quantities move across 0.3-0.7 rather than leaving it as an assumption.

    Do not "fix" this by hardcoding 0.5615 here. That threshold was calibrated for the
    v3_weighted bundle on its own held-out image and means nothing applied to a different
    classifier; the fallback is correct, its consequences are the thing worth documenting.
    """
    import joblib

    bundle = joblib.load(model_path)
    if proba_threshold is None:
        proba_threshold = bundle.get("threshold", 0.5)
    scaler, clf = bundle["scaler"], bundle["clf"]
    df = df.copy()
    if len(df) == 0:
        df["IsCrack"] = pd.Series(dtype=bool)
        df["CrackProbability"] = pd.Series(dtype=float)
        return df
    X = scaler.transform(df[FEATURE_COLUMNS].values)
    proba = clf.predict_proba(X)[:, 1]
    df["CrackProbability"] = proba
    df["IsCrack"] = proba >= proba_threshold

    # A wide-open void (now caught by the absolute-darkness test) can be one
    # giant, high-solidity blob -- nothing like the thin/elongated shapes the
    # model learned "crack" from, since this candidate type didn't exist
    # before that fix. No training data means no reliable model opinion here;
    # a solid dark region this large is essentially never anything BUT a real
    # crack/void (artifacts -- pores, inclusions -- don't get this big), so it
    # gets force-kept the same way an obviously-long thin candidate does.
    df.loc[df["Area"] >= force_keep_area, "IsCrack"] = True
    return df


def train_model(df, train_labels_path, model_path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    import joblib

    labels_df = pd.read_csv(train_labels_path)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    if "Label" not in labels_df.columns or "IsCrack" not in labels_df.columns:
        raise ValueError("--train-labels CSV must have 'Label' and 'IsCrack' columns")

    merged = df.merge(labels_df[["Label", "IsCrack"]], on="Label", how="inner", suffixes=("", "_manual"))
    if len(merged) < 10:
        raise ValueError(f"Only {len(merged)} labeled rows matched by Label -- need more for training")

    y = merged["IsCrack"].astype(bool).values
    X_raw = merged[FEATURE_COLUMNS].values
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0)
    if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 3:
        scores = cross_val_score(clf, X, y, cv=min(5, min(np.bincount(y))))
        print(f"Cross-validated accuracy on {len(y)} labeled examples: {scores.mean():.3f} +/- {scores.std():.3f}")
    clf.fit(X, y)

    importances = sorted(zip(FEATURE_COLUMNS, clf.feature_importances_), key=lambda t: -t[1])
    print("Feature importances:")
    for name, imp in importances:
        print(f"  {name:16s} {imp:.3f}")

    joblib.dump({"scaler": scaler, "clf": clf, "feature_names": FEATURE_COLUMNS}, model_path)
    print(f"Saved trained model: {model_path}")

    # Apply the freshly trained model back to every candidate in this image.
    df = df.copy()
    proba = clf.predict_proba(scaler.transform(df[FEATURE_COLUMNS].values))[:, 1]
    df["CrackProbability"] = proba
    df["IsCrack"] = proba >= 0.5
    return df


def _crop_with_pad(img, x0, y0, x1, y1, pad_value=255):
    h, w = img.shape
    out = np.full((y1 - y0, x1 - x0), pad_value, dtype=img.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    if sx0 < sx1 and sy0 < sy1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def save_review_sheets(img8, df, out_dir, base, cols=10, rows_per_page=15, thumb_px=110):
    """
    Contact sheets of every candidate cropped from the ORIGINAL image, so you
    can quickly eyeball which ones are mislabeled without hunting for tiny
    dots in the full-resolution overlay.

    Border color = current classification (red = crack, cyan = artifact).
    Sorted with the most AMBIGUOUS candidates (probability closest to 0.5)
    first -- those are the ones most likely to be wrong, so you spend your
    review time where it matters.

    Each thumbnail is captioned "L<Label> p=<prob>" -- <Label> is the number
    to look up in the candidates CSV when you correct its IsCrack value.
    """
    if len(df) == 0:
        return []

    order = (df["CrackProbability"] - 0.5).abs().sort_values().index
    ordered = df.loc[order].reset_index(drop=True)

    per_page = cols * rows_per_page
    cell = thumb_px + 34
    paths = []
    for start in range(0, len(ordered), per_page):
        chunk = ordered.iloc[start:start + per_page].reset_index(drop=True)
        n_rows = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (cols * cell, n_rows * cell), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in chunk.iterrows():
            r, c = divmod(i, cols)
            crop_px = int(np.clip(3 * max(row["Width"], row["Height"]), 50, 300))
            half = crop_px // 2
            cx, cy = int(row["X"]), int(row["Y"])
            patch = _crop_with_pad(img8, cx - half, cy - half, cx + half, cy + half)
            patch_img = Image.fromarray(patch).convert("RGB").resize((thumb_px, thumb_px))
            px, py = c * cell, r * cell
            border_color = (220, 0, 0) if row["IsCrack"] else (0, 160, 220)
            draw.rectangle([px + 2, py + 2, px + thumb_px + 6, py + thumb_px + 6],
                            outline=border_color, width=3)
            sheet.paste(patch_img, (px + 5, py + 5))
            caption = f"L{int(row['Label'])} p={row['CrackProbability']:.2f}"
            draw.text((px + 5, py + thumb_px + 10), caption, fill="black")
        path = get_unique_filename(os.path.join(out_dir, f"{base}_review_page{start // per_page + 1}.png"))
        sheet.save(path)
        paths.append(path)
    return paths


def _cheapest_path(flat, pA, pB, margin=15):
    """The DARKEST available route between two points (Dijkstra through a
    brightness-as-cost field, via skimage's route_through_array) -- not the
    straight line between them. A real crack's rim can bend between
    fragments, so insisting on the exact straight line rejects genuine
    curved connections; routing toward whatever's darkest finds a real
    winding crack if one is there. Returns the path as absolute (row, col)
    coordinates into the full `flat` image."""
    r0 = max(0, min(pA[0], pB[0]) - margin)
    r1 = min(flat.shape[0], max(pA[0], pB[0]) + margin + 1)
    c0 = max(0, min(pA[1], pB[1]) - margin)
    c1 = min(flat.shape[1], max(pA[1], pB[1]) + margin + 1)
    cost = flat[r0:r1, c0:c1].astype(np.float64)
    start = (pA[0] - r0, pA[1] - c0)
    end = (pB[0] - r0, pB[1] - c0)
    indices, _ = route_through_array(cost, start, end, fully_connected=True)
    indices = np.array(indices)
    return indices + [r0, c0]


def _cheapest_path_darkness(flat, pA, pB, margin=15, percentile=75):
    """75th-percentile brightness along the darkest available route between
    two points -- NOT the mean. A path with genuinely dark endpoints but a
    real bright gap in the middle can average out to a deceptively low
    MEAN brightness even though a solid stretch of it is plain background;
    a high percentile catches "is there a substantial bright stretch
    anywhere on this route" instead of averaging it away."""
    path = _cheapest_path(flat, pA, pB, margin=margin)
    return float(np.percentile(flat[path[:, 0], path[:, 1]], percentile))


def merge_large_cracks(labeled, df, flat, min_area_px=1000, max_gap_px=80, width_scale=0.6,
                        max_bridge_darkness=160):
    """
    The main crack is often segmented into several large fragments (the void
    itself, plus ragged rim pieces around its edge that don't quite touch
    it) -- each counts and displays as a separate crack even though they're
    obviously one. This connects them with a THIN connector whose width is
    measured LOCALLY (via a distance transform right at the connection
    point, not the whole fragment's shape) -- unlike disk-dilation bridging,
    this can't balloon into a round "paintbrush" blob at any gap size, and a
    fragment already massively wide (like a big void) doesn't force a huge
    connector, because only the width AT the actual junction point matters.

    Being merely nearby isn't enough, though -- two unrelated dark spots that
    happen to sit within max_gap_px of each other would otherwise get a
    connector drawn across whatever intact surface sits between them, even
    if nothing crack-like is actually there. Before a candidate pair is even
    eligible for the spanning tree, the DARKEST available route between them
    (Dijkstra through the flattened brightness field, not the straight line
    -- a real crack's rim can bend) must have its 75th-PERCENTILE brightness
    at or below max_bridge_darkness -- not its mean. A path with genuinely
    dark endpoints but a real bright gap partway along can still average out
    to a deceptively low mean; requiring most of the route (not just the
    average) to be dark catches that case while still following genuine
    curved connections. The bridge that actually gets drawn follows this
    same validated route (via cv2.polylines), not a straight-line shortcut
    between the endpoints, so what's drawn always matches what was checked.

    Only candidates already classified as crack AND at or above min_area_px
    are eligible -- ordinary small microcracks scattered across the image
    never enter this at all. Returns (df, bridge_mask); df gets a
    CrackGroupID column (>=0 for any candidate connected to at least one
    other; -1 otherwise) so the true crack count (fragments merged) can be
    reported, without changing any Label or existing IsCrack value.
    """
    df = df.copy()
    df["CrackGroupID"] = -1

    large = df[(df["IsCrack"]) & (df["Area"] >= min_area_px)]
    labels = large["Label"].tolist()
    if len(labels) < 2:
        return df, np.zeros(labeled.shape, dtype=bool)

    coords, dist_transform = {}, {}
    for lbl in labels:
        mask = labeled == lbl
        coords[lbl] = np.column_stack(np.where(mask))
        dist_transform[lbl] = ndi.distance_transform_edt(mask)

    def local_width(lbl, point, radius=12):
        dt = dist_transform[lbl]
        r, c = point
        r0, r1 = max(0, r - radius), min(dt.shape[0], r + radius + 1)
        c0, c1 = max(0, c - radius), min(dt.shape[1], c + radius + 1)
        patch = dt[r0:r1, c0:c1]
        nonzero = patch[patch > 0]
        return float(nonzero.max() * 2) if len(nonzero) else 4.0

    n = len(labels)
    dist_matrix = np.full((n, n), np.inf)
    nearest_points = {}
    bridge_paths = {}
    for i in range(n):
        tree = cKDTree(coords[labels[i]])
        for j in range(i + 1, n):
            d, idx = tree.query(coords[labels[j]])
            k = int(np.argmin(d))
            if d[k] <= max_gap_px:
                pB, pA = coords[labels[j]][k], coords[labels[i]][idx[k]]
                path = _cheapest_path(flat, tuple(pA), tuple(pB))
                if np.percentile(flat[path[:, 0], path[:, 1]], 75) > max_bridge_darkness:
                    continue  # a real chunk of even the darkest available route is bright, intact surface
                dist_matrix[i, j] = dist_matrix[j, i] = d[k]
                nearest_points[(i, j)] = (pB, pA)
                bridge_paths[(i, j)] = path

    if np.isinf(dist_matrix).all():
        return df, np.zeros(labeled.shape, dtype=bool)

    sparse = lil_matrix(np.where(np.isinf(dist_matrix), 0, dist_matrix))
    mst = minimum_spanning_tree(sparse).toarray()

    bridge_mask = np.zeros(labeled.shape, dtype=np.uint8)
    union_parent = list(range(n))

    def find(x):
        while union_parent[x] != x:
            union_parent[x] = union_parent[union_parent[x]]
            x = union_parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if mst[i, j] > 0 or mst[j, i] > 0:
                pB, pA = nearest_points[(i, j)]
                wA, wB = local_width(labels[i], tuple(pA)), local_width(labels[j], tuple(pB))
                w = int(np.clip((wA + wB) / 2 * width_scale, 4, 20))
                # Draw the ACTUAL validated route, not a straight line between
                # the endpoints -- the darkest path can curve away from the
                # straight line to follow real dark pixels, so a straight
                # line here could cut back across bright surface even though
                # a genuinely dark connection exists nearby.
                path = bridge_paths[(i, j)]
                pts = path[:, [1, 0]].astype(np.int32).reshape(-1, 1, 2)  # (row, col) -> (x, y) for cv2
                cv2.polylines(bridge_mask, [pts], isClosed=False, color=1, thickness=w)
                union_parent[find(i)] = find(j)

    roots = {find(i) for i in range(n)}
    for root in roots:
        members = [labels[i] for i in range(n) if find(i) == root]
        if len(members) > 1:
            df.loc[df["Label"].isin(members), "CrackGroupID"] = root

    return df, bridge_mask.astype(bool)


def save_bw_image(labeled, kept_labels, out_path, bridge_mask=None):
    crack_mask = np.isin(labeled, list(kept_labels)) if kept_labels else np.zeros(labeled.shape, dtype=bool)
    if bridge_mask is not None:
        crack_mask = crack_mask | bridge_mask
    bw = np.where(crack_mask, 0, 255).astype(np.uint8)
    out_path = get_unique_filename(out_path)
    tifffile.imwrite(out_path, bw)
    return out_path, crack_mask


def save_overlay(img8, labeled, df, out_path, dpi=300, bridge_mask=None):
    out_path = get_unique_filename(out_path)
    fig, ax = plt.subplots(figsize=(img8.shape[1] / 300, img8.shape[0] / 300), dpi=dpi)
    ax.imshow(img8, cmap="gray")

    kept = df[df["IsCrack"]]
    rejected = df[~df["IsCrack"]]

    overlay = np.zeros((*labeled.shape, 4))
    kept_mask = np.isin(labeled, kept["Label"].tolist()) if len(kept) else np.zeros(labeled.shape, dtype=bool)
    if bridge_mask is not None:
        kept_mask = kept_mask | bridge_mask
    rejected_mask = np.isin(labeled, rejected["Label"].tolist()) if len(rejected) else np.zeros(labeled.shape, dtype=bool)
    overlay[kept_mask] = [1, 0, 0, 0.55]     # red = kept crack
    overlay[rejected_mask] = [0, 0.8, 1, 0.45]  # cyan = rejected artifact
    ax.imshow(overlay)

    # A merged group is one visual crack, so it should carry exactly one
    # number, not one per fragment that happened to compose it -- label at
    # its largest member's position (the most visually prominent point).
    if "CrackGroupID" in kept.columns and (kept["CrackGroupID"] >= 0).any():
        merged = kept[kept["CrackGroupID"] >= 0]
        standalone = kept[kept["CrackGroupID"] < 0]
        to_label = pd.concat([
            merged.loc[merged.groupby("CrackGroupID")["Area"].idxmax()],
            standalone,
        ])
    else:
        to_label = kept

    for _, row in to_label.iterrows():
        ax.annotate(
            str(int(row["Label"])), (row["X"], row["Y"]), color="#00FF00", fontsize=6, fontweight="bold",
            ha="left", va="bottom", xytext=(3, 3), textcoords="offset points",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
        )
    ax.set_axis_off()
    ax.text(
        0.01, 0.01, "red = crack   cyan = rejected artifact",
        transform=ax.transAxes, color="white", fontsize=8,
        bbox=dict(facecolor="black", alpha=0.6, pad=3),
    )
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def detect_cracks(
    image_path,
    out_dir=None,
    bg_sigma=40,
    denoise_sigma=1.0,
    open_radius=1,
    close_radius=3,
    min_area_px=40,
    mode="auto",
    model_path=None,
    train_labels_path=None,
    proba_threshold=0.5,
    force_keep_length=150,
    force_keep_area=50000,
    vessel_sigma_min=1,
    vessel_sigma_max=6,
    make_review_sheets=False,
    autocrop=True,
    hard_overrides=None,
    absolute_dark_thresh=10,
    merge_large_cracks_enabled=True,
    merge_min_area=1000,
    merge_max_gap=80,
    merge_max_bridge_darkness=160,
    contrast_low_pct=1.0,
    contrast_high_pct=99.5,
):
    out_dir = out_dir or os.path.dirname(os.path.abspath(image_path))
    base = os.path.splitext(os.path.basename(image_path))[0]

    img8 = load_as_uint8(image_path, low_pct=contrast_low_pct, high_pct=contrast_high_pct)
    # Segmentation may need a gentle (e.g. 0/100) stretch to avoid clipping real
    # background to artificial black on low-black-content images, but that same
    # gentle stretch makes the image look washed-out next to normally-stretched
    # ones. Overlays/review sheets are pure visualization, so always render them
    # from the standard 1/99.5 stretch -- every image looks the same "brightness"
    # when browsing, regardless of what contrast detection actually used.
    is_default_contrast = (contrast_low_pct == 1.0 and contrast_high_pct == 99.5)
    img8_display = img8 if is_default_contrast else load_as_uint8(image_path, low_pct=1.0, high_pct=99.5)
    if autocrop:
        x0, y0, x1, y1 = find_field_of_view(img8)
        if (x0, y0, x1, y1) != (0, 0, img8.shape[1], img8.shape[0]):
            print(f"Auto-cropped out info bar / aperture vignette: "
                  f"{img8.shape[1]}x{img8.shape[0]} -> box ({x0},{y0})-({x1},{y1})")
        img8 = img8[y0:y1, x0:x1]
        img8_display = img8_display[y0:y1, x0:x1]

    flat = flatten_background(img8, sigma=bg_sigma)
    dark_mask = segment_dark_regions(flat, denoise_sigma=denoise_sigma, img8=img8,
                                      absolute_dark_thresh=absolute_dark_thresh)
    clean = clean_mask(dark_mask, open_radius=open_radius, close_radius=close_radius,
                        min_area_px=max(5, min_area_px // 3))
    vesselness = compute_vesselness(flat, sigma_min=vessel_sigma_min, sigma_max=vessel_sigma_max)
    labeled, df = extract_candidates(clean, flat, vesselness, min_area_px=min_area_px)

    print(f"Candidate regions after segmentation: {len(df)}")

    if mode == "train":
        if not train_labels_path or not model_path:
            raise ValueError("--mode train requires both --train-labels and --model")
        df = train_model(df, train_labels_path, model_path)
    elif mode == "apply":
        if not model_path:
            raise ValueError("--mode apply requires --model")
        df = classify_with_model(df, model_path, proba_threshold=proba_threshold, force_keep_area=force_keep_area)
    else:
        df = classify_auto(df, force_keep_length=force_keep_length, force_keep_area=force_keep_area)

    if hard_overrides:
        # Manual corrections are a stronger signal than any model's probability --
        # a borderline case (e.g. p=0.57) can survive weighted training and still
        # get predicted the "wrong" way. Force the exact rows a human corrected,
        # no matter what the model itself thinks.
        for label, is_crack in hard_overrides.items():
            mask = df["Label"] == label
            if mask.any():
                df.loc[mask, "IsCrack"] = is_crack
                df.loc[mask, "CrackProbability"] = 1.0 if is_crack else 0.0

    bridge_mask = None
    if merge_large_cracks_enabled:
        df, bridge_mask = merge_large_cracks(labeled, df, flat, min_area_px=merge_min_area,
                                              max_gap_px=merge_max_gap,
                                              max_bridge_darkness=merge_max_bridge_darkness)

    df = df.sort_values("Area", ascending=False).reset_index(drop=True)
    kept_labels = set(df.loc[df["IsCrack"], "Label"].tolist())

    bw_path, crack_mask = save_bw_image(labeled, kept_labels, os.path.join(out_dir, f"{base}_cracks_bw.tif"),
                                         bridge_mask=bridge_mask)
    overlay_path = save_overlay(img8_display, labeled, df, os.path.join(out_dir, f"{base}_cracks_overlay.png"),
                                 bridge_mask=bridge_mask)
    csv_path = get_unique_filename(os.path.join(out_dir, f"{base}_cracks.csv"))
    df.to_csv(csv_path, index=False)

    n_kept = int(df["IsCrack"].sum())
    n_merged_groups = df.loc[(df["IsCrack"]) & (df["CrackGroupID"] >= 0), "CrackGroupID"].nunique() if "CrackGroupID" in df.columns else 0
    n_standalone = int((df["IsCrack"] & (df.get("CrackGroupID", -1) < 0)).sum())
    n_reported_cracks = n_merged_groups + n_standalone
    total_crack_area_px = int(crack_mask.sum())
    frac = total_crack_area_px / crack_mask.size

    print(f"Kept {n_kept} of {len(df)} candidates as real cracks (mode={mode}).")
    if merge_large_cracks_enabled and n_merged_groups:
        print(f"  -> merged into {n_reported_cracks} distinct crack(s) "
              f"({n_merged_groups} merged group(s) + {n_standalone} standalone)")
    print(f"Total crack area: {total_crack_area_px} px  ({frac*100:.3f}% of image)")
    print(f"Black & white mask : {bw_path}")
    print(f"Overlay (QC)       : {overlay_path}")
    print(f"Measurements CSV   : {csv_path}")

    review_paths = []
    if make_review_sheets:
        review_paths = save_review_sheets(img8_display, df, out_dir, base)
        if review_paths:
            print(f"Review sheet(s)    : {len(review_paths)} page(s), e.g. {review_paths[0]}")
        else:
            print("Review sheet(s)    : none (no candidates found)")

    print("Tip: to improve accuracy further, open the review sheet(s), find any")
    print("wrong red/cyan borders, flip that row's IsCrack value in the CSV, then:")
    print(f'  --mode train --train-labels "{csv_path}" --model crack_clf.joblib')

    return {"count": n_kept, "n_reported_cracks": n_reported_cracks, "bw_path": bw_path,
            "overlay_path": overlay_path, "csv_path": csv_path, "review_paths": review_paths, "dataframe": df}


def main():
    parser = argparse.ArgumentParser(description="Detect cracks in an SEM/TXM micrograph with ML-based artifact rejection.")
    parser.add_argument("image", help="Path to the input TIFF image.")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: same folder as input).")
    parser.add_argument("--bg-sigma", type=float, default=40, help="Gaussian sigma for background flattening.")
    parser.add_argument("--denoise-sigma", type=float, default=1.0, help="Gaussian sigma before thresholding.")
    parser.add_argument("--open-radius", type=int, default=1, help="Morphological opening disk radius.")
    parser.add_argument("--close-radius", type=int, default=3, help="Morphological closing disk radius.")
    parser.add_argument("--min-area", type=int, default=40, help="Minimum region area (px) to consider as a candidate.")
    parser.add_argument("--vessel-sigma-min", type=int, default=1, help="Frangi filter minimum scale (px).")
    parser.add_argument("--vessel-sigma-max", type=int, default=6, help="Frangi filter maximum scale (px).")
    parser.add_argument("--mode", choices=["auto", "train", "apply"], default="auto",
                         help="'auto' = unsupervised clustering (no labels needed, default); "
                              "'train' = fit+save a RandomForest classifier from corrected labels; "
                              "'apply' = classify using a previously trained model.")
    parser.add_argument("--model", default=None, help="Path to save (train) or load (apply) the classifier (.joblib).")
    parser.add_argument("--train-labels", default=None,
                         help="CSV with 'Label' and 'IsCrack' columns (from a reviewed/corrected candidates CSV).")
    parser.add_argument("--proba-threshold", type=float, default=None,
                        help="Probability threshold in --mode apply. Default: use the "
                             "threshold the model was calibrated at (0.5 for older models).")
    parser.add_argument("--force-keep-length", type=float, default=150,
                         help="Regions with major-axis length >= this (px) are always kept in --mode auto.")
    parser.add_argument("--review-sheet", action="store_true",
                         help="Also save paginated thumbnail contact sheets of every candidate for quick manual review.")
    parser.add_argument("--no-autocrop", action="store_true",
                         help="Disable auto-detection/cropping of instrument info bars and aperture vignettes.")
    parser.add_argument("--contrast-low-pct", type=float, default=1.0,
                         help="Low percentile for the initial 16-bit->8-bit contrast stretch (default 1.0). "
                              "Lower this (e.g. to 0) for images with almost no true black pixels, where the "
                              "default clips a chunk of ordinary texture to artificial pure black.")
    parser.add_argument("--contrast-high-pct", type=float, default=99.5,
                         help="High percentile for the initial contrast stretch (default 99.5).")
    args = parser.parse_args()

    detect_cracks(
        args.image,
        out_dir=args.out_dir,
        bg_sigma=args.bg_sigma,
        denoise_sigma=args.denoise_sigma,
        open_radius=args.open_radius,
        close_radius=args.close_radius,
        min_area_px=args.min_area,
        mode=args.mode,
        model_path=args.model,
        autocrop=not args.no_autocrop,
        train_labels_path=args.train_labels,
        proba_threshold=args.proba_threshold,
        force_keep_length=args.force_keep_length,
        vessel_sigma_min=args.vessel_sigma_min,
        vessel_sigma_max=args.vessel_sigma_max,
        make_review_sheets=args.review_sheet,
        contrast_low_pct=args.contrast_low_pct,
        contrast_high_pct=args.contrast_high_pct,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        detect_cracks("../original/260708_316_H_b2_front_CBS_002.tif", make_review_sheets=True)
        sys.exit(0)
    main()
