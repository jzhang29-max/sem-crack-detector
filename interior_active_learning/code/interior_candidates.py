"""
Generate "interior candidate" regions: gaps between/within already-confirmed
crack fragments that the production pipeline's brightness-based mask doesn't
cover, but that might genuinely be part of the crack's interior.

Two deliberately BOUNDED candidate shapes, chosen after an earlier "fill the
convex hull of every merged group" approach was tried and rejected for
over-filling unrelated background on winding/curved cracks:

  A) "concavity" -- for each SINGLE confirmed-crack connected component, the
     gap between that region's own convex hull and its actual footprint.
     Bounded by that one region's own hull, so it can never spill across
     unrelated background the way a hull over a whole multi-fragment,
     possibly-winding group could.

  B) "bridge_corridor" -- for each pair of confirmed-crack regions close
     enough to have been (or been eligible to be) bridged by the production
     merge_large_cracks logic, a buffered corridor around the SAME validated
     darkest-available-route (Dijkstra through brightness, 75th-percentile
     darkness check) merge_large_cracks itself uses -- just wider than the
     thin connector line it actually draws. Bounded by the corridor's
     length x width, and gated by the same darkness check, so it only
     proposes corridors along genuinely dark routes.

Every candidate here is a PROPOSAL for a human to accept or reject via the
active-learning review step, not something auto-committed to any mask.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import joblib
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import morphology, measure
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    ORIGINAL_DIR, PROD_MODEL_PATH, CANDIDATES_DIR, MODELS_DIR, INTERIOR_LABEL_OFFSET,
    contrast_kwargs_for, load_hard_overrides, load_correction_mask,
)
from labeling_overlay import draw_labels
from active_learning_select import INTERIOR_FEATURE_COLUMNS

from detect_cracks import (
    load_as_uint8, find_field_of_view, flatten_background, segment_dark_regions,
    clean_mask, compute_vesselness, extract_candidates, classify_with_model,
    merge_large_cracks, _cheapest_path,
)


def apply_pixel_corrections(labeled, df, correction_mask):
    """Apply paint-app corrections at the PIXEL level, not the whole-Label
    level. correction_mask is 0 (no correction) / 1 (force CRACK) / 2 (force
    NOT-CRACK, still a labeled "artifact" candidate) / 3 (ERASE -- not a
    candidate at all, back to plain unmarked background) per pixel, same
    shape as `labeled`.

    A whole-Label flip ("this candidate's verdict is now X") breaks down the
    moment one connected "crack" region is enormous -- e.g. the entire dark
    background matrix comes back as a SINGLE Label spanning most of the
    image (confirmed: this is exactly what a user hit trying to erase part
    of one image's overlay). Painting over a small patch of that region
    would either do nothing or flip the ENTIRE region once ingested -- never
    just the painted patch, because there was only one boolean IsCrack for
    the whole thing to flip.

    Instead: for each existing Label touched by a correction, split off
    exactly the corrected pixels into a brand-new Label carrying the forced
    verdict (or, for an erasure, no Label at all -- those pixels just go
    back to 0/background), and leave the untouched remainder under the
    original Label with its original verdict. If a correction happens to
    cover an entire Label (the common case for small/medium candidates), no
    split is needed -- it's just flipped/dropped in place.

    Only Area/X/Y/Label/IsCrack/CrackProbability are recomputed for a split
    piece; every other feature column is copied from the parent row as a
    placeholder. This is safe here because downstream consumers of this
    in-memory df (build_simple_overlay, the interior-candidate generators,
    and merge_large_cracks -- verified by reading merge_large_cracks itself)
    only ever read Label/IsCrack/Area from df, recomputing actual geometry
    from `labeled` directly; nothing here is written back to the production
    results/*.csv this tool stays read-only against.

    A force-crack (value=1) pixel that lands on BLANK background (Label==0,
    not an existing candidate at all) is handled separately, by
    _merge_blank_force_crack_into_touching_cracks: if it touches an
    existing crack region it's folded into that SAME crack (so painting a
    red stroke that connects to an existing red crack makes them one crack,
    not two coincidentally-adjacent ones) rather than staying an unrelated
    standalone patch."""
    if correction_mask is None or not correction_mask.any():
        return labeled, df

    labeled = labeled.copy()
    df = df.copy()
    next_label = int(df["Label"].max()) + 1 if len(df) else 1
    drop_indices = []

    for target_value, forced_is_crack in ((1, True), (2, False), (3, None)):
        mask = correction_mask == target_value
        if not mask.any():
            continue
        new_rows = []
        touched = sorted(int(lbl) for lbl in np.unique(labeled[mask]) if lbl != 0)
        for lbl in touched:
            row_idx = df.index[df["Label"] == lbl]
            if len(row_idx) == 0:
                continue
            row = df.loc[row_idx[0]]
            if forced_is_crack is not None and bool(row["IsCrack"]) == forced_is_crack:
                continue  # already the desired verdict -- nothing to correct
            label_mask = labeled == lbl
            corrected_part = label_mask & mask
            if not corrected_part.any():
                continue
            remainder = label_mask & ~corrected_part

            if forced_is_crack is None:
                # erase -- these pixels stop being a candidate of any kind
                labeled[corrected_part] = 0
                if not remainder.any():
                    drop_indices.append(row_idx[0])
                else:
                    ys2, xs2 = np.where(remainder)
                    df.loc[row_idx[0], "Area"] = int(remainder.sum())
                    df.loc[row_idx[0], "X"] = float(xs2.mean())
                    df.loc[row_idx[0], "Y"] = float(ys2.mean())
                continue

            if not remainder.any():
                # the correction covers this Label's entire footprint -- flip in place
                df.loc[row_idx[0], "IsCrack"] = forced_is_crack
                df.loc[row_idx[0], "CrackProbability"] = 1.0 if forced_is_crack else 0.0
                continue
            labeled[corrected_part] = next_label
            ys, xs = np.where(corrected_part)
            new_row = row.to_dict()
            new_row.update({
                "Label": next_label,
                "Area": int(corrected_part.sum()),
                "X": float(xs.mean()),
                "Y": float(ys.mean()),
                "IsCrack": forced_is_crack,
                "CrackProbability": 1.0 if forced_is_crack else 0.0,
            })
            new_rows.append(new_row)
            ys2, xs2 = np.where(remainder)
            df.loc[row_idx[0], "Area"] = int(remainder.sum())
            df.loc[row_idx[0], "X"] = float(xs2.mean())
            df.loc[row_idx[0], "Y"] = float(ys2.mean())
            next_label += 1

        # New rows from any split above must land in df BEFORE the
        # blank-merge check below -- otherwise a blank patch touching a
        # label that was JUST split off this same pass (already baked into
        # `labeled`, but not yet a df row) would look up an empty result
        # and crash with "single positional indexer is out-of-bounds".
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        if forced_is_crack is True:
            df = _merge_blank_force_crack_into_touching_cracks(labeled, df, mask)

    if drop_indices:
        df = df.drop(index=drop_indices)
    return labeled, df


def _merge_blank_force_crack_into_touching_cracks(labeled, df, mask):
    """A force-crack correction can't "correct" a BLANK (Label==0) pixel the
    way it corrects an existing candidate -- there's no existing row to
    flip. But if the blank patch touches an EXISTING crack region, the
    right behavior isn't to leave it as an unrelated new candidate either:
    a red stroke drawn right up against an existing red crack should join
    it into the SAME crack, not become a second disconnected one that only
    happens to look contiguous because both render red. Checked per
    connected component (not per pixel) so one stroke that only touches at
    one end doesn't get fragmented; if a patch touches MULTIPLE existing
    crack fragments at once, all of them are merged into one survivor,
    mirroring what production's own merge_large_cracks does for bridged
    fragments. A patch touching no crack at all is left alone here --
    that's the genuinely-new-candidate case, handled elsewhere."""
    blank_force_crack = mask & (labeled == 0)
    if not blank_force_crack.any():
        return df
    comp_labeled = measure.label(blank_force_crack, connectivity=2)
    for region in measure.regionprops(comp_labeled):
        comp_mask = comp_labeled == region.label
        dil_comp = morphology.binary_dilation(comp_mask, morphology.disk(1))
        touching = sorted(int(lbl) for lbl in np.unique(labeled[dil_comp]) if lbl != 0)
        # skip any label `labeled` claims but df has no row for -- shouldn't
        # happen once new_rows is flushed before this runs, but silently
        # skipping a not-yet-registered neighbor is safer than crashing on
        # an empty .iloc[0] lookup
        touching_crack = [lbl for lbl in touching
                           if not df.loc[df["Label"] == lbl].empty and bool(df.loc[df["Label"] == lbl, "IsCrack"].iloc[0])]
        if not touching_crack:
            # An isolated patch: the user painted a crack the model never proposed,
            # and it touches nothing the model did. This used to `continue`, on the
            # basis that the "genuinely-new-candidate case" was handled elsewhere.
            # It was not: the patch stayed at label 0, so it had no row in df, and
            # every export and region CSV built from df omitted it -- the one thing
            # a reviewer most wants recorded, "the model missed this", disappeared
            # on the next re-render.
            #
            # The model-feature columns are copied from an existing row and are NOT
            # measured for this region; only Label/Area/X/Y/IsCrack are real, and
            # CandidateType says "painted" so that is visible. Training is
            # unaffected either way -- build_training_data computes features from
            # the correction mask itself, never from these columns.
            if not len(df):
                continue
            new_label = int(df["Label"].max()) + 1
            labeled[comp_mask] = new_label
            ys, xs = np.where(comp_mask)
            nr = df.iloc[0].to_dict()
            nr.update({"Label": new_label, "Area": int(comp_mask.sum()),
                       "X": float(xs.mean()), "Y": float(ys.mean()),
                       "IsCrack": True, "CrackProbability": 1.0,
                       "CandidateType": "painted", "ParentLabels": ""})
            df = pd.concat([df, pd.DataFrame([nr])], ignore_index=True)
            continue
        survivor = touching_crack[0]
        combined = comp_mask.copy()
        for lbl in touching_crack:
            combined |= (labeled == lbl)
        labeled[combined] = survivor
        ys, xs = np.where(combined)
        idx = df.index[df["Label"] == survivor][0]
        df.loc[idx, "Area"] = int(combined.sum())
        df.loc[idx, "X"] = float(xs.mean())
        df.loc[idx, "Y"] = float(ys.mean())
        if len(touching_crack) > 1:
            df = df[~df["Label"].isin(touching_crack[1:])]
    return df


def run_production_pipeline(image_name):
    """Reproduce the deployed pipeline's crack/artifact call for one image,
    read-only (same functions, same production model, same manual
    corrections) -- this is the "ground truth per the CURRENT model" that
    interior candidates are proposed on top of."""
    image_path = os.path.join(ORIGINAL_DIR, f"{image_name}.tif")
    img8 = load_as_uint8(image_path, **contrast_kwargs_for(image_name))
    x0, y0, x1, y1 = find_field_of_view(img8)
    img8 = img8[y0:y1, x0:x1]
    flat = flatten_background(img8)
    dark_mask = segment_dark_regions(flat, img8=img8)
    clean = clean_mask(dark_mask, min_area_px=13)
    vesselness = compute_vesselness(flat)
    labeled, df = extract_candidates(clean, flat, vesselness, min_area_px=40)
    df = classify_with_model(df, PROD_MODEL_PATH)
    overrides = load_hard_overrides(image_name)
    if overrides:
        for label, is_crack in overrides.items():
            m = df["Label"] == label
            if m.any():
                df.loc[m, "IsCrack"] = is_crack
    # Pixel-level corrections made by painting/erasing in the paint app --
    # applied AFTER the main ledger's overrides (so a paint correction can
    # override an old ledger entry too) and BEFORE merging, so bridging
    # logic also reflects the corrected verdict, and every subsequent render
    # (paint template, quicklook, before/after) shows the correction instead
    # of the same mistake reappearing every time. See
    # apply_pixel_corrections()'s docstring for why this operates on a
    # per-pixel mask rather than a whole-Label flip.
    correction_mask = load_correction_mask(image_name, labeled.shape)
    if correction_mask is not None:
        labeled, df = apply_pixel_corrections(labeled, df, correction_mask)
    df, bridge_mask = merge_large_cracks(labeled, df, flat, min_area_px=1000, max_gap_px=80)
    return dict(img8=img8, flat=flat, labeled=labeled, df=df, bridge_mask=bridge_mask, vesselness=vesselness)


# interior_fill has only 2 negative examples (vs dozens positive) as of this
# writing -- nowhere near enough for a plain ML boundary to be trustworthy
# (confirmed: a plain 0.5 cut accepts ~95% of all interior_fill candidates).
# If train_interior_model.py's calibrate_interior_fill_rule() found a valid
# hybrid rule (ML floor + distance/brightness gate), that's loaded from the
# model bundle and used instead of this flat fallback -- see that function's
# docstring for why. This fallback only fires if the calibration couldn't run
# (e.g. fewer than 2 negatives labeled yet), in which case it's intentionally
# very conservative.
INTERIOR_FILL_FALLBACK_THRESHOLD = 0.9


def score_interior_candidates(stage, threshold=0.5):
    """Generate this image's interior candidates fresh (concavity/
    bridge_corridor/interior_fill) and score them with
    models/interior_model.joblib, applying the exact same hybrid rule for
    interior_fill (ML floor + distance/brightness gate, or the conservative
    flat fallback if no rule has been calibrated yet) and flat threshold for
    the other two types. This is the single source of truth for "what does
    the model currently accept" -- both apply_interior_model.py's standalone
    report and run_enhanced_pipeline() (below) call this rather than each
    keeping their own copy, so the two can never quietly drift apart.

    Returns (accepted, rejected), each a list of (mask_bool, CandidateType,
    feats) -- feats is the full _region_features() dict already computed, so
    a caller logging one of these into candidates/*_interior.csv never needs
    to recompute it. Returns ([], []) if no model has been trained yet."""
    model_path = os.path.join(MODELS_DIR, "interior_model.joblib")
    if not os.path.exists(model_path):
        return [], []
    bundle = joblib.load(model_path)
    interior_fill_rule = bundle.get("interior_fill_rule")

    labeled, df, flat, img8, vesselness = stage["labeled"], stage["df"], stage["flat"], stage["img8"], stage["vesselness"]
    crack_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
    dil_crack = morphology.binary_dilation(crack_mask, morphology.disk(1))
    dist_to_crack = ndi.distance_transform_edt(~crack_mask)

    raw = concavity_candidates(crack_mask)
    raw += bridge_corridor_candidates(labeled, df, flat, crack_mask)
    raw += interior_fill_candidates(labeled, df, img8, crack_mask, dist_to_crack)

    accepted, rejected = [], []
    for mask_bool, ctype, parent in raw:
        feats = _region_features(mask_bool, flat, img8, vesselness, crack_mask, dil_crack, dist_to_crack)
        X = bundle["scaler"].transform([[feats[c] for c in INTERIOR_FEATURE_COLUMNS]])
        proba = bundle["clf"].predict_proba(X)[0, 1]
        if ctype == "interior_fill":
            if interior_fill_rule is not None:
                is_accepted = (proba >= interior_fill_rule["floor"]
                               and feats["MeanDistToCrack"] <= interior_fill_rule["dist_thr"]
                               and feats["MeanFlatBrightness"] <= interior_fill_rule["bri_thr"])
            else:
                is_accepted = proba >= INTERIOR_FILL_FALLBACK_THRESHOLD
        else:
            is_accepted = proba >= threshold
        (accepted if is_accepted else rejected).append((mask_bool, ctype, feats))
    return accepted, rejected


def run_enhanced_pipeline(image_name, stage=None, threshold=0.5):
    """run_production_pipeline() plus whatever models/interior_model.joblib
    currently accepts, folded directly into labeled/df as real crack Labels.
    This is what the paint template (apply_paint_annotations.make_template),
    results/, and review/applied/ all render from -- so an interior
    candidate the model accepted is visible and paintable exactly like any
    other crack, closing the loop the plain review-sheet flow couldn't: if
    the model was WRONG about one, painting cyan over it (or red, if it
    missed something adjacent) routes back into candidates/*_interior.csv as
    a properly-typed label via apply_paint_annotations.ingest() -- which is
    exactly the additional negative-example signal interior_fill has been
    starved of all along (see models/checkpoints/README.md) -- rather than
    the model's mistakes being invisible until the next full review-sheet
    round.

    Only claims currently-BLANK pixels for each accepted candidate (never
    steals pixels from an existing crack OR artifact Label) -- interior
    candidates are already generated as gaps outside crack_mask, but
    interior_fill's flood-fill can still graze an unrelated existing
    artifact region; leaving that portion under its original verdict is a
    simpler and safer choice than reconciling two Labels' bookkeeping over a
    partial overlap, and nothing downstream depends on this candidate's
    Area/X/Y being exact (everything recomputes geometry from `labeled`
    directly -- see apply_pixel_corrections's docstring).

    Returns run_production_pipeline()'s stage dict plus:
      - "interior_origin": {Label: (CandidateType, feats)} for every
        accepted region folded in here.
      - "n_interior_total": accepted + rejected candidate count, for
        reporting.
    stage: an already-computed run_production_pipeline() result, reused as
    given (same caching rationale as apply_pixel_corrections/ingest)."""
    if stage is None:
        stage = run_production_pipeline(image_name)
    accepted, rejected = score_interior_candidates(stage, threshold=threshold)

    labeled, df = stage["labeled"].copy(), stage["df"].copy()
    next_label = int(df["Label"].max()) + 1 if len(df) else 1
    interior_origin = {}
    new_rows = []
    for mask_bool, ctype, feats in accepted:
        claimable = mask_bool & (labeled == 0)
        if not claimable.any():
            continue  # fully overlapped by an earlier accepted candidate or an existing Label
        labeled[claimable] = next_label
        row = dict(feats)
        row.update({
            "Label": next_label, "SourceImage": image_name, "CandidateType": ctype,
            "ParentLabels": "", "IsCrack": True, "CrackProbability": 1.0,
        })
        new_rows.append(row)
        interior_origin[next_label] = (ctype, feats)
        next_label += 1
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return dict(stage, labeled=labeled, df=df, interior_origin=interior_origin,
                n_interior_total=len(accepted) + len(rejected))


def build_simple_overlay(stage):
    """Red=crack / cyan=artifact only, whatever IsCrack says for each Label
    in the given stage -- no separate color for interior-candidate types.
    Whether a still-experimental interior candidate shows up as plain red
    depends entirely on which stage this is given: pass a plain
    run_production_pipeline() stage for the pure production result, or a
    run_enhanced_pipeline() stage to also render whatever
    models/interior_model.joblib currently accepts, folded in as real crack
    Labels (see run_enhanced_pipeline's docstring for why -- this is what
    lets you paint-correct one of the model's mistakes directly). Rendered
    directly from the same img8/labeled arrays used everywhere else in this
    file (not the production save_overlay()'s matplotlib PNG, whose saved
    pixel dimensions don't exactly match the source array -- verified on two
    images, off by 9-21px per axis -- which would silently misalign painted
    coordinates with the actual candidate mask)."""
    img8, labeled, df = stage["img8"], stage["labeled"], stage["df"]
    crack_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
    artifact_mask = np.isin(labeled, df.loc[~df["IsCrack"], "Label"].tolist())

    # Exactly matches the production save_overlay()'s own tint colors/alphas
    # (detect_cracks.py: overlay[kept_mask] = [1,0,0,0.55], overlay[rejected_mask]
    # = [0,0.8,1,0.45]) -- verified these two didn't actually match before
    # (this had crack alpha 0.6 vs production's 0.55, and a dimmer custom
    # teal (0,0.7,0.9) instead of production's brighter (0,0.8,1) cyan).
    rgb = np.stack([img8] * 3, axis=-1).astype(float) / 255.0
    red = np.array([1, 0, 0])
    cyan = np.array([0, 0.8, 1.0])
    rgb[artifact_mask] = rgb[artifact_mask] * 0.55 + cyan * 0.45
    rgb[crack_mask] = rgb[crack_mask] * 0.45 + red * 0.55
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _region_features(mask_bool, flat, img8, vesselness, crack_mask, dil_crack, dist_to_crack):
    # dil_crack and dist_to_crack are precomputed ONCE per image and passed
    # in -- both are full-frame operations (dilation, distance transform),
    # and computing either of them fresh per-candidate (as an earlier
    # version of this function did) redundantly reruns a whole-image
    # operation once per candidate, which is the same class of blowup fixed
    # in the production merge_large_cracks earlier this session (verified:
    # OOM-killed on a 25M-pixel image with 100+ candidates).
    ys, xs = np.where(mask_bool)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    area = int(mask_bool.sum())
    props = measure.regionprops(mask_bool[y0:y1, x0:x1].astype(np.uint8))[0]
    minor = props.axis_minor_length if props.axis_minor_length > 0 else 0.5
    elongation = props.axis_major_length / minor
    perim = props.perimeter if props.perimeter > 0 else 1.0
    circularity = min(4 * np.pi * props.area / (perim ** 2), 1.0)

    boundary = mask_bool[y0:y1, x0:x1] & ~morphology.binary_erosion(mask_bool[y0:y1, x0:x1])
    by, bx = np.where(boundary)
    touching_crack = int(dil_crack[y0:y1, x0:x1][by, bx].sum()) if len(by) else 0
    frac_touching_crack = touching_crack / max(1, len(by))

    mean_dist_to_crack = float(dist_to_crack[mask_bool].mean())

    return {
        "Area": area,
        "Width": x1 - x0,
        "Height": y1 - y0,
        "X": xs.mean(),
        "Y": ys.mean(),
        "Elongation": elongation,
        "Solidity": props.solidity,
        "Eccentricity": props.eccentricity,
        "Extent": props.extent,
        "Circularity": circularity,
        "MeanRawBrightness": float(img8[mask_bool].mean()),
        "MeanFlatBrightness": float(flat[mask_bool].mean()),
        "MeanVesselness": float(vesselness[mask_bool].mean()),
        "FracBoundaryTouchingCrack": frac_touching_crack,
        "MeanDistToCrack": mean_dist_to_crack,
        "LogArea": float(np.log10(max(area, 1))),
    }


def concavity_candidates(crack_mask, close_radius=12, min_area=30, max_area=6000):
    """Small local notches along a crack's own boundary only -- NOT a
    global convex hull. A global hull on one long, curvy fragment (which
    real crack fragments in this dataset often are) cuts straight across
    the concave inside of every bend, overfilling badly (confirmed visually
    on CBS_002: the hull swallowed a wide swath along the main crack's top
    edge). Morphological closing with a small, fixed-radius disk can only
    bridge gaps/notches up to ~2*close_radius px wide, so it's insensitive
    to the fragment's overall shape/curvature -- it fills a jagged nick in
    the boundary but can't run away across a bend the way a hull does."""
    out = []
    crack_labeled = measure.label(crack_mask, connectivity=2)
    for region in measure.regionprops(crack_labeled):
        if region.area < 20:
            continue
        y0, x0, y1, x1 = region.bbox
        pad = close_radius + 2
        yy0, yy1 = max(0, y0 - pad), min(crack_labeled.shape[0], y1 + pad)
        xx0, xx1 = max(0, x0 - pad), min(crack_labeled.shape[1], x1 + pad)
        sub = crack_labeled[yy0:yy1, xx0:xx1] == region.label
        closed = morphology.binary_closing(sub, morphology.disk(close_radius))
        gap = closed & ~sub
        if not gap.any():
            continue
        gap_labeled = measure.label(gap, connectivity=2)
        for g in measure.regionprops(gap_labeled):
            if g.area < min_area or g.area > max_area:
                continue
            full_mask = np.zeros(crack_labeled.shape, dtype=bool)
            full_mask[yy0:yy1, xx0:xx1] |= (gap_labeled == g.label)
            out.append((full_mask, "concavity", str(region.label)))
    return out


def bridge_corridor_candidates(labeled, df, flat, crack_mask, min_fragment_area=1000,
                                max_gap_px=40, max_bridge_darkness=170, corridor_width=20,
                                min_area=30, max_area=30000):
    """Only fragments already at least merge_large_cracks's own min_area_px
    (1000px) are eligible -- mirroring exactly which fragments production
    bridging would even consider, so this doesn't propose connecting small
    unrelated microcracks that the production pipeline itself would never
    treat as candidates for merging. max_gap_px is also tighter than
    merge_large_cracks's 80px (confirmed too permissive here: it was
    linking small cracks that clearly shouldn't be connected)."""
    large = df[(df["IsCrack"]) & (df["Area"] >= min_fragment_area)]
    labels = large["Label"].tolist()
    if len(labels) < 2:
        return []

    coords = {}
    for lbl in labels:
        mask = labeled == lbl
        boundary = mask & ~morphology.binary_erosion(mask)
        coords[lbl] = np.column_stack(np.where(boundary))

    out = []
    n = len(labels)
    for i in range(n):
        tree = cKDTree(coords[labels[i]])
        for j in range(i + 1, n):
            d, idx = tree.query(coords[labels[j]])
            k = int(np.argmin(d))
            if d[k] > max_gap_px:
                continue
            pB, pA = coords[labels[j]][k], coords[labels[i]][idx[k]]
            path = _cheapest_path(flat, tuple(pA), tuple(pB))
            p75 = float(np.percentile(flat[path[:, 0], path[:, 1]], 75))
            if p75 > max_bridge_darkness:
                continue
            corridor = np.zeros(labeled.shape, dtype=np.uint8)
            pts = path[:, [1, 0]].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(corridor, [pts], isClosed=False, color=1, thickness=corridor_width)
            gap = corridor.astype(bool) & ~crack_mask
            area = int(gap.sum())
            if area < min_area or area > max_area:
                continue
            out.append((gap, "bridge_corridor", f"{labels[i]}+{labels[j]}"))
    return out


def interior_fill_candidates(labeled, df, img8, crack_mask, dist_from_crack, bg_ring_px=150,
                              max_reach_px=500, darkness_k=2.0, max_group_multiple=4.0,
                              min_area=200, max_area=2_000_000):
    """A crack's WIDE interior (the visible gray "damaged zone" between its
    two rim edges, as opposed to the darkest rim itself) is often much
    brighter, in the background-FLATTENED image, than the rim -- flattening
    is a high-pass filter, and a feature wider than its ~120-160px blur
    radius gets partly subtracted away along with genuine illumination
    drift (this is segment_dark_regions's own documented blind spot).
    Verified on CBS_016: in `flat` the interior (median 168) is barely
    distinguishable from clean background (median 150), but in the RAW
    percentile-stretched img8 the same interior (median 173) is clearly
    darker than clean background (median 235) -- a ~60pt gap. So this
    candidate type thresholds img8, not flat.

    IMPORTANT CAVEAT (measured on CBS_016): brightness fades from the crack
    outward as a smooth GRADIENT, not a sharp edge -- median img8 goes
    150 (0-20px out) -> 189 (20-40px) -> 216 (40-60px) -> 225 (60-90px) ->
    230 (90-120px) -> 236 (320-400px, clean). There is no "true" cutoff to
    find; darkness_k just picks a point on that gradient, and different
    values change the total proposed area by ~30% without a clear
    "correct" answer. That's exactly why this is a human-reviewed
    candidate rather than an auto-applied fill -- the review labels are
    what should ultimately calibrate how far out to fill, not this
    heuristic.

    Bounded three ways so a leaky per-image threshold can't run away:
    1) the threshold itself is calibrated per-image, from THIS image's own
       clean-background statistics (median/MAD sampled far from any crack);
    2) only pixels CONNECTED to the existing crack mask through a contiguous
       run of "loosely dark" pixels are kept (a flood-fill from the crack,
       not a blind global threshold) -- an unrelated dark spot elsewhere in
       the frame that never touches a crack is never included. A first
       version also required every included pixel be within a small fixed
       radius (60px) of the crack mask itself, meant as an extra safety
       margin -- but that cut the flood-fill off partway across any band
       wider than ~120px, missing exactly the wide-interior case this exists
       for (confirmed on CBS_016: it only filled a thin halo along each
       edge, not the band's actual middle). max_reach_px is now a much
       larger backstop (500px) purely against a pathological thin accidental
       bridge to something far away -- not a width limit on legitimate fill.
    3) the total filled area for a whole crack group is capped at
       max_group_multiple times that group's own already-confirmed area,
       which is what actually catches a leaky threshold (it scales with
       how big the crack already is, unlike a fixed pixel radius).
    """
    if not crack_mask.any():
        return []

    # dist_from_crack is computed ONCE by the caller and shared with
    # _region_features's own distance-to-crack feature -- a distance
    # transform (already the fast, proven approach used elsewhere in this
    # file) computes "everywhere within radius R of the crack" in one O(N)
    # pass; calling binary_dilation with a large-radius disk structuring
    # element (R=150) directly over a 25M-pixel image is equivalent but was
    # measured taking 4+ minutes without finishing.
    bg_far = dist_from_crack > bg_ring_px
    if bg_far.sum() < 1000:
        return []
    bg_median = float(np.median(img8[bg_far]))
    bg_mad = float(np.median(np.abs(img8[bg_far].astype(np.float32) - bg_median)) * 1.4826)
    bg_mad = max(bg_mad, 3.0)
    threshold = bg_median - darkness_k * bg_mad

    loose_dark = (img8 < threshold) & (dist_from_crack <= max_reach_px)

    loose_labeled = measure.label(loose_dark | crack_mask, connectivity=2)
    touched_group_ids = set(np.unique(loose_labeled[crack_mask]))
    touched_group_ids.discard(0)

    out = []
    for gid in touched_group_ids:
        blob = (loose_labeled == gid)
        new_area_mask = blob & ~crack_mask
        if not new_area_mask.any():
            continue
        parent_area = int((blob & crack_mask).sum())
        if new_area_mask.sum() > parent_area * max_group_multiple:
            continue  # this group's threshold leaked too far -- skip rather than propose a runaway blob
        parent_labels = sorted(int(x) for x in np.unique(labeled[blob & crack_mask]) if x != 0)

        comp_labeled = measure.label(new_area_mask, connectivity=2)
        for c in measure.regionprops(comp_labeled):
            if c.area < min_area or c.area > max_area:
                continue
            full_mask = comp_labeled == c.label
            out.append((full_mask, "interior_fill", "+".join(map(str, parent_labels))))
    return out


def _load_existing_labels(csv_path):
    """Split an existing candidates CSV into (already-labeled rows,
    user_painted rows) so a regeneration can preserve both -- discovered
    the hard way: re-running this on an already-labeled image silently
    wiped every verdict (including painted-in ones, which aren't
    regenerated by `raw` at all) because the old CSV was just overwritten
    wholesale."""
    if not os.path.exists(csv_path):
        return None, None
    old = pd.read_csv(csv_path)
    labeled = old[old["IsCrack"].astype(str).str.strip().str.upper().isin(["TRUE", "FALSE"])].copy()
    painted = old[old["CandidateType"] == "user_painted"].copy()
    return labeled, painted


def _carry_forward_labels(new_df, old_labeled, tol_px=5, area_tol_frac=0.15):
    """Match old labeled rows to freshly-regenerated candidates by
    (CandidateType, position, area) rather than assuming Label numbers
    line up -- Label is just an insertion-order counter, so it's only
    stable across regeneration if nothing about the candidate-generation
    code or its inputs changed. Position+area matching is robust to a
    reordering even when the numbering isn't; an unmatched old verdict is
    reported rather than silently dropped."""
    new_df = new_df.copy()
    new_df["IsCrack"] = new_df["IsCrack"].astype(object)
    n_matched = 0
    for _, old_row in old_labeled.iterrows():
        same_type = new_df["CandidateType"] == old_row["CandidateType"]
        close = ((new_df["X"] - old_row["X"]).abs() <= tol_px) & ((new_df["Y"] - old_row["Y"]).abs() <= tol_px)
        area_close = (new_df["Area"] - old_row["Area"]).abs() <= max(old_row["Area"] * area_tol_frac, 5)
        match = new_df.index[same_type & close & area_close]
        if len(match) >= 1:
            new_df.loc[match[0], "IsCrack"] = old_row["IsCrack"]
            new_df.loc[match[0], "CrackProbability"] = old_row["CrackProbability"]
            n_matched += 1
    return new_df, n_matched


def build_interior_candidates_for_image(image_name, save_quicklook=True):
    stage = run_production_pipeline(image_name)
    labeled, df, flat, img8, vesselness = stage["labeled"], stage["df"], stage["flat"], stage["img8"], stage["vesselness"]
    crack_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())

    # Computed ONCE per image and reused everywhere below -- see
    # _region_features / interior_fill_candidates docstrings for why
    # per-candidate (or per-call) recomputation of these full-frame ops is a
    # correctness-preserving but very expensive mistake.
    dil_crack = morphology.binary_dilation(crack_mask, morphology.disk(1))
    dist_to_crack = ndi.distance_transform_edt(~crack_mask)

    raw = concavity_candidates(crack_mask)
    raw += bridge_corridor_candidates(labeled, df, flat, crack_mask)
    raw += interior_fill_candidates(labeled, df, img8, crack_mask, dist_to_crack)

    records = []
    next_label = INTERIOR_LABEL_OFFSET
    quicklook_mask = np.zeros(labeled.shape, dtype=np.uint8)
    for mask_bool, ctype, parent in raw:
        feats = _region_features(mask_bool, flat, img8, vesselness, crack_mask, dil_crack, dist_to_crack)
        feats.update({
            "Label": next_label,
            "SourceImage": image_name,
            "CandidateType": ctype,
            "ParentLabels": parent,
            "IsCrack": "",       # to be filled by human review
            "CrackProbability": 0.5,  # unlabeled -> treated as maximally uncertain
        })
        records.append(feats)
        quicklook_mask[mask_bool] = {"concavity": 1, "bridge_corridor": 2, "interior_fill": 3}[ctype]
        next_label += 1

    cols = ["Label", "SourceImage", "CandidateType", "ParentLabels", "Area", "Width", "Height",
            "X", "Y", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
            "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
            "FracBoundaryTouchingCrack", "MeanDistToCrack", "LogArea",
            "IsCrack", "CrackProbability"]
    out_df = pd.DataFrame.from_records(records, columns=cols)
    out_csv = os.path.join(CANDIDATES_DIR, f"{image_name}_interior.csv")

    old_labeled, old_painted = _load_existing_labels(out_csv)
    if old_labeled is not None and len(old_labeled) > 0:
        out_df, n_matched = _carry_forward_labels(out_df, old_labeled)
        print(f"  carried forward {n_matched}/{len(old_labeled)} previously-labeled interior verdicts")
        if n_matched < len(old_labeled):
            print(f"  WARNING: {len(old_labeled) - n_matched} previous verdict(s) could not be matched "
                  f"to a regenerated candidate -- candidate generation may have changed for this image")
    if old_painted is not None and len(old_painted) > 0:
        out_df = pd.concat([out_df, old_painted], ignore_index=True)
        print(f"  preserved {len(old_painted)} user_painted candidate(s)")

    out_df.to_csv(out_csv, index=False)

    if save_quicklook:
        artifact_mask = np.isin(labeled, df.loc[~df["IsCrack"], "Label"].tolist())

        rgb = np.stack([img8] * 3, axis=-1).astype(float) / 255.0
        red = np.array([1, 0, 0])
        cyan = np.array([0, 0.7, 0.9])
        yellow = np.array([1, 0.85, 0])
        orange = np.array([1, 0.5, 0])
        magenta = np.array([0.85, 0, 0.85])
        rgb[artifact_mask] = rgb[artifact_mask] * 0.55 + cyan * 0.45
        rgb[crack_mask] = rgb[crack_mask] * 0.4 + red * 0.6
        rgb[quicklook_mask == 1] = rgb[quicklook_mask == 1] * 0.35 + yellow * 0.65
        rgb[quicklook_mask == 2] = rgb[quicklook_mask == 2] * 0.35 + orange * 0.65
        rgb[quicklook_mask == 3] = rgb[quicklook_mask == 3] * 0.35 + magenta * 0.65
        pil_img = Image.fromarray((rgb * 255).astype(np.uint8))

        # One number per merged crack GROUP (not one per fragment that
        # composes it), same simplification the production overlay uses --
        # otherwise a single long crack merged from many fragments would be
        # covered in overlapping numbers instead of carrying one.
        kept = df[df["IsCrack"]]
        if "CrackGroupID" in kept.columns and (kept["CrackGroupID"] >= 0).any():
            merged = kept[kept["CrackGroupID"] >= 0]
            standalone = kept[kept["CrackGroupID"] < 0]
            crack_to_label = pd.concat([merged.loc[merged.groupby("CrackGroupID")["Area"].idxmax()], standalone])
        else:
            crack_to_label = kept
        rejected = df[~df["IsCrack"]]

        draw_labels(pil_img, zip(crack_to_label["X"], crack_to_label["Y"], crack_to_label["Label"]),
                    color=(0, 255, 0))
        draw_labels(pil_img, zip(rejected["X"], rejected["Y"], rejected["Label"]), color=(0, 255, 255))
        if len(out_df):
            draw_labels(pil_img, zip(out_df["X"], out_df["Y"], out_df["Label"]), color=(255, 255, 255))

        pil_img.save(os.path.join(CANDIDATES_DIR, f"{image_name}_quicklook.png"))

    return out_df, stage


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image_name")
    args = ap.parse_args()
    df, _ = build_interior_candidates_for_image(args.image_name)
    print(f"{args.image_name}: {len(df)} interior candidates "
          f"({(df['CandidateType']=='concavity').sum()} concavity, "
          f"{(df['CandidateType']=='bridge_corridor').sum()} bridge_corridor, "
          f"{(df['CandidateType']=='interior_fill').sum()} interior_fill)")
