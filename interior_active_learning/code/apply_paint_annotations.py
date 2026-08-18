"""
Manual annotation mechanism, using the SAME two colors as the production
overlay itself: RED = crack, CYAN = artifact/not-crack. One unified rule
handles both adding new regions and correcting existing ones:

  - Paint red or cyan over BLANK background (not part of any existing
    candidate) -> creates a new CandidateType="user_painted" candidate
    with that verdict -- UNLESS the red stroke touches an existing crack
    region, in which case it's merged into that same crack instead (see
    interior_candidates._merge_blank_force_crack_into_touching_cracks) --
    a red stroke connecting to an existing red crack becomes part of THAT
    crack, not a second, coincidentally-adjacent one. Cyan/artifact strokes
    are never merged this way (production doesn't merge artifacts either).
  - Paint red over an EXISTING cyan (artifact) region, or cyan over an
    EXISTING red (crack) region -> corrects that PAINTED PATCH's verdict
    (flip it) rather than creating a new one, leaving the rest of that
    candidate's region untouched. This is a per-pixel correction (see
    interior_candidates.apply_pixel_corrections), not a whole-candidate
    flip -- a single existing "crack" region can be huge (e.g. the entire
    dark background of an image comes back as ONE connected region), so
    painting over part of it must correct only that part. Persisted in
    paint/<image>_correction_mask.png, which run_production_pipeline()
    applies on every future render, so the correction sticks.
  - Painting a color over a region that's ALREADY that same color is a
    no-op (nothing to add or correct).

Since _color_mask already excludes pixels that were already that color in
the template, "new red paint" can only ever land on either blank
background or an existing CYAN region -- never on existing red -- so which
case applies is unambiguous from the paint color and whatever Label (if
any) is underneath.

The Eraser tool is separate from red/cyan: it doesn't recolor a candidate
to the OPPOSITE verdict (still a labeled "artifact"), it removes it from
candidacy entirely -- back to plain, unmarked background, for the case
where a region is neither a real crack nor a meaningful artifact and
shouldn't be highlighted as either. Internally it's tracked as MAGENTA
paint (never exposed as a color swatch, and never naturally occurring in
the red/cyan-tinted template's color space, so detecting it is just as
unambiguous as red/cyan) and applied via the same per-pixel correction
mask as red/cyan corrections (see interior_candidates.apply_pixel_corrections).

The paint template is the PRODUCTION overlay only (red=confirmed crack,
cyan=artifact -- no concavity/bridge_corridor/interior_fill colors mixed
in; those are separate, still-experimental proposals reviewed through the
active-learning flow instead). Rendered fresh from the same arrays
run_production_pipeline() uses everywhere else, not the actual
results/*_cracks_overlay.png file, whose saved pixel dimensions don't
exactly match the source image (verified on two images, off by 9-21px per
axis) -- using it directly would silently misalign painted coordinates
with the real candidate mask.

Usage
-----
1. Make a paint template for an image:
     python3 apply_paint_annotations.py make-template 260708_316_H_b2_front_CBS_002
   This writes paint/<image>_paint_template.png.

2. Copy it to paint/<image>_painted.png and paint (in the paint app, or an
   external editor) PURE RED (255,0,0) = crack, PURE CYAN (0,204,255) =
   not-crack (matches the template's own artifact tint hue exactly), or
   PURE MAGENTA (255,0,255) = erase (remove from candidacy entirely,
   whether it's currently red or cyan) -- over whatever needs adding,
   correcting, or removing. Save in place. If you overdraw, undo (Preview:
   Cmd+Z) or re-copy the template over your painted file to start that
   image fresh -- the template itself is never modified.

3. Ingest it:
     python3 apply_paint_annotations.py ingest 260708_316_H_b2_front_CBS_002
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    CANDIDATES_DIR, PAINT_DIR, PAINT_LABEL_OFFSET, ORIGINAL_PAINT_CORRECTIONS_PATH,
    load_correction_mask, save_correction_mask,
)
from interior_candidates import (
    run_production_pipeline, build_simple_overlay,
    _region_features, apply_pixel_corrections,
)
# UNIFIED MODEL EXPERIMENT: substituted in place of interior_candidates.run_
# enhanced_pipeline (the two-separate-models version) -- see unified_pipeline.py's
# module docstring for what changed and why. This file is otherwise identical
# to the production apply_paint_annotations.py.
from unified_pipeline import run_unified_pipeline as run_enhanced_pipeline
from scipy import ndimage as ndi
from skimage import morphology

RED = np.array([255, 0, 0])
# Matches production save_overlay()'s own pre-blend artifact color exactly
# ([0, 0.8, 1.0] * 255 -- see detect_cracks.py) rather than pure cyan
# (0,255,255), which is why painted cyan used to look a shade off from the
# template's own cyan tint.
CYAN = np.array([0, 204, 255])
# Internal-only marker for the Eraser tool -- never offered as a paint color
# swatch. Mathematically can't occur in the template's own color space (a
# grayscale/red or grayscale/cyan blend, or plain grayscale, can never
# average out to pure magenta), so detecting it is exactly as unambiguous
# as detecting real red/cyan paint.
ERASE_MARKER = np.array([255, 0, 255])
COLOR_TOLERANCE = 40
MIN_ERASE_OVERLAP_PX = 15  # ignore a stray few pixels of paint accidentally grazing a neighboring candidate


def make_template(image_name):
    # run_enhanced_pipeline (not the bare production pipeline) so whatever
    # models/interior_model.joblib currently accepts shows up as plain red,
    # paintable and correctable exactly like any other candidate -- see its
    # docstring for why this is what closes the active-learning loop.
    # Degrades to the plain production result automatically if no model has
    # been trained yet (score_interior_candidates returns nothing to add).
    stage = run_enhanced_pipeline(image_name)
    template_path = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    build_simple_overlay(stage).save(template_path)
    print(f"Wrote {template_path}")
    print(f"Copy it to {os.path.join(PAINT_DIR, image_name + '_painted.png')}, paint RED "
          f"(255,0,0) = crack, CYAN (0,204,255) = not-crack -- over blank area to add a new "
          f"candidate, or over an existing opposite-colored region to correct it -- or MAGENTA "
          f"(255,0,255) to erase a region from candidacy entirely -- save, then run:")
    print(f"  python3 apply_paint_annotations.py ingest {image_name}")


def _find_next_label():
    """Highest existing user_painted Label across all candidate CSVs, +1."""
    import glob
    max_label = PAINT_LABEL_OFFSET
    for f in glob.glob(os.path.join(CANDIDATES_DIR, "*_interior.csv")):
        d = pd.read_csv(f)
        painted = d[d["CandidateType"] == "user_painted"]
        if len(painted):
            max_label = max(max_label, int(painted["Label"].max()))
    return max_label + 1


def _color_mask(painted, template, color):
    """Pixels freshly painted `color`, excluding any that were already that
    colour in the template -- which is what makes "new red paint" unambiguous.

    Compares SQUARED integer distances rather than float Euclidean norms.
    dist < T and dist**2 < T**2 are equivalent for non-negative values, but the
    float version converted a (H, W, 3) uint8 array to float32 first: on a
    6144x4096 image that is a 300 MB temporary, allocated twice per call and
    called three times per ingest. That was the single largest cost in saving a
    correction -- ~2.2s of a 7s save, before counting the memory pressure.
    int16 is wide enough: channel differences are within +/-255 and the squared
    sum of three of them fits in int32.
    """
    # int32, NOT int16. A channel difference reaches 255, whose square is 65025 --
    # already past int16's 32767 -- and np.einsum accumulates in the INPUT dtype,
    # so an int16 version silently wrapped negative and reported far-apart colours
    # as matches. Observed: pure red paint on a white template was classified as
    # cyan, and only 418 of 3200 painted pixels were seen as red.
    tol2 = np.int32(int(COLOR_TOLERANCE) ** 2)
    col = np.asarray(color, dtype=np.int32)
    d = painted.astype(np.int32) - col
    mask = np.einsum("ijk,ijk->ij", d, d) < tol2
    dt = template.astype(np.int32) - col
    mask &= np.einsum("ijk,ijk->ij", dt, dt) >= tol2
    return mask


def _split_new_vs_correction(mask, labeled):
    """A painted mask splits into: pixels over existing candidates (label
    != 0, each already necessarily the OPPOSITE color per _color_mask's
    exclusion -- i.e. a correction) vs pixels over blank background
    (label == 0 -- new candidate material)."""
    over_existing = mask & (labeled != 0)
    over_blank = mask & (labeled == 0)
    return over_existing, over_blank


def _filter_by_overlap(over_existing_mask, labeled, min_overlap):
    """Drop touched-Label pixels whose total overlap with the painted mask is
    below min_overlap -- ignores a stray few pixels of paint accidentally
    grazing a neighboring candidate rather than treating it as an intentional
    correction. Returns (filtered_mask, sorted list of qualifying Labels)."""
    if not over_existing_mask.any():
        return over_existing_mask, []
    touched_labels, counts = np.unique(labeled[over_existing_mask], return_counts=True)
    qualifying = sorted(int(lbl) for lbl, count in zip(touched_labels, counts) if lbl != 0 and count >= min_overlap)
    if not qualifying:
        return np.zeros_like(over_existing_mask), []
    filtered = over_existing_mask & np.isin(labeled, qualifying)
    return filtered, qualifying


def _new_candidates_from_mask(mask, is_crack, next_label, flat, img8, vesselness,
                               crack_mask, dil_crack, dist_to_crack, min_area, labeled):
    """labeled is mutated in place: each qualifying new candidate's pixels
    are stamped with its Label number, same as any other candidate --
    otherwise these pixels stay Label 0 forever, and build_simple_overlay
    (which only ever renders np.isin(labeled, df["Label"])) has no way to
    color them at all. Confirmed this exact gap: a cyan stroke painted on
    blank background correctly became a real, labeled row in
    candidates/*_interior.csv (so training data was fine), but visually
    reverted to plain, uncolored background the moment the template was
    regenerated -- "looks like I painted nothing there" -- because the
    pixel array never actually got told about the new Label."""
    records = []
    comp_labeled = measure.label(mask, connectivity=2)
    for c in measure.regionprops(comp_labeled):
        if c.area < min_area:
            continue
        mask_bool = comp_labeled == c.label
        feats = _region_features(mask_bool, flat, img8, vesselness, crack_mask, dil_crack, dist_to_crack)
        feats.update({
            "Label": next_label,
            "SourceImage": None,  # filled in by caller
            "CandidateType": "user_painted",
            "ParentLabels": "",
            "IsCrack": is_crack,
            "CrackProbability": 1.0 if is_crack else 0.0,
        })
        records.append(feats)
        labeled[mask_bool] = next_label
        next_label += 1
    return records, next_label


def _log_touched_labels(image_name, labels, verdict):
    """Append (SourceImage, Label) pairs to ORIGINAL_PAINT_CORRECTIONS_PATH --
    purely so active_learning_select.py can keep a corrected candidate from
    resurfacing for review. NOT authoritative for the actual verdict anymore
    (the per-pixel correction mask is) -- CorrectedTo (True/False/"erased")
    is kept only as a human-readable hint of what the correction was."""
    if not labels:
        return
    new_df = pd.DataFrame([{"SourceImage": image_name, "Label": lbl, "CorrectedTo": verdict} for lbl in labels])
    if os.path.exists(ORIGINAL_PAINT_CORRECTIONS_PATH):
        existing = pd.read_csv(ORIGINAL_PAINT_CORRECTIONS_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(ORIGINAL_PAINT_CORRECTIONS_PATH, index=False)


def _log_interior_origin_corrections(image_name, interior_origin, touched_labels, is_crack):
    """When a paint correction touches a Label that run_enhanced_pipeline
    folded in as an accepted interior candidate (see its docstring in
    interior_candidates.py), record the NEW verdict as a properly-typed row
    in candidates/<image>_interior.csv -- reusing the CandidateType and
    features already computed when it was accepted, rather than letting it
    fall through as a generic "user_painted" placeholder. This is what
    actually closes the loop: overturning one of the interior model's own
    mistakes becomes a real training example for THAT specific candidate
    type the next time train_interior_model.py runs, rather than just a
    visual flip that teaches the model nothing (interior_fill in particular
    has been starved of real negative examples all along -- see
    models/checkpoints/README.md)."""
    if not interior_origin:
        return
    rows = []
    for lbl in touched_labels:
        origin = interior_origin.get(lbl)
        if origin is None:
            continue  # an ordinary production/user_painted Label, not one of ours
        ctype, feats = origin
        row = dict(feats)
        row.update({
            "Label": lbl, "SourceImage": image_name, "CandidateType": ctype,
            "ParentLabels": "", "IsCrack": is_crack, "CrackProbability": 1.0 if is_crack else 0.0,
        })
        rows.append(row)
    if not rows:
        return
    new_df = pd.DataFrame.from_records(rows)
    csv_path = os.path.join(CANDIDATES_DIR, f"{image_name}_interior.csv")
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        # see the matching comment further down in ingest() -- same dtype
        # guard, same reason.
        existing["IsCrack"] = existing["IsCrack"].astype(object)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)


def ingest(image_name, min_area=20, stage=None):
    """stage: an already-computed run_enhanced_pipeline() result, if the
    caller has one cached (paint_server.py does, per open image) -- avoids
    re-running the ENTIRE production pipeline (background flattening,
    vesselness, ML classification, merge_large_cracks -- the genuinely slow
    parts) on every single ingest, which is what made repeated paint/erase/
    check cycles painfully slow. Recomputed fresh only if not provided
    (e.g. the CLI usage, which has no cache to reuse)."""
    template_path = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    painted_path = os.path.join(PAINT_DIR, f"{image_name}_painted.png")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"{template_path} -- run make-template first")
    if not os.path.exists(painted_path):
        raise FileNotFoundError(f"{painted_path} -- copy the template, paint it, save it here first")

    painted = np.array(Image.open(painted_path).convert("RGB"))
    template = np.array(Image.open(template_path).convert("RGB"))
    if painted.shape != template.shape:
        raise ValueError(f"painted image is {painted.shape}, template is {template.shape} -- "
                          f"don't resize/crop the file, just paint on top of it")

    red_mask = _color_mask(painted, template, RED)
    cyan_mask = _color_mask(painted, template, CYAN)
    erase_mask = _color_mask(painted, template, ERASE_MARKER)
    print(f"Detected {int(red_mask.sum())} red (crack) + {int(cyan_mask.sum())} cyan (not-crack) "
          f"+ {int(erase_mask.sum())} erased pixels")
    if not red_mask.any() and not cyan_mask.any() and not erase_mask.any():
        print("No paint detected -- nothing to ingest.")
        return {"n_candidates": 0, "n_corrections": 0, "message": "No paint detected -- nothing to ingest."}

    if stage is None:
        stage = run_enhanced_pipeline(image_name)
    labeled, df, flat, img8, vesselness = stage["labeled"], stage["df"], stage["flat"], stage["img8"], stage["vesselness"]
    crack_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
    dil_crack = morphology.binary_dilation(crack_mask, morphology.disk(1))
    dist_to_crack = ndi.distance_transform_edt(~crack_mask)

    red_over_existing, red_over_blank = _split_new_vs_correction(red_mask, labeled)
    cyan_over_existing, cyan_over_blank = _split_new_vs_correction(cyan_mask, labeled)
    # erasing blank background (nothing there to begin with) is a no-op --
    # only pixels actually over an existing candidate matter
    erase_over_existing, _ = _split_new_vs_correction(erase_mask, labeled)

    # A new red stroke that TOUCHES an existing crack region should extend/
    # join that crack rather than becoming a second, unrelated candidate
    # that only looks contiguous because both happen to render red -- route
    # those pixels through the correction mask (as force-crack) instead of
    # the brand-new-candidate path; see
    # interior_candidates._merge_blank_force_crack_into_touching_cracks.
    # Checked per connected component so one stroke that only touches at
    # one end isn't fragmented into a merged part and an isolated part.
    red_blank_components = measure.label(red_over_blank, connectivity=2)
    touching_component_ids = [
        r.label for r in measure.regionprops(red_blank_components)
        if ((red_blank_components == r.label) & dil_crack).any()
    ]
    red_blank_touching = (np.isin(red_blank_components, touching_component_ids)
                           if touching_component_ids else np.zeros_like(red_over_blank))
    red_blank_isolated = red_over_blank & ~red_blank_touching

    # Corrections are applied at the PIXEL level (merged into a persistent
    # correction mask that apply_pixel_corrections() reads on every future
    # run_production_pipeline call), not as a whole-Label flip -- painting
    # cyan over (or erasing) a small patch of an enormous "crack" region
    # should correct only that patch, not the entire region. See
    # apply_pixel_corrections()'s docstring in interior_candidates.py for why.
    red_correct_mask, red_touched = _filter_by_overlap(red_over_existing, labeled, MIN_ERASE_OVERLAP_PX)
    cyan_correct_mask, cyan_touched = _filter_by_overlap(cyan_over_existing, labeled, MIN_ERASE_OVERLAP_PX)
    erase_correct_mask, erase_touched = _filter_by_overlap(erase_over_existing, labeled, MIN_ERASE_OVERLAP_PX)
    # Brand-new regions painted on BLANK background are recorded here too.
    #
    # They were previously handled only by _new_candidates_from_mask, which
    # stamps them into `labeled` and appends a row to the candidates CSV -- so
    # they rendered correctly and were clickable, but never entered the
    # correction mask. build_training_data.py builds training rows exclusively
    # from that mask, so a crack the model missed ENTIRELY -- exactly the case a
    # user most wants to teach it -- contributed nothing to the next retrain.
    # Corrections to regions the model already proposed always worked; only
    # from-scratch paint was silently non-teaching.
    #
    # Recording them as verdicts is consistent with what the mask means: a
    # per-pixel human judgement, independent of whether the pipeline happened to
    # propose that pixel. It also makes them survive a re-render, which the CSV
    # row alone did not guarantee.
    any_write = (red_correct_mask.any() or cyan_correct_mask.any()
                 or erase_correct_mask.any() or red_blank_touching.any()
                 or red_blank_isolated.any() or cyan_over_blank.any())
    if any_write:
        merged = load_correction_mask(image_name, labeled.shape)
        merged = merged.copy() if merged is not None else np.zeros(labeled.shape, dtype=np.uint8)
        merged[red_correct_mask] = 1
        merged[cyan_correct_mask] = 2
        merged[erase_correct_mask] = 3
        merged[red_blank_touching] = 1
        # Order is irrelevant here, and the earlier claim that it made
        # corrections "win" was wrong: _split_new_vs_correction partitions each
        # painted mask by label != 0 vs label == 0, so the correction masks and
        # these blank-background masks are disjoint by construction. Red and
        # cyan cannot overlap either -- _color_mask requires a pixel to be
        # within tolerance of one colour, and the two are far apart.
        merged[red_blank_isolated] = 1
        merged[cyan_over_blank] = 2
        save_correction_mask(image_name, merged)
    _log_touched_labels(image_name, red_touched, True)
    _log_touched_labels(image_name, cyan_touched, False)
    _log_touched_labels(image_name, erase_touched, "erased")
    interior_origin = stage.get("interior_origin", {})
    _log_interior_origin_corrections(image_name, interior_origin, red_touched, True)
    _log_interior_origin_corrections(image_name, interior_origin, cyan_touched, False)
    n_corrections = len(red_touched) + len(cyan_touched) + len(erase_touched)
    n_extended = len(touching_component_ids)

    # red (crack) and cyan (not-crack) are opposite verdicts, so unlike two
    # same-colored strokes that touch, they're ALWAYS kept as separate
    # candidates -- each gets its own independent connected-component pass.
    # red_blank_isolated (not red_over_blank) is used here because the
    # touching portion was already routed into the crack-extension merge
    # above -- creating a separate candidate for it too would duplicate it.
    next_label = _find_next_label()
    records, next_label = _new_candidates_from_mask(
        red_blank_isolated, True, next_label, flat, img8, vesselness, crack_mask, dil_crack, dist_to_crack, min_area, labeled)
    more_records, next_label = _new_candidates_from_mask(
        cyan_over_blank, False, next_label, flat, img8, vesselness, crack_mask, dil_crack, dist_to_crack, min_area, labeled)
    records += more_records
    for r in records:
        r["SourceImage"] = image_name
    if records:
        # Fold these new rows into the in-memory df too (not just the CSV
        # below) -- labeled now has their pixels stamped with real Label
        # numbers (see _new_candidates_from_mask), so df needs matching rows
        # or the commit step's build_simple_overlay would render an
        # orphaned Label that matches nothing in df, which np.isin just
        # silently treats as "not a candidate" -- the same invisible-paint
        # symptom this fix exists to close, just moved one step later.
        df = pd.concat([df, pd.DataFrame.from_records(records)], ignore_index=True)

    if records:
        new_df = pd.DataFrame.from_records(records)
        csv_path = os.path.join(CANDIDATES_DIR, f"{image_name}_interior.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            # If every row in the existing file is still unlabeled, IsCrack
            # round-trips through CSV as an all-NaN float64 column -- concat
            # against new_df's real Python True/False then silently
            # downcasts them to 1.0/0.0 to match, which
            # load_labeled_interior()'s string-based filter doesn't
            # recognize as a label at all (confirmed: this exact bug
            # silently dropped 20 real corrections, 9 of them cyan, from
            # two images before it was caught). Cast to object first, same
            # fix ingest_labels.py already applies for the same reason.
            existing["IsCrack"] = existing["IsCrack"].astype(object)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(csv_path, index=False)

    # Commit: fold this ingest's corrections into labeled/df, regenerate the
    # template from that committed state, and reset every pixel this ingest
    # just consumed (a correction, an erasure, OR a brand new candidate) so a
    # resumed/reloaded session shows the true committed result instead of a
    # stale leftover stroke. Without this: (a) an erased patch would still
    # show its erase marker after saving, (b) a new candidate painted on
    # blank background looks IDENTICAL to an un-ingested stroke and would
    # get detected -- and duplicated -- all over again on this image's next
    # ingest (labeled/df already know about it now -- see
    # _new_candidates_from_mask -- so this reset just makes painted.png
    # match its real, rendered color instead of stripping it back to a
    # stale, unpainted background).
    full_correction_mask = load_correction_mask(image_name, labeled.shape)
    if full_correction_mask is not None:
        committed_labeled, committed_df = apply_pixel_corrections(labeled, df, full_correction_mask)
    else:
        committed_labeled, committed_df = labeled, df
    stage["labeled"], stage["df"] = committed_labeled, committed_df
    new_template_img = build_simple_overlay({"img8": img8, "labeled": committed_labeled, "df": committed_df})
    new_template_arr = np.array(new_template_img)
    consumed = red_mask | cyan_mask | erase_mask
    painted[consumed] = new_template_arr[consumed]
    # compress_level=1: these are ~23 MB overlays rewritten on every single
    # correction, and the default level 6 spends seconds squeezing a file that
    # is read back locally milliseconds later.
    Image.fromarray(painted).save(painted_path, compress_level=1)
    new_template_img.save(template_path, compress_level=1)

    n_pos = sum(1 for r in records if r["IsCrack"])
    n_neg = len(records) - n_pos
    parts = []
    if records:
        parts.append(f"{len(records)} new candidate(s) ({n_pos} crack, {n_neg} not-crack)")
    if n_corrections:
        parts.append(f"{n_corrections} existing-candidate correction(s)")
    if n_extended:
        parts.append(f"{n_extended} crack region(s) extended/joined")
    msg = ("Added " + ", ".join(parts) + ".") if parts else "Paint detected but nothing met min_area/overlap thresholds."
    print(msg)
    return {"n_candidates": len(records), "n_corrections": n_corrections, "n_extended": n_extended, "message": msg}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("make-template")
    p1.add_argument("image_name")
    p2 = sub.add_parser("ingest")
    p2.add_argument("image_name")
    args = ap.parse_args()
    if args.cmd == "make-template":
        make_template(args.image_name)
    else:
        ingest(args.image_name)
