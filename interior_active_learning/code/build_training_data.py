"""
Turn the per-pixel correction masks into training rows for the whole dataset.

The correction masks are the authoritative record of every human verdict
(1 = crack, 2 = not-crack, 3 = erased from candidacy, 0 = never reviewed).
Only 1 and 2 become training rows: an erased region is not a material
feature at all, and an unreviewed one is not evidence either way.

Features are read STRAIGHT OUT of extract_candidates' own dataframe rather
than recomputed here. That is deliberate: an earlier version of a similar
script recomputed MeanDarkness as mean(flat) when extract_candidates
defines it as mean(255 - flat), i.e. sign-flipped, which silently corrupted
every downstream number until the discrepancy was chased down. Reading the
pipeline's own columns makes that class of bug impossible.

    python3 build_training_data.py
Writes ../../training_data/labeled_regions.csv
"""
import os
import sys
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import numpy as np
import pandas as pd

from common import ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, contrast_kwargs_for, load_correction_mask
from detect_cracks import (
    load_as_uint8, find_field_of_view, flatten_background, segment_dark_regions,
    clean_mask, compute_vesselness, extract_candidates,
    region_features_from_labeled,
)

OUT_CSV = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")

# The production classifier's own 8 features, all supplied directly by
# extract_candidates -- no recomputation, no placeholders.
FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity",
            "Extent", "Circularity", "MeanDarkness", "MeanVesselness"]


def labeled_images():
    names = []
    for f in sorted(os.listdir(PAINT_DIR)):
        if not f.endswith("_correction_mask.png"):
            continue
        n = f.replace("_correction_mask.png", "")
        if os.path.exists(os.path.join(ORIGINAL_DIR, f"{n}.tif")):
            names.append(n)
    return names


def process(image_name):
    try:
        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{image_name}.tif"),
                             **contrast_kwargs_for(image_name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        flat = flatten_background(img8)
        clean = clean_mask(segment_dark_regions(flat, img8=img8), min_area_px=13)
        ves = compute_vesselness(flat)
        # DELIBERATELY NOT CALLED -- serving does not call it either.
        #
        # unified_pipeline.py removed exclude_border_background() because it deleted main
        # cracks (35-37% of the user's hand marks on two images). Leaving it here made the
        # training builder and the inference pipeline run DIFFERENT segmenters over the
        # same images, with two consequences:
        #
        #   (a) The accepted false-positive class became unlearnable. Off-specimen
        #       background flooding the MAR Cast frames is tolerated on the grounds that
        #       the user paints it not-crack once and the correction sticks. It does stick
        #       for that image, via the pixel mask -- but this builder deleted exactly
        #       those large, border-touching, low-vesselness regions BEFORE the mask vote,
        #       so they produced zero negative rows. The user's most repeated correction
        #       could never become training data, which is why the accepted cost was not
        #       decaying with use.
        #   (b) Region Label IDs diverged. extract_candidates renumbers survivors 1..N in
        #       scan order, so removing regions shifts every later ID -- and the override
        #       ledger is keyed by ID.
        labeled, df = extract_candidates(clean, flat, ves, min_area_px=40)

        mask = load_correction_mask(image_name, labeled.shape)
        if mask is None:
            return image_name, None

        rows = []
        for _, r in df.iterrows():
            lbl = int(r["Label"])
            vals = mask[labeled == lbl]
            if vals.size == 0:
                continue
            counts = np.bincount(vals, minlength=4)
            if counts[1:].sum() == 0:
                continue                      # never reviewed
            verdict = int(counts[1:].argmax() + 1)
            if verdict not in (1, 2):
                continue                      # erased
            rec = {f: float(r[f]) for f in FEATURES}
            rec.update({"IsCrack": verdict == 1, "SourceImage": image_name,
                         "Label": lbl, "Area": int(r["Area"])})
            rows.append(rec)

        # Human-painted regions the SEGMENTER never proposed.
        #
        # The loop above iterates extract_candidates' output, so a verdict only
        # becomes a training row if the darkness threshold happened to propose
        # that region. A crack painted from scratch on blank background has no
        # candidate -- apply_pixel_corrections deliberately leaves an isolated
        # blank force-crack patch alone ("the genuinely-new-candidate case") --
        # so the single most informative annotation a user can make, "you missed
        # this entirely", produced no training data at all.
        #
        # This matters more now that SAM is in the pipeline: SAM proposes regions
        # the darkness threshold misses, and the same classifier scores them, so
        # teaching it these shapes directly improves that path.
        # Labelled PER VERDICT, not over both at once. Labelling
        # isin(mask, (1, 2)) together merges a red patch and a cyan patch that
        # touch into ONE component, and the majority vote below then assigns the
        # whole thing whichever verdict has more pixels -- so painting "not crack"
        # next to "crack" could produce a training row claiming the not-crack area
        # IS a crack. Each component now contains one verdict by construction, so
        # the vote is a formality kept only for robustness.
        n_extra = 0
        from skimage import measure as _measure
        for verdict in (1, 2):
            uncovered = (mask == verdict) & (labeled == 0)
            if not uncovered.any():
                continue
            comp = _measure.label(uncovered, connectivity=2)
            for pr in _measure.regionprops(comp):
                if pr.area < 40:
                    continue
                sl = pr.slice
                sub = pr.image
                _, fd = region_features_from_labeled(
                    sub.astype(np.int32), flat[sl], ves[sl], min_area_px=40)
                if not len(fd):
                    continue
                r = fd.iloc[0]
                rec = {f: float(r[f]) for f in FEATURES}
                # Labels must not collide between the two verdict passes, which
                # both number their components from 1.
                rec.update({"IsCrack": verdict == 1, "SourceImage": image_name,
                            "Label": -(int(pr.label) * 10 + verdict), "Area": int(pr.area)})
                rows.append(rec)
                n_extra += 1

        out = pd.DataFrame(rows)
        n_pos = int(out["IsCrack"].sum()) if len(out) else 0
        print(f"{image_name:32s} {len(out):5d} rows ({n_pos} crack / {len(out)-n_pos} not-crack"
              + (f", {n_extra} painted-from-scratch" if n_extra else "") + ")", flush=True)
        return image_name, out
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{image_name:32s} FAILED: {e}", flush=True)
        return image_name, None


if __name__ == "__main__":
    names = labeled_images()
    print(f"{len(names)} images have correction masks AND a local image file\n")
    with Pool(3) as pool:
        results = pool.map(process, names) if names else []

    frames = [o for _, o in results if o is not None and len(o)]
    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # MERGE rather than replace. A distributed checkout ships this CSV but not
    # the source images (they are gigabytes), so recomputing from local images
    # alone would throw away every previously-labelled region and, on a fresh
    # clone with no images at all, produce nothing -- which broke the app's
    # Retrain button entirely.
    #
    # Rows for images present locally are recomputed (the mask is authoritative
    # and may have changed); rows for images that are absent are carried over
    # untouched. So a user who clones, adds their own images and corrects them
    # ADDS to the shipped labels instead of destroying them.
    prior = pd.DataFrame()
    if os.path.exists(OUT_CSV):
        try:
            prior = pd.read_csv(OUT_CSV)
        except Exception:
            prior = pd.DataFrame()
    if len(prior):
        recomputed = set(fresh["SourceImage"].unique()) if len(fresh) else set()
        carried = prior[~prior["SourceImage"].isin(recomputed)]
        n_carry_imgs = carried["SourceImage"].nunique() if len(carried) else 0
        if n_carry_imgs:
            print(f"\ncarrying over {len(carried)} rows from {n_carry_imgs} image(s) "
                  f"whose source file is not present locally")
        allrows = pd.concat([carried, fresh], ignore_index=True) if len(fresh) else carried
    else:
        allrows = fresh

    if not len(allrows):
        print("NO ROWS PRODUCED -- no correction masks with local images, and no "
              "existing training_data/labeled_regions.csv to build on")
        sys.exit(1)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    allrows.to_csv(OUT_CSV, index=False)

    n_pos = int(allrows["IsCrack"].sum())
    per = allrows.groupby("SourceImage")["IsCrack"].agg(["size", "sum"])
    per["neg"] = per["size"] - per["sum"]
    print(f"\nTOTAL {len(allrows)} rows: {n_pos} crack / {len(allrows)-n_pos} not-crack "
          f"across {allrows['SourceImage'].nunique()} images")
    print(f"images with >=1 negative: {int((per['neg'] > 0).sum())}  "
          f"(grouped CV needs at least ~3 to be meaningful)")
    print(f"\nWrote {OUT_CSV}")

    # ACCOUNT FOR EVERY MASK. 7 of 39 hand-drawn masks used to contribute zero rows and
    # nothing said so: the per-image worker catches any exception, prints FAILED and
    # returns None while the pool carries on, the merge silently carries prior rows for
    # images whose .tif is absent, and the process still exits 0. That is 18% of the
    # labelling effort in the repo disappearing into a successful-looking run.
    #
    # Masks are the one artifact here a human made by hand. A build that drops one must say
    # which, and must not report success.
    masks = {f[:-len("_correction_mask.png")] for f in os.listdir(PAINT_DIR)
             if f.endswith("_correction_mask.png")}
    represented = set(allrows["SourceImage"].unique())
    missing = sorted(masks - represented)
    print(f"\nmasks on disk: {len(masks)}   represented in the training data: "
          f"{len(masks & represented)}")
    if missing:
        print(f"\n{len(missing)} HAND-DRAWN MASK(S) CONTRIBUTED NO ROWS -- "
              f"that labelling effort is not reaching the model:")
        for n in missing:
            _tif = os.path.join(ORIGINAL_DIR, f"{n}.tif")
            # A mask holding only erase (3) or only unreviewed (0) pixels is legitimately
            # row-free and is not evidence of a fault; saying "broken" about it sends the
            # user hunting for a bug that is not there.
            _reason = "source .tif missing" if not os.path.exists(_tif) else None
            if _reason is None:
                try:
                    import numpy as _np
                    from PIL import Image as _Im
                    _a = _np.asarray(_Im.open(os.path.join(PAINT_DIR,
                                                           f"{n}_correction_mask.png")))
                    if _a.ndim > 2:
                        _a = _a[..., 0]
                    _has_verdict = bool((_a == 1).any() or (_a == 2).any())
                    _reason = ("no crack/not-crack verdicts in the mask (only erased or "
                               "unreviewed pixels) -- nothing to learn from, not a fault"
                               if not _has_verdict else
                               "mask has verdicts but produced no rows: shape mismatch "
                               "against the current render, or every marked pixel fell "
                               "outside a candidate region")
                except Exception as _e:
                    _reason = f"mask unreadable: {type(_e).__name__}"
            why = _reason
            print(f"    {n:36s} {why}")
        # NOT a non-zero exit. The first version of this block called sys.exit(2), and on
        # this repo the condition is ALWAYS true -- 39 masks on disk, 32 represented, 7
        # unbuildable and committed -- so app_endpoints saw returncode 2, raised
        # RuntimeError, and the Retrain button failed permanently for every user with a
        # message that read like corruption. Nothing the researcher paints removes those 7,
        # so it could never clear.
        #
        # The CSV is already written and correct at this point. Reporting a real problem
        # must not destroy the capability the report is about, so this warns and carries the
        # list in the printed output for the UI to surface.
        print("\nThe rows above were still written; retraining continues. Fix the listed "
              "masks to get their labels into the model.")
