"""
Single-model variant of the two-stage production pipeline -- built to let
manual correction/retraining happen against ONE shared model instead of a
separate 8-feature step-E classifier and an 11-feature step-H classifier.
See ../../UNIFIED_MODEL_EXPERIMENT_NOTES.md for the validation behind this
(pooling 71 human-verified original-candidate corrections with the existing
243 interior examples doesn't weaken the interior_fill leave-one-out safety
margin -- if anything it holds slightly tighter).

IMPORTANT STRUCTURAL POINT, worth being clear about: using one model does
NOT collapse this into one pass. concavity/bridge_corridor/interior_fill
candidates are only DEFINABLE relative to an already-decided crack mask --
that's a property of the PROBLEM (you can't ask "how close is this to a
confirmed crack" before anything is confirmed), not of how many models are
used. What's unified here is the MODEL -- one set of learned weights, one
training/retraining process, one joblib file -- not the number of passes.

Pass 1: classify the original darkness-threshold candidates with the
PRODUCTION 8-feature classifier (models/crack_classifier.joblib), exactly as
the production pipeline's own Step E does. An earlier version used the
11-feature unified model here with training-mean placeholders for the two
crack-context features, which are undefined before any crack is confirmed;
measured against hand labels that cost specificity 74.1% -> 39.8% for a
+5% recall gain, and made the paint app's rendered template disagree with
its own seeded proposals. See score_pass1_candidates' docstring.

Pass 2: generate concavity/bridge_corridor/interior_fill candidates
relative to THAT crack_mask (the same generator functions the production
pipeline already uses), compute their REAL 11-feature vector (now
genuinely meaningful, since a crack mask exists), and score them with the
SAME unified model + its own calibrated interior_fill hybrid rule. Accepted
regions get folded into the final crack mask, same as the two-stage system.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from scipy import ndimage as ndi
from skimage import morphology, measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import external_mask
from common import (
    ORIGINAL_DIR, CANDIDATES_DIR, MODELS_DIR, PROD_MODEL_PATH, contrast_kwargs_for,
    load_hard_overrides, load_correction_mask,
)
from detect_cracks import (
    load_as_uint8, find_field_of_view, flatten_background, segment_dark_regions,
    clean_mask, compute_vesselness, extract_candidates, merge_large_cracks,
    classify_with_model, exclude_border_background,
)
from active_learning_select import INTERIOR_FEATURE_COLUMNS
from interior_candidates import (
    apply_pixel_corrections, concavity_candidates, bridge_corridor_candidates,
    interior_fill_candidates, _region_features,
)


def _unified_model_path():
    return os.path.join(MODELS_DIR, "unified_model.joblib")


def _load_unified_bundle():
    path = _unified_model_path()
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _score(bundle, feats_list, ctype):
    """feats_list: list of feature dicts. Returns list of (is_accepted, proba)."""
    if not feats_list:
        return []
    X = bundle["scaler"].transform([[f[c] for c in INTERIOR_FEATURE_COLUMNS] for f in feats_list])
    probas = bundle["clf"].predict_proba(X)[:, 1]
    rule = bundle.get("interior_fill_rule")
    out = []
    for feats, proba in zip(feats_list, probas):
        if ctype == "interior_fill" and rule is not None:
            accepted = (proba >= rule["floor"] and feats["MeanDistToCrack"] <= rule["dist_thr"]
                        and feats["MeanFlatBrightness"] <= rule["bri_thr"])
        else:
            _t2 = (THRESHOLD_OVERRIDE if THRESHOLD_OVERRIDE is not None
                   else bundle.get("threshold_default", 0.5))
            accepted = proba >= _t2
        out.append((accepted, float(proba)))
    return out


#: Set to a float to run BOTH passes at that decision threshold instead of at whatever
#: each model bundle carries. None -- the default, and the only value the app or any
#: existing script ever sees -- means "use the bundle's own threshold", so this is inert
#: unless a caller deliberately sets it.
#:
#: It exists because the batch CLI exposes --threshold, and the alternative was for a
#: shipped entry point to monkeypatch classify_with_model at runtime. A patch that reaches
#: into another module's namespace works right up until someone reorders an import, and
#: then it fails by silently running at the default -- which is the exact class of error
#: this pipeline is supposed to make impossible. BOTH passes must honour it: patching Pass
#: 1 alone would report half a detector.
THRESHOLD_OVERRIDE = None


def score_pass1_candidates(df, labeled, img8, flat, vesselness, bundle=None):
    """Pass 1: classify the ORIGINAL darkness-threshold candidates using the
    production 8-feature classifier (models/crack_classifier.joblib) via
    detect_cracks.classify_with_model() -- the same call the production
    pipeline itself makes.

    This deliberately does NOT use the 11-feature unified model, even though
    Pass 2 does. Two of the unified model's features (MeanDistToCrack,
    FracBoundaryTouchingCrack) are undefined here by construction -- there is
    no confirmed crack yet to measure distance from -- so an earlier version
    substituted the training-pool mean for both. Measured on 1285
    hand-labelled candidates of AS_24hr_BSE_Side_008, that placeholder
    approach costs far more than it gains:

        8-feature production model : recall 89.3%, specificity 74.1%
        11-feature + placeholders  : recall 94.3%, specificity 39.8%

    i.e. it accepts ~5% more real cracks while wrongly accepting 60% of
    non-cracks -- on one real image that flipped 159 of 282 proposed regions
    relative to the production classifier. Since the paint app rewrites its
    on-disk template from whatever this returns, that disagreement showed up
    directly as regions changing colour the moment a user clicked anything.

    Pass 2 keeps the unified model, where both context features are real.

    Returns a COPY of df with IsCrack/CrackProbability set; merging, hard
    overrides and pixel corrections happen in run_unified_pipeline() after
    this returns. `bundle` is accepted and ignored, so existing callers
    (run_unified_pipeline, pipeline_stages_unified) need no change."""
    if not len(df):
        df = df.copy()
        df["IsCrack"] = False
        df["CrackProbability"] = 0.0
        return df
    # force_keep_area is classify_with_model's own default (50000): a dark
    # region that large is essentially always a real crack/void regardless
    # of what a size-sensitive model says.
    return classify_with_model(df, PROD_MODEL_PATH, proba_threshold=THRESHOLD_OVERRIDE)


def run_unified_pipeline(image_name, stage=None):
    """Same return shape as interior_candidates.run_enhanced_pipeline() (a
    dict with img8/flat/labeled/df/vesselness/interior_origin/
    n_interior_total), so this can be dropped into the paint app / apply
    script as a straight substitute -- the only difference is HOW labeled/df
    get decided (one shared model, two passes, instead of two separately-
    trained models)."""
    bundle = _load_unified_bundle()

    if stage is None:
        image_path = os.path.join(ORIGINAL_DIR, f"{image_name}.tif")
        img8 = load_as_uint8(image_path, **contrast_kwargs_for(image_name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        flat = flatten_background(img8)
        dark_mask = segment_dark_regions(flat, img8=img8)
        clean = clean_mask(dark_mask, min_area_px=13)
        vesselness = compute_vesselness(flat)
        # exclude_border_background() USED TO RUN HERE. It is deliberately not
        # called any more, and the reason is measured, not stylistic.
        #
        # It removed large, border-touching, low-vesselness regions as off-specimen
        # background. But a wide open crack IS large, border-touching and smooth, so
        # it deleted main cracks -- verified against the human masks: 2,500,241 px
        # removed on ..._front_CBS_004 including 69,497 px the user had marked BY HAND
        # as crack (35% of their marks on that image), and 1,189,583 px on _003
        # including 167,607 hand-marked px (37%). Those pixels never reached the
        # classifier, so the main crack rendered black no matter what the model did.
        #
        # Recalibrating max_vesselness cannot save it. Measured across all 62 images,
        # the two populations overlap completely: deleted regions that are
        # hand-marked crack span vesselness 0.000027-0.002260, while large
        # unmarked (candidate background) regions span 0.000181-0.002344. There is
        # no cut point. The feature does not separate crack from background here,
        # which is exactly why the archived pipeline never called this.
        #
        # Off-specimen background does still get proposed on some edge captures --
        # the real problem this was written for, and on the MAR Cast frames it floods
        # most of the lower frame red. That is a visible, correctable false positive
        # (paint it not-crack once and the correction overrides the model for good),
        # whereas deleting the main crack was silent and uncorrectable.
        #
        # THREE discriminators were measured on this dataset and ALL of them overlap,
        # so do not reintroduce a region-level rule expecting one to work:
        #   vesselness      crack 0.000027-0.002260   vs  background 0.000181-0.002344
        #   raw darkness    INVERTED -- the main crack on ..._front_CBS_004 is mean
        #                   1.44 / local std 0.53, DARKER and SMOOTHER than the
        #                   off-specimen background at 5.02 / 2.09
        #   perimeter touch crack 28.6% (..._front_CBS_005) vs background 28.5%
        #                   (MAR_Amb_Cast_CBS_0001) -- a 0.1 point margin
        # Physically this is expected: a wide-open crack void and empty background are
        # the same thing to the detector, no returned signal. What separates them is
        # whether the region is enclosed by specimen, which none of these measure.
        #
        # Worse, on MAR_Amb_Cast_CBS_0005 the background region and the crack are ONE
        # connected component containing 976,295 hand-marked crack pixels, so no
        # region-level exclusion can drop it without destroying human labels.
        # AN IMPORTED MASK REPLACES THE DETECTOR.
        #
        # A survey of the field put this project's detector last: ilastik's Random Forest
        # over a multi-scale filter bank and micro-sam's ViT both produce better masks than
        # a darkness threshold plus a LogisticRegression over 8 features, and the deployed
        # operating point misses roughly 40% of crack pixels. Everything this project does
        # that nobody else does -- refusing calibration, unreviewed-aware metrics, a gated
        # retrain, per-CSV provenance, one row per crack with opening width and tortuosity
        # -- sits DOWNSTREAM of the mask. So the mask can come from whatever segments best.
        #
        # Authority order is unchanged: human correction > imported mask > built-in
        # detector. The import is fed through extract_candidates exactly as the built-in
        # candidates are, so the df schema, the correction machinery and the merge step all
        # behave identically; only the source of the regions differs. min_area_px=1 because
        # dropping regions another tool deliberately produced is not this layer's decision.
        _ext = external_mask.load(image_name, clean.shape)
        _from_external = _ext is not None
        if _from_external:
            labeled, df = extract_candidates(_ext > 0, flat, vesselness, min_area_px=1)
            # The importing tool already decided what is crack. Re-scoring its regions with
            # the weaker built-in classifier would throw away the reason for importing.
            df["IsCrack"] = True
        else:
            labeled, df = extract_candidates(clean, flat, vesselness, min_area_px=40)

            # --- PASS 1: score original candidates with the unified model ---
            df = score_pass1_candidates(df, labeled, img8, flat, vesselness, bundle)

        overrides = load_hard_overrides(image_name)
        if overrides:
            for label, is_crack in overrides.items():
                m = df["Label"] == label
                if m.any():
                    df.loc[m, "IsCrack"] = is_crack
        correction_mask = load_correction_mask(image_name, labeled.shape)
        if correction_mask is not None:
            labeled, df = apply_pixel_corrections(labeled, df, correction_mask)
        df, bridge_mask = merge_large_cracks(labeled, df, flat, min_area_px=1000, max_gap_px=80)
        stage = dict(img8=img8, flat=flat, labeled=labeled, df=df, bridge_mask=bridge_mask, vesselness=vesselness)

    labeled, df, flat, img8, vesselness = stage["labeled"], stage["df"], stage["flat"], stage["img8"], stage["vesselness"]

    # --- PASS 2: generate + score interior-type candidates relative to the
    # now-real crack mask, with their FULL, genuinely-meaningful features ---
    interior_origin = {}
    n_interior_total = 0
    # Pass 2 proposes interior/concavity/bridge candidates and scores them with the
    # built-in model. With an imported mask that would mix the detector this import exists
    # to replace back into the result, so the mask would no longer be what the source tool
    # said. Skipped, and the skip is recorded in the stage.
    _skip_pass2 = bool(locals().get("_from_external") or
                       external_mask.has_external(image_name))
    if bundle is not None and not _skip_pass2:
        crack_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
        dil_crack = morphology.binary_dilation(crack_mask, morphology.disk(1))
        dist_to_crack = ndi.distance_transform_edt(~crack_mask)

        raw = concavity_candidates(crack_mask)
        raw += bridge_corridor_candidates(labeled, df, flat, crack_mask)
        raw += interior_fill_candidates(labeled, df, img8, crack_mask, dist_to_crack)
        n_interior_total = len(raw)

        labeled = labeled.copy()
        df = df.copy()
        next_label = int(df["Label"].max()) + 1 if len(df) else 1
        new_rows = []

        # Pixels the human took off the table: 3 = erased from candidacy,
        # 2 = marked not-crack. Erasing sets labeled to 0, and cyan painted onto
        # blank background is not-crack at labeled 0 too -- so Pass 2's "claim any
        # unlabeled pixel" rule below re-proposed exactly those pixels as fresh
        # crack candidates, and the human's verdict silently flipped back on the
        # next re-render. The README promises corrections always override the
        # model; this is the one place that was not true.
        #
        # Re-read here rather than relying on the variable above, because that is
        # only assigned on the path that computes the stage from scratch; when a
        # caller passes stage= it was never defined.
        _cm = load_correction_mask(image_name, labeled.shape)
        protected = (np.isin(_cm, (2, 3)) if _cm is not None
                     else np.zeros(labeled.shape, dtype=bool))

        for mask_bool, ctype, parent in raw:
            feats = _region_features(mask_bool, flat, img8, vesselness, crack_mask, dil_crack, dist_to_crack)
            accepted, proba = _score(bundle, [feats], ctype)[0]
            if not accepted:
                continue
            claimable = mask_bool & (labeled == 0) & ~protected
            if not claimable.any():
                continue
            labeled[claimable] = next_label
            row = dict(feats)
            row.update({"Label": next_label, "SourceImage": image_name, "CandidateType": ctype,
                        "ParentLabels": "", "IsCrack": True, "CrackProbability": proba})
            new_rows.append(row)
            interior_origin[next_label] = (ctype, feats)
            next_label += 1
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return dict(stage, labeled=labeled, df=df, interior_origin=interior_origin,
                n_interior_total=n_interior_total,
                mask_source=("external" if external_mask.has_external(image_name)
                             else "built-in"))
