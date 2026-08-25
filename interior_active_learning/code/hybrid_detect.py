"""
The best-measured detector, as one callable: pipeline + Pass 2 + SAM union.

Measured on the 5 images with enough human not-crack marks for specificity to
mean anything, pixel-level over adjudicated pixels only, classifier trained on
other images:

    Pass 1 only                     f1 0.697   recall 0.575   spec 0.476
    Pass 1 + Pass 2  (was deployed) f1 0.715   recall 0.597   spec 0.476
    Pass 1 + SAM                    f1 0.768   recall 0.667   spec 0.395
    Pass 1 + Pass 2 + SAM  <- this  f1 0.776   recall 0.678   spec 0.395

SAM adds +0.061 f1 on top of the full pipeline, improving 4 of 5 images. Pass 2
and SAM turn out to be complementary rather than redundant, which was not the
expected result -- the prediction going in was that Pass 2 would already cover
what SAM finds, and it does not.

The gain is entirely recall: +8.1 points of recall for -8.1 of specificity,
precision flat. That trade suits a human-in-the-loop workflow, where removing a
false positive is one click and noticing a missed crack is not.

Honest limits, because they bound how hard this should be pushed: n=5 images, so
no statistical test can clear p=0.0625; specificity rests on small negative
pools (one image has 4,416 not-crack pixels, another has tn=0); and only 1.8-4%
of each image is human-adjudicated, so most of SAM's added area is unmeasurable
either way.

SAM costs ~3 min/image on Apple MPS versus ~40s for the pipeline alone, so
use_sam is a parameter, not a hardcoded choice. The app runs the pipeline first
so something is on screen immediately, then upgrades with SAM if asked.

The void/bright filter is left ON by default even though the unfiltered union
scored marginally higher (0.757 vs 0.749 on the Pass-1 comparison). That gap is
well inside noise at n=5, while the filter removes regions that are physically
incapable of being cracks -- the near-black empty background beyond the
specimen edge, and ridges brighter than the image median -- which were 34% of
SAM's added area. Keeping provably-impossible regions to chase 0.008 of f1 is
the wrong trade for something a person has to look at.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import numpy as np

from common import ORIGINAL_DIR, PROD_MODEL_PATH, contrast_kwargs_for, is_test_image
from detect_cracks import region_features_from_labeled
from unified_pipeline import run_unified_pipeline

TILE, STRIDE = 1024, 896
MIN_AREA = 40
MAX_AREA_FRAC_OF_TILE = 0.15
MAX_BORDER_FRAC = 0.30
VOID_MAX = 25

_SAM = {}


def sam_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _get_sam():
    """Load SAM once per process. ViT-Huge: measured 73% standalone crack
    detection vs 31% for ViT-Base, which is why the small variant is not used
    despite being ~4x faster."""
    if "gen" in _SAM:
        return _SAM["gen"]
    import torch
    from transformers import SamModel, SamProcessor, pipeline
    # cuda -> mps -> cpu. This used to test MPS only, so a Linux machine with an NVIDIA GPU
    # fell through to CPU with nothing on screen saying so -- the failure mode is "why is this
    # so slow", which is the hardest kind to diagnose remotely. torch.backends.mps imports
    # fine on Linux and is_available() simply returns False, so the order is safe on both.
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    # float32 is required: the mask-generation pipeline hits a float64 op on MPS
    # otherwise, which fails outright.
    model = SamModel.from_pretrained("facebook/sam-vit-huge", torch_dtype=torch.float32).to(device)
    _SAM["gen"] = pipeline("mask-generation", model=model,
                           image_processor=proc.image_processor, device=device)
    _SAM["device"] = device
    return _SAM["gen"]


def _as_np2d(m):
    import torch
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    while m.ndim > 2:
        m = m[0]
    return m.astype(bool)


def sam_crack_mask(img8, flat, vesselness, bundle, progress=None):
    """SAM masks that the crack classifier accepts, as one boolean mask.

    Features come from region_features_from_labeled -- the same function
    extract_candidates uses -- so a SAM mask and a pipeline candidate of the
    same shape get identical features. Hand-rolling this produced a
    sign-flipped MeanDarkness and a phantom result earlier in this project.

    No force-keep-by-area here, unlike classify_with_model: that rule is safe
    for darkness-thresholded candidates but not for SAM, which will happily
    return a 500k-px mask of a bright grain.
    """
    from PIL import Image as _Image
    gen = _get_sam()
    h, w = img8.shape
    med = float(np.median(img8))
    accepted = np.zeros((h, w), bool)

    ys = list(range(0, max(h - TILE, 0) + 1, STRIDE)) or [0]
    xs = list(range(0, max(w - TILE, 0) + 1, STRIDE)) or [0]
    if ys[-1] + TILE < h:
        ys.append(max(h - TILE, 0))
    if xs[-1] + TILE < w:
        xs.append(max(w - TILE, 0))
    total = len(ys) * len(xs)
    done = n_kept = 0

    for cy in ys:
        for cx in xs:
            tile = img8[cy:cy + TILE, cx:cx + TILE]
            done += 1
            if progress:
                progress(done, total, n_kept)
            if min(tile.shape) < 16:
                continue
            try:
                res = gen(_Image.fromarray(np.stack([tile] * 3, -1)), points_per_batch=64)
            except Exception:
                continue
            th, tw = tile.shape
            for mm in res["masks"]:
                mm = _as_np2d(mm)
                if mm.shape != tile.shape:
                    continue
                area = int(mm.sum())
                if area < MIN_AREA or area > MAX_AREA_FRAC_OF_TILE * th * tw:
                    continue
                border = (mm[0, :].sum() + mm[-1, :].sum() + mm[:, 0].sum() + mm[:, -1].sum())
                if border / float(2 * (th + tw)) > MAX_BORDER_FRAC:
                    continue
                ys_l, xs_l = np.nonzero(mm)
                y0, y1 = ys_l.min(), ys_l.max() + 1
                x0, x1 = xs_l.min(), xs_l.max() + 1
                sub = mm[y0:y1, x0:x1]
                gy0, gx0 = cy + y0, cx + x0
                gy1, gx1 = min(gy0 + sub.shape[0], h), min(gx0 + sub.shape[1], w)
                sub = sub[:gy1 - gy0, :gx1 - gx0]
                if sub.sum() < MIN_AREA:
                    continue
                raw_mean = float(img8[gy0:gy1, gx0:gx1][sub].mean())
                # cannot be a crack: empty background, or brighter than the
                # image as a whole
                if raw_mean <= VOID_MAX or raw_mean >= med:
                    continue
                _, fd = region_features_from_labeled(
                    sub.astype(np.int32), flat[gy0:gy1, gx0:gx1],
                    vesselness[gy0:gy1, gx0:gx1], min_area_px=MIN_AREA)
                if not len(fd):
                    continue
                X = bundle["scaler"].transform(
                    fd[bundle["feature_names"]].values[:1])
                if bundle["clf"].predict_proba(X)[0, 1] >= bundle.get("threshold", 0.5):
                    accepted[gy0:gy1, gx0:gx1] |= sub
                    n_kept += 1
    if progress:
        progress(total, total, n_kept)
    return accepted, n_kept


def detect(image_name, use_sam=False, progress=None):
    """Run the full detector. Returns the run_unified_pipeline stage dict with
    a `sam_mask` and `crack_mask` added.

    progress(stage:str, frac:float, note:str) is called so a UI can show
    something during the slow parts.
    """
    import joblib

    def rep(stage, frac, note=""):
        if progress:
            progress(stage, frac, note)

    rep("pipeline", 0.05, "segmenting and classifying")
    # Pass 2 is skipped silently when interior_active_learning/models/unified_model.joblib
    # is absent -- _load_unified_bundle() just returns None. That costs 27% of all
    # crack regions and it happened for real: an early version of make_package.sh
    # copied only the top-level models/ directory, so every clone ran a weaker
    # detector than the benchmarked one with nothing on screen to say so. Warn.
    from unified_pipeline import _load_unified_bundle as _lub
    if _lub() is None:
        msg = ("Pass-2 model missing (interior_active_learning/models/unified_model.joblib) "
               "-- running Pass 1 only, which finds ~27% fewer crack regions")
        print(f"WARNING: {msg}", flush=True)
        rep("pipeline", 0.05, msg)

    stage = run_unified_pipeline(image_name)
    labeled, df = stage["labeled"], stage["df"]
    pipe_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
    stage["pipeline_mask"] = pipe_mask
    stage["crack_mask"] = pipe_mask
    stage["sam_mask"] = None
    stage["n_sam_regions"] = 0
    rep("pipeline", 0.5, f"{int(df['IsCrack'].sum())} crack regions")

    if use_sam and sam_available():
        bundle = joblib.load(PROD_MODEL_PATH)

        def sp(done, total, kept):
            rep("sam", 0.5 + 0.5 * done / max(total, 1),
                f"tile {done}/{total}, {kept} regions kept")

        sam, n = sam_crack_mask(stage["img8"], stage["flat"], stage["vesselness"],
                                bundle, progress=sp)
        stage["sam_mask"] = sam
        stage["n_sam_regions"] = n
        stage["crack_mask"] = pipe_mask | sam
        rep("done", 1.0, f"pipeline + {n} SAM regions")
    else:
        rep("done", 1.0, "pipeline only")
    return stage


def fold_sam_into_candidates(stage, image_name):
    """Turn SAM's accepted pixels into real candidate regions in the stage.

    Without this a SAM region renders red but has no row in `df`, so it cannot be
    clicked, corrected or counted -- visible but not editable. Extracted here so
    the interactive path (/api/process) and the batch path
    (regenerate_templates.py --with-sam) share ONE implementation. They had
    diverged: only the interactive path ran SAM at all, which meant Re-apply and
    the post-retrain re-render silently dropped every SAM region.
    """
    sam = stage.get("sam_mask")
    if sam is None or not sam.any():
        return 0
    import pandas as pd
    from skimage import measure
    labeled, df = stage["labeled"], stage["df"]
    new = sam & (labeled == 0)
    if not new.any():
        return 0
    lab_new = measure.label(new, connectivity=2)
    nxt = int(df["Label"].max()) + 1 if len(df) else 1
    rows = []
    for pr in measure.regionprops(lab_new):
        if pr.area < MIN_AREA:
            continue
        labeled[pr.slice][pr.image] = nxt
        rows.append({"Label": nxt, "Area": int(pr.area), "IsCrack": True,
                     "CrackProbability": 1.0, "SourceImage": image_name,
                     "CandidateType": "sam"})
        nxt += 1
    if rows:
        stage["df"] = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return len(rows)


def render_and_record(image_name, use_sam=False, progress=None):
    """Detect, fold SAM in, write the paint template, update candidate counts.

    The single entry point for producing an overlay, so the interactive and
    batch paths cannot disagree about what an overlay contains.
    """
    import json
    from interior_candidates import build_simple_overlay
    from common import PAINT_DIR

    stage = detect(image_name, use_sam=use_sam, progress=progress)
    n_sam = fold_sam_into_candidates(stage, image_name)
    import template_writer
    _tp = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    template_writer.discard(_tp)      # this render supersedes anything queued
    build_simple_overlay(stage).save(_tp)
    df = stage["df"]
    counts_path = os.path.join(PAINT_DIR, "candidate_counts.json")
    counts = {}
    if os.path.exists(counts_path):
        try:
            counts = json.load(open(counts_path))
        except Exception:
            counts = {}
    # Test fixtures must not enter this tracked file; see common.is_test_image.
    if not is_test_image(image_name):
        counts[image_name] = {"n_candidates": int(len(df)),
                              "n_crack": int(df["IsCrack"].sum())}
    json.dump(counts, open(counts_path, "w"), indent=2)
    n_interior = len(stage.get("interior_origin", {}))
    return {"image": image_name, "n_candidates": int(len(df)),
            "n_crack": int(df["IsCrack"].sum()), "n_sam_regions": n_sam,
            "n_interior": n_interior, "used_sam": bool(use_sam and stage.get("sam_mask") is not None)}
