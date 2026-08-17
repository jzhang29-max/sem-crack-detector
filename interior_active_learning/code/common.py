"""
Shared setup for the interior-fill active-learning experiment. Everything in
this folder is READ-ONLY with respect to the main project's INPUTS: it
imports the production pipeline functions from ../../code/detect_cracks.py
and reads ../../original/*.tif + the production model, but never writes to
../../training_data or ../../models (the production classifier). All
experiment output (candidates, labels, models, review sheets) stays inside
this folder, with one deliberate, narrow exception: apply_interior_model.py
also writes {image}_final_result.png/_bw.png into ../../results/, alongside
(never overwriting) the production pipeline's own {image}_cracks_overlay.png
etc., so the corrected result has one place to look without discarding the
pure production baseline as a reference point.
"""
import os
import sys
import numpy as np

EXP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(EXP_ROOT)
MAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "code")
sys.path.insert(0, MAIN_CODE_DIR)

ORIGINAL_DIR = os.path.join(PROJECT_ROOT, "original")
PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crack_classifier.joblib")
LEDGER_PATH = os.path.join(PROJECT_ROOT, "manual_corrections_ledger.csv")
PROD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CANDIDATES_DIR = os.path.join(EXP_ROOT, "candidates")
LABELS_DIR = os.path.join(EXP_ROOT, "labels")
MODELS_DIR = os.path.join(EXP_ROOT, "models")
REVIEW_DIR = os.path.join(EXP_ROOT, "review")
PAINT_DIR = os.path.join(EXP_ROOT, "paint")

for d in (CANDIDATES_DIR, LABELS_DIR, MODELS_DIR, REVIEW_DIR, PAINT_DIR):
    os.makedirs(d, exist_ok=True)

# Interior candidates get Label numbers starting above any real
# detect_cracks() region label (max seen across all 25 images: 947), so the
# two can be safely mixed in one review batch / CSV without colliding.
# Kept as small as safely possible (rather than e.g. 500_000) so numbers
# stay readable -- collision avoidance across DIFFERENT images' interior
# candidates (which all number from this same offset) is handled by always
# matching on (SourceImage, Label) together, never Label alone; see
# already_labeled_interior_pairs() in active_learning_select.py.
INTERIOR_LABEL_OFFSET = 1_000
# Painted candidates use a running global max (see _find_next_label() in
# apply_paint_annotations.py), so this only needs enough headroom above the
# largest realistic per-image interior candidate count to avoid colliding
# with regular interior candidates WITHIN one image's CSV.
PAINT_LABEL_OFFSET = 2_000

# Images that need the gentler 0/100 contrast stretch in the main pipeline
# (kept in sync with process_one_round4.py / run_round4_all25.sh).
GENTLE_CONTRAST_IMAGES = {
    "260708_316_H_b2_front_CBS_009",
    "260708_316_H_b2_front_CBS_010",
    "260622_316_H_b2_front_CBS_04",
}


def contrast_kwargs_for(name):
    if name in GENTLE_CONTRAST_IMAGES:
        return {"low_pct": 0.0, "high_pct": 100.0}
    return {"low_pct": 1.0, "high_pct": 99.5}


def load_hard_overrides(image_name):
    import pandas as pd
    if not os.path.exists(LEDGER_PATH):
        return None
    ledger = pd.read_csv(LEDGER_PATH)
    g = ledger[ledger["SourceImage"] == image_name]
    if len(g) == 0:
        return None
    is_crack = g["CorrectedTo"].astype(str).str.strip().str.lower().isin(["true", "1"])
    return dict(zip(g["Label"].astype(int), is_crack))


# A log of (SourceImage, Label) pairs touched by a paint-app correction --
# used ONLY to keep a corrected original candidate from resurfacing in a
# future active-learning review batch (active_learning_select.py reads this
# directly). This is NOT the authoritative record of what the correction
# actually was -- that's the per-pixel correction mask below -- because a
# single Label can be only PARTIALLY corrected (see apply_pixel_corrections's
# docstring for why a whole-Label CorrectedTo flag doesn't work when one
# connected "crack" region spans most of the image).
ORIGINAL_PAINT_CORRECTIONS_PATH = os.path.join(LABELS_DIR, "original_paint_corrections.csv")


def _correction_mask_path(image_name):
    return os.path.join(PAINT_DIR, f"{image_name}_correction_mask.png")


def load_correction_mask(image_name, shape):
    """Returns a uint8 array (same shape as `labeled`) where 0 = no paint
    correction, 1 = force this pixel to CRACK, 2 = force this pixel to
    NOT-CRACK (still a labeled "artifact" candidate), 3 = ERASE this pixel
    from candidacy entirely (back to plain, unmarked background) -- or None
    if this image has no saved corrections yet. Stored as a single-channel
    PNG (compresses extremely well since most pixels are 0) so it persists
    across paint sessions and pipeline runs."""
    from PIL import Image
    path = _correction_mask_path(image_name)
    if not os.path.exists(path):
        return None
    mask = np.array(Image.open(path))
    if mask.shape != shape:
        return None  # stale mask from a differently-cropped/sized render -- ignore rather than misapply
    return mask


def save_correction_mask(image_name, mask):
    from PIL import Image
    Image.fromarray(mask.astype(np.uint8)).save(_correction_mask_path(image_name))


def list_original_images():
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR) if f.lower().endswith(".tif")
    )
