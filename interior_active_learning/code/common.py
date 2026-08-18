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


import threading as _threading

# One lock per image, guarding the read-modify-write of that image's correction
# mask. Two overlapping corrections used to interleave: both read the same mask,
# both wrote, and the second silently discarded the first's verdict -- and with a
# non-atomic write the file could be left truncated. The frontend fires
# flip_region straight from mousedown with no in-flight flag, so two quick clicks
# reach the server concurrently.
_MASK_LOCKS = {}
_MASK_LOCKS_GUARD = _threading.Lock()


def mask_lock(image_name):
    with _MASK_LOCKS_GUARD:
        return _MASK_LOCKS.setdefault(image_name, _threading.Lock())


def save_png_atomic(pil_img, path, **kw):
    """Write a PNG via temp-and-rename so a concurrent reader never sees a
    half-written file.

    Templates, painted layers and masks were all written straight to their final
    path, and these files are 4-33 MB: a reader arriving mid-write gets a
    truncated PNG. format="PNG" is explicit because PIL infers format from the
    extension and rejects a .tmp name.
    """
    tmp = path + ".tmp"
    kw.setdefault("format", "PNG")
    pil_img.save(tmp, **kw)
    os.replace(tmp, path)


def save_json_atomic(obj, path):
    """Same for JSON. candidate_counts.json is rewritten after every image, so a
    reader (or a second writer) could catch it mid-dump and get invalid JSON."""
    import json as _json
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        _json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


# Images whose stored mask does not fit the current render. Recorded so the
# server can report them instead of the condition being invisible: two of the 35
# shipped masks are in this state (painted against a differently sized render),
# holding 371,227 hand-marked pixels that contributed nothing to training and
# that nothing warned about.
UNUSABLE_MASKS = {}


def correction_mask_state(image_name, shape):
    """('absent'|'ok'|'shape_mismatch', shape_on_disk_or_None).

    Callers used to get None for both "no corrections yet" and "corrections
    exist but do not fit", which is what made the second case silent AND made
    it look safe to overwrite the file.
    """
    from PIL import Image
    path = _correction_mask_path(image_name)
    if not os.path.exists(path):
        return "absent", None
    m = np.array(Image.open(path))
    while m.ndim > 2:
        m = m[..., 0]
    return ("ok" if m.shape == tuple(shape) else "shape_mismatch"), m.shape


def load_correction_mask(image_name, shape):
    """Returns a uint8 array (same shape as `labeled`) where 0 = no paint
    correction, 1 = force this pixel to CRACK, 2 = force this pixel to
    NOT-CRACK (still a labeled "artifact" candidate), 3 = ERASE this pixel
    from candidacy entirely (back to plain, unmarked background) -- or None
    if this image has no usable saved corrections. Stored as a single-channel
    PNG (compresses extremely well since most pixels are 0) so it persists
    across paint sessions and pipeline runs.

    A mask whose shape does not match the current render still returns None --
    applying it would misplace every verdict -- but it is now RECORDED in
    UNUSABLE_MASKS and logged, because returning a bare None meant the user was
    never told their labels had stopped counting.
    """
    from PIL import Image
    path = _correction_mask_path(image_name)
    if not os.path.exists(path):
        return None
    mask = np.array(Image.open(path))
    while mask.ndim > 2:
        mask = mask[..., 0]
    if mask.shape != tuple(shape):
        n_marked = int(np.isin(mask, (1, 2, 3)).sum())
        if UNUSABLE_MASKS.get(image_name) != (mask.shape, tuple(shape)):
            UNUSABLE_MASKS[image_name] = (mask.shape, tuple(shape))
            print(f"WARNING: {image_name}: saved corrections are "
                  f"{mask.shape[0]}x{mask.shape[1]} but this render is "
                  f"{shape[0]}x{shape[1]} -- {n_marked:,} hand-marked pixels are "
                  f"NOT being used. They will not be overwritten; re-mark this "
                  f"image or realign the mask.", flush=True)
        return None
    UNUSABLE_MASKS.pop(image_name, None)
    return mask


def save_correction_mask(image_name, mask):
    """Write the correction mask, atomically, without ever destroying an
    existing mask that merely does not fit the current render.

    Both halves matter. The write used to go straight to the final path, so a
    reader could be served a half-written PNG. And because load_correction_mask
    returned None for a shape mismatch, every writer treated "corrections exist
    but do not fit" as "no corrections yet", built a fresh zero array and
    overwrote the file -- one click on such an image erased 148,068 hand-marked
    pixels with no snapshot on that path.
    """
    from PIL import Image
    path = _correction_mask_path(image_name)
    mask = np.asarray(mask).astype(np.uint8)
    if os.path.exists(path):
        state, on_disk = correction_mask_state(image_name, mask.shape)
        if state == "shape_mismatch":
            # Preserve the human verdicts under a name that says what they fit,
            # rather than silently replacing them.
            aside = path.replace(".png", f".stale-{on_disk[0]}x{on_disk[1]}.png")
            n = 1
            while os.path.exists(aside):
                aside = path.replace(".png", f".stale-{on_disk[0]}x{on_disk[1]}.{n}.png")
                n += 1
            os.rename(path, aside)
            print(f"WARNING: {image_name}: existing {on_disk[0]}x{on_disk[1]} corrections "
                  f"do not fit this {mask.shape[0]}x{mask.shape[1]} render; moved to "
                  f"{os.path.basename(aside)} instead of overwriting them.", flush=True)
    # format="PNG" is required, not decorative: PIL infers the format from the
    # file extension and raises ValueError("unknown file extension: .tmp") on a
    # temp name, which turned every save into a 500 the first time this ran.
    tmp = path + ".tmp"
    Image.fromarray(mask).save(tmp, format="PNG")
    os.replace(tmp, path)


def list_original_images():
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR) if f.lower().endswith(".tif")
    )
