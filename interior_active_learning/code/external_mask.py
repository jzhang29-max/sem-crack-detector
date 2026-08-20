"""Accept a crack mask produced by another tool, and keep everything downstream.

WHY THIS EXISTS
A survey of the field put this project's detector last. ilastik's Random Forest over a
multi-scale filter bank, micro-sam's ViT, and the commercial CNNs all produce better masks
than a darkness threshold plus a LogisticRegression over 8 morphology features -- the
deployed operating point misses roughly 40% of crack pixels. Meanwhile the things no
competitor does are all downstream of the mask:

  * a calibration that REFUSES when the scale bar and HFW disagree by more than 5%
  * reporting PIXELS, and saying so, when any image in a group is uncalibrated
  * UNREVIEWED pixels never scored as negatives
  * a promotion gate that refuses without a valid out-of-sample baseline
  * a provenance record bound to every exported CSV
  * one row per crack with skeleton length, opening width, tortuosity, branching

Competing on mask quality is a losing fight. Composing is not. Import a mask from whatever
segments best, and this project becomes the measurement and audit layer on top of it --
which is the one arrangement where it is better than all of them, because none of them will
do the measurement side and it no longer has to win the segmentation side.

WHAT AN IMPORTED MASK IS AND IS NOT
It replaces the DETECTOR, not the human. Order of authority is unchanged:

    human correction  >  imported mask  >  built-in detector

An imported mask is treated exactly as the built-in detector's output would be: correction
masks still override it pixel by pixel, erased pixels stay erased, and 0 in a correction
mask still means UNREVIEWED. Importing does not touch, move or rewrite any correction mask.

PROVENANCE IS MANDATORY, NOT OPTIONAL
Every import records the source tool, the file it came from, that file's SHA-256, the
pixel count, and when it happened. A measurement CSV whose mask came from somewhere else
but does not say so is worse than no measurement: the number looks native. So
provenance_for() feeds the sidecar that already ships beside every CSV, and any exported
row set carries `mask_source`.

ACCEPTED FORMS
Any 2D image the same shape as the cropped frame. Non-zero means crack -- so a binary
0/255 PNG, a 0/1 mask, or a labelled instance map all work, and a labelled map's identities
are preserved as separate regions rather than being merged. A shape mismatch is REFUSED
rather than resampled: silently resampling somebody else's segmentation is how a mask ends
up half a pixel off everywhere, and nothing downstream would show it.
"""
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image

from common import PAINT_DIR, VERSION, save_json_atomic, save_png_atomic

Image.MAX_IMAGE_PIXELS = None

#: Tools whose output has been considered here. Anything else is accepted with the name
#: recorded verbatim -- the point is composability, not a whitelist.
KNOWN_SOURCES = ("ilastik", "micro-sam", "napari", "fiji", "cvat", "dragonfly", "avizo",
                 "mipar", "zen", "custom")


def mask_path(image_name):
    return os.path.join(PAINT_DIR, f"{image_name}_external_mask.png")


def meta_path(image_name):
    return os.path.join(PAINT_DIR, f"{image_name}_external_mask.json")


def has_external(image_name):
    return os.path.exists(mask_path(image_name)) and os.path.exists(meta_path(image_name))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(image_name, shape):
    """The imported mask as an int label array, or None.

    Shape mismatch REFUSES loudly rather than resampling. A resampled foreign segmentation
    is subtly wrong everywhere and nothing downstream can see it, which is the same reason
    load_correction_mask refuses a mismatched correction mask instead of stretching it.
    """
    if not has_external(image_name):
        return None
    a = np.asarray(Image.open(mask_path(image_name)))
    if a.ndim > 2:
        a = a[..., 0]
    if a.shape != tuple(shape):
        raise ValueError(
            f"{image_name}: imported mask is {a.shape} but the current frame is "
            f"{tuple(shape)}. Re-export it from the source tool at the frame's size; it is "
            f"NOT resampled, because a resampled foreign segmentation is wrong everywhere "
            f"and invisible downstream.")
    return a.astype(np.int32)


def store(image_name, src_path, source_tool, shape, note=""):
    """Import a mask file for this image. Returns the provenance record.

    Refuses a shape mismatch, an empty mask, and a mask that is entirely non-zero -- the
    last two are the signatures of exporting the wrong layer, and both would otherwise
    produce a confidently wrong measurement table.
    """
    a = np.asarray(Image.open(src_path))
    if a.ndim > 2:
        a = a[..., 0]
    if a.shape != tuple(shape):
        raise ValueError(
            f"mask is {a.shape} but {image_name}'s cropped frame is {tuple(shape)}. "
            f"Export at the frame size -- masks are never resampled here.")
    nz = int((a != 0).sum())
    if nz == 0:
        raise ValueError("the mask is entirely zero: nothing would be measured. Check that "
                         "you exported the segmentation layer and not an empty one.")
    if nz == a.size:
        raise ValueError("the mask is entirely non-zero, i.e. every pixel is crack. That is "
                         "the signature of exporting the wrong layer (a probability map or "
                         "an image, not a mask).")

    lab = a.astype(np.int32)
    n_regions = int(len(np.unique(lab)) - 1)
    save_png_atomic(Image.fromarray(lab.astype(np.uint16) if lab.max() > 255
                                    else lab.astype(np.uint8)), mask_path(image_name))
    rec = {
        "image": image_name,
        "source_tool": source_tool,
        "source_file": os.path.basename(src_path),
        "source_sha256": _sha256(src_path),
        "shape": [int(shape[0]), int(shape[1])],
        "nonzero_px": nz,
        "nonzero_fraction": round(nz / a.size, 6),
        "n_regions": n_regions,
        "labelled_instances": bool(lab.max() > 1),
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "tool_version": VERSION,
        "note": note,
        # Stated explicitly so nobody reads an imported result as this project's detector.
        "authority": "replaces the built-in detector; human corrections still override it",
    }
    save_json_atomic(rec, meta_path(image_name))
    return rec


def clear(image_name):
    """Drop the import and fall back to the built-in detector."""
    gone = False
    for p in (mask_path(image_name), meta_path(image_name)):
        if os.path.exists(p):
            os.remove(p)
            gone = True
    return gone


def provenance_for(image_name):
    """The import record, for the sidecar beside an exported CSV."""
    if not has_external(image_name):
        return {"mask_source": "built-in detector"}
    try:
        with open(meta_path(image_name)) as fh:
            rec = json.load(fh)
    except (ValueError, OSError):
        return {"mask_source": "imported, but its provenance file is unreadable"}
    return {
        "mask_source": f"imported from {rec.get('source_tool', 'unknown tool')}",
        "mask_import": rec,
    }
