"""SAM 2 as a boundary refiner for the built-in detector's candidates.

WHY THIS IS IN THE PRODUCTION TREE AND NOT ONLY IN experiments/
Measured on the ten both-class frames, scored on adjudicated pixels only, SAM 2 refinement
beats the shipped detector on ALL FOUR metrics:

    arm            f1     recall  specificity  precision
    pipeline       0.638  0.534   0.460        0.970
    sam2_refine    0.676  0.561   0.569        0.976     <- dominates on every one
    hybrid_or      0.707  0.604   0.445        0.970     <- better f1, WORSE specificity

That is a Pareto improvement with nothing retrained, which is a stronger result than any of
this project's own model work produced. `hybrid_or` reaches a higher f1 but it is a union and
unions only add positives, so it gives back specificity; "refine" is the mode that is better
in every direction and is therefore the default.

HOW IT IS PROMPTED. transformers exposes SAM 2 as a promptable segmenter with no automatic
mask generator, which suits this problem: automatic generation proposes whole objects and a
crack is a thin dark filament. On a synthetic 3-px filament, a BOX prompt scored IoU 0.742
against 0.604 for a single point and 0.545 for ten points along it, so each candidate region
is passed as its bounding box and SAM 2 redraws the boundary in a padded window. The detector
proposes; SAM 2 refines.

SAM 2 returns three variants per prompt. The one with the highest MODEL-PREDICTED IoU is
taken -- never the one closest to the human mask, which would be scoring against an oracle
that does not exist at inference. A variant covering more than half its window is discarded
as the background object rather than a crack.

AUTHORITY ORDER IS UNCHANGED: human correction > imported mask > SAM 2 > built-in detector.
Refinement is applied only where the human has not ruled, so a painted verdict still wins.
"""
import os

import numpy as np

#: Window padding around a candidate box, in pixels, and a floor on window size. SAM 2 needs
#: context around the object; too little and it cannot separate filament from field.
PAD = 48
MIN_WIN = 128

#: Smallest checkpoint by default. It is what the measurement above was made with, and a
#: bigger one is an untested lever rather than a known improvement.
DEFAULT_MODEL = "facebook/sam2.1-hiera-tiny"

_CACHE = {}


def available():
    """Can SAM 2 run here? Never raises -- callers use this to decide, not to diagnose."""
    try:
        import torch  # noqa: F401
        from transformers import Sam2Model, Sam2Processor  # noqa: F401
        return True
    except Exception:
        return False


def _load(model_id):
    if model_id in _CACHE:
        return _CACHE[model_id]
    import torch
    from transformers import Sam2Model, Sam2Processor
    proc = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    _CACHE[model_id] = (proc, model.to(dev), dev, torch)
    return _CACHE[model_id]


def boxes_for(labeled, labels):
    """Bounding box per region label, as (y0, x0, y1, x1)."""
    from scipy import ndimage as ndi
    if not len(labels):
        return []
    objs = ndi.find_objects(labeled)
    out = []
    for lb in labels:
        i = int(lb)
        sl = objs[i - 1] if 0 < i <= len(objs) else None
        if sl is not None:
            out.append((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop))
    return out


def refine_mask(img8, boxes, model_id=DEFAULT_MODEL, progress=None):
    """Union of SAM 2's boundary for each box prompt, in full-frame coordinates."""
    proc, model, dev, torch = _load(model_id)
    H, W = img8.shape
    out = np.zeros((H, W), dtype=bool)
    for i, (y0, x0, y1, x1) in enumerate(boxes):
        wy0, wx0 = max(0, y0 - PAD), max(0, x0 - PAD)
        wy1, wx1 = min(H, y1 + PAD), min(W, x1 + PAD)
        if (wy1 - wy0) < MIN_WIN:
            c = (wy0 + wy1) // 2
            wy0, wy1 = max(0, c - MIN_WIN // 2), min(H, c + MIN_WIN // 2)
        if (wx1 - wx0) < MIN_WIN:
            c = (wx0 + wx1) // 2
            wx0, wx1 = max(0, c - MIN_WIN // 2), min(W, c + MIN_WIN // 2)
        win = img8[wy0:wy1, wx0:wx1]
        if win.size == 0:
            continue
        box = [[[float(x0 - wx0), float(y0 - wy0), float(x1 - wx0), float(y1 - wy0)]]]
        try:
            inp = proc(images=np.dstack([win] * 3), input_boxes=box,
                       return_tensors="pt").to(dev)
            with torch.no_grad():
                res = model(**inp, multimask_output=True)
            masks = proc.post_process_masks(res.pred_masks.cpu(), inp["original_sizes"])[0]
            m = np.asarray(masks).astype(bool)
            while m.ndim > 3:
                m = m[0]
            sc = res.iou_scores.detach().cpu().numpy().reshape(-1)
            k = int(np.argmax(sc[:m.shape[0]])) if sc.size else 0
            chosen = m[k]
            if chosen.mean() > 0.5:
                continue
            out[wy0:wy1, wx0:wx1] |= chosen
        except Exception:
            # One prompt failing must not lose the other several hundred.
            continue
        if progress and (i + 1) % 100 == 0:
            progress(i + 1, len(boxes))
    return out


def apply_to_stage(stage, mode, correction_mask=None, model_id=DEFAULT_MODEL,
                   progress=None):
    """Refine a pipeline stage's crack mask. Returns (mask, info) and mutates nothing.

    mode "refine"  SAM 2's boundaries for the ACCEPTED candidates, replacing theirs
    mode "hybrid"  the union of the detector's mask and SAM 2's -- higher f1, lower
                   specificity, so not the default
    """
    if mode in (None, "off"):
        return None, {"sam2_mode": "off"}
    lab, df = stage["labeled"], stage["df"]
    acc = df.loc[df["IsCrack"], "Label"].tolist()
    boxes = boxes_for(lab, acc)
    base = np.isin(lab, acc)
    sam = refine_mask(stage["img8"], boxes, model_id=model_id, progress=progress)
    out = (base | sam) if mode == "hybrid" else sam

    # THE HUMAN STILL WINS. Refinement runs after pixel corrections, so anywhere the human
    # painted a verdict their answer is restored: 1 forces crack, 2 and 3 force not-crack.
    # Without this, SAM 2 could quietly overturn a painted decision, and the README's promise
    # that corrections always override the model would stop being true.
    forced_on = forced_off = 0
    if correction_mask is not None and correction_mask.shape == out.shape:
        human_yes = (correction_mask == 1)
        human_no = np.isin(correction_mask, (2, 3))
        forced_on = int((human_yes & ~out).sum())
        forced_off = int((human_no & out).sum())
        out = (out | human_yes) & ~human_no
    return out, {"sam2_mode": mode, "sam2_model": model_id,
                 "sam2_prompts": len(boxes),
                 "sam2_px": int(sam.sum()), "detector_px": int(base.sum()),
                 "px_restored_by_human_crack": forced_on,
                 "px_removed_by_human_not_crack": forced_off}
