"""SAM 2 as a boundary refiner for the built-in detector's candidates. OPT-IN, not default.

READ THIS BEFORE MAKING IT THE DEFAULT AGAIN. It was the default briefly, on the strength of
the table below, and that was wrong. If you are here to turn it back on, the second table is
the reason not to.

On ADJUDICATED pixels -- the ~8% of a frame a human has marked -- refinement wins on all four
metrics over the ten both-class frames:

    arm            f1     recall  specificity  precision
    pipeline       0.638  0.534   0.460        0.970
    sam2_refine    0.676  0.561   0.569        0.976
    hybrid_or      0.707  0.604   0.445        0.970

On the WHOLE FRAME it fragments the mask. Measured on 260708_316_H_b2_front_CBS_002:

    crack pixels            257,148 -> 294,706   +14.6%
    connected components        111 -> 203        +82.9%
    skeleton length px        9,479 -> 18,657     +96.8%

Fewer pixels, twice the components, more than double the skeleton: solid crack regions become
lacy and break apart. Per region the trimming is near-universal -- 144 of 170 regions lose
pixels, -4.1% on the main through-crack and up to -62.3% on smaller ones -- while the frame
TOTAL rises, because refinement claims 54,217 px of new area outside the candidates it was
given. It trims what it was pointed at and spills elsewhere.

WHY THE f1 ROSE ANYWAY. Trimming a region removes false positives from the small marked
not-crack pool faster than it loses true positives from the marked crack pool, so specificity
climbs and carries f1 with it. Crack COUNT and crack LENGTH -- the two headline quantities this
tool produces -- both get worse at the same time, and neither is in the objective. A user
looking at overlays caught what the metric could not.

So the four-metric win is a statement about reviewed pixels, not about the mask. Before this
could be a default it needs an objective that counts fragmentation; sam2_hybrid.py now reports
components and skeleton length beside the metrics so a future comparison can see it.

HOW IT IS PROMPTED. transformers exposes SAM 2 as a promptable segmenter with no automatic
mask generator, which suits this problem: automatic generation proposes whole objects and a
crack is a thin dark filament. On a synthetic 3-px filament a BOX prompt scored IoU 0.742
against 0.604 for a single point and 0.545 for ten points along it, so each candidate region
is passed as its bounding box and SAM 2 redraws the boundary in a padded window.

SAM 2 returns three variants per prompt. The one with the highest MODEL-PREDICTED IoU is
taken -- never the one closest to the human mask, which would be scoring against an oracle
that does not exist at inference. A variant covering more than half its window is discarded as
the background object rather than a crack.

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
    # cuda -> mps -> cpu; see hybrid_detect for why MPS-only was wrong on Linux.
    from common import pick_torch_device
    dev = pick_torch_device(torch)
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
                   specificity; neither mode is the default, see the module docstring
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


#: Cached failure state. A missing checkpoint or no network must degrade to the shipped
#: detector, once, quietly, and on the record -- not raise on every image.
_FAILED = {}


def status():
    """What SAM 2 will actually do here, without trying to make it happen."""
    if not available():
        return {"usable": False, "why": "torch/transformers not importable"}
    if _FAILED:
        return {"usable": False, "why": next(iter(_FAILED.values()))}
    return {"usable": True, "model": DEFAULT_MODEL,
            "loaded": DEFAULT_MODEL in _CACHE}


def refine_labeled(labeled, df, img8, correction_mask=None, mode="refine",
                   model_id=DEFAULT_MODEL, progress=None):
    """Replace each ACCEPTED region's pixels with SAM 2's boundary, keeping its label ID.

    This is the form the pipeline needs. Returning a bare mask would lose the region
    structure that makes a detection clickable, correctable and countable; returning a
    relabelled image would renumber every region and invalidate the override ledger, which
    is keyed by label ID. So the labels are preserved and only their PIXELS move.

    Guarantees, in order of precedence:
      * a region SAM 2 declines to segment keeps its original pixels -- refinement never
        deletes a detection outright
      * a pixel is claimed by at most one label, first-come by iteration order
      * the human wins: pixels marked not-crack or erased are cleared, and pixels marked
        crack are restored to the label they had before refinement

    Returns (new_labeled, info). Raises nothing: on failure it returns the input unchanged
    with the reason in info, because a detector that crashes is worse than one that does
    not improve.
    """
    info = {"sam2_mode": mode, "sam2_model": model_id}
    if mode in (None, "off"):
        return labeled, {"sam2_mode": "off"}
    if not available():
        return labeled, dict(info, sam2_mode="unavailable",
                             why="torch/transformers not importable")
    if model_id in _FAILED:
        return labeled, dict(info, sam2_mode="unavailable", why=_FAILED[model_id])
    try:
        _load(model_id)
    except Exception as e:
        _FAILED[model_id] = f"{type(e).__name__}: {str(e)[:120]}"
        return labeled, dict(info, sam2_mode="unavailable", why=_FAILED[model_id])

    acc = [int(x) for x in df.loc[df["IsCrack"], "Label"].tolist()]
    boxes = dict(zip(acc, boxes_for(labeled, acc)))
    out = labeled.copy()
    kept_original = 0
    moved = 0
    for i, lb in enumerate(acc):
        box = boxes.get(lb)
        if box is None:
            continue
        old = (labeled == lb)
        try:
            sam = refine_mask(img8, [box], model_id=model_id)
        except Exception:
            continue
        new = (old | sam) if mode == "hybrid" else sam
        if not new.any():
            kept_original += 1
            continue                      # never delete a detection outright
        out[old] = 0
        free = (out == 0)
        out[new & free] = lb
        if not (out == lb).any():
            # Everything SAM 2 returned for this region was already claimed by an earlier
            # label, so the region would vanish. The "never delete a detection outright"
            # guard above only covered SAM 2 returning nothing; this is the other way to
            # lose one, and a detection disappearing silently is the worst outcome here.
            out[old & (out == 0)] = lb
            kept_original += 1
            continue
        moved += 1
        if progress and (i + 1) % 100 == 0:
            progress(i + 1, len(acc))

    forced_off = forced_on = 0
    if correction_mask is not None and correction_mask.shape == out.shape:
        human_no = np.isin(correction_mask, (2, 3))
        acc_now = np.isin(out, acc)
        forced_off = int((human_no & acc_now).sum())
        out[human_no & acc_now] = 0
        human_yes = (correction_mask == 1)
        # Restore a painted crack pixel to whichever accepted label held it before
        # refinement, so a hand mark cannot be erased by a model.
        lost = human_yes & ~np.isin(out, acc) & np.isin(labeled, acc)
        forced_on = int(lost.sum())
        out[lost] = labeled[lost]

    info.update({"sam2_regions_refined": moved,
                 "sam2_regions_kept_original": kept_original,
                 "px_before": int(np.isin(labeled, acc).sum()),
                 "px_after": int(np.isin(out, acc).sum()),
                 "px_removed_by_human_not_crack": forced_off,
                 "px_restored_by_human_crack": forced_on})
    return out, info
