"""
Build a SAM-union overlay for every image, so the union can be inspected
visually rather than only as a recall number.

Union definition. SAM is class-agnostic -- it segments grains, background and
databar edges just as happily as cracks -- so "SAM's cracks" is not something
SAM gives you. Each SAM mask is therefore scored with the SAME classifier the
pipeline uses, and the union is:

    union = (pipeline's accepted crack regions) OR (SAM masks the classifier accepts)

Features for SAM masks come from detect_cracks.region_features_from_labeled --
literally the function extract_candidates calls -- so a SAM mask and a pipeline
candidate of identical shape get identical features. This matters more than it
looks: an earlier hand-rolled version of this computed MeanDarkness as
mean(flat) when the pipeline defines it as mean(255 - flat), which inverted the
feature and produced a phantom "+16% recall from union" that survived until the
numbers were cross-checked against an independent measurement. Sharing the
function makes that class of error impossible rather than merely unlikely.

Overlay colours, chosen so SAM's marginal contribution is the visually obvious
thing:
    red     = pipeline only
    yellow  = SAM only          <- what the union actually adds
    orange  = both agree

Expected result, stated in advance so the output is not read as a surprise: the
controlled union test on 45 hand-confirmed cracks found 0 cracks that SAM
catches and the pipeline misses (pipeline 91%, SAM 49%, union 91%). Yellow
should therefore be rare, and where it appears it is more likely a false
positive than a missed crack.

    python3 sam_union_overlays.py [--only IMAGE] [--limit N]
Writes ../../figures/sam_union/<image>_union.png  and  sam_union_stats.csv
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline

from common import (ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, PROD_MODEL_PATH,
                     contrast_kwargs_for, load_correction_mask)
from detect_cracks import (load_as_uint8, find_field_of_view, flatten_background,
                           compute_vesselness, region_features_from_labeled)

Image.MAX_IMAGE_PIXELS = None

OUT_DIR = os.path.join(PROJECT_ROOT, "figures", "sam_union")
STATS = os.path.join(PROJECT_ROOT, "figures", "sam_union", "sam_union_stats.csv")
SAM_ID = "facebook/sam-vit-huge"
TILE, STRIDE = 1024, 896          # 128 px overlap so tile-edge cracks aren't split
MIN_AREA = 40                     # same floor the pipeline uses
# Reject "the background of this tile" masks. The cap must be a fraction of the
# TILE, not of the whole image: a mask covering an entire 1024x1024 tile is
# 1.05M px, which is only 16.7% of a 6.3M-px image and so sailed under an
# image-relative 20% ceiling, producing tile-shaped blobs with straight edges.
MAX_AREA_FRAC_OF_TILE = 0.15
# A SAM mask may touch at most this fraction of its tile's border. Genuine
# cracks cross a tile edge at one or two places; a background region hugs it.
MAX_BORDER_FRAC = 0.30
VIEW_MAX = 2048                   # downscale overlays for viewing


def crack_mask_from_template(image_name):
    """Read the pipeline's final crack mask out of the paint template instead
    of recomputing it with run_unified_pipeline().

    Purely a cost fix, and a large one. Profiling one 4096x6144 image: SAM
    inference is ~5s/tile (~3 min/image), but run_unified_pipeline costs ~126
    min on the same image -- exclude_border_background 285s (five dilations
    with a radius-40 disk over 25 Mpx), merge_large_cracks 56s, and ~117 min
    in Pass-2, which scores each interior candidate one at a time against
    FULL-IMAGE arrays. None of that work is needed here: the union only needs
    the finished mask, and regenerate_templates.py already wrote exactly that
    mask for all 63 images using the app's own pipeline.

    build_simple_overlay alpha-blends rather than writing a flat colour, so
    the mask is recovered algebraically, not by a brightness threshold (RGB
    thresholds are precisely what produced false readings earlier in this
    project). With g the underlying grey:
        crack     R = g*0.45 + 140.25, G = B = g*0.45  -> G == B, R-G in {140,141}
        artifact  G-R ~ 91.8, B-R ~ 114.75             -> G != B
        untouched R == G == B                          -> R-G == 0
    so the crack test is exact and brightness-independent. Verified against
    run_unified_pipeline on two images (one MAR, one steel): identical masks,
    0 pixels missing and 0 extra.

    Corrections come along for free -- run_unified_pipeline applies the
    correction mask before the template is written, so a reviewed verdict is
    already baked in, matching how the first 39 images were processed.
    """
    path = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    if not os.path.exists(path):
        return None
    a = np.array(Image.open(path).convert("RGB")).astype(np.int16)
    d = a[..., 0] - a[..., 1]
    return (a[..., 1] == a[..., 2]) & ((d == 140) | (d == 141))


def as_np2d(m):
    """SAM masks come back as torch tensors; numpy and torch disagree about
    what .nonzero() returns, which has already broken this once."""
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    while m.ndim > 2:
        m = m[0]
    return m.astype(bool)


def sam_masks_for(img8, gen):
    """Tile the image and collect SAM masks in GLOBAL coordinates."""
    h, w = img8.shape
    out = []
    ys = list(range(0, max(h - TILE, 0) + 1, STRIDE)) or [0]
    xs = list(range(0, max(w - TILE, 0) + 1, STRIDE)) or [0]
    if ys[-1] + TILE < h:
        ys.append(max(h - TILE, 0))
    if xs[-1] + TILE < w:
        xs.append(max(w - TILE, 0))
    for cy in ys:
        for cx in xs:
            tile = img8[cy:cy + TILE, cx:cx + TILE]
            if tile.shape[0] < 16 or tile.shape[1] < 16:
                continue
            pil = Image.fromarray(np.stack([tile] * 3, -1))
            try:
                res = gen(pil, points_per_batch=64)
            except Exception as e:
                print(f"    tile ({cy},{cx}) SAM failed: {type(e).__name__}", flush=True)
                continue
            for m in res["masks"]:
                mm = as_np2d(m)
                if mm.shape != tile.shape:
                    continue
                if mm.sum() < MIN_AREA:
                    continue
                out.append((cy, cx, mm))
    return out, len(ys) * len(xs)


def score_sam_masks(masks, flat, vesselness, bundle, shape):
    """Score each SAM mask independently, cropped to its own bounding box.

    Per-mask rather than one merged labelled array on purpose: SAM masks
    overlap heavily, so merging them into a binary image and re-labelling
    would fuse most of the image into a single blob and destroy exactly the
    instance separation being evaluated.
    """
    H, W = shape
    accepted = np.zeros((H, W), dtype=bool)
    n_acc = 0
    rej = {"too_small": 0, "too_big": 0, "border_hugging": 0}
    feats, keep = [], []
    for cy, cx, mm in masks:
        th, tw = mm.shape
        max_area = MAX_AREA_FRAC_OF_TILE * th * tw
        # fraction of the tile's own border occupied by this mask
        border = (mm[0, :].sum() + mm[-1, :].sum() + mm[:, 0].sum() + mm[:, -1].sum())
        border_frac = border / float(2 * (th + tw))
        ys, xs = np.nonzero(mm)
        if not len(ys):
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = mm[y0:y1, x0:x1]
        area = int(sub.sum())
        if area < MIN_AREA:
            rej["too_small"] += 1
            continue
        if area > max_area:
            rej["too_big"] += 1
            continue
        if border_frac > MAX_BORDER_FRAC:
            rej["border_hugging"] += 1
            continue
        gy0, gx0 = cy + y0, cx + x0
        gy1, gx1 = min(gy0 + sub.shape[0], H), min(gx0 + sub.shape[1], W)
        sub = sub[:gy1 - gy0, :gx1 - gx0]
        if sub.sum() < MIN_AREA:
            continue
        lab = sub.astype(np.int32)
        _, d = region_features_from_labeled(lab, flat[gy0:gy1, gx0:gx1],
                                            vesselness[gy0:gy1, gx0:gx1],
                                            min_area_px=MIN_AREA)
        if not len(d):
            continue
        feats.append(d.iloc[0])
        keep.append((gy0, gy1, gx0, gx1, sub, area))
    if not feats:
        return accepted, 0, 0, rej
    fdf = pd.DataFrame(feats)
    n_scored = len(fdf)
    p = bundle["clf"].predict_proba(bundle["scaler"].transform(fdf[bundle["feature_names"]].values))[:, 1]
    thr = bundle.get("threshold", 0.5)
    # NOTE: deliberately NO force-keep-by-area here. classify_with_model applies
    # one (>=50000 px auto-accept) and it is right to to do so for candidates
    # produced by a DARKNESS threshold -- a dark region that large really is
    # always a crack or void. SAM masks carry no such guarantee: it will return
    # a 500k-px mask of a bright grain, and force-keeping those was what turned
    # this image's crack area from 1.6% into 33%. The classifier decides alone.
    for (gy0, gy1, gx0, gx1, sub, area), prob in zip(keep, p):
        if prob >= thr:
            accepted[gy0:gy1, gx0:gx1] |= sub
            n_acc += 1
    return accepted, n_scored, n_acc, rej


def make_overlay(img8, pipe_mask, sam_mask):
    both = pipe_mask & sam_mask
    only_p = pipe_mask & ~sam_mask
    only_s = sam_mask & ~pipe_mask
    rgb = np.stack([img8] * 3, -1).astype(np.uint8)
    rgb[only_p] = (220, 30, 30)      # red    -- pipeline only
    rgb[only_s] = (255, 240, 20)     # yellow -- SAM only (the union's addition)
    rgb[both] = (255, 140, 0)        # orange -- agreement
    im = Image.fromarray(rgb)
    if max(im.size) > VIEW_MAX:
        s = VIEW_MAX / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    names = ([a.only] if a.only else
             sorted(os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR)
                    if f.lower().endswith(".tif")))
    if a.limit:
        names = names[:a.limit]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = SamProcessor.from_pretrained(SAM_ID)
    model = SamModel.from_pretrained(SAM_ID, torch_dtype=torch.float32).to(device)
    gen = pipeline("mask-generation", model=model,
                   image_processor=proc.image_processor, device=device)
    bundle = joblib.load(PROD_MODEL_PATH)
    print(f"SAM {SAM_ID} on {device} | classifier {os.path.basename(PROD_MODEL_PATH)} "
          f"| {len(names)} images\n", flush=True)

    # resume: keep whatever a previous run already produced
    prev = pd.read_csv(STATS) if os.path.exists(STATS) else pd.DataFrame()
    rows = prev.to_dict("records") if len(prev) else []
    done = set(prev["SourceImage"]) if len(prev) else set()
    todo = [n for n in names if n not in done]
    print(f"{len(done)} already done, {len(todo)} to go\n", flush=True)

    for i, name in enumerate(todo, 1):
        t0 = time.time()
        try:
            pipe_mask = crack_mask_from_template(name)
            if pipe_mask is None:
                print(f"[{i}/{len(todo)}] {name}: no template, skipped", flush=True)
                continue
            img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{name}.tif"),
                                 **contrast_kwargs_for(name))
            x0, y0, x1, y1 = find_field_of_view(img8)
            img8 = img8[y0:y1, x0:x1]
            if img8.shape != pipe_mask.shape:
                print(f"[{i}/{len(todo)}] {name}: shape mismatch "
                      f"{img8.shape} vs template {pipe_mask.shape}, skipped", flush=True)
                continue
            flat = flatten_background(img8)
            ves = compute_vesselness(flat)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {name}: prep FAILED {type(e).__name__}: {e}", flush=True)
            continue

        masks, n_tiles = sam_masks_for(img8, gen)
        sam_mask, n_scored, n_acc, rej = score_sam_masks(masks, flat, ves, bundle, img8.shape)

        tot = img8.size
        union = pipe_mask | sam_mask
        only_s = int((sam_mask & ~pipe_mask).sum())
        only_p = int((pipe_mask & ~sam_mask).sum())
        rec = {
            "SourceImage": name, "tiles": n_tiles, "sam_masks_raw": len(masks),
            "sam_masks_scored": n_scored, "sam_masks_accepted": n_acc,
            "rej_too_big": rej["too_big"], "rej_border": rej["border_hugging"],
            "pipeline_px": int(pipe_mask.sum()), "sam_px": int(sam_mask.sum()),
            "union_px": int(union.sum()), "sam_only_px": only_s, "pipeline_only_px": only_p,
            "pipeline_area_pct": 100.0 * pipe_mask.sum() / tot,
            "union_area_pct": 100.0 * union.sum() / tot,
            "sam_only_area_pct": 100.0 * only_s / tot,
            "union_gain_pct_of_pipeline": (100.0 * only_s / max(int(pipe_mask.sum()), 1)),
        }

        # Ground-truth verdict on what SAM adds. The correction mask is the
        # human record (1 = crack, 2 = not-crack, 0 = never reviewed), so this
        # turns "do the yellow regions look wrong" into a measurement: pixels
        # SAM adds that land on human-confirmed crack are a real gain, pixels
        # landing on human-confirmed not-crack are false positives the union
        # would introduce.
        cm = load_correction_mask(name, img8.shape)
        for k in ("sam_only_on_crack", "sam_only_on_notcrack", "sam_only_unreviewed"):
            rec[k] = np.nan
        if cm is not None:
            so = sam_mask & ~pipe_mask
            rec["sam_only_on_crack"] = int((so & (cm == 1)).sum())
            rec["sam_only_on_notcrack"] = int((so & (cm == 2)).sum())
            rec["sam_only_unreviewed"] = int((so & (cm == 0)).sum())
        rows.append(rec)
        make_overlay(img8, pipe_mask, sam_mask).save(
            os.path.join(OUT_DIR, f"{name}_union.png"))
        pd.DataFrame(rows).to_csv(STATS, index=False)
        print(f"[{i}/{len(todo)}] {name:30s} {n_tiles:>3d} tiles, {len(masks):>5d} SAM masks "
              f"-> {n_acc:>4d} accepted | area pipe {rec['pipeline_area_pct']:5.2f}% "
              f"union {rec['union_area_pct']:5.2f}% | SAM-only {rec['sam_only_area_pct']:5.3f}% "
              f"(+{rec['union_gain_pct_of_pipeline']:.1f}% rel) | {time.time()-t0:5.1f}s",
              flush=True)

    if not rows:
        print("no images processed"); return
    d = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print(f"{len(d)} images")
    print(f"pipeline crack px total   {d.pipeline_px.sum():>12,d}")
    print(f"union crack px total      {d.union_px.sum():>12,d}")
    print(f"SAM-only px total         {d.sam_only_px.sum():>12,d}  "
          f"({100.0*d.sam_only_px.sum()/max(d.pipeline_px.sum(),1):.2f}% of pipeline)")
    print(f"mean area%, pipeline      {d.pipeline_area_pct.mean():>12.3f}")
    print(f"mean area%, union         {d.union_area_pct.mean():>12.3f}")
    print(f"SAM masks accepted total  {d.sam_masks_accepted.sum():>12,d} "
          f"of {d.sam_masks_scored.sum():,d} scored")

    g = d.dropna(subset=["sam_only_on_crack"])
    if len(g):
        oc, on, ou = (int(g.sam_only_on_crack.sum()), int(g.sam_only_on_notcrack.sum()),
                      int(g.sam_only_unreviewed.sum()))
        tot = max(oc + on + ou, 1)
        print(f"\nwhat SAM ADDS, judged against your correction masks "
              f"({len(g)} reviewed images):")
        print(f"  on human-confirmed CRACK      {oc:>12,d} px  ({100.0*oc/tot:5.1f}%)  <- real gain")
        print(f"  on human-confirmed NOT-crack  {on:>12,d} px  ({100.0*on/tot:5.1f}%)  <- false positive")
        print(f"  never reviewed                {ou:>12,d} px  ({100.0*ou/tot:5.1f}%)  <- unknown")
        if oc + on:
            print(f"  of the ADJUDICATED pixels, {100.0*oc/(oc+on):.1f}% are real crack")
    print(f"\noverlays in {OUT_DIR}")
    print(f"stats: {STATS}")


if __name__ == "__main__":
    main()
