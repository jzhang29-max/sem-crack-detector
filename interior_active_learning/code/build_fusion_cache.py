"""
Cache everything needed to evaluate SAM+classifier combinations, so that the
expensive part is paid once instead of once per experiment.

SAM inference costs ~5s per 1024px tile, ~3 min for a large image. A search
over {classifier} x {region source} x {fusion rule} x {SAM filter} would re-run
identical inference dozens of times if each trial started from the image. So
this runs the pipeline and SAM once per evaluation image and stores, for every
candidate region from either source:

    source ('pipeline' | 'sam'), bbox, the region mask packed to bits,
    the 8 production features, mean raw brightness, and the human verdict

Regions are stored as bbox + packed bits rather than full-frame masks: the sum
of all region areas is a small fraction of the frame, so this is ~100x smaller
than one int32 label array per image and any combination can still be rebuilt
exactly by OR-ing the selected crops.

Features for BOTH sources come from detect_cracks.region_features_from_labeled,
the same function extract_candidates uses, so a SAM mask and a pipeline
candidate of identical shape get identical features. Hand-rolling this for SAM
is what produced a phantom "+16% recall from union" earlier in this project.

Ground truth is the per-pixel correction mask (1=crack, 2=not-crack,
0=unreviewed). Only images with both classes present are useful as evaluation
targets, since specificity is meaningless without negatives.

    python3 build_fusion_cache.py [--images A B ...]
Writes ../../fusion_cache/<image>.npz
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline

from common import (ORIGINAL_DIR, PROJECT_ROOT, contrast_kwargs_for, load_correction_mask)
from detect_cracks import (load_as_uint8, find_field_of_view, flatten_background,
                            segment_dark_regions, clean_mask, compute_vesselness,
                            exclude_border_background, extract_candidates,
                            region_features_from_labeled)

Image.MAX_IMAGE_PIXELS = None
OUT_DIR = os.path.join(PROJECT_ROOT, "fusion_cache")
FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity",
            "Extent", "Circularity", "MeanDarkness", "MeanVesselness"]
SAM_ID = "facebook/sam-vit-huge"
TILE, STRIDE = 1024, 896
MIN_AREA = 40
# the 5 both-class images with the most negatives -- the only places
# specificity can be measured at all
DEFAULT = ["AS_24hr_BSE_Side_008", "260708_316_H_b2_front_CBS_012",
           "260708_316_H_b2_front_CBS_013", "260708_316_H_b2_front_CBS_015",
           "MAR_Amb_HIP_CBS_0006"]


def verdict_for(mask, cm):
    """majority human verdict over a region: 1 crack, 2 not-crack, 0 unreviewed"""
    v = cm[mask]
    if v.size == 0:
        return 0
    c = np.bincount(v, minlength=4)
    if c[1:].sum() == 0:
        return 0
    return int(c[1:].argmax() + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = SamProcessor.from_pretrained(SAM_ID)
    model = SamModel.from_pretrained(SAM_ID, torch_dtype=torch.float32).to(device)
    gen = pipeline("mask-generation", model=model,
                   image_processor=proc.image_processor, device=device)
    print(f"SAM {SAM_ID} on {device}\n", flush=True)

    for name in a.images:
        t0 = time.time()
        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{name}.tif"),
                             **contrast_kwargs_for(name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        flat = flatten_background(img8)
        clean = clean_mask(segment_dark_regions(flat, img8=img8), min_area_px=13)
        ves = compute_vesselness(flat)
        clean = exclude_border_background(clean, ves)
        labeled, df = extract_candidates(clean, flat, ves, min_area_px=MIN_AREA)

        cm = load_correction_mask(name, img8.shape)
        if cm is None:
            print(f"{name}: no correction mask, skipped"); continue

        recs = []

        def add(src, m):
            ys, xs = np.nonzero(m)
            if not len(ys):
                return
            y0b, y1b, x0b, x1b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sub = m[y0b:y1b, x0b:x1b]
            if sub.sum() < MIN_AREA:
                return
            lab = sub.astype(np.int32)
            _, fd = region_features_from_labeled(lab, flat[y0b:y1b, x0b:x1b],
                                                 ves[y0b:y1b, x0b:x1b], min_area_px=MIN_AREA)
            if not len(fd):
                return
            r = fd.iloc[0]
            recs.append({
                "source": src, "bbox": [int(y0b), int(y1b), int(x0b), int(x1b)],
                "shape": [int(sub.shape[0]), int(sub.shape[1])],
                "bits": np.packbits(sub.ravel()),
                "feats": np.array([float(r[f]) for f in FEATURES], np.float64),
                "area": int(sub.sum()),
                "mean_raw": float(img8[y0b:y1b, x0b:x1b][sub].mean()),
                "verdict": verdict_for(m, cm),
            })

        for lbl in df["Label"].values:
            add("pipeline", labeled == int(lbl))
        n_pipe = len(recs)

        h, w = img8.shape
        ys_t = list(range(0, max(h - TILE, 0) + 1, STRIDE)) or [0]
        xs_t = list(range(0, max(w - TILE, 0) + 1, STRIDE)) or [0]
        if ys_t[-1] + TILE < h: ys_t.append(max(h - TILE, 0))
        if xs_t[-1] + TILE < w: xs_t.append(max(w - TILE, 0))
        for cy in ys_t:
            for cx in xs_t:
                tile = img8[cy:cy + TILE, cx:cx + TILE]
                if min(tile.shape) < 16:
                    continue
                try:
                    res = gen(Image.fromarray(np.stack([tile] * 3, -1)), points_per_batch=64)
                except Exception as e:
                    print(f"  tile ({cy},{cx}) SAM failed {type(e).__name__}", flush=True)
                    continue
                for mm in res["masks"]:
                    mm = mm.detach().cpu().numpy() if isinstance(mm, torch.Tensor) else np.asarray(mm)
                    while mm.ndim > 2:
                        mm = mm[0]
                    mm = mm.astype(bool)
                    if mm.shape != tile.shape or mm.sum() < MIN_AREA:
                        continue
                    if mm.sum() > 0.15 * TILE * TILE:
                        continue
                    full = np.zeros(img8.shape, bool)
                    full[cy:cy + tile.shape[0], cx:cx + tile.shape[1]] = mm
                    add("sam", full)

        out = os.path.join(OUT_DIR, f"{name}.npz")
        np.savez_compressed(
            out,
            image=name, shape=np.array(img8.shape),
            feature_names=np.array(FEATURES),
            source=np.array([r["source"] for r in recs]),
            bbox=np.array([r["bbox"] for r in recs], np.int32),
            rshape=np.array([r["shape"] for r in recs], np.int32),
            bits=np.array([r["bits"] for r in recs], dtype=object),
            feats=np.array([r["feats"] for r in recs], np.float64),
            area=np.array([r["area"] for r in recs], np.int64),
            mean_raw=np.array([r["mean_raw"] for r in recs], np.float64),
            verdict=np.array([r["verdict"] for r in recs], np.int8),
            human_crack_bits=np.packbits((cm == 1).ravel()),
            human_notcrack_bits=np.packbits((cm == 2).ravel()),
            img_median=float(np.median(img8)),
        )
        v = np.array([r["verdict"] for r in recs])
        print(f"{name:32s} {n_pipe:>5d} pipeline + {len(recs)-n_pipe:>5d} SAM regions | "
              f"verdicts crack {int((v==1).sum()):>4d} / not {int((v==2).sum()):>4d} / "
              f"unreviewed {int((v==0).sum()):>5d} | {time.time()-t0:5.1f}s", flush=True)

    print(f"\ncache in {OUT_DIR}")


if __name__ == "__main__":
    main()
