#!/usr/bin/env python3
"""The best-performing crack detector found on this corpus, as one runnable function.

    detect(grey_uint8) -> boolean mask

CONFIGURATION (chosen by leave-one-frame-out over 44 labelled frames, never on the test frame):
    Meijering ridge filter, sigmas 1-4, black_ridges=True
    keep the top 2% of the response (quantile 98)
    drop connected components of 32 px or fewer

WHY THIS ONE. Scored on clIoU_adapt -- tolerant centreline IoU with the tolerance set per frame
to half that frame's label brush -- it is the top arm at 0.2848 (LOFO, 95% CI [0.2040, 0.3450]),
and it beats Otsu by +0.0741 at p = 0.0049, which survives Bonferroni over the six pairwise
tests. Sato ridge is statistically indistinguishable from it (p = 0.889) and is an equally
defensible choice; everything else is not significantly different from anything.

WHAT "BEST" DEPENDS ON, because it genuinely does here:
    clIoU_adapt   Meijering 0.2848  > Sato 0.2752 > threshold 0.2024 > Otsu 0.1925
    cont_lift     threshold  3.18   > Meijering 2.07 > Sato 1.99 > Otsu 1.71
    pixel IoU     Otsu       0.2945 > threshold 0.1815 > Sato 0.1097 > Meijering 0.0957

Otsu wins pixel IoU and comes LAST on the metric that suits these labels. That is not a
paradox: its PAR is 1.45, so it over-predicts area, and against a brush of median 59 px pixel
IoU rewards exactly that. A perfect 3 px trace down the label centreline scores pixel IoU
0.1662 on this corpus (iou_ceiling.py), so any pixel-IoU number above ~0.17 is a measure of
thickness, not of detection. Meijering's PAR of 0.553 means it predicts about half the labelled
area -- which is what a correct thin detection looks like against a broad brush.

DO NOT read the absolute numbers as quality. Every metric here is scored against region
assertions, not pixel-precise ground truth. They rank methods; they do not measure correctness.

Reproduce: expand_corpus.py -> expanded_bench.py -> this file's config is best_config.json.
"""
import os, json
import numpy as np
from skimage.filters import meijering
from skimage.morphology import remove_small_objects

SC = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(f"{SC}/best_config.json")) if os.path.exists(f"{SC}/best_config.json") else {
    "sigmas": [1, 2, 3, 4], "quantile": 98, "min_object_px": 32}


def detect(grey, quantile=None, sigmas=None, min_object_px=None):
    """Return a boolean crack mask for a uint8 greyscale SEM tile.

    `grey` must be the RAW micrograph. Never pass a rendered overlay: doing so is what
    invalidated this project's first benchmark, where the annotation was burned into the input
    and a bare threshold recovered it at recall 1.000 (see LEAK_POSTMORTEM.md).
    """
    q = CFG["quantile"] if quantile is None else quantile
    sg = CFG["sigmas"] if sigmas is None else sigmas
    mn = CFG["min_object_px"] if min_object_px is None else min_object_px
    g = grey.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    r = meijering(g, sigmas=np.asarray(sg, dtype=float), black_ridges=True)
    mask = r >= np.percentile(r, q)
    return remove_small_objects(mask, max_size=mn) if mn else mask


if __name__ == "__main__":
    from PIL import Image
    meta = json.load(open(f"{SC}/tiles_all/meta.json"))
    print(f"config: {CFG}\n")
    print(f"  {'frame':<40} {'predicted %':>12} {'labelled %':>11} {'PAR':>6}")
    for m in meta[:8]:
        g = np.array(Image.open(f"{SC}/tiles_all/{m['tile']}_gray.png").convert("L"))
        gt = np.array(Image.open(f"{SC}/tiles_all/{m['tile']}_gt.png")) > 127
        p = detect(g)
        print(f"  {m['tile'][:38]:<40} {100*p.mean():>11.2f}% {100*gt.mean():>10.2f}% "
              f"{p.sum()/max(gt.sum(),1):>6.2f}")
