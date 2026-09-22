#!/usr/bin/env python3
"""The best-performing crack detector found on this corpus, as one runnable function.

NOT THE SHIPPED DETECTOR. This is the winning arm of the offline evaluation in this
directory. The application in interior_active_learning/ does not import this module -- its
detector is the two-pass pipeline documented in the root README. Calling this "the model"
conflated the two; they are different code with different numbers.

    detect(grey_uint8) -> boolean mask

CONFIGURATION (chosen by leave-one-frame-out over 44 labelled frames, never on the test frame):
    Meijering ridge filter, sigmas 1-4, black_ridges=True
    keep the top 1% of the response (quantile 99)
    drop connected components of 32 px or fewer

    The quantile is 99 for WHOLE FRAMES, which is the deployment setting and what
    best_config.json ships. On the label-centred TILES the optimum was 98, because a tile
    cropped around a label contains proportionally more crack and so tolerates a fatter cut.
    This block said 98 for four commits after the deployed value became 99.

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

AN OPTIONAL SCRATCH REJECTOR is available via reject_scratches=True. IT IS OFF BY DEFAULT AND
SHOULD PROBABLY STAY OFF. Read this before enabling it.

With a leave-one-out orientation axis, an abstain gate for short components, an R >= 0.40
precondition and a Kulpa-metric tortuosity, it reaches median clIoU_adapt 0.1691 against 0.1514
for no filter at p = 0.027 over 38 frames, keeping 91% of predicted pixels.
(p and both medians are from artefact_filter.json, which is the source for this whole
paragraph; it stores the 38 per-frame scores with the filter ON but not the matching OFF
series, so the per-frame win count cannot be rechecked from the committed artefact and is
not quoted here. An earlier version of this line said p = 0.018, which appears nowhere in
the artefact.)

That is significant BY RANK and worthless BY MASS. The mean paired delta is -0.0001. One frame,
AS_24hr_BSE_Side_008, collapses from 0.5120 to 0.2289 -- a single loss 28x the median gain,
which cancels all 27 improvements. The signed-rank test counts ranks, so it cannot see this.

The mechanism is not fixable by tuning. That frame has orientation coherence R = 0.827 and the
filter deletes 455 components holding 60.5% of predicted pixels, because its cracks are long,
straight and mutually parallel -- the scratch signature exactly. R says there IS a dominant
direction; it cannot say whether that direction belongs to polishing marks or to cracks in a
directionally solidified or rolled microstructure. The leave-one-out axis does not help when
hundreds of parallel crack segments each set the axis for the others.

A WIDTH-UNIFORMITY GUARD WAS ADDED AND REMOVED. The idea was sound: a polishing mark has
constant width along its length, a crack tapers and branches, so requiring uniform width before
deleting should protect cracks. Measured from the binary mask it carries NO signal at all --
after a q99 cut both populations are 3-4 px wide because the cut sets the width, giving CV
0.271 for cracks against 0.284 for striations, indistinguishable and backwards. Measured from
the GREY, as the sigma of maximum single-scale Meijering response, it does carry signal: 0.269
against 0.208, AUC 0.680, correct direction.

It still does not work, and the reason is worth keeping. Forced onto the frame it was meant to
save, it recovers the score almost completely -- 0.2289 -> 0.5002 against 0.5118 for no filter
-- but it does so while keeping 97% of predicted pixels. It rescues the frame by switching the
filter OFF, not by telling cracks from scratches. Tightening it is monotonically harmful across
the corpus (0.1691 -> 0.1669 -> 0.1654 -> 0.1605) and leave-one-frame-out rejects it outright.
AUC 0.680 is real but too weak: every setting that protects cracks also stops the filter doing
anything.

Enable reject_scratches only on material where you know cracking is NOT directional, and check
the affected frames by eye.

REJECTING ROUND COMPONENTS MAKES THINGS WORSE and is not offered. Every setting that included
an eccentricity cut scored below filter-off (0.1416, 0.1379, 0.1336, 0.1309 against 0.1514).
Round components are not simply carbides: crack junctions and blobby crack mouths are round
too. That was half of the filter I designed, and the data rejected it.

DO NOT read the absolute numbers as quality. Every metric here is scored against region
assertions, not pixel-precise ground truth. They rank methods; they do not measure correctness.

Reproduce: expand_corpus.py -> expanded_bench.py -> this file's config is best_config.json.
"""
import os, json
import numpy as np
from skimage.filters import meijering
from skimage.morphology import remove_small_objects

SC = os.path.dirname(os.path.abspath(__file__))
# The fallback MUST match what best_config.json ships. It used to say quantile 98, so if the
# JSON were ever absent or unreadable detect() would silently become a different detector from
# the deployed one -- the failure mode this file exists to prevent.
_CFG_PATH = f"{SC}/best_config.json"
CFG = json.load(open(_CFG_PATH)) if os.path.exists(_CFG_PATH) else {
    "sigmas": [1, 2, 3, 4], "quantile": 99, "min_object_px": 32}


def detect(grey, quantile=None, sigmas=None, min_object_px=None, reject_scratches=False):
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
    mask = remove_small_objects(mask, max_size=mn) if mn else mask
    if reject_scratches:
        # OFF BY DEFAULT and deliberately so: it raises the median clIoU_adapt from 0.1514 to
        # 0.1691 at p = 0.027 over 38 frames. Turn it on if you would rather lose a straight
        # crack than keep a polishing scratch; leave it off if recall on straight cracks
        # matters. (These figures are for the align_deg = 30 call below -- the fourth
        # positional argument of apply_filter is align_deg, not min_len, which an earlier
        # version of this comment got wrong. It also described the superseded align_deg = 20
        # run -- 0.1684, p = 0.064 -- for one commit after the call changed.)
        # see the module docstring: significant by rank, net-zero by mass, and it can delete
        # 60% of a correct prediction on a frame whose cracks are parallel.
        # sys.path, not a bare import: SC is computed above but never added, so
        # `from artefact_filter import ...` only resolved when the caller's CWD happened to be
        # this directory. Importing best_detector from anywhere else raised ModuleNotFoundError
        # the moment reject_scratches was switched on.
        import sys as _sys
        if SC not in _sys.path:
            _sys.path.insert(0, SC)
        from artefact_filter import apply_filter
        mask, _ = apply_filter(mask, 0.0, 1.08, 30)
    return mask


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
