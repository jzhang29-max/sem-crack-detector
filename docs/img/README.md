# How the README figures were made

Recorded because the same frame looks materially different depending on which
configuration produced it — f1 0.715 without SAM, 0.776 with it — so an unlabelled
figure invites a reader to attribute the better number to a picture made by the
weaker setup. Each figure states its own configuration in its footer bar; this file
gives the commands.

## detection.png

`260622_316_H_b2_front_CBS_02`, Pass 1 + Pass 2 (no SAM). **No human labels from this
image were used**: it has no correction mask and contributes no rows to
`training_data/labeled_regions.csv`, so the right panel is unaided model output on
an image the classifier never saw.

    python3 code/build_figures.py 260622_316_H_b2_front_CBS_02 \
        --out docs/img/detection.png --frac 0.28 \
        --note "Pass 1 + Pass 2; no correction mask and no training rows from this image"

The first version of this figure used `..._CBS_01` and was rejected as a poor
demonstration, correctly: its red sat on grey grain faces as flat polygonal blobs,
one of them rectangular from a SAM tile boundary, while obvious dark features went
unmarked. Candidates for the replacement were ranked on morphology rather than by
eye -- share of red area in components with elongation >= 3, no single component
dominating, and raw-image contrast -- and `_CBS_02` won on all three. Two other
strong-looking candidates were rejected on honesty grounds: `HIP_24hr_SE_Side_006`
tracks its crack beautifully but 64% of that red is hand-painted, and
`MAR_Amb_HIP_CBS_0010` is one large red wash.

The crop is chosen by the script, not by eye: it takes the window with the most
crack pixels. A 6144 px frame downscaled to README width renders a 3 px crack at
half a pixel, so the figure has to be a crop and the crop has to come from the data.

## review.png

`AS_24hr_BSE_Side_008`, same pipeline, but this image **is** human-reviewed and **is**
in the training set — 1,285 of the 4,128 training rows come from it. Built with
`--reviewed`, which labels the right panel "after review" and its cyan "marked
not-crack" rather than "rejected". Mislabelling this one as model output would
present corrected pixels as unaided accuracy.

    python3 build_figures.py AS_24hr_BSE_Side_008 \
        --out docs/img/review.png --frac 0.32 --reviewed \
        --note "in the training set; 134,039 px adjudicated not-crack by hand"

## app.png

A screenshot of the running app at a 1440x900 viewport, with
`260622_316_H_b2_back_CBS_01` open. Not scripted. To retake it: start the app,
size the window to 1440x900, open that image, and capture at 1x rather than
retina — a 2880x1800 grab is four times the pixels for no legibility a reader
gains at the rendered size.

## benchmark/

The ten figures embedded in `../MODEL_VALIDATION_BENCHMARK.md`, produced by the
scripts in `interior_active_learning/code/experiments/`. They previously lived at
`interior_active_learning/benchmark_figures*/`, which was never tracked, so every
one of them rendered as a broken image.

## Conventions

Both composites are single PNGs with the panels and labels **baked in**, rather
than two images laid out by a markdown table. A table breaks differently across
GitHub, editors and other renderers, and loses every label the moment one panel is
dragged into a slide. PNG, not JPEG: the overlays are saturated red and cyan a few
pixels wide, exactly where JPEG's chroma subsampling fringes worst.

1400 px wide, ~500 KB each. That is roughly 2x GitHub's README column, so they stay
sharp on retina displays without needing an HTML `width` attribute.
