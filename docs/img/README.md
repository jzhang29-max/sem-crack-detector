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

    ./.venv/bin/python3 code/build_figures.py 260622_316_H_b2_front_CBS_02 \
        --out docs/img/detection.png --frac 0.55 \
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
in the training set — 1,285 of the 4,787 training rows come from it. Built with
`--reviewed`, which labels the right panel "after review" and its cyan "marked
not-crack" rather than "rejected". Mislabelling this one as model output would
present corrected pixels as unaided accuracy.

    ./.venv/bin/python3 code/build_figures.py AS_24hr_BSE_Side_008 \
        --out docs/img/review.png --frac 0.55 --reviewed \
        --note "in the training set; 134,039 px adjudicated not-crack by hand"

## app.png

The running app at a 1440x900 viewport with `260622_316_H_b2_back_CBS_01` open. Retaken
2026-08-23; the previous one dated from 2026-08-17 and showed a threshold of 0.400 and a
sidebar model panel that no longer exist — the app now reports
`LogisticRegression / thr 0.500 / no SAM` in the status bar.

It used to say "not scripted", which is why it went stale unnoticed. It is scriptable, so
here is the command:

    ./run &                       # or: make run
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
        --headless=new --disable-gpu --hide-scrollbars \
        --virtual-time-budget=25000 --window-size=1440,900 \
        --screenshot=docs/img/app.png http://127.0.0.1:8767/

The virtual-time budget matters: the overlay is fetched and drawn after load, so a shorter
budget captures an empty canvas. The app reopens whatever image was last painted, so open that
frame once in a real browser first if you want the same one in the shot.


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

## pipeline.png

The architecture diagram at the top of the top-level README.

    ./.venv/bin/python3 code/generate_pipeline_diagram.py

Generated rather than drawn on purpose: the stage order lives in the `STAGES` list in that
script, so changing the pipeline and forgetting the picture is a diff someone sees in review
instead of a diagram that quietly stops being true.

## Both figures were rebuilt on 2026-08-23, and why

They had been built on 2026-08-17/18, which put them **before** commit `ce11f25`
("drop the exclusion that deleted main cracks"). The old `detection.png` therefore showed a
detector that left the large through-cracks unmarked — the very thing that commit fixed. Any
figure that predates a detection change is evidence for a detector nobody runs.

Rebuilding also forced a crop change. `--frac` is the crop height as a fraction of the frame,
and the window is still chosen by `densest()` rather than by eye. At the old `0.28` the denser
mask put that window *inside* the main crack, so the figure was a solid red rectangle that
demonstrated nothing. `0.55` keeps the same data-picked centre and pulls back far enough to show
the crack against correctly-unmarked material, which is what the figure is for. The frame-level
number in the caption (9.8% of area marked crack, up from 2.5%) comes from the whole image, not
the crop, so it is unaffected by the zoom.

## model_card.png

The sidebar model card, opened, showing the Performance rows the app now carries: held-out AUC,
grouped-CV AUC with its spread, pixel f1, and a dash for the false-call rate that this corpus
cannot support. Every row has a `title=` tooltip with the protocol behind it.

The card is collapsed by default, so the URL fragment `#model` opens it on load purely so this
screenshot is scriptable:

    ./run &
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
        --headless=new --disable-gpu --hide-scrollbars \
        --virtual-time-budget=25000 --window-size=1440,900 \
        --screenshot=docs/img/model_card.png "http://127.0.0.1:8767/#model"

That fragment exists because of what happened to `app.png`: its doc said "not scripted" and it
sat six days stale, showing a threshold and a sidebar panel that no longer existed. A panel
that can only be photographed by hand is one whose screenshot will rot the same way.
