# SEM Crack Detector

Drop SEM images in a browser window, see cracks detected, fix what's wrong by
painting, press one button to retrain on your corrections.

```bash
git clone https://github.com/jzhang29-max/sem-crack-detector.git && cd sem-crack-detector && ./run
```

That's the whole setup. `./run` creates a virtualenv, installs dependencies and
opens the browser. First run takes a few minutes to install; after that it starts
in seconds. `make` does the same if you prefer.

## The loop

1. **Drop images** onto the window — `.tif .tiff .png .jpg`. Each is converted to
   greyscale and run through the current model automatically.
2. **Look at the overlay.** Red = crack, cyan = rejected candidate.
3. **Correct it.** Red brush marks a crack, cyan marks not-a-crack, × removes a
   region from consideration. Switch **Mode** to *Whole region* to set an entire
   connected region in one click — essential for large ones.
4. **Nothing to save.** Marks commit automatically about a second after you stop
   drawing; the footer says *All changes saved*. Switching images flushes first,
   so work cannot be lost.
5. **Retrain & re-overlay** rebuilds the training set from every correction you
   have made, retrains, and re-renders every image.

Corrections are stored per-pixel and **always override the model**. Retraining
never discards them.

Keyboard: <kbd>⌘Z</kbd>/<kbd>Ctrl-Z</kbd> undo · <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd>
crack/not-crack/erase · <kbd>F</kbd> fit.

## Exports

**Export** gives a B&W mask (crack = black), the overlay burned into the image, a
per-region CSV (area, centroid, bbox, length, width, aspect, eccentricity,
orientation, solidity, perimeter), or every image as a `.zip` with a cross-image
`summary.csv`.

All lengths are in **pixels**. The scale bar lives in the SEM databar this
pipeline crops before analysis, so no µm/px factor is recoverable here — multiply
by the microscope's own value.

## How well it works

Measured on the 5 images with enough human not-crack marks for specificity to
mean anything. Pixel-level, scored only on pixels a human adjudicated, classifier
trained on *other* images:

| detector | f1 | recall | specificity |
|---|---|---|---|
| Pass 1 (darkness threshold + classifier) | 0.697 | 0.575 | 0.476 |
| Pass 1 + Pass 2 (interior/bridge fills) | 0.715 | 0.597 | 0.476 |
| **Pass 1 + Pass 2 + SAM** — the default | **0.776** | **0.678** | 0.395 |

SAM adds +0.061 f1 over the full pipeline, improving 4 of 5 images. The gain is
entirely recall (+8.1 points) traded against specificity (−8.1), precision flat —
a good trade when a human reviews the output.

**Limits, plainly:** n = 5 images, so no statistical test clears p = 0.0625.
Specificity rests on small negative pools — one image has 4,416 not-crack pixels,
another none. Only 1.8–4% of each image is adjudicated, so most of what SAM adds
is unmeasurable either way. Treat +0.061 as real but loosely bounded.

Method, including the measurement mistakes made along the way and what they
invalidated, is in [docs/MODEL_VALIDATION_BENCHMARK.md](docs/MODEL_VALIDATION_BENCHMARK.md).

## SAM is optional

Untick **Options → Use SAM** for ~40 s/image instead of ~3 min, at f1 0.715. If
PyTorch is absent the box disables itself and the pipeline runs alone.

```bash
pip install torch transformers torchvision   # torchvision is required, not extra:
                                             # SAM's post-processing calls its NMS
```

## Retraining is guarded

**Retrain** does not blindly deploy. The candidate is compared against the
current model on a held-out image and only promoted if it scores at least as
well. Not hypothetical — it caught a regression during development where 18
extra rows moved held-out AUC from 0.9252 to 0.9153. The previous model is kept
at `models/crack_classifier_PREV.joblib`.

## What ships here

Code, trained models, and the human correction masks — the labels are the part
nobody can regenerate. Source images, overlays and results are excluded: large
and reproducible from images plus labels.

## Notes for maintainers

- The threshold lives **in the model bundle**, not in call sites.
  `classify_with_model` reads it; hardcoding one discards any recalibration.
- Features for SAM masks go through `region_features_from_labeled`, the same
  function the pipeline uses. Re-implementing them once produced a sign-flipped
  `MeanDarkness` and a phantom result.
- There are **two** model directories. `models/` holds Pass 1;
  `interior_active_learning/models/unified_model.joblib` holds Pass 2, and when
  it is missing Pass 2 is skipped — costing 27% of crack regions. The app warns.
- Undo stores only each stroke's bounding box. It used to snapshot the whole
  canvas (100 MB per stroke on a 6144×4096 image, 25 deep) which is what made
  painting feel slow.
- The server caches pipeline output per image; restart it after editing a
  correction mask by hand.
