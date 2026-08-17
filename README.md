# SEM Crack Detector

Drop SEM images into a browser window, see cracks detected, fix what's wrong by
painting. Corrections save themselves. One button retrains the model on them.

```bash
git clone https://github.com/jzhang29-max/sem-crack-detector.git && cd sem-crack-detector && ./run
```

That is the entire setup — one command. `./run` creates a virtualenv, installs
dependencies, and opens the browser. The first run takes a few minutes to
install; afterwards it starts in seconds. `make` does the same. Nothing to
configure, no paths to edit.

## The loop

1. **Drop images** anywhere on the window — `.tif .tiff .png .jpg`. Each is
   converted to greyscale and run through the current model automatically.
2. **Look at the result.** Red = crack, cyan = a candidate the model rejected.
   Untick **Show result** to see the plain image underneath.
3. **Correct it** with the three tools: **Add crack**, **Not crack**, **Erase**
   (remove from consideration entirely). Switch the second control from
   **Brush** to **Whole region** to set an entire connected region in one
   click — essential for large ones, where brushing would take hundreds of
   strokes.
4. **Nothing to save.** Marks commit by themselves about a second after you stop
   drawing; the status bar says *All changes saved*. Switching images flushes
   first, and closing the tab with unsaved marks warns you.
5. **Retrain** rebuilds the training set from every correction you have ever
   made, retrains, and re-renders every image.

Corrections are stored per-pixel and **always override the model**. Retraining
never discards them.

### Undo

<kbd>⌘Z</kbd> undoes your last brush stroke. Once local strokes run out it
undoes the **last saved correction** — a server-side stack ten deep per image.
This matters because saving is automatic: without it, a mistaken mark would
become permanent a second after you drew it.

Also: <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> pick Add crack / Not crack / Erase,
<kbd>F</kbd> fits the image to the window.

## Speed, and the one place it still waits

| action | large image (6144×4096) |
|---|---|
| opening an image | 0.03 s |
| a correction saving itself | **6.2 s** |
| background preparation after opening | ~204 s |

Saving a correction needs the image's pipeline result. Building that costs ~204 s
on the largest images, so it starts the moment you open one, in the background,
while you are still looking. **If you open a large image and paint within the
first few minutes, that first save waits for it to finish.** Smaller images
prepare in ~25 s. Everything after the first save is ~6 s.

Whole-region clicks are immediate; they never go through this path.

## Exports

**Download** gives a black & white mask (crack = black), the overlay burned into
the image, a per-region CSV (area, centroid, bbox, length, width, aspect ratio,
eccentricity, orientation, solidity, perimeter), or every image as a `.zip` with
a cross-image `summary.csv`.

All lengths are in **pixels**. The scale bar lives in the SEM databar this
pipeline crops off before analysis, so no µm/px factor is recoverable here —
multiply by the microscope's own value.

## Managing models

The sidebar's model card shows what is live — type, threshold, how many reviewed
regions it was trained on, and its held-out AUC. The dropdown switches models,
which is how you **roll back** a retrain you did not want; the model being
replaced is backed up first. **Advanced → Re-apply model** re-renders every image
with whichever model is selected.

**Retrain will not deploy a worse model.** The candidate is scored against the
current one on a held-out image and only promoted if it does at least as well.
That is not hypothetical — during development 18 extra training rows moved
held-out AUC from 0.9252 to 0.9153 and the gate declined to deploy.

## How well it works

Measured on the 5 images with enough human not-crack marks for specificity to
mean anything. Pixel-level, scored only on pixels a human actually adjudicated,
with the classifier trained on *other* images:

| detector | f1 | recall | specificity |
|---|---|---|---|
| Pass 1 (darkness threshold + classifier) | 0.697 | 0.575 | 0.476 |
| Pass 1 + Pass 2 (interior / bridge fills) | 0.715 | 0.597 | 0.476 |
| **Pass 1 + Pass 2 + SAM** — the default | **0.776** | **0.678** | 0.395 |

SAM adds +0.061 f1 over the full pipeline, improving 4 of 5 images. The gain is
entirely recall (+8.1 points) traded against specificity (−8.1), precision flat —
a good trade when a human reviews the output, since deleting a false positive is
one click and noticing a missed crack is not.

**Limits, plainly:** n = 5 images, so no statistical test clears p = 0.0625.
Specificity rests on small negative pools — one image has 4,416 not-crack pixels,
another none at all. Only 1.8–4% of each image is adjudicated, so most of what
SAM adds is unmeasurable in either direction. Treat +0.061 as real but loosely
bounded.

Full method, including the measurement mistakes made along the way and what they
invalidated, is in [docs/MODEL_VALIDATION_BENCHMARK.md](docs/MODEL_VALIDATION_BENCHMARK.md).
A comparison against the sibling TXM app — what was adopted from it and what was
declined — is in [docs/APP_COMPARISON.md](docs/APP_COMPARISON.md).

## SAM is optional

Untick **Advanced → Use SAM on new images** for ~40 s/image instead of ~3 min, at
f1 0.715 instead of 0.776. If PyTorch is not installed the checkbox disables
itself and the pipeline runs alone — nothing breaks.

```bash
pip install torch transformers torchvision   # torchvision is required, not extra:
                                             # SAM's post-processing calls its NMS
```

## What ships here

Code, the trained models, and the human correction masks — the labels are the
only part nobody can regenerate. Source images, overlays and results are excluded:
they are large and reproducible from the images plus the labels. A clone with no
images still retrains, using the shipped labels.

Removing an image from the list **moves** it to `removed_images/` rather than
deleting it, and keeps its corrections, so a misclick is recoverable.

## Testing

```bash
PORT=8799 ./run &
BASE=http://127.0.0.1:8799 python3 interior_active_learning/code/test_app.py
```

66 checks covering upload, detection, exports, correction precedence, region
isolation, threshold plumbing, the retrain gate, autosave, undo, and the
performance fixes. `make test` runs the same thing.

## Notes for maintainers

- **The threshold lives in the model bundle**, not in call sites.
  `classify_with_model` reads it; hardcoding one silently discards any
  recalibration.
- **Features for SAM masks go through `region_features_from_labeled`**, the same
  function the pipeline uses. Re-implementing them once produced a sign-flipped
  `MeanDarkness` and an entirely phantom result.
- **There are two model directories.** `models/` holds Pass 1;
  `interior_active_learning/models/unified_model.joblib` holds Pass 2, and when
  it is missing Pass 2 is skipped — costing 27% of crack regions. The app warns
  rather than failing silently.
- **Undo stores only each stroke's bounding box.** It used to snapshot the whole
  canvas: 100 MB per stroke on a 6144×4096 image, 25 deep.
- **Colour masks use integer squared distances.** The float32 version allocated a
  300 MB temporary per call, three calls per save.
- **Editing the frontend needs only a browser refresh** — the server re-reads
  `paint_frontend.py` when its mtime changes.
- **Editing a correction mask by hand needs a server restart.** The per-image
  pipeline cache does not watch that file, so the old verdict survives in memory.
