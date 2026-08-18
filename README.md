# SEM Crack Detector

Drop SEM images into a browser window, see cracks detected, fix what's wrong by
painting. Corrections save themselves. One button retrains the model on them.

## What it looks like

![The app: a sidebar of SEM images with crack counts, and a large through-crack marked in red on the open image](docs/img/app.png)

The sidebar lists every image with how much of it is flagged; the toolbar is the
entire workflow. The card at the bottom names the model that is actually live —
family, threshold, how many reviewed regions it was trained on — so the number on
screen is never ambiguous about which model produced it.

The image open here is `260622_316_H_b2_back_CBS_01`, where 14.9% of the frame is
marked crack. **13.2 of those 14.9 points the detector found on its own**; a human
forced only 10.9% of the red. Nothing in that image was marked *not*-crack, so the
review there ran one way only — the reviewer added, and never overruled.

## Install

```bash
git clone https://github.com/jzhang29-max/sem-crack-detector.git && cd sem-crack-detector && ./run
```

That is the entire setup — one command. `./run` creates a virtualenv, installs
dependencies, and opens the browser. The first run takes a few minutes to
install; afterwards it starts in seconds. `make` does the same. Nothing to
configure, no paths to edit.

The clone downloads **~1.2 GB** and occupies **~2.2 GB** on disk once checked out
(git keeps its own compressed copy alongside the working files). That is because
the 62 source SEM images ship with it, so the detector and every number below can
be reproduced rather than taken on trust.

## The loop

The 62 images ship **without their overlays** — those are derived, and at ~30 MB
each they would quadruple the download. So the first time you click an image, the
detector runs on it right then, with a progress bar and a running note of which
stage it is in. Measured: **20 s for a 5.8-megapixel frame, 94 s for a
26.9-megapixel one**. After that the overlay is cached on disk and the same image
reopens in under a second. The counts in the sidebar come from the repo, so you can see what is worth
opening before anything is rendered.

1. **Drop images** anywhere on the window — `.tif .tiff .png .jpg`. Each is
   converted to greyscale and run through the current model automatically.
2. **Look at the result.** Red = crack, cyan = a candidate the model rejected.
   Untick **Show result** to see the plain image underneath.
3. **Correct it** with the three tools: **Add crack**, **Not crack**, **Erase**
   (remove from consideration entirely). Switch the second control from
   **Brush** to **Whole region** to set an entire connected region in one
   click — essential for large ones, where brushing would take hundreds of
   strokes.

![The same frame before and after human review, cracks in red and hand-marked not-crack regions in cyan](docs/img/review.png)

   Left, the micrograph; right, the result after review. This is
   `AS_24hr_BSE_Side_008`, the most heavily adjudicated image in the set — 1,285 of
   the 4,128 training rows come from it, and 134,039 pixels in it were marked
   not-crack by hand. Red tracks the cracks closely here precisely *because* it was
   reviewed, which is why this figure is labelled "after review" and the one above
   is not: read this one as what a finished image looks like, not as unaided
   accuracy.

4. **The status bar is the source of truth.** Every long action — detecting,
   saving, retraining, re-applying — reports there and on the thin progress bar
   under the toolbar, naming the stage rather than just spinning.
5. **Nothing to save.** Marks commit by themselves about a second after you stop
   drawing; the status bar says *All changes saved*. Switching images flushes
   first, and closing the tab with unsaved marks warns you.
6. **Retrain** rebuilds the training set from every correction you have ever
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

### A crack it finds on an image it has never seen

![An SEM micrograph beside the same frame with detected cracks in red and rejected candidates in cyan](docs/img/detection.png)

Left is the micrograph as acquired. Right is the model's own output: red where it
calls crack, cyan where it proposed a region and then turned it down. This image
carries no human labels and contributed no training rows, so nothing in it was
fitted. The cyan discs are the informative part — those are round pores, and the
classifier rejected them while keeping the elongated dark features beside them.

Now the failure in the same frame. The broad red patch in the upper third, the one
straddling the centre line, has a dead-straight horizontal top edge — and that edge
is a **SAM tile boundary**. SAM runs on 1024 px tiles at 896 px stride and scores
each tile independently, so a crack accepted in one tile can fall below threshold in
the next and the region is cut off mid-crack. At y = 2688, which is exactly 3 x 896,
86% of that patch's width is red; one pixel row above it, 0% is. Three of this
image's eight largest regions end exactly on a tile line. The 128 px overlap between
tiles softens this but plainly does not remove it.

Measured on the 5 images with enough human not-crack marks for specificity to
mean anything. Pixel-level, scored only on pixels a human actually adjudicated,
with the classifier trained on *other* images:

| detector | f1 | recall | specificity |
|---|---|---|---|
| Pass 1 (darkness threshold + classifier) | 0.697 | 0.575 | 0.476 |
| **Pass 1 + Pass 2** — what the shipped overlays use | **0.715** | 0.597 | 0.476 |
| **Pass 1 + Pass 2 + SAM** — best, opt-in | **0.776** | **0.678** | 0.395 |

**SAM is not part of the model.** `crack_classifier.joblib` is a LogisticRegression
over 8 features and is identical whether or not SAM is installed. SAM is a
separate stage run at detection time: it proposes regions the darkness threshold
never found, and the same classifier scores them. So "+ SAM" is a pipeline
configuration, not a different trained model.

Which means it only applies where it is actually run:

| action | includes SAM? |
|---|---|
| dropping in a new image | yes, if **Use SAM** is ticked |
| **Re-apply model** | only if you choose it when asked |
| the re-render after **Retrain** | only if you ask for it |
| the overlays shipped in this repo | **no** — pipeline only, f1 0.715 |

Including SAM costs **~8 min per image** instead of ~40 s, so a 62-image
re-render is **~8.5 hours** rather than ~15 minutes. (Measured on an Apple M-series
GPU via MPS at 6144×4096; an earlier estimate of ~3 min/image was taken from a
smaller image and was optimistic by roughly 3×.) Re-apply asks which you want and
shows the estimate for your image count.

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

## Turning SAM on

If PyTorch is not installed, **Advanced** shows an **Enable SAM** button that
installs it into this app's own virtualenv — no terminal needed. It is ~2.5 GB and
takes a few minutes; restart the app afterwards. Until then the checkbox is
disabled and the pipeline runs alone, which is a working detector, just the 0.715
one rather than 0.776.

Equivalent by hand, if you prefer:

```bash
pip install torch transformers torchvision   # torchvision is required, not extra:
                                             # SAM's post-processing calls its NMS
```

## What ships here

Code, the trained models, the human correction masks, and **the 62 source
images**. The masks are the part nobody can regenerate; the images are what makes
every result here checkable instead of merely claimed.

The images are **losslessly compressed** — zlib inside the TIFF, 2.55 GB down to
1.17 GB, pixel values bit-identical (asserted per image at packaging time, not
assumed). The pipeline reads exactly the same numbers it would from the
uncompressed originals.

Derived outputs are excluded because they are regenerable and large: overlays,
results, figures, caches. Retraining also works in a clone with no images at all,
from the shipped labels alone.

Removing an image from the list **moves** it to `removed_images/` rather than
deleting it, and keeps its corrections, so a misclick is recoverable.

## Where things are

Every `.py` in this repo is in exactly one of three categories. Nothing is left
unexplained, and nothing that the app does not use sits next to code that it does.

**The app — 18 modules, this is the whole live path:**

| file | what it is |
|---|---|
| `interior_active_learning/code/paint_server.py` | the Flask app; `./run` starts this |
| `…/paint_frontend.py` | the entire UI, one file, hot-reloads on save |
| `…/app_endpoints.py` | upload, detect, retrain, job polling |
| `…/app_exports.py` | mask / overlay / region CSV / zip |
| `…/app_extras.py` | thumbnails, model picker, re-apply, remove, SAM install |
| `…/app_undo.py` | the server-side correction-mask undo stack |
| `…/apply_paint_annotations.py` | painted PNG → per-pixel correction mask |
| `…/build_training_data.py` | correction masks → training rows |
| `…/train_v3_weighted.py` | trains Pass 1 with per-image weighting |
| `…/regenerate_templates.py` | batch re-render; **the only** overlay renderer |
| `…/hybrid_detect.py` | Pass 1 + Pass 2 + SAM, as one call |
| `…/unified_pipeline.py` | orchestrates the two passes |
| `…/interior_candidates.py` | Pass 2 candidate generation |
| `…/common.py` | paths and per-image contrast settings |
| `…/labeling_overlay.py`, `…/active_learning_select.py` | overlay drawing, image ranking |
| `…/test_app.py` | the test suite |
| `code/detect_cracks.py` | Pass 1 segmentation and the 8 features |

**Kept because they regenerate something that ships here.** Not imported by the
app, so nothing breaks if you ignore them — but delete them and an artifact in
this repo becomes unreproducible:

- `train_unified_model.py`, `unified_data.py`, `build_original_ledger_unified_features.py`
  → rebuild the Pass 2 model, `interior_active_learning/models/unified_model.joblib`
- `train_interior_model.py`, `apply_interior_model.py` → rebuild and inspect
  `interior_model.joblib`, which `interior_candidates.py` loads at runtime
- `code/pipeline_stages_unified.py`, `code/generate_full_workflow_diagram_unified.py`
  → rebuild `docs/diagram/full_workflow_unified_*.svg`
- `code/build_figures.py` → rebuilds the two composite figures above. The crop is
  picked from the data (the densest crack window), not by eye, so the figures can be
  regenerated after a model change without re-deciding what to show.
  `docs/img/README.md` records the exact command behind each one
- `crack_measurements.py`, `extended_features.py` → the extended-feature study
- `ingest_labels.py`, `ingest_marginal_verdicts.py` → the CSV-era label ingest paths
- `interior_active_learning/code/experiments/` (25 scripts) → produced every figure
  and table in the benchmark doc

**`archive/` — nothing imports it, and nothing should.** Superseded code, models
kept as counterexamples, and one-off analyses that are the evidence behind the
numbers above. `archive/README.md` says what each item proved. Safe to delete
wholesale if you don't care about reproducing the reasoning.

## Testing

```bash
PORT=8799 ./run &
BASE=http://127.0.0.1:8799 python3 interior_active_learning/code/test_app.py
```

84 checks covering upload, detection, exports, correction precedence, region
isolation, threshold plumbing, the retrain gate, autosave, undo, first-render
routing, and the performance fixes. `make test` runs the same thing.

## Working on this repo

This repo is the whole project: edit here, commit, push. There is no build or
packaging step — earlier there was a separate research folder that a
`make_package.sh` assembled this repo from, and the two have been merged, so that
script is gone. `docs/APP_COMPARISON.md` and a comment in `hybrid_detect.py` still
mention it in the past tense, describing bugs it once caused.

Two consequences worth knowing:

- **The overlays and per-image results are not here.** They are derived: the app
  rebuilds an image's overlay the first time you open it, and **Download → all**
  regenerates masks, overlays and per-region CSVs. An earlier archive of them
  lives in the private `CBS_Crack_Detection_All` repo, which is archived
  (read-only) on GitHub.
- **`interior_active_learning/paint/candidate_counts.json` is a cached count**,
  not ground truth. 34 of the 62 entries were measured with SAM enabled and the
  rest with the pipeline alone, so the sidebar understates those 28. Each entry is
  rewritten the next time that image is processed, so it self-corrects as you work.

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

## License

The **software** is MIT — see [LICENSE](LICENSE).

The **data** — the 62 micrographs, the human correction masks, the label and
training CSVs, and the models derived from them — is
[CC BY 4.0](LICENSE-DATA): reuse it, including commercially, with attribution.
`LICENSE-DATA` also records what the dataset is and the two ways it is easy to
misread, the important one being that a correction mask value of 0 means *never
reviewed*, not *not a crack*.
