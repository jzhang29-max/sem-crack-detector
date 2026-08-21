# SEM app vs TXM app — what was adopted, and two defects it exposed

The sibling TXM project has its own correction app, and `docs/APP_COMPARISON.md`
there already listed what this one was missing. Rather than argue with that list,
each item was checked against this codebase and the real ones were fixed. Two of
them were not cosmetic — they were defects that made this app quietly worse than
its own published benchmark.

## Defects the comparison surfaced

**1. The package shipped only half the model.** `make_package.sh` copied
`models/*.joblib` and stopped. There are two model directories:

| directory | model | role |
|---|---|---|
| `models/` | `crack_classifier.joblib` | Pass 1 |
| `interior_active_learning/models/` | `unified_model.joblib` | **Pass 2** |

`unified_pipeline._load_unified_bundle()` returns `None` when the second is
absent, and Pass 2 is then skipped **with no error and nothing on screen**. A
fresh clone therefore ran Pass 1 only — losing 27% of all crack regions and
dropping measured f1 from 0.776 to 0.768 — while reporting success. Verified by
cloning: `_load_unified_bundle() -> None`.

Fixed in the packager, and `hybrid_detect.detect()` now prints and reports a
warning when the Pass-2 model is missing, because a silent 27% loss is worse than
a crash.

**2. A fresh clone could not retrain at all.** `build_training_data.py` rebuilt
the training CSV from local images only. At the time the package shipped 35
correction masks but no source images, so on a clone it found 0 usable images,
printed `NO ROWS PRODUCED`, exited 1 — and the Retrain button failed. (The package
now ships all 62 images too, which removes the trigger; the fix below still
matters, because a user who clones and works on their OWN images hits exactly the
same path.) It now
MERGES: rows for images present locally are recomputed, rows for absent images are
carried over. A user who clones, adds their own images and corrects them now ADDS
to the shipped 4,128 labels instead of destroying them. Verified on a clone with
zero images: 4,128 rows carried, training completes.

## Adopted from TXM

**`KMP_DUPLICATE_LIB_OK=TRUE` in the launcher.** Not a tuning knob: scikit-learn
and torch each vendor an OpenMP runtime and loading both into one process aborts
on macOS. This app loads both — scikit-learn for the classifier, torch for SAM —
so it was a latent crash. Also `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` so the SAM
pass can use all of unified memory, and a warning when no model file is present.

**In-app exports**, which this app had none of — masks and overlays could only be
produced by running scripts from a terminal. Now: per-image B&W mask (crack
black, matching TXM so results are interchangeable), burned-in overlay, a
per-region CSV (area, centroid, bbox, length, width, aspect, eccentricity,
orientation, solidity, perimeter), and an all-images `.zip` with a cross-image
`summary.csv`. For anyone measuring crack growth across a series, that region
table is the actual product.

Verified for internal consistency rather than just HTTP 200 — on one image the
mask's black pixel count, the overlay's red pixel count and the CSV's summed area
are all exactly 181,943.

Units are pixels only, deliberately. The scale bar lives in the SEM databar that
this pipeline crops before analysis, so no µm/px factor is recoverable; inventing
one would put a fabricated number in a results table.

**⌘Z undo.** The frontend had *no* keyboard handlers at all. Added ⌘Z/Ctrl-Z, plus
`F` to fit and `1/2/3` to pick crack / not-crack / erase. `preventDefault()`
matters: without it the browser's own undo also fires and reverts the image
dropdown and brush slider, which reads as the app losing its place.

## Deliberately not adopted

**A live threshold slider.** TXM exposes one and defaults post-processing off,
because there post-processing measurably deletes thin crack. The intent here was
that a threshold is calibrated per model and stored in its bundle, so exposing a
slider would invite moving off a calibrated operating point without the
measurement that justified it.

That reasoning is sound and it does not describe what ships. **The deployed
bundle, `models/crack_classifier.joblib`, carries no `threshold` key at all**, so
`bundle.get("threshold", 0.5)` returns the fallback and production runs at exactly
**0.5** — a library default reached by omission. (An earlier draft of this
paragraph cited 0.40 chosen on a held-out image; the bundle that does carry a
calibrated threshold, `crack_classifier_v3_weighted.joblib`, stores 0.5615 and is
not the one in use.) So the argument against a slider currently protects a number
nobody chose. `experiments/threshold_sensitivity.py` measures what that number is
worth: moving it across 0.3–0.7 changes crack count by 1.28–1.41× while leaving
the ordering of conditions intact. If a slider is wanted later, the honest version
ships that curve beside it.

**A model registry with history.** This app keeps one backup
(`crack_classifier_PREV.joblib`) plus the pre-deployment
`crack_classifier_PRE_V3_BACKUP.joblib`, and the retrain gate refuses to deploy a
model that scores worse on held-out data. That covers the failure this would
protect against; a full registry is a nice-to-have, not a gap.

## Where this app is ahead

- **Click-to-flip whole regions**, which TXM has since copied. One click corrects
  a region that would take hundreds of brush strokes.
- **A retrain gate that refuses regressions** rather than warning. It earned its
  keep during development: 18 extra training rows moved held-out AUC from 0.9252
  to 0.9153, and the gate declined to deploy.
- **An 84-check end-to-end test suite** (`interior_active_learning/code/test_app.py`)
  covering upload, detection, export, correction precedence, region isolation,
  threshold plumbing and the retrain guard.
