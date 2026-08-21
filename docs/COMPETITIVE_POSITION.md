# Where this tool stands against the field

Written from a survey of ilastik 1.4.2, Fiji/ImageJ2 (TWS, Labkit, Ridge Detection,
MorphoLibJ, AnalyzeSkeleton, em-scalebartools), napari + micro-sam 1.8.9, CVAT / Label
Studio / Roboflow, the commercial suites (Dragonfly, Avizo, ZEN Intellesis, PRECiV+TruAI,
MIPAR, Aivia), and the 2022–2026 literature on crack segmentation in micrographs.

Its purpose is to stop this repo from overclaiming. If you are deciding whether to use this
tool, the honest summary is in the first two sections.

## Is there an existing tool like this?

No single tool occupies this niche, but **the closest substitute is free and one assembly
step away**: Fiji/ImageJ2 with `em-scalebartools` + `AnalyzeSkeleton` + `MorphoLibJ` +
Trainable Weka Segmentation or Labkit. That stack already gives trainable pixel
classification with brush annotation, scale-bar calibration, calibrated skeleton path
length, calibrated tortuosity and geodesic diameter, area and orientation, and one-click
FEI/TFS databar cropping.

What it does not give: a calibration that **refuses** when two independent readings
disagree, a unit-mixing guard, an `UNREVIEWED` pixel class distinct from background, a
correction layer that provably survives re-running the classifier, a gated model promotion,
or a provenance record bound to each CSV. What it costs you instead is assembly — three to
five plugins, a table join by label, and a macro only its author can run.

The niche is narrow because it is the intersection of three constituencies that are each
well served alone: 2D SEM micrographs of metals; metrology-grade unit discipline; and audit
of the model and the calibration. Bio-imaging tooling owns the first and ignores the second.
Commercial metallography owns the second and sells the third as a paid compliance module.
Annotation platforms own the human workflow and ignore both.

## What this tool wins at, and what it does not

**Genuinely distinctive** — no surveyed competitor does these:

- Calibration that refuses on a >5% scale-bar vs HFW disagreement instead of picking one
- Reporting **PIXELS** and saying so when any image in a group is uncalibrated
- `UNREVIEWED` pixels never scored as negatives in any reported metric
- Gated model promotion: refuses without a valid out-of-sample baseline rather than deploying
- A provenance sidecar bound to every CSV naming model, mtime, calibration and its source
- One row per crack carrying all seven morphometrics without joining three tables
- Mean/max crack **opening width** per crack (ilastik has none; Fiji needs an unmaintained
  plugin with a live anisotropy bug)
- A CC-BY-4.0 corpus of SEM crack masks — the only such release found

**Loses outright:**

- **Mask quality.** ilastik's Random Forest over a multi-scale filter bank, micro-sam's ViT,
  and the commercial CNNs are all stronger detectors than a darkness threshold plus a
  LogisticRegression over 8 features. Recall 0.597 means roughly 40% of crack pixels are
  missed.
- **3D, serial sections, tomography, mosaic stitching.** Absent here, and decisive whenever
  a crack is longer than one field of view — which happens silently in this corpus.
- **Scripting and a plugin API.** Everyone else has them. Batch and a CLI are now here
  (`code/semcrack.py`), but an importable, documented Python API and a plugin surface
  are not, and "import our module and read the source" is not an API.
- **Multi-annotator and inter-rater agreement.** CVAT self-hosted does this for free, and it
  is exactly the evidence this tool cannot produce.
- **Calibration convenience.** Fiji reads FEI/ZEISS metadata automatically and MIPAR
  auto-detects the bar. This now reads FEI/ZEISS tags too, so the *capability* gap is closed,
  but the convenience gap is not: on files whose tags survived, Fiji is one click and this is
  one flag; on files like this corpus, both are useless and a human still marks two points.
  Claim the refusal and the cross-check, not the calibration.

## What changed after this survey

The survey's highest-leverage recommendation was to stop competing on mask quality and start
composing. That is now implemented: `code/import_mask.py` and
`POST /api/external_mask/<image>` accept a crack mask from ilastik, micro-sam, Fiji or
anything else, and it replaces the built-in detector while leaving the human's corrections
authoritative. The three strongest competitors become upstream suppliers, and the comparison
moves off "my detector vs theirs" — ground this project loses — onto the measurement and
audit layer it actually owns.

The other recommendation acted on: `experiments/naive_baselines.py` now compares the
deployed pipeline against global Otsu, Otsu+cleanup, Frangi ridges and Frangi∩dark on
identical adjudicated pixels. The pipeline wins by +0.101 f1 over the best naive method, and
is the only one not degenerate at one end. That question had never been asked before.

### The survey's own list, closed out

Seven items came out of the survey. Six are now done, and saying which is which matters
more than saying six.

| item | state |
|---|---|
| FEI/ZEISS metadata as a third calibration source | **done**, with a caveat that matters more than the feature: `calibration.read_instrument_metadata` reads FEI/Thermo INI blocks and ZEISS `CZ_SEM` tags and cross-checks them against a hand reading, but it returns `None` on all 62 images here. Every file carries an `ImageDescription` of exactly `{"shape": [h, w]}` — a numpy re-save destroyed the instrument's own record of the field width before any of this code ran. So the feature is tested against synthetic vendor files, and the corpus fact is documented rather than hidden. |
| leave-one-**specimen**-out in the promotion gate | **done**. Sibling frames from one session are near-duplicate leakage, so the gate holds out a whole specimen. |
| boundary-touching flag | **done**. `TouchesBoundary` / `BoundaryPx` / `LengthIsCensored` per crack, and `aggregate.py` refuses longest-crack-per-frame where censoring is unknown rather than taking a maximum over lower bounds. |
| specimen as the default statistical unit | **done**. `n_specimens` beside `n_cracks`, means averaged per specimen, dispersion refused below three independent units. |
| headless batch CLI | **done**: `code/semcrack.py`, with the refusals in the README's table and a run manifest pinning model, threshold and directories. |
| threshold sensitivity | **done, and measured**: sweeping 0.3–0.7 through both passes on 9 frames moves crack count **1.28–1.41×** and area fraction only **1.02–1.17×**; the count is **2.1–3.8× more sensitive** than the area within a specimen (a draft said 15x, which took the max of a ratio whose denominator is how little the area moved, and so selected a specimen whose area is mostly off-specimen background). The condition ordering did **not** flip, which was the prediction, so the claim is "state your threshold" rather than "comparisons are wrong". The override was found not to reach Pass 2's `interior_fill` branch, so the first run measured a partly-frozen detector; the sweep was re-run with a pre-flight check proving the override reaches every branch, and on the three frames complete under both versions the effect is **2 cracks in 1,665 (0.120%) on one frame, with every reported span unchanged to five decimal places** — the changes fell at interior threshold levels while a max/min span is fixed by the endpoints. The shipped operating point turned out to be the `0.5` fallback of `bundle.get("threshold", 0.5)` — a library default reached by omission, under every number here. |
| pixel-size sensitivity | **not a table, and here is why.** Length scales with the pixel size and area with its square; a table of that is arithmetic, and `experiments/failure_mode_magnitudes.py` already measures the case that bites (a mis-read bar: up to +136% length, +458% area). What is genuinely missing is not sensitivity but **uncertainty**: a hand-marked bar span carries a few pixels of aiming error at each end, so every µm column is a point estimate with an interval nobody reports. That is the version worth building. |
| Zenodo DOI | **needs a person.** Nothing in code produces one. |

## Claims this repo should not make

Recorded so they do not creep back in:

- **"Retrain from the browser"** — ilastik, micro-sam, Label Studio, Roboflow, MIPAR and ZEN
  all retrain from a UI. The gate is the differentiator, not the retraining.
- **"Scale-bar calibration"** as a strength — table stakes, and less convenient here than in
  Fiji. The refusal is the strength.
- **Info-bar cropping** — shipped one-click by `em-scalebartools`. Only the vendor-agnostic
  detection is even mildly novel, and that is a footnote.
- **Brush / bucket / erase / undo / autosave latency / CSV export / skeleton length / area /
  orientation** — Fiji has done all of it for twenty years; napari gives paint/fill/erase/undo
  for free. Latency is UX colour, not a differentiator.
- **Test count** — the suite says nothing about whether the detector finds cracks. Several
  of those tests exist only because an earlier check *could not fail*.
- **Any specificity, precision, area-fraction or crack-density figure** — 27 of 38 labelled
  images carry no not-crack label, so specificity 0.476 rests on approximately one frame.
  That is not a measurement.
- **AUC as detection quality** — lead with the deployed operating point (f1 0.715, recall
  0.597, specificity 0.476 at threshold 0.5), not an AUC.
- **Tortuosity, boundary roughness and branch counts as material descriptors** — self-defined,
  no standard, no round-robin, no scale-dependence analysis. MorphoLibJ already ships
  calibrated tortuosity and geodesic diameter.
- **The six-classifier bake-off as validation of the pipeline** — it compares classifiers on
  this project's own features. See `experiments/naive_baselines.py` for the comparison that
  actually addresses the question.
- **"Human corrections always override the model"** as an accuracy claim — it is an *honesty*
  claim. The correct framing: the human is the instrument; the tool is the ruler and the
  audit log.

## The one thing that would change a reviewer's mind

**A frozen, blinded, fully-adjudicated holdout** — every pixel labelled, by at least two
annotators, on specimens and sessions never used in training — with the resulting bias and
confidence interval propagated into the µm columns that appear in a paper's table.

Everything distinctive here is process hygiene about numbers, and process hygiene has a
ceiling: the promotion gate can keep improving agreement with one operator's brush habits
while accuracy against physical reality stays flat, and nothing in the current design can
detect that. A blinded holdout is the only artifact that turns the four refusal behaviours
from "admirably careful" into "measured, with an error bar" — and no surveyed competitor has
one for SEM cracks in metals either.

It cannot be built in this repo. It needs a second annotator and a few unseen specimens.

## Known measurement caveats a reviewer will raise

- **Fragmentation bias.** A main crack merged from fragments was reported per fragment until
  recently; the connector geometry is now honoured, but the residual magnitude at recall
  0.597 has not been quantified. See `docs/MEASUREMENTS.md`.
- **Right-censoring.** Any crack touching the frame edge is censored. `longest-crack-per-frame`
  is not a valid cross-condition comparable without a boundary-touching flag, and the max
  statistic scales with frame count and field width.
- **Statistical unit.** Aggregation pools regions across frames, which invites a
  pseudo-replication objection. The specimen, not the region, is the right unit.
- **Leave-one-image-out leakage.** The promotion gate holds out one *image*; sibling frames
  from the same session remain in training, so the gate's AUC is optimistic. Leave-one-
  *specimen*-out would be sounder — the filenames already encode specimen and session.
- **Sensitivity.** No threshold or pixel-size sensitivity table exists. Every reported
  measurement should be re-run at 0.4/0.5/0.6 and at 2× downsample before publication.
