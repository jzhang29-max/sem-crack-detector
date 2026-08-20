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
- **Batch, CLI, scripting, plugin API.** Everyone else has them.
- **Multi-annotator and inter-rater agreement.** CVAT self-hosted does this for free, and it
  is exactly the evidence this tool cannot produce.
- **Calibration convenience.** Fiji reads FEI/ZEISS metadata automatically and MIPAR
  auto-detects the bar. Here a human clicks two points. Claim the refusal, not the
  calibration.

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
- **Test count** — 155 checks says nothing about whether the detector finds cracks. Several
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
