# A publishable claim, and what it still needs

Written after a survey of the field concluded that this tool is **not** publishable as a
tool: the closest substitute is free and assemblable from four Fiji plugins in an afternoon,
and "our software is careful" is not a finding. This is the paper that *is* here.

## The claim

> Quantitative micrograph metrology has silent failure modes in which software substitutes a
> plausible default for an honest "cannot tell". They are measurable, they move published
> quantities by amounts larger than the effects being studied, and they are preventable by
> instrument design that refuses rather than guesses.

This is a claim about **everyone's numbers**, not about this repo's software. That is what
makes it a contribution rather than a release note.

## Results, all measured on the shipped corpus

Every magnitude below is reproducible from a committed script. Every failure mode was
*observed* in this project — each one shipped, was caught, and was fixed — rather than
hypothesised. That provenance is uncomfortable and it is also the point: these are the
errors a careful person makes anyway, which is why documenting the magnitude matters more
than documenting the fix.

| # | failure mode | measured magnitude | script |
|---|---|---|---|
| 1 | Unlabelled pixels scored as background | specificity **+0.488**, precision **−0.591**, recall **exactly unchanged** (n=10) | `experiments/scoring_convention_bias.py` |
| 2 | Calibration accepted without a cross-check | length error up to **+136%**, area up to **+458%** | `experiments/failure_mode_magnitudes.py` |
| 3 | Model promotion gated on an in-sample baseline | **+0.029** AUC bar that every honest candidate must clear | same |
| 4 | Train/serve preprocessing skew | **13.8%** of labelled regions unreachable; 6 of 38 images contributed nothing | same |

### Result 1 is the headline, and it has a control

Only **8.16%** of pixels are adjudicated in the images that carry both classes (6.83% across
all 38 masks; 92.83% UNREVIEWED). Treating unlabelled as background inflates the negative
class ~2721× in aggregate and up to 2.5×10⁷× on one frame.

The convention moves specificity and precision **in opposite directions**: a paper reporting
specificity this way flatters itself by nearly half, one reporting precision punishes itself
by six tenths, and neither states which convention it used. Two frames move from specificity
0.000 to 0.961 and 0.812.

**The control:** recall is identical to the last floating-point bit under both conventions,
because `tp/(tp+fn)` contains no negative term. This is asserted in the script, not merely
printed — if recall ever moves, the experiment raises instead of reporting a finding. It is
the evidence that the effect runs through the negative class and is not an artifact of this
pipeline.

### Why result 2 is worse than it looks

A 16% scale error is a 30% area error, and the worst plausible reading gives +136% length /
+458% area. Crucially **none of the three readings is detectable from inside the program** —
each looked sound in isolation. Only a second, independent reading disagreeing exposes it.
That is the argument for refusal over best-effort: the fix is not a better bar detector, it is
requiring two routes to agree.

## What the instrument design contributes

Four refusal behaviours, each attached to a measured failure mode rather than to taste:

1. `UNREVIEWED` is a distinct mask state; metrics are computed on adjudicated pixels only
2. Calibration refuses when scale-bar and field-width readings disagree by >5%
3. Promotion refuses without a valid **out-of-sample** baseline, and refuses a baseline
   measured on a different held-out image
4. A group containing any uncalibrated image reports **pixels** and says so

Plus the composability result: the measurement layer is detector-agnostic — a mask imported
from ilastik or micro-sam flows through calibration, measurement, provenance and aggregation
unchanged, with the human's corrections still authoritative. That is the argument that this is
**infrastructure**, not one more segmenter, and it is why the weak built-in detector does not
undermine the paper.

## Limitations, stated before a referee states them

- **n = 10 for result 1.** Only 10 of 39 masks carry both classes, and specificity does not
  exist without both. This supports a methodological demonstration, not a general claim about
  the field. **Raising it requires marking not-crack regions on more images — no code
  substitutes.**
- **No blinded holdout.** Everything here measures *self-consistency* failures. Accuracy
  against physical truth is unmeasured, so the paper cannot claim the refusals improve
  correctness — only that their absence introduces quantified distortion. A referee will ask.
- **Single annotator.** No inter-rater agreement, so "the human is the instrument" is an
  assumption, not a characterised quantity.
- **No traceable calibration standard.** The 5% cross-check catches internal disagreement;
  both readings can still be wrong together against a certified reference specimen.
- **Detector is weak.** Recall 0.597 at the deployed operating point. Stated openly; the
  composability path is the answer, not a defence.
- **Failure modes are from one codebase.** Whether they generalise is argued, not
  demonstrated. Strengthening this means auditing another tool's handling of unlabelled
  pixels — feasible and not yet done.

## What is needed from a person, not from more code

1. **Not-crack marks on ~20 more images.** Takes n from 10 to 30. Roughly 30 minutes per
   image with bucket-fill. This is the single highest-value hour available.
2. **A second annotator on a handful of frames.** Yields inter-rater agreement (κ/Dice) and
   turns the central assumption into a measurement. Export the masks to CVAT self-hosted and
   compute agreement there rather than building a multi-annotator workflow.
3. **Two to four unseen specimens, fully adjudicated, blinded.** The one artifact that
   converts every refusal from "admirably careful" into "measured, with an error bar".

## Venue fit

- **Good:** *Ultramicroscopy*, *Journal of Microscopy*, *Materials Characterization* — a
  measurement-methodology contribution with a corpus.
- **Good if led by the dataset:** *Scientific Data* — the CC-BY SEM crack corpus with masks
  appears to be the only such release found in the survey.
- **Poor:** *npj Computational Materials*, or any venue expecting a better model. The
  detector is the weakest component here and the paper should not pretend otherwise.

## What must not be claimed

See `docs/COMPETITIVE_POSITION.md`. In particular: not a better segmenter; not a novel
measurement of tortuosity or roughness (self-defined, no standard); not any specificity or
crack-density figure as a property of the material rather than of the annotation effort.
