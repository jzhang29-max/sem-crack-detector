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
| 1 | Unlabelled pixels scored as background | macro: specificity **+0.488**, precision **−0.591**; micro: **+0.266** / **−0.629**; recall **exactly unchanged** under both (n=10) | `experiments/scoring_convention_bias.py` |
| 2 | Calibration accepted without a cross-check | length error up to **+136%**, area up to **+458%** | `experiments/failure_mode_magnitudes.py` |
| 3 | Model promotion gated on an in-sample baseline | **+0.029** AUC bar that every honest candidate must clear | same |
| 1b | **Sparsity sensitivity of result 1** | gap **+0.488 [+0.285, +0.689]** at 8.16% adjudicated, and **+0.490** at 0.163% — invariant across a 50× thinning; interval excludes zero at every level | `experiments/sparsity_sensitivity.py` |
| 4 | Train/serve preprocessing skew | **13.8%** of labelled regions unreachable; 6 of 38 images contributed nothing | same |
| 5 | Decision threshold inherited as a library default | crack count **1.34–1.38×** across 0.3–0.7; area fraction **1.02–1.23×**; the count is **1.4–3.5× more sensitive** than the area within a specimen; condition ordering **stable** | `experiments/threshold_sensitivity.py` |

### Result 1 generalises beyond this codebase

`docs/UNLABELLED_PIXEL_AUDIT.md` audits six external targets from primary source. Training is
the stage the field mostly gets right; export fails in zero-of-six default paths; scoring is
*structural* -- `sklearn.metrics` has no `ignore_index`, `torchmetrics` has one in
`classification/` and none in `segmentation/`, and `segmentation_models_pytorch.get_stats`
raises `ValueError` for it in binary mode.

The sharpest form: CVAT scores correctly and exports destructively; ilastik trains correctly
and hides the faithful export. Metric hygiene inside a tool protects nobody, because the
export imposes the convention on every downstream consumer -- an information-theoretic
argument, not a quality judgement.

**Obligation this creates.** The void label is fourteen-year-old prior art (PASCAL VOC 2011,
Cityscapes, ADE20K, COCO-panoptic) and must be cited before a referee does it. The closest
match to these exact semantics is **nnU-Net v2's ignore label**, not VOC's -- VOC void is
boundary ambiguity, "a human looked and declined", where this is "a human never looked".
What is new is the map of where the convention stopped, not the mechanism.

### Result 1 is the headline, and it has a control

Pixel census across the 38 masks, summing to 100%: crack **6.727%**, not-crack **0.034%**,
UNREVIEWED **92.901%**, erased **0.338%**. So 6.761% is adjudicated, and treating the
unreviewed remainder as negative inflates the negative class ~2721x in aggregate and up to
2.5x10^7 on one frame. Across the 10 both-class images the adjudicated fraction is 8.16%.

Three denominators appear in this paper -- 62 images, 38 masks, 10 both-class masks -- and
each must be stated with its own number rather than blended.

**The fragility, beside the finding:** adjudicated *negatives* are 0.034% of pixels, about
three parts in ten thousand (145,441 px pooled, against 169,965,542 under the dense
convention -- a factor of 1,169). Specificity under exclusion is estimated from that pool and
is therefore high-variance. The dense convention's real attraction is that it always returns
a number, which is why it persists; saying so makes this a diagnosis rather than an
accusation.

The convention moves specificity and precision **in opposite directions**: a paper reporting
specificity this way flatters itself by nearly half, one reporting precision punishes itself
by six tenths, and neither states which convention it used. Two frames move from specificity
0.000 to 0.961 and 0.812.

**The control:** recall is identical to the last floating-point bit under both conventions,
because `tp/(tp+fn)` contains no negative term. This is asserted in the script, not merely
printed — if recall ever moves, the experiment raises instead of reporting a finding. It is
the evidence that the effect runs through the negative class and is not an artifact of this
pipeline.

### Result 1b answers the two objections the point estimate could not

A single pair of numbers invites two immediate objections, and both are answerable from data
already on disk.

*"0.4599 from 10 images and an adjudicated-negative class of three parts in ten thousand — how
certain is that?"* Bootstrap over **images**, the unit of independence: the specificity gap is
**+0.488 [+0.285, +0.689]**. The interval excludes zero, so the effect is not a noise artefact
at this n.

*"Your corpus is 8% reviewed. What happens at 1%? Is this a knife edge?"* Thinning the
adjudicated region by up to 50× moves the gap from **+0.488 to +0.490**. It is a broad effect,
which means it transfers to corpora far sparser than this one rather than being an artefact of
how much this particular annotator happened to review.

The design has a built-in asymmetry worth stating in the paper: the dense convention is
**invariant to thinning by construction**, because it never consults the adjudication. It is
therefore a fixed reference line, and every bit of movement comes from the exclusion side. The
dense number is not merely different — it is *indifferent to how much work the human did*.

**A prediction that failed, and the better conclusion that replaced it.** We expected the
exclusion estimate's confidence interval to widen as review thinned — the honest price of
declining to guess. It did not: 0.425 → 0.413, essentially flat. Thinning removes pixels, and
even 0.163% of a 25-megapixel frame is tens of thousands of them, so each image's estimate
stays stable. The interval is dominated by **between-image variance at n=10**, not by
within-image sampling. The consequence is a labelling instruction that reverses the intuitive
one: *to tighten the interval, mark not-crack on more IMAGES; marking existing images more
thoroughly will not do it.*

### Result 5 is a partial negative, and that is how it is reported

Both passes accept a region when its probability clears a threshold. The shipped bundle
carries **no threshold key**, so `bundle.get("threshold", 0.5)` returns the fallback and the
operating point under every number in this repository is **0.5 by omission** — not a
calibration decision, and absent from every figure it produces.

The falsifiable prediction was that the ordering of the three superalloy conditions
(AS / Cast / HIP, one specimen each) would flip somewhere in 0.3–0.7. **It did not**, on any
of four quantities. So the honest claim is *state your threshold and report its sensitivity*,
not *published comparisons are wrong*. Overstating this would be the same error the paper is
about, committed in the paper about it.

Three things keep it from being a null result.

**Which quantity is fragile is not the one you would guess.** Within a single specimen — same
frames, same threshold move — crack count moves 33.7% while area fraction moves 23.4% on
`MAR_Amb_AS`, and 37.9% against 10.7% on `MAR_Amb_HIP`: the count is **1.4–3.5× more
sensitive**. Raising the threshold deletes marginal regions, which changes *how many objects
there are* more than *how much dark area there is*. A paper reporting crack **area fraction**
is the more robust of the two; one reporting crack **density** — a count, and the more
commonly published figure — is not. The two are routinely treated as interchangeable measures
of "how cracked" a specimen is, and under an undocumented threshold they are not.

> **A number in an earlier draft of this section was an artefact, and the mechanism is worth
> keeping.** That draft said **15×**, taking the maximum of the ratio across specimens. But
> the ratio's denominator is *how much the area moved*, so maximising it selects whichever
> specimen's area moved least — and an area that barely moves can mean the area is robust or
> that the area is mostly not crack. `MAR_Amb_Cast` reports a crack-area fraction of **24.5%
> against 2.4% and 6.6%** for the other two, because off-specimen background floods the lower
> frame on those captures, a failure `unified_pipeline.py` already documents. That large,
> threshold-insensitive non-crack region dilutes the area movement to 2.1% and inflates the
> ratio to **16.3×**. The screen is now a relative one — a specimen whose area fraction exceeds 3×
> the across-specimen median is excluded and named — and the headline is the range over the
> specimens that survive it. Selecting the extreme of a ratio is a way to report a
> segmentation failure as a sensitivity result.

**The movement is not evenly spread.** Among the specimens that pass the screen above, the
largest single step is **10.0%** of `MAR_Amb_HIP`'s crack count, in the move from 0.6 to 0.7
alone. An operating point can sit beside a step change that a range-averaged sensitivity
figure would hide, which is the argument for publishing the curve rather than one ratio. (An
earlier draft cited 19.8% of a total crack length; that was measured on the excluded specimen
and inherited the same contamination.)

> **A defect in the sweep's own instrument was found and its effect measured.**
> `--threshold` and `THRESHOLD_OVERRIDE` did not reach Pass 2's `interior_fill` branch,
> which kept its own calibrated floor of 0.65 regardless — so one of three Pass-2 candidate
> types could not respond to the threshold, and the first run of this experiment measured a
> partly-frozen detector. That branch is not a formality: on the smallest corpus frame its
> floor rejects 19 of 22 candidates, so it is the most selective gate in Pass 2.
>
> The override now provably reaches every branch (the sweep refuses to start unless a
> pre-flight check confirms it, and that check was verified to fail against the old logic),
> and the experiment was re-run. **Three frames are complete under both versions, giving 75
> comparisons across five thresholds and five quantities. Eight differ, all on one frame:
> crack count moved by +1 at t=0.4 and +2 at t=0.6, a largest change of 2 in 1,665
> (0.120%).**
>
> Every span reported above is unchanged to five decimal places — 1.38404× before and after
> on the affected frame, and identically for total length and area fraction. The mechanism
> is worth stating because it is a general caution about this kind of summary: a max/min
> span is determined by the ENDPOINTS of the sweep, and both changes landed at interior
> levels. So the curve genuinely moved and the statistic reporting it did not. A different
> summary — mean absolute change across levels — would have registered the difference. The
> conclusion here is that the defect did not affect the published figures, not that the
> defect had no effect.

**The sample is complete, and that resolved an open question.** Nine frames, three per
specimen, every one measured at all five thresholds — density **110–2,678 cracks (median
740)**.

An earlier run of this experiment lost its three densest frames to a timeout, and since the
slowest frames are the densest, that exclusion correlated with the variable most likely to
drive sensitivity: a crowded frame has more marginal regions to gain or lose. The spans were
therefore published as a lower bound in case sensitivity rose with density. It does not
appear to. With the densest frame (2,678 cracks) now included, the crack-count spans are
**1.34–1.38×** against **1.28–1.41×** measured on the sparser sample — comparable, and if
anything narrower. That is a measured answer to the caveat rather than its removal.

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
5. The **specimen**, not the crack, is the statistical unit. A group of 2,915 cracks from 23
   frames on 3 specimens reports `n_specimens` beside the pooled count, means averaged per
   specimen, and below three independent units refuses dispersion outright -- per-crack
   spread there measures variation *within* one specimen
6. Longest-crack-per-frame is **refused** where edge-censoring is unknown, rather than
   reported as a maximum taken over lower bounds; the CSV cell is left empty rather than
   filled beside a flag a reader may miss
7. A µm column carries the scale's own uncertainty (a 200 px hand-marked bar fixes it to
   1.06%, so `61.40 µm` is 61.4 ± 0.7), and the routes that cannot know their precision
   record **absent** rather than zero

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
  both readings can still be wrong together against a certified reference specimen. The
  uncertainty now reported is **instrument-only and small** (about 1% on a well-marked bar);
  segmentation error -- whether the skeleton was measured from the real crack edge -- is
  larger and is quantified nowhere in this work. Reporting the small interval while staying
  silent about the large one would misrepresent the total more than reporting no interval,
  which is why the sidecar states the scope in the same sentence as the number.
- **Vendor metadata was destroyed before this project began.** All 62 TIFFs carry
  `Software: tifffile.py` and an `ImageDescription` of exactly `{"shape": [h, w]}`: a re-save
  with a tool that does not preserve private tags threw away the instrument's own record of
  the field width, silently. So automatic calibration is impossible on this corpus and the
  metadata reader is tested against synthetic FEI/ZEISS files instead. It is the paper's own
  thesis happening to the paper's own data, and worth one sentence in the discussion.
- **Detector is weak.** Recall 0.597 at the deployed operating point. Stated openly; the
  composability path is the answer, not a defence.
- **The external audit is now run, with two named exceptions.** `experiments/
  three_state_conformance.py` pushes one three-state fixture (512 crack / 128 adjudicated-
  negative / 3,456 unreviewed px) through ten probes and records what each library actually
  does, with the installed version of every one of them: scikit-learn 1.9.0, scikit-image
  0.26.0, torch 2.13.0, torchmetrics 1.9.0,
  segmentation_models_pytorch 0.5.0, MONAI 1.6.0, datumaro 1.13.8 (the CVAT export path),
  the Label Studio brush converter, elf/micro-sam matching, and mask-file round-trip. The
  reference pair the fixture is built to separate is specificity **0.875 adjudicated against
  0.949 dense**, so any tool silently choosing the dense convention shows up as that
  difference rather than as an opinion.
  Two things could not be executed and are named rather than glossed: **ilastik** has no PyPI
  distribution (conda/binary only), so the source-level finding stands unexecuted; and the
  **micro-sam annotator GUI** needs an interactive napari session, though its scorer is
  covered by the elf probe.
- **Preferential review.** Excluding unreviewed pixels is unbiased only if the reviewed region
  is representative, and annotators plausibly review the ambiguous, feature-dense parts. So
  0.4599 is itself computed on a non-random sample and the true full-image specificity is
  unknowable from this corpus. The honest deliverable is an **interval**: the two conventions
  bracket the estimate, the bracket is enormous (specificity 0.46-0.95), and reporting either
  endpoint without naming the convention is uninterpretable.

## What is needed from a person, not from more code

1. **Not-crack marks on ~20 more images.** Takes n from 10 to 30. Roughly 30 minutes per
   image with bucket-fill. This is the single highest-value hour available — and
   `sparsity_sensitivity.py` now shows *why* it is images rather than thoroughness: the
   interval is dominated by between-image variance, so breadth tightens it and depth does not.
   That was measured, not assumed, and it reverses the intuitive advice.
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
