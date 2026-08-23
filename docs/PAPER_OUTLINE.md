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

> **Which detector every result below was measured with: the bare two-pass detector**
> (`--sam2 off`), which is what ships. Every result was ALSO run under SAM 2 refinement and
> both columns are given, because the two do not always agree.
>
> SAM 2 refinement was briefly the default and is now opt-in: it wins on adjudicated pixels
> but fragments the mask on the whole frame (connected components +83%, skeleton length
> +97%, and 54,217 px claimed outside the detector's own candidates). So the **left** column
> is the shipped configuration.
>
> | | **bare detector (`--sam2 off`) — SHIPPED** | `--sam2 refine` (opt-in) |
> |---|---|---|
> | result 1, macro specificity gap | **+0.4879** | +0.3717 |
> | result 1, micro specificity gap | **+0.2662** | +0.2001 |
> | result 1, precision gap | **−0.5905** | −0.5942 |
> | result 1, per-frame direction | **10/10 positive** | 9/10 positive, 1 negative |
> | result 1c, erased-as-negative gap | **+0.5988** | +0.4964 |
> | result 1b, i.i.d. thinning, 8.16% → 0.163% | **+0.488 → +0.470** | +0.372 → +0.364 |
> | result 1b, whole-region, worst level | **+0.156 [−0.014, +0.391]** | +0.136 [−0.024, +0.384] |
> | result 1b, levels whose interval includes zero | **1 of 6** | 2 of 6 |
> | result 5, crack-count span across 0.3–0.7 | **1.34–1.38×** | 1.38× |
> | result 5, count-vs-area sensitivity | **1.4–3.5×** | 1.3–2.0× |
> | result 5, condition ordering | **stable** | stable |
>
> **1b and 5 were re-measured against predictions written down first, and both held.** 1b was
> predicted to get *noisier*, not cleaner, because refinement shrinks the gap so the sparsest
> levels sit closer to zero: two of six whole-region levels now include zero against one
> before. 5 was predicted to keep its ordering stable and keep crack count the fragile
> quantity, because SAM 2 sits downstream of the threshold and redraws boundaries for
> whichever candidates the threshold accepted rather than changing which quantity responds.
> Stating the expectation first is the only thing that makes a confirmation worth anything.
>
> **Results 2, 3 and 4 are detector-independent, and that was checked rather than argued.**
> Run under `--sam2 off` and `--sam2 refine`, all **25 scalar fields** across the three are
> identical. The independence is structural at two levels: none of them reads the predicted
> mask (result 2 is arithmetic on scale-bar readings, result 3 refits the classifier on stored
> training rows, result 4 counts rows and enumerates mask files), and their input
> `labeled_regions.csv` is built by `build_training_data.py` calling `extract_candidates`
> **directly** — upstream of where refinement happens — so even the training data is
> untouched by the detector default.
>
> **The skew that could have crept in was checked too.** Overlays are now refined while
> training labels are still voted onto *unrefined* candidates, so a human painting on a
> refined boundary might in principle have missed the candidate that receives the verdict —
> which is precisely the class of bug result 4 is about. On
> `260708_316_H_b2_front_CBS_002`, **100% of the human's 141,662 crack pixels lie inside an
> accepted region under both configurations and 0 px are covered by only one**, because
> `sam2_refine.refine_labeled` restores painted verdicts after refinement. No new skew.

## Results, all measured on the shipped corpus

Every magnitude below is reproducible from a committed script. Every failure mode was
*observed* in this project — each one shipped, was caught, and was fixed — rather than
hypothesised. That provenance is uncomfortable and it is also the point: these are the
errors a careful person makes anyway, which is why documenting the magnitude matters more
than documenting the fix.

| # | failure mode | measured magnitude | script |
|---|---|---|---|
| 1 | Unlabelled pixels scored as background | specificity gap **+0.488** macro / **+0.266** micro on the shipped bare detector (**+0.678** out-of-sample; **+0.372** / **+0.200** under opt-in SAM 2 refinement); precision **−0.594**; recall **exactly unchanged** under every configuration (n=10) | `scoring_convention_bias.py`, `oos_convention_gap.py` |
| 2 | Calibration accepted without a cross-check | length error up to **+136%**, area up to **+458%** | `experiments/failure_mode_magnitudes.py` |
| 3 | Model promotion gated on an in-sample baseline | **+0.029** AUC bar that every honest candidate must clear | same |
| 1b | **Sparsity sensitivity of result 1** | gap **+0.488 [+0.279, +0.689]** at 8.16% adjudicated. Invariant under i.i.d. pixel thinning (**+0.470** at 0.163%) but **NOT** under whole-region thinning, where it swings to **+0.156 [−0.014, +0.391]** — the invariance was a property of the sampling model | `experiments/sparsity_sensitivity.py` |
| 4 | Train/serve preprocessing skew | **13.8%** of labelled regions unreachable and 6 more images reached, both measured against a frozen 2026 snapshot; **0** marked images contribute no rows today, by enumeration | same |
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
three parts in ten thousand. The negative CLASS is 208,763 px against 176,519,519 under the
dense convention, a factor of **846**; an earlier draft published 1,169, which is the ratio
of the two *true-negative* counts (145,441 against 169,965,542) presented as the ratio of the
classes. Specificity under exclusion is estimated from that pool and
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

### Result 1c: the headline gap depends on a convention INSIDE the adjudicated convention

The adjudicated negative pool was defined as `m == 2` — pixels a human painted "not crack".
The correction mask has a fourth state, **3 = erased**, and on the ten scored frames erased
pixels number **435,058 against 208,763** marked not-crack: 2.08× more, and 24.9× on one
frame. The detector does not distinguish them. `unified_pipeline.py` protects
`np.isin(_cm, (2, 3))` from being re-proposed as crack under a single comment — "pixels the
human took off the table" — so the pipeline reads an erasure as a human "no" while the
experiment scoring the pipeline did not.

`experiments/erased_state_sensitivity.py` measures both readings on all ten frames:

| negative pool | macro specificity | macro precision | pooled negatives |
|---|---|---|---|
| erased excluded (as shipped) | **0.4599** | 0.9696 | 208,763 |
| erased counted as negative | **0.3490** | 0.8861 | 643,821 |
| unlabelled = background | 0.9478 | 0.3791 | 176,519,519 |

So the reported gap of **+0.4879 becomes +0.5988** — 23% larger — under the other reading of
the same marks. **The finding survives either reading; only its size moves.** That is the
useful form: an effect whose direction is robust to a defensible re-reading of the annotation
is a stronger claim than a point estimate that happened to pick one.

**We predicted this the wrong way round.** The expectation, written into the script before it
ran, was that erasures would add *easy* negatives and inflate specificity — the dense
convention's error in miniature. Specificity *fell* by 0.111. An erased region is one the
detector **proposed**; that is why a human was looking at it with the erase tool. Those pixels
are therefore disproportionately predicted-positive, so they enter the negative pool as false
positives rather than as easy true negatives. Direction is frame-dependent for the same
reason: three frames have no erasures and do not move, two rise (0.0000 → 0.6016 on one), five
fall (0.9312 → 0.0782 on one).

**Neither number is "the" answer, and the corpus cannot settle it.** An erasure is either "this
is not a crack" or "do not consider this" — off-specimen background, a charging artefact,
something outside the area of interest — and nothing on disk distinguishes them. Closing it
needs the annotator, or an interface that records *why* a region was erased. That is a
one-line addition to the paint tool and it is the cheapest of the outstanding annotation asks.

### Result 1 was measured in-sample, and out-of-sample it is larger

The experiment neutralises human input at the **pixel** level: `load_correction_mask` and
`load_hard_overrides` are patched so the prediction cannot read the verdicts it is scored
against. Necessary, and not sufficient. The Pass-1 classifier doing the predicting was
**fitted on region labels derived from those same masks**, and all ten scored frames
contributed training rows — 10 of 10, from 33 rows on one frame to 1,285 on another. The
headline was measured on training data.

`experiments/oos_convention_gap.py` refits Pass 1 per frame with that frame's **whole
specimen** held out — sibling frames from one session are near-duplicate leakage — using the
trainer's own estimator factory and per-image weighting, so the refit is the shipped procedure
rather than a local invention.

| arm | macro adjudicated specificity | macro specificity gap |
|---|---|---|
| in-sample (as published) | 0.4599 | +0.4879 |
| volume-matched control | 0.3933 | +0.5527 |
| **out-of-sample** | **0.2664** | **+0.6781** |

**The direction was predicted in writing before the run.** A model that has seen a frame's
labels makes fewer false positives there; the adjudicated negative pool is small and
fp-sensitive so its specificity rises; the dense specificity is dominated by the vast
unreviewed area and barely moves. Gap = dense − adjudicated, so circularity should *shrink*
the gap and +0.488 should be an underestimate. It held: **+0.488 → +0.678.**

**A confound in that design, and its size.** Holding out a whole specimen removes both the
leakage and 5.2–26.8% of the training rows, so a gap that grows could be the absence of
leakage or merely a worse model trained on less. The volume-matched control drops the *same
number* of rows at random from *other* specimens, leaving the scored frame's specimen in the
fit. Of the +0.1902 total move, **+0.1255 (66%) is the leakage and +0.0648 is the reduced
training volume** — real, but not the explanation.

**Per frame it is not universal**, and that belongs in the record: the gap grows on 6 frames,
shrinks on 4, and two of those four cannot move at all because their adjudicated specificity
is already 0.0000 and cannot fall. On one frame the collapse is dramatic — adjudicated
specificity 0.8866 → 0.1385, gap +0.0192 → +0.7578.

### Result 1b: one objection answered, the other now answered against us

Two objections faced the point estimate, and only the first survives as first reported.

*"0.4599 from 10 images and an adjudicated-negative class of three parts in ten thousand — how
certain is that?"* Bootstrap over **images**: the specificity gap is **+0.488 [+0.279,
+0.689]**, and all ten images show it in the same direction.

**What the interval does not establish.** The earlier write-up said "the interval excludes
zero, so the effect is not a noise artefact". That cannot fail. All ten per-image gaps are
strictly positive, and a percentile bootstrap of a strictly-positive sample cannot return a
non-positive bound, so `lo > 0` is arithmetic rather than evidence. The interval characterises
the *size* of the effect; the evidence that it is real is the 10-for-10 direction agreement,
which is a different and better argument.

*"Your corpus is 8% reviewed. What happens at 1%? Is this a knife edge?"* **The original answer
was an artefact of how thinning was modelled, and the honest answer is less comfortable.**

`_thin()` drew an i.i.d. random sample of adjudicated *pixels*. That is an unbiased,
very-low-variance estimator of the statistic on the full mask, so a sweep built on it can
hardly return anything but invariance — and duly did, +0.488 → +0.470 across a 50× thinning.
The finding was a property of the sampling model.

A reviewer who stops early leaves no scattered pixel sample. They mark a region, then another,
and stop; what remains unreviewed is contiguous and whole-region shaped. Thinning by whole
marked **regions** instead:

| kept | mean adjudicated | negative pool px | specificity gap |
|---|---|---|---|
| 1.00 | 8.160% | 256–134,039 | +0.488 [+0.271, +0.685] |
| 0.50 | 2.311% | 130–67,100 | +0.524 [+0.299, +0.749] |
| 0.25 | 1.554% | 130–33,745 | +0.362 [+0.102, +0.658] |
| 0.10 | 0.917% | 126–13,812 | +0.522 [+0.204, +0.813] |
| 0.05 | 0.447% | 126–7,388 | **+0.156 [−0.014, +0.391]** |
| 0.02 | 1.089% | 126–3,855 | +0.405 [+0.122, +0.707] |

The gap swings between +0.156 and +0.524, the intervals widen, and at one level the interval
**includes zero**. (Kept-fraction is non-monotonic because whole regions cannot be dropped to
hit an arbitrary target; at the sparsest level one large region survives.)

So the defensible claim is narrower than the one published: **the effect is present at every
level of review effort tested, but its magnitude is not stable under realistic sparsity, and at
5% region-level review it is not separable from zero.** A paper working on a corpus reviewed as
thinly as most are cannot assume the effect transfers at the size measured here.

**The published explanation for the flat interval was also measuring the wrong pool.** It read:
"even 0.163% of a 25-megapixel frame is tens of thousands of pixels, so each image's estimate
stays stable." That is the *adjudicated* count, which is crack-dominated. Specificity rests only
on the negative part, and at the sparsest level that is **5 to 2,681 pixels per frame** — on one
frame, five.

The claim that survives intact is the design asymmetry: the dense specificity is computed once
from the unthinned mask and never re-derived, so it is a fixed reference line and all movement
comes from the exclusion side. Note the precise reason, since the earlier wording had it wrong:
it is invariant because it is *exempted from the thinning*, not because it "never consults the
adjudication" — `~crack` is the adjudication.


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

### Result 4's two figures are not the same kind of number

`13.8% of labelled regions unreachable` and `6 of 38 images contributed nothing` are both
comparisons against a **frozen snapshot** — 4,128 rows across 32 images, measured once while
`build_training_data.py` still called `exclude_border_background()`. The live side of each
comparison is read from `labeled_regions.csv` at run time, so any later change to that file
for any other reason (more marking, a re-ingest, a different vote rule) is silently credited
to the train/serve skew and the percentage drifts without the code changing. They describe a
historical change, they are correct as such, and neither names an image.

What can be established **now**, by enumeration rather than subtraction: of the images whose
correction mask carries at least one marked pixel, **0** produce no training rows. That is the
claim about the current state, and it is the one worth quoting.

Two denominator notes that came out of building the enumeration. There are **39 mask files but
38 masks** — the app writes a full-size all-`UNREVIEWED` file when an image is opened, so
`MAR_Amb_HIP_CBS_0007` has a mask file with zero marks in it. The first version of this
enumeration counted that file as labelled and duly reported one image "contributing no rows",
manufacturing the failure it was looking for; the corpus figure of 38 used elsewhere in this
outline is the correct one.

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
