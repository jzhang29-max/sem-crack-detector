# The labels are region assertions, not crack outlines

**This is the most consequential finding in this analysis, and it changes what the
dataset is and what can be benchmarked with it.**

## 1. What I did

The competing papers all report Dice or IoU. This pipeline has never been scored that
way — it reports LOIO/grouped-CV AUC only — and it cannot be scored against the
exported masks, because corrections are already merged into those, making the
comparison circular.

So I scored the **raw model output** (`run_enhanced_pipeline`, which runs before
`apply_pixel_corrections`) against the human correction masks, with
`SEMCRACK_PAINT_DIR` pointed at an empty scratch directory so no correction could load
into the model and nothing could be written to the real `paint/`.

`MAR_Amb_HIP_CBS_0005`, 20.05 % of the frame reviewed:

| metric | value |
|---|---|
| IoU | 0.083 |
| Dice | 0.154 |
| clDice | 0.116 |
| precision | 1.000 |
| recall | **0.083** |
| raw model crack area | 2.67 % |
| human crack marks | 5,046,494 px (20.05 % of frame) |
| human not-crack marks | **0** |

Precision of 1.000 is meaningless here — with zero not-crack marks a false positive is
structurally impossible. The real number is **recall 8.3 %**.

## 2. But that is not a model failure — the two things are not comparable

Rendering the human marks next to the model output
(`analysis/crops/label_granularity.png`) shows the human marks are **broad brush /
flood fill**: scalloped brush-stroke edges and circular brush holes, covering 94.8 % of
a 1200 × 1200 px window. The model finds narrow features. Low IoU between a flood fill
and a thin curve is arithmetic, not error.

Measured across all 44 correction masks with usable marks — median stroke thickness,
as 2 × distance transform on the marked region's skeleton:

| stroke thickness | frames | share of all 70 M marked px |
|---|---|---|
| ≤25 px — plausibly tracing a crack | 9 | **2.1 %** |
| 25–80 px | 17 | 6.8 % |
| **>80 px — broad brush / flood fill** | **18** | **91.1 %** |

Distribution: min 4 px, p25 28 px, median **59 px**, p75 142 px, max **413 px**.

The frames carrying the most labels are the coarsest:

```
MAR_Amb_AS_CBS_0003            12,780,833 px   median stroke 413 px
MAR_Amb_AS_ETD_0003            12,388,443 px   median stroke 268 px
260708_316_H_b2_front_CBS_016   5,322,088 px   median stroke 230 px
MAR_Amb_HIP_CBS_0005            5,046,494 px   median stroke 223 px
MAR_Amb_AS_ETD_0002             4,957,080 px   median stroke 208 px
MAR_Amb_Cast_ETD_0004           4,739,397 px   median stroke 174 px
Cast_24hr_SE_Side_006           3,709,448 px   median stroke 289 px
```

## 3. What this invalidates

**The "70.4 million hand-corrected pixels" figure is real as a pixel count but 91 % of
it is region-level assertion, not crack delineation.** Consequences:

1. **The corpus is not a pixel-precise segmentation benchmark.** Dice/IoU against these
   labels is not comparable to Schmies (2023), Gerçek/Chekhonin (2025) or cigRockSEM,
   all of which used pixel-precise annotation. Publishing IoU against these labels
   would be misleading.
2. **On heavily painted frames, the exported "crack area fraction" is measuring the
   brush.** `MAR_Amb_AS_CBS_0003` reads 51.17 % crack area with "99.2 % human
   confirmed" — that is 12.78 M px painted with a 413-px-thick brush. It is not a
   measured crack area. This weakens the area fractions in `CRACK_ANALYSIS.md` further,
   on top of the magnification correction.
3. **It explains several earlier observations at once**: the 200–520 px "crack widths",
   ρ(area, width) = +0.86, box-counting dimension ≈ 1.9 on high-area frames, and the
   "human-confirmed" high-area frames. On those frames the mask *is* the brush stroke.
4. **`ConfirmedShareOfCrackPct` in my review audit overstates verification.** It
   measures overlap with painted regions, which is near-1 by construction where the
   human flood-filled.

## 4. The constructive read — this is a weak-supervision dataset

Coarse region marks are not worthless; they are **scribble / region-level weak
labels**, and learning precise segmentation from weak labels is an active, respectable
research area with its own literature and metrics. That reframing fits the data as it
actually is:

> *Learning pixel-precise crack segmentation in SEM from coarse region-level
> annotations, with a small precisely-labelled validation subset.*

> ⛔ **The framing claim in this section is WITHDRAWN (2026-09-18); the measurement is not.**
> That these labels are region-level weak labels is measured and stands -- it is the whole
> point of this file. What does not stand is the claim that reframing the work this way is a
> *genuine methods question* or an unoccupied opening. It is an occupied one: the formal
> setting is **Superset Label Learning** (Liu & Dietterich, ICML 2014, PMLR 32:1629-1637),
> the segmentation machinery is **box supervision with a tightness prior** (Kervadec et al.,
> MIDL 2020, arXiv:2004.06816) -- box supervision *is* superset supervision -- and the crack
> domain already ships a shrink module for exactly this. `analysis/README.md` records the
> weak-supervision framing among the recommendations that died. Use this section to
> understand what the labels are, not to claim a contribution.

Concretely:

1. **Build a small precise-label set.** The 9 fine-stroke frames (≤25 px) already
   carry ~1.5 M px. Re-label 3–5 fields at pixel precision with a thin brush, on
   magnification-matched fields, to serve as the evaluation set. **This is the single
   highest-priority task** — nothing can be benchmarked without it.
2. **Treat the 91 % coarse marks as weak supervision**, not ground truth: use them as
   region constraints or partial cross-entropy over marked pixels only, which is
   standard practice for scribble supervision.
3. **Report clDice and Boundary IoU, not just pixel Dice.** For thin curvilinear
   structures pixel Dice is dominated by width — exactly the failure mode already
   documented in this corpus. clDice is the established alternative.
4. **Recompute area fractions from raw model output**, and report them separately from
   the correction-merged export. Right now the two are conflated.
5. **Separate "reviewed" from "delineated"** in the app's own bookkeeping, so a coarse
   confirm is not counted as a precise label.

## 5. What still stands

The paired-detector results are untouched by this, because they compare **model output
to model output** on the same field, with no human labels in the loop: CBS finds 2.04×
more crack extent than ETD with narrower features (p = 0.016 / 0.039), on frames
verified matched at 10.00 kV / 1.6 nA / 6.0 mm WD / matched HFW. Likewise the
digitization-bias analysis, the censoring analysis and the two-population region
structure are all model-side and unaffected.

## 6. One environment note

The shipped `crack_classifier.joblib` fails to run under scikit-learn 1.7.2
(`AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`) and
requires the repo's own `.venv` (scikit-learn 1.9.0). For a public repo that is a
reproducibility trap — pin scikit-learn in the requirements and state the version in
the model provenance, or re-export the model in a version-stable format.
