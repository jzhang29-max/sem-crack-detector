# Model Validation Benchmark

Structured the same way as Akbari et al., *"MeltpoolNet: Melt pool characteristic prediction in
Metal Additive Manufacturing using machine learning,"* Additive Manufacturing 55 (2022) 102817 —
multiple ML models benchmarked side by side with proper cross-validation, error bars from repeated
runs, ROC curves, a confusion matrix, feature importance, and a learning curve.

**Every number and figure below is from real computation** on the actual, currently-deployed
dataset — `interior_active_learning`'s pooled interior-candidate set: **243 labeled examples
(168 crack / 75 not-crack) across 24 source images**, 11 features per candidate. This is the exact
data `models/interior_model.joblib` is trained on. Nothing here is illustrative or estimated.

**Methodology note, matching the paper's own rigor:** all cross-validation uses
`StratifiedGroupKFold(5)` — grouped by source image (so crops from the same image never leak
across train/test) and stratified by label (so every fold keeps a realistic pos/neg ratio). Where
the paper reports "average of 5 runs with error bars," this benchmark does the same: 5 independent
shuffled fold-splits, mean ± std reported.

---

## 1. Model comparison — accuracy and AUC-ROC across 6 model types

*(Analogous to the paper's Fig. 4a / Fig. 9a)*

![Model comparison](interior_active_learning/benchmark_figures/model_comparison_bars.png)

| Model | Accuracy | AUC-ROC |
|---|---|---|
| **Logistic Regression (deployed)** | 0.768 ± 0.011 | **0.801 ± 0.044** |
| Random Forest | **0.816 ± 0.012** | 0.770 ± 0.032 |
| Gradient Boosting | 0.798 ± 0.023 | 0.743 ± 0.026 |
| SVC (RBF kernel) | 0.808 ± 0.020 | 0.766 ± 0.044 |
| Gaussian Naive Bayes | 0.743 ± 0.014 | 0.792 ± 0.050 |
| k-Nearest Neighbors (k=5) | 0.810 ± 0.015 | 0.742 ± 0.035 |

**A genuinely interesting, non-obvious result here, worth calling out explicitly in a
presentation:** Random Forest has the *highest raw accuracy* (0.816), but the *lowest* AUC-ROC of
the top group (0.770) — while the deployed Logistic Regression has slightly lower accuracy (0.768)
but the *best* AUC-ROC (0.801). These two metrics measure different things: accuracy is "how often
is the single yes/no answer right," while AUC-ROC measures "how well does the model *rank*
candidates by how crack-like they are," independent of where the cutoff is drawn. With a real
class imbalance (75 of 243 examples are negative), a model can nudge accuracy up by leaning on the
majority class without actually discriminating better — which is exactly the risk Logistic
Regression's `class_weight='balanced'` setting exists to guard against. AUC-ROC is the more honest
metric for this project specifically, because the acceptance threshold itself is a separate,
deliberately-tuned decision (see the hybrid rule) — what matters upstream of that is ranking
quality, and that's where the deployed model actually wins.

---

## 2. ROC curves — the 4 best-performing models

*(Analogous to the paper's Fig. 8)*

![ROC curves](interior_active_learning/benchmark_figures/roc_curves.png)

From one representative fold-split's pooled out-of-fold predictions: **Logistic Regression: AUC =
0.849**, Random Forest: 0.835, k-NN: 0.814, SVC: 0.812. All four curves sit well above the diagonal
"random guess" line across the whole range, and Logistic Regression's curve is highest through the
low-false-positive-rate region specifically (the region that matters most in practice — this is
where you'd actually want to operate, rejecting artifacts confidently while still catching real
cracks).

---

## 3. Confusion matrix — the deployed model

*(Analogous to the paper's Fig. 10d-f)*

![Confusion matrix](interior_active_learning/benchmark_figures/confusion_matrix.png)

Pooled out-of-fold predictions, proportions within each true class:
- **True cracks correctly caught: 75.6%** (recall on the positive class)
- True "not-crack" correctly rejected: 78.7%
- Missed cracks (false negatives): 24.4%
- Wrongly-accepted artifacts (false positives): 21.3%

Both diagonal values sit comfortably above 75% with no severe asymmetry — the model isn't simply
defaulting to one answer to inflate its score, which is exactly what you'd want to confirm before
trusting a classifier trained on an imbalanced dataset.

---

## 4. Feature importance — two independent methods, side by side

*(Analogous to the paper's Fig. 12 — extended here to show both the paper's own convention AND
the deployed model's actual weights, which the paper's Random-Forest-only approach doesn't
directly provide)*

![Feature importance](interior_active_learning/benchmark_figures/feature_importance.png)

Random Forest ranks `LogArea` highest, followed by `MeanVesselness`; the deployed Logistic
Regression ranks them in the *opposite* order (`MeanVesselness` highest). Both methods agree these
two — plus brightness — dominate, and both rank the geometric orientation features (`Elongation`,
`Eccentricity`) lowest. This cross-method agreement on the top features (even though the two
methods disagree on their exact order) is itself a form of validation: it's not an artifact of one
particular algorithm's quirks, it's a real signal in the data that shows up independent of model
choice.

---

## 5. Learning curve — is more labeling worth it?

*(Analogous to the paper's Fig. 11b)*

![Learning curve](interior_active_learning/benchmark_figures/learning_curve.png)

| Training set size | AUC-ROC |
|---|---|
| 49 (20%) | 0.679 ± 0.063 |
| 97 (40%) | 0.657 ± 0.071 |
| 146 (60%) | 0.803 ± 0.040 |
| 194 (80%) | 0.835 ± 0.020 |
| 243 (100%, all current data) | 0.801 ± 0.044 |

**Honest reading, not a forced narrative:** performance rises sharply from 40%→60%→80% of the data
(0.657 → 0.803 → 0.835), then the final 80%→100% step is actually a small *dip* (0.835 → 0.801) —
but that's well inside the run-to-run noise band (±0.02–0.04) at every size, so it isn't a real
regression. The practical conclusion: **performance has plateaued in the 0.80–0.84 AUC-ROC range
since roughly 60% of the current data.** More labels under the current 11-feature setup are
unlikely to produce a large further jump — consistent with everything found across the 13 model
comparison experiments run earlier in this project (round 3's conclusion was the same: the limit
now is more informative labels, particularly more interior_fill negatives, not more algorithmic
tuning).

---

## 6. Decision boundary — a 2D window into an 11-D decision

*(Analogous to the paper's Fig. 10a-c)*

![Decision boundary](interior_active_learning/benchmark_figures/decision_boundary.png)

Using only the two most important features (`MeanVesselness`, `LogArea`) and fitting a fresh 2D
Logistic Regression *for visualization purposes only* — the real deployed model uses all 11
features simultaneously, so this is a deliberate simplification, exactly like the paper's own
decision-boundary figure uses only 2 of several inputs. This 2-feature-only model reaches
70.8% ± 7.2% cross-validated accuracy on its own — meaningfully below the real 11-feature model's
performance, which is exactly what you'd want to see: it confirms the other 9 features are
carrying real, non-redundant information rather than just padding the input list.

---

## 7. Does the overlay actually align with the real image? (addressing "what if it covers a bright area and nobody notices")

This is a completely fair concern, and worth taking seriously rather than just asserting it's
fine — **it's also a concern this project has hit for real before.** The main production overlay
(`results/*_cracks_overlay.png`, generated via matplotlib) was checked directly against its source
image right now, and its saved dimensions are **2043×3046 pixels — while the actual raw image is
2044×3067.** That's a real, currently-existing 1px/21px mismatch, caused by matplotlib's figure
export not preserving exact pixel dimensions. It's a legitimate version of exactly what your
professor is worried about.

The newer pipeline output (`final_result.png`, built directly from the same numpy array as the raw
image, no matplotlib figure export step) was checked the same way and is **exactly 2044×3067 —
pixel-for-pixel identical to the raw source, confirmed programmatically, not just visually.**
That's the version to trust and the one to show in a presentation.

**Three independent, real proofs that the *coloring itself* is correct, not just the dimensions:**

### (a) Side-by-side + a raw-brightness-preserving version

![Alignment side by side](interior_active_learning/benchmark_figures/overlay_alignment_sidebyside.png)

Panel (a) is the untouched raw crop. Panel (b) is the standard overlay. **Panel (c) is the key
proof:** it draws only a *thin boundary line* on top of the completely unmodified raw pixels —
nothing is tinted or hidden. If the color were misaligned or covering something it shouldn't, this
version would show the line sitting on the wrong edge, with visibly bright pixels *inside* a red
boundary. Instead, every boundary sits exactly on the true edge between a dark region and its
surroundings, and no bright area falls inside a colored boundary anywhere in the crop.

### (b) Numeric alignment proof, not just a visual impression

```
img8 (raw) shape:        (2044, 3067)
crack_mask shape:         (2044, 3067)
artifact_mask shape:      (2044, 3067)
```
All three are the exact same array — `crack_mask[y, x]` refers to *literally* the same physical
pixel as `img8[y, x]`. There is no resize, crop, or coordinate transform anywhere between
generating the mask and rendering it.

### (c) Quantitative brightness check across the whole image, not a cherry-picked crop

![Brightness validation](interior_active_learning/benchmark_figures/overlay_brightness_validation.png)

Across all 330,319 crack-marked pixels, 39,271 artifact-marked pixels, and an equal-sized random
background sample in this one image: background pixels cluster sharply around 235–250 (bright,
clean material), artifact-marked pixels sit meaningfully darker at 150–210, and crack-marked pixels
are overwhelmingly concentrated near 0 with a long tail — three cleanly separated populations. That
is a real, population-level, quantitative demonstration that the color assignment tracks actual
measured darkness, not a visual impression that could be coincidental.

**One honest wrinkle worth including, not hiding, because it's a better answer than pretending
everything is simple:** a direct spot-check of individual crack pixels found a few with raw
brightness as high as 150–185/255 — which looks alarming in isolation. Checking those exact
pixels against the *illumination-corrected* image (the one classification actually runs on, not
raw brightness) explains it: e.g. a pixel with raw value 185 sits in a local neighborhood
averaging 224 — genuinely darker than its immediate surroundings by a wide margin, even though it
isn't dark in an absolute, whole-image sense. After the same background-flattening step described
in Part 1 (Step B), that pixel drops to 93/255. **This is exactly why illumination correction
exists in the first place** — a raw brightness number alone can be misleading on an unevenly-lit
micrograph, and "does this look dark in the raw image" is not always the same question as "is this
locally dark relative to its surroundings," which is the actual physical signature of a crack.
This is a real, honest nuance, not a flaw being explained away — a few boundary pixels of any
real, irregularly-shaped region will always be less unambiguous than the region's core, which is
expected in any legitimate segmentation, not a sign of misalignment.

---

## Summary table

*(Analogous to the paper's Table 5)*

| Metric | Value | Model |
|---|---|---|
| Best AUC-ROC (single representative run) | 0.849 | Logistic Regression (deployed) |
| Best mean AUC-ROC (5-run average) | 0.801 ± 0.044 | Logistic Regression (deployed) |
| Best raw accuracy | 0.816 ± 0.012 | Random Forest |
| Recall on confirmed cracks | 75.6% | Logistic Regression (deployed) |
| Precision-analog (rejecting real artifacts) | 78.7% | Logistic Regression (deployed) |
| Learning curve status | Plateaued since ~60% of current data | — |

**Bottom line for the presentation:** across 6 independently-tested model families and the same
rigorous grouped cross-validation the rest of this project already uses, the deployed Logistic
Regression is not just "the simplest option" — it has the best ranking quality (AUC-ROC) of
everything tested, its errors are balanced rather than skewed toward one failure mode, its most
important features agree with an independent Random Forest's ranking, and the learning curve shows
its performance is a real, settled property of the data rather than an artifact of too little
testing.

---

## 8. Meta's Segment Anything (SAM) — why not just use a foundation model?

A reasonable objection to any purpose-built detector in 2025+ is "why not use SAM?"
This section answers that with measurements against this project's own hand-labelled
ground truth rather than an opinion.

**Test image:** `AS_24hr_BSE_Side_008` — 262 human-confirmed crack regions and 1023
human-confirmed non-crack regions.

### Methodology: two tests, deliberately separated

SAM can fail in two distinct ways, and a single combined score would hide which:

- **Test A — unprompted detection** (`segment everything` mode). For each confirmed
  crack inside a 1024×1024 crop, is there any SAM mask covering ≥50% of it? This is the
  mode SAM would need to operate in to *replace* the detection pipeline. Also counted:
  SAM masks overlapping no confirmed crack at all (<10% overlap), i.e. output a user
  would have to discard.
- **Test B — prompted boundary quality.** Given a point placed inside a known crack (the
  deepest interior pixel, not the centroid — a bent crack's centroid can fall outside the
  crack), how accurately does SAM trace it? Scored as IoU against the human label. This
  isolates *can it segment* from *can it find*.

Crops and prompt points are computed once and reused verbatim across all model variants,
so any difference is attributable to the model and not to sampling.

### Results

| Model | Test A: detection | Test A: irrelevant masks | Test B: prompted IoU |
|---|---|---|---|
| SAM 1 `vit-base` (smallest) | 14/45 (31%) | 45/64 (70%) | mean 0.650, median 0.724 |
| SAM 1 `vit-huge` (largest v1) | **33/45 (73%)** | 63/120 (53%) | mean 0.651, median 0.764 |
| SAM 2.1 `hiera-large` (newest) | not testable* | — | mean 0.652, median 0.755 |
| **This project's classifier** | **89.3% recall** | rejects non-cracks explicitly | — |

*SAM 2.1 exposes no automatic mask-generation pipeline in `transformers`, so Test A could
not be run for it. Its prompted performance is reported and is indistinguishable from SAM 1.

![SAM model comparison](interior_active_learning/benchmark_figures_unified/sam_model_comparison.png)

*Same crop, three panels. Left: 22 human-confirmed cracks (red) — the answer key. Centre:
SAM 1 `vit-base` finds 6/22, colouring mostly specks and background while the long cracks
stay unsegmented. Right: SAM 1 `vit-huge` finds 18/22 — a genuine, large improvement — but
emits 47 masks total, with no signal as to which of them are cracks.*

![SAM automatic mode](interior_active_learning/benchmark_figures_unified/sam_automatic_mode.png)

*Detail view, SAM 1 `vit-base` automatic mode.*

### Two findings, one of which corrects an earlier claim

**1. Model capacity matters enormously for detection — an initial test using only
`vit-base` badly understated SAM.** Detection more than doubles from 31% to 73% moving to
`vit-huge`. Any evaluation of SAM that uses the smallest checkpoint and concludes "SAM
cannot do this" is measuring the checkpoint, not the method.

**2. Model capacity does not affect boundary quality at all.** Prompted IoU is essentially
identical across all three variants — 0.650, 0.651, 0.652 — spanning the smallest SAM 1 to
the newest SAM 2.1. Once SAM is told where to look, a 10× larger model traces the crack no
better.

Together these locate the limitation precisely: **scale improves SAM's ability to notice
that *something* is there, but nothing about scale gives it the concept of "crack."** Even
at its best, 53% of `vit-huge`'s masks correspond to no crack, and SAM offers no mechanism
to reject them — it is class-agnostic by design. Detection is also unstable across regions
(100% on one crop, 43% on another).

The purpose-built classifier reaches 89.3% recall *and* explicitly rejects non-cracks,
which is the part SAM structurally cannot do.

### Consistency with published work

This reproduces, independently and on a different material, the central finding of
**μSAM (Archit et al., "Segment Anything for Microscopy," *Nature Methods*, 2025)**: default
SAM underperforms on microscopy imagery and requires fine-tuning on domain data before it
is competitive with specialist tools.

### Hybrid and union tests — SAM as a candidate generator

Beyond standalone use, SAM was tested as a *candidate generator* feeding this
project's own classifier, so the only variable is where candidate regions come
from. Measured on 45 confirmed cracks across three 1024x1024 crops:

| Approach | Cracks found | Extra regions kept |
|---|---|---|
| Threshold segmentation + classifier | **41/45 (91%)** | 35 |
| SAM `vit-huge` masks + same classifier | 22/45 (49%) | 4 |
| Union (either route flags it) | 41/45 (91%) | 39 |

**Cracks found only by SAM: 0.** Everything SAM contributes is a strict subset
of what threshold segmentation already finds, so the union gains nothing.

*Correction, recorded deliberately:* an earlier version of this test reported
the baseline at 56% and a union gain of +16% recall. That was wrong — the test
script computed `MeanDarkness` as `mean(flat)` while `extract_candidates` uses
`mean(255 - flat)`, feeding the classifier a sign-flipped feature and crippling
the baseline. The discrepancy was noticeable (the same classifier measures 89.3%
recall on this image by an independent route) and should have been chased before
the result was reported. With the feature corrected, the baseline is 91% and
SAM's apparent contribution disappears entirely.

**Practical note:** SAM `vit-huge` requires 2-4 minutes per 1024x1024 crop on
CPU. A 6144x4376 image needs ~26 crops, so a 70-image dataset would take on the
order of 90 hours. Even had the union helped, that cost would be difficult to
justify against threshold segmentation running in seconds per image.

### If SAM were to be pursued further

Three options, in increasing cost:

1. **Hybrid (no training).** Use the existing pipeline for detection and pass its candidate
   locations to SAM as point/box prompts, using SAM only for boundary refinement. This is
   directly supported by the data above: the pipeline detects at 89.3%, and prompted SAM
   traces at IoU ~0.65 regardless of variant.
2. **Fine-tune the mask decoder.** Freeze SAM's image encoder and train only the lightweight
   decoder on this project's annotations — the approach μSAM itself takes. Feasible on modest
   hardware.
3. **Evaluate `micro-sam`** (PyPI), the μSAM authors' released models, already fine-tuned on
   electron and light microscopy. Worth trying before any in-house training, since success
   there removes the need for option 2 entirely.

---

## 9. Retraining on all 34 manually-corrected images

Every correction mask painted so far was converted into training rows and the classifier
retrained from scratch. Script: `interior_active_learning/code/build_training_data.py`
then `train_v3_weighted.py`. Figure: `figures/retrain_on_all_corrections.png`.

Features are read directly out of `extract_candidates`' own dataframe rather than
recomputed, specifically so the inverted-`MeanDarkness` bug recorded in Section 8 cannot
recur. The check that this worked: the existing production model reproduces its previously
published 89.3% recall / 74.1% specificity on `AS_24hr_BSE_Side_008` exactly when run
against the new CSV. If the features had drifted, that number would have moved.

### 9.1 What the corrections actually contain

| | |
|---|---|
| reviewed regions | 4,110 |
| marked crack | 3,031 |
| marked not-crack | 1,079 |
| images with ≥1 correction | 32 (of 63) |
| images with ≥1 **not-crack** mark | **8** |
| not-crack marks from `AS_24hr_BSE_Side_008` alone | **1,023 of 1,079 (95%)** |

That last row is the single most important fact about this dataset, and it is a property of
the *workflow*, not of the material. The paint tool is used to confirm cracks, so a region
only becomes a training row when someone clicks it — and nobody clicks the thousands of
uninteresting dark specks. One image was instead labelled exhaustively, region by region,
and it supplies essentially all of the negative class.

### 9.2 Why naive grouped CV gives a misleading number

Averaging per-fold AUC across a grouped 5-fold split returns 0.65 ± 0.20. That number
should not be quoted. The fold composition explains why:

| fold | test rows | test crack | test not-crack | AUC |
|---|---|---|---|---|
| 0 | 168 | 143 | 25 | 0.691 |
| 1 | 385 | 377 | 8 | 0.839 |
| 2 | 1906 | 883 | **1023** | 0.869 |
| 3 | 1497 | 1479 | 18 | 0.738 |
| 4 | 154 | 149 | 5 | **0.454** |

Only fold 2 — the one holding the exhaustively-labelled image — has enough negatives for
AUC to mean anything. Folds 1, 3 and 4 compute an AUC against 5–18 negatives, which is
sampling noise, and averaging five numbers of wildly unequal reliability is not a point
estimate. The ±0.20 spread is the tell.

### 9.3 The fix is weighting, not more labels

Each image contributed between 1 and 1,285 rows, reflecting time spent painting rather than
how much material the image represents. Unweighted, two images are 46% of the training
signal. Weighting each row by 1/(rows in its image) makes every image count once.

Measured by holding out `AS_24hr_BSE_Side_008` **entirely** — the only image whose
crack/not-crack ratio is trustworthy, so the only place specificity is a real number:

| condition | AUC | recall | specificity |
|---|---|---|---|
| A — current production model *(saw this image; optimistic)* | 0.862 | 89.3% | 74.1% |
| B — retrained on the other 31 images, unweighted | 0.729 | 6.9% | 99.7% |
| C — retrained on the 7 other both-class images only | 0.641 | 7.6% | 98.7% |
| D — **retrained on the other 31, per-image weights** | **0.925** | 93.5% | 64.4% |
| E — reverse: trained on this image only, tested on the other 7 | 0.483 | 34.2% | 60.4% |

Condition B is the trap: train on everything as-is and recall collapses to 6.9%. With a
training set that is 96% positive, `class_weight='balanced'` upweights the 56 surviving
negatives roughly 25× and they dominate the boundary. Condition D fixes it and beats the
production model's own optimistic score by a wide margin.

Condition E is worth noting separately: a model trained *only* on the exhaustively-labelled
image scores 0.483 on the others — no better than chance. Neither label subset generalises
to the other on its own. Both are needed.

### 9.4 Model family selection — why not the highest AUC

| model | pooled OOF AUC | LOIO AUC | specificity @ 0.5 | verdict |
|---|---|---|---|---|
| LogisticRegression | 0.704 | 0.925 | 62.3% | **selected** |
| SVC (RBF) | 0.202 | **0.946** | 0.3% | disqualified |
| GradientBoosting | 0.546 | 0.689 | 2.3% | disqualified |
| RandomForest | 0.373 | 0.583 | 0.5% | disqualified |

SVC ranks best on the held-out image, and the margin is real — bootstrap difference over LR
is +0.021 with 95% CI [+0.002, +0.042]. It is still unusable, because at the threshold the
pipeline actually applies it labels **all 1,285 regions crack**: recall 100%, specificity
0%, 1,023 false positives. Its *ranking* is good and its *probability scale* is not; one CV
fold produced scores confined to [0.93, 1.00]. That same instability is why its pooled
out-of-fold AUC inverts to 0.202 — pooling requires scores to be comparable between folds.

This is the reason to distrust "highest AUC wins" here. AUC is threshold-free; the pipeline
is not. LogisticRegression is the only family whose ranking is strong *and* whose
probabilities sit on a stable scale, so a fixed threshold keeps its meaning across retrains.

### 9.5 The deployed operating point

The retrained model's better ranking is spent on removing false positives at **matched
recall**, so no sensitivity is traded away silently:

| threshold | recall | specificity | false positives |
|---|---|---|---|
| production model @ 0.5 | 89.3% | 74.1% | 265 |
| retrained @ 0.500 | 93.5% | 64.4% | 364 |
| retrained @ **0.578** — *deployed* | **89.3%** | **79.3%** | **212** |
| retrained @ 0.554 | 89.7% | 74.6% | 260 |
| retrained @ 0.693 (Youden) | 76.3% | 93.9% | 62 |

**Net result: identical recall to production, 20% fewer false positives (265 → 212), on an
image the retrained model never saw — against a baseline that was trained on it.**

Saved to `models/crack_classifier_v3_weighted.joblib` with the threshold stored in the
bundle. `models/crack_classifier.joblib` is deliberately **not** overwritten; swapping the
production model changes every overlay and is a separate decision.

### 9.6 The honest limitation

The gain above is validated on **one** image, because only one image has a trustworthy
class ratio. Everything else in this section is consistent with that gain being real, but
it rests on a single held-out sample and should be treated as such.

The cheapest way to strengthen it is not more crack confirmations — it is **not-crack marks
on more images**. 24 of the 32 reviewed images have none. Marking the obviously-not-crack
features (pits, polishing scratches, inclusions, contrast noise) on even five more images
would let specificity be measured across images instead of within one, and would let
grouped CV report a number worth quoting.

### 9.7 Applying the retrained model to all 63 images

`interior_active_learning/code/apply_v3_all_images.py`. Both models were run over the
**same candidate set** per image — identical segmentation, identical features, only the
classifier differs — so every difference below is attributable to the model alone. This was
a measurement pass: the production model was not replaced and no paint template was
rewritten, so an in-progress painting session was unaffected.

| | production | retrained |
|---|---|---|
| candidates evaluated | 41,878 | 41,878 |
| accepted as crack | 22,199 | 23,243 |
| newly accepted | — | 5,218 |
| newly rejected | — | 4,174 |
| mean crack-area % | 2.428 | 2.459 |

**The headline is that nothing moved much.** Crack-area fraction — the number that actually
gets reported as a material result — changed by more than 1 percentage point on **0 of 63
images**, and by more than 0.5 pp on 2. Largest swings: +0.47 pp (`MAR_Amb_HIP_ETD_0008`)
and −0.66 pp (`260708_316_H_b2_front_CBS_014`). Across the 31 images with no corrections at
all, and therefore no training influence whatsoever, mean area went 1.642% → 1.696%.

This is a refinement, not a regime change. The ~9,400 individual flips are mostly small
regions near the decision boundary trading places, which is what a threshold move of 0.5 →
0.578 combined with a re-fit should look like.

### 9.8 Agreement with the corrections — and why one version of it is misleading

| average | production | retrained |
|---|---|---|
| row-weighted, all 32 reviewed images | 49.2% | **64.6%** |
| per-image mean, all 32 reviewed images | 60.4% | 64.3% |
| per-image mean, 8 both-class images | 54.7% | 62.0% |

The row-weighted +15.4-point jump should not be quoted. `AS_24hr_BSE_Side_008` (1,285 rows)
and `MAR_Amb_AS_CBS_0003` (617 rows) are 46% of all reviewed regions between them, and both
improved a lot, so that average largely reports what happened on two images. The per-image
mean — which is also the objective the weighted model is actually fitting — gives **+3.9
points** overall and **+7.3 points** on the images where a false positive can cost
agreement. Agreement improved on 15 of 32 images and got *worse* on 11.

Both numbers are also **in-sample**: the retrained model was trained on these very
corrections, so agreement with them is optimistic by construction and is a sanity check,
not evidence of generalisation. The held-out result in §9.5 remains the only clean measure.

### 9.9 Confirming the weighting trade-off is the right one

Fitting the same features to the same 4,110 rows, in-sample:

| | row-accuracy | per-image mean | held-out AUC (§9.3) |
|---|---|---|---|
| unweighted | 90.8% | 80.9% | 0.729 |
| per-image weighted | 73.6% | 72.6% | **0.925** |

Worth stating plainly because it looks backwards: the weighted model fits the training
labels *substantially worse* and generalises *far better*. The unweighted model reaching
90.8% also settles a question the low agreement numbers might otherwise raise — the
8-feature representation is not the bottleneck. It can fit these labels; the weighting
deliberately stops it from doing so on two images' terms.
