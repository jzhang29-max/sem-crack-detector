# Head-to-head verdict: is this method better?

Six competitor papers read on a fixed rubric, full text obtained for five.
**Short answer: no on segmentation accuracy, yes on measurement validity — and the
competitors are broken on exactly the axes where this work is strong.**

## 1. Your "mostly biological" objection — adjudicated

It **fails** for the papers that matter:

| paper | domain |
|---|---|
| Schmies 2023 (SEM crack features, fatigue-cracked metal) | **materials** |
| Gerçek/Chekhonin 2025 (SEM steel carbides) | **materials** |
| Taufique 2025, npj (SEM+EBSD, 316L FSP) | **materials** |
| cigRockSEM 2025 (mudstone/sandstone/shale) | **geological** |
| clDice / OmniCrack30k / Skeleton Recall cluster | method-agnostic (concrete cracks, roads, vessels) |
| ilastik / Cellpose 2.0 / RootPainter / micro_sam | **biological** |

Only the last row is bio — and its relevance is not that it does what you do, but that
**not one of the six clusters has a guardrail on model promotion.** So your objection
rescues the *tool* framing partially, and nothing else.

## 2. Where you lose, honestly

**Metric legibility — universal and fatal as currently reported.** Every competitor
reports Dice/IoU/F1 with numbers. You report AUC. Schmies: mean IoU + per-class IoU +
P/R/F1. Gerçek: Dice with median, IQR and a non-parametric test. Taufique: F1 to 0.62
plus HD95 (26.3–55.7 px) and MAE in µm. cigRockSEM: accuracy, F1, recall, mIoU.
Cellpose 2.0: AP 0.36–0.76. RootPainter: Dice ≈0.9 **with external validation against
independent manual grid counts (R² 0.78–0.92)**.

**Other real losses:** Taufique has *instrumental* ground truth (EBSD at 50 nm step,
MTEX-reconstructed) and a genuine 4-way out-of-distribution test, with cross-detector
transfer *explained* by CASINO Monte Carlo escape depths (17±4 nm at 5 keV → 190±29 nm
at 20 keV). cigRockSEM has ~26× more independent fields and 3.92 B exhaustively
labelled pixels. Gerçek ships Apache-2.0 code, CC-BY data with a DOI, the checkpoint,
the Optuna study, calibrated uncertainty (temperature scaling, T=1.87117) **and** a
fully-specified classical baseline. Schmies has a calibrated **height** channel from a
4-quadrant BSE detector via shape-from-shading, acquired in the *same scan* as SE and
BSE — inherently co-registered, zero drift.

## 3. Where you genuinely win — and it is consistent across all six reads

1. **Nobody has a human-in-the-loop loop with a guardrail. Not one of the six.**
   Gerçek is one-shot annotate-train-done. Schmies has no correction interface. The
   clDice cluster has none. ilastik/TWS have brush-based interaction but no promotion
   gate and no erase channel. Your specificity floor + recall-matched threshold
   transfer catches failures a single foreground-F1 comparison cannot.
2. **Nobody measures acquisition sensitivity on matched fields.** The topology-metrics
   reviewer put it plainly: *"Nobody asks whether the same physical object imaged twice
   under different acquisition conditions yields the same measurement."* Your 15
   index-matched CBS/ETD pairs at verified 10.00 kV / 1.6 nA / 6.0 mm WD are unique in
   this set.
3. **Full-frame reasoning.** Your Dijkstra + MST merge has reach to the frame diagonal
   (~7,385 px). ilastik's filter halo is ~35 px; Gerçek runs 128×128 tiles; clDice's
   largest 2D images are 1500×1500; OmniCrack30k drives a 256×256 patch. No tiled
   network can link fragments 5,000 px apart. This is architectural, not incidental.
4. **Your evaluation protocol is more rigorous than all three deep-learning papers.**
   Gerçek's `random_split` over 1,920 pooled tiles from 12 micrographs puts
   *neighbouring crops of the same image* in train and test. cigRockSEM uses a single
   unstratified 8:2 split over 512×512 crops with no grouping by field, well, sample or
   magnification, and no validation set. Taufique's in-distribution 3-fold CV draws
   train and test chips from **the same 1792×1280 field**. You use grouped/leave-one-out
   by image. You are the only one with a leakage-aware protocol.

## 4. The competitors' scale handling is broken — an auditable finding

This is worth a paper section on its own, and it is verifiable from their own code:

- **cigRockSEM's advertised "magnification standardisation" does not standardise.**
  The final dataset still spans 0.0008–0.9773 µm/px — a **1,222× residual spread**.
- **cigRockSEM's metrics are scored on the wrong class.** `SegEvalMetrics.py` calls
  jaccard/precision/recall/f1 with `pos_label=255`, and their own Table 2 defines
  white(255) = **background**. Table 5 is also internally arithmetically impossible
  (U-Net shale acc 0.9753, recall 0.9936, F1 0.9769, mIoU 0.6358 cannot coexist), and
  `precision_score` is computed in their code but **omitted from the results table** —
  the one metric that would expose over-segmentation.
- **Gerçek hardcodes a wrong pixel size.** `pixel_size_um = 1/180` for all images, but
  JFL is 1/143.2 — every JFL carbide length is **20.4 % short**, every area **1.56×
  too small**. And their 0.5 channel merge, `(SE2//2)+(inlens//2)`, *erases*
  cross-detector disagreement instead of measuring it — exactly the quantity you have.
- **Taufique's headline 0.34 µm MAE is an ensemble artefact.** Per-fold values for the
  identical configuration are 7.93, 0.85 and 1.84 µm; the 3-model sum happens to land
  at 0.34. A constant predictor at the EBSD mean scores ≈0.9 µm on the same set.

You cannot beat these papers on accuracy. You can beat them on **being right**.

## 5. Two corrections to numbers I have been quoting

**The AUC is 0.714, not 0.884.** `models/crack_classifier_v3_metrics.json`:
`pooled_auc = 0.7144 ± 0.0278` (grouped CV, 45 images, 6,408 pos / 1,097 neg), while
`loio_auc_exhaustive_image = 0.8840` is leave-one-image-out on **one** exhaustively
labelled frame. I repeated 0.884 throughout this analysis; the defensible headline is
**0.714**. Corroborating that the single-image LOIO is unreliable: SVC(RBF) scores
*higher* on it (0.921) while its pooled AUC is 0.276.

**The repo already records losing to Otsu — with an important caveat.**
`docs/COMPETITIVE_POSITION.md` states the pipeline loses f1 by 0.054 to Otsu +
small-object cleanup, and self-corrects an earlier draft that had claimed +0.101 from a
two-frame subset. But it also records that Otsu families **collapse specificity**
(0.113 vs your 0.455; one scores f1 0.989 at specificity 0.000 by predicting almost
everything) and Frangi families collapse recall. The repo's own conclusion is the
right one: *the pipeline is the only method competitive on both sides of the confusion
matrix* — not that it wins on f1. Keep that framing; it survives review.

## 6. The plan, in dependency order

1. **Pixel-precise evaluation set — the blocker.** 8 fields inside one HFW stratum,
   with explicit negative regions, plus a 3-field overlap with a **second annotator** so
   inter-rater Dice can be reported. Do *not* reuse the 9 fine-stroke frames as-is: they
   are selected *by* stroke width, which biases the set. **5–8 annotator-days.** Nothing
   below can be reported without this, and it beats Gerçek on a rubric axis he vacates
   entirely (N=1 annotator, who was also the microscopist).
2. **Publish the cross-detector reproducibility ceiling as the headline.** 1 day; the
   numbers exist in `analysis/paired_detector.csv`.
3. **Parse HFW from the databar and re-parameterise every filter constant in µm.**
   `_detect_databar_top` in `code/detect_cracks.py` already finds the panel — read it
   instead of cropping it away. 2–3 days. Beats Gerçek decisively on scale correctness.
4. **Re-split by FIELD, not frame** (`train_v3_weighted.py` groups by SourceImage;
   CBS/ETD pairs of the same field currently straddle the split). 0.5 day.
5. **Replace the AUC headline with clDice + Boundary IoU at 2 px and 5 px**, plus IoU /
   Dice / precision / recall at the deployed operating point, and always report
   connected-component count and total skeleton length in µm alongside overlap. 3–4 days.
6. **Run Gerçek's own published classical baseline** (SimpleITK, Gaussian σ=1.0,
   top-hat r=30, Otsu, hole fill, min-size 3 px) so the comparison is exact. 1–1.5 days.
7. **Fix skeleton length** with a corner-count or Kulpa/Vossepoel–Smeulders estimator.
   0.5–1 day. Defect repair, but load-bearing for everything measurement-side.

Optional and larger: match Schmies' architecture deliberately
(`smp.Unet(resnet34, imagenet, in_channels=4)`) with four *never-averaged* channels and
train the coarse marks as weak supervision (partial BCE over marked pixels + soft-clDice).
3–5 days plus 20–40 rented A100-hours; Apple Silicon MPS will not do this.

## 7. The one-sentence claim a 2026 reviewer would accept

> *On identical fields of view at matched beam conditions, changing only the SEM
> detector changes measured crack extent by 2.04× — larger than the material
> differences such studies report — and three published SEM segmentation pipelines
> propagate scale incorrectly, so image-derived crack metrics require an acquisition
> uncertainty budget before they can support materials conclusions.*

Note this claim does **not** depend on your detector being good. That is why it holds.
