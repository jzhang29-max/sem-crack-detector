# Does the unlabelled-pixel problem generalise beyond this codebase?

A referee will ask: you demonstrated this on your own software, so how do you know it is not
just your bug? This is the answer. Six external targets were audited **from primary source**
— repository files, cited by path and line — with a second pass that fetched each citation
and marked a claim refuted if the line did not say what was claimed.

Only claims that survived verification appear here. Line numbers are branch-tip as audited
(2026-08) and **must be re-pinned to commit SHAs before submission**.

## The answer: yes, and the split is informative

Three sub-questions must be kept apart, because tools routinely get one right and another
wrong.

**Training — the field mostly gets this right.** ilastik's Random Forest path is
structurally incapable of training on unlabelled pixels: 0 is a reserved sentinel, classes
are 1..N, and `lazyflow/operators/opFeatureMatrixCache.py:262` gathers training rows via
`numpy.nonzero(labels[..., 0])`, so a zero-valued pixel can never become a row in the design
matrix. `torch.nn.CrossEntropyLoss` ships `ignore_index=-100` on by default. mmsegmentation,
nnU-Net v2 and `segmentation_models_pytorch.losses.DiceLoss` all implement exclusion.

**Export — the failure is essentially universal.** **Zero of six** annotation or analysis
tools preserve an "a human has not looked here" value in their *default* mask output.

- micro-sam initialises its canvas `np.zeros(...)`, commits only `seg != 0`, and leaves the
  zarr fill value at 0.
- CVAT's segmentation exporters build a label map with exactly one extra entry —
  `background`, `#000000` — and no void; every uncovered pixel becomes index 0. Datumaro,
  the library underneath, *does* carry `ignored = 255`: a capability no CVAT user can reach.
  A frame with zero annotations gets no mask at all and is dropped from the split list, so a
  fully-unreviewed image silently disappears while a partially-reviewed one is exported as
  though its unreviewed area were background.
- Label Studio writes per-region uint8 alpha layers, zero elsewhere, with no aggregate mask
  and no void value. Its mask **import** thresholds at `> 128`, so a PASCAL/Cityscapes void
  pixel (255) round-trips as **foreground**.
- All four crack repositories audited export binary 0/255 or 0/1.
- QuPath's `LabeledImageServer` assigns unannotated pixels label 0 (`*Background*`), with the
  ignored-class filter commented out in its label-export path.
- MetalDAM — 42 hand-annotated SEM steel micrographs, the closest published analogue to this
  corpus — has five classes and no void value.

**Scoring — structural, not incidental.** In the dominant APIs there is no first-class way
to express "unlabelled" in the binary-segmentation path:

| library | `ignore_index` in the segmentation path |
|---|---|
| `sklearn.metrics` | **zero occurrences** |
| `torchmetrics` | present in `classification/`, **absent in `segmentation/`** |
| MONAI | zero occurrences; input contract requires binarised one-hot, which cannot encode a third state; `DiceCELoss` documents it as "not supported" |
| `segmentation_models_pytorch` | `get_stats` **raises `ValueError`** for `ignore_index` in binary mode |

That last one is refusal, not oversight — and binary mode is exactly the object-versus-
background case of a micrograph.

## Tools that get it right, which strengthens the argument

Naming them matters: it converts "everyone is wrong" into "the fix is known and unadopted",
which is both truer and much harder to dismiss.

- **mmsegmentation** `IoUMetric` masks out `ignore_index` before aAcc/Acc/Precision/Dice/IoU;
  the default is 255.
- **nnU-Net v2** states plainly that metric computation in validation is done only on
  annotated pixels.
- **CREMI** VOI defaults to `ignore_groundtruth=[0]`.
- **Cell Tracking Challenge** SEG averages IoU over reference objects only, so unannotated
  objects cannot be penalised.
- **CVAT** cannot mis-score by construction: its quality metrics are annotation-level over a
  confusion matrix with an explicit "unmatched" pseudo-label, so no pixel true-negative term
  exists anywhere inside CVAT.
- **ilastik** reports no metrics at all, so it cannot inflate anything, and its one
  quantitative number (Random Forest out-of-bag error) is computed over labelled voxels only.
- **`skimage.metrics`** — three of ten functions (`adapted_rand_error`, `contingency_table`,
  `variation_of_information`) carry `ignore_labels`, defaulting to `(0,)` for the first.
  These are connectomics/EM metrics, from the community that pioneered sparse dense-EM
  annotation.

**And here is the sharpest form of the result:** the two tools that score correctly — CVAT
and ilastik — *still* ship a default export that destroys the distinction. Metric hygiene
inside a tool protects nobody, because the export imposes the convention on every downstream
consumer. That is an information-theoretic argument, not a quality judgement, and it cannot
be answered with "but tool X is fine".

## Prior art that must be credited, not claimed

The idea of a void/ignore label is **fourteen-year-old prior art** and the paper must say so
in its own voice before a referee does. It is honoured in the released ground truth *and* the
official scorer for PASCAL VOC (`VOCevalseg.m:55`, `locs = gtim<255;`, devkit dated
2011-05-18), Cityscapes (`ignoreInEval`), ADE20K (label 0, with the comment that detections
in unlabeled portions should not be penalised), COCO-Stuff and COCO-panoptic (`VOID = 0`;
predicted segments over 50% VOID are not counted as false positives).

**The correct citation for these exact semantics is nnU-Net v2's ignore label, not VOC's.**
VOC and Cityscapes void is largely boundary ambiguity and out-of-ROI — "a human looked and
declined to commit" — whereas the state here is "a human never looked". ADE20K's label 0 and
COCO's VOID are closer.

What is new is therefore not the mechanism but **the map of where it stopped**: the
convention propagated through natural-image benchmarks into general frameworks
(mmsegmentation's default) and into biomedical imaging (nnU-Net, CREMI, CTC), and did **not**
reach materials-micrograph benchmarks (MetalDAM, OmniCrack30k) or the default export surface
of any microscopy or annotation tool audited. The split is domain-shaped, not
competence-shaped.

## Measured effect on this corpus

Same prediction, same ground truth, two conventions, over the 10 masks carrying both classes.
Both averagings are reported because they are not interconvertible and the conclusion should
not depend on which a reader assumes.

| metric | macro adj | macro dense | Δ | micro adj | micro dense | Δ |
|---|---|---|---|---|---|---|
| recall | 0.5338 | 0.5338 | **+0.0000** | 0.2013 | 0.2013 | **+0.0000** |
| specificity | 0.4599 | 0.9478 | +0.4879 | 0.6967 | 0.9629 | +0.2662 |
| precision | 0.9696 | 0.3791 | −0.5905 | 0.9827 | 0.3542 | −0.6285 |
| f1 | 0.6378 | 0.3515 | −0.2863 | 0.3341 | 0.2567 | −0.0774 |

Four-cell counts, pooled: `tp 3,594,024  fp 63,322  fn 14,262,153  tn 145,441` under
exclusion, against `tn 169,965,542` under the dense convention — a factor of **1,169**.

**Recall is invariant to the last floating-point bit under both averagings**, and this is
asserted in the experiment rather than merely reported: every affected metric is affected
through the false-positive term (IoU, Dice, precision) or the true-negative term (specificity,
accuracy), and recall touches neither. The audit predicts this independently of the data.

## What must not be claimed

- **No novelty for the ignore label.** Cite VOC, Cityscapes, ADE20K, COCO-panoptic,
  mmsegmentation, nnU-Net v2.
- **No universal failure.** ilastik trains correctly and offers a coverage-preserving export;
  CVAT cannot mis-score; mmsegmentation, nnU-Net, CREMI and CTC are correct.
- **Do not say ilastik destroys the unreviewed state.** `ExportNames.LABELS` preserves 0
  verbatim. The accurate claim: its default and headline export discards it while a
  secondary, undocumented export preserves it. Partial credit, stated as such.
- **Do not conflate prediction with ground truth.** A classifier predicts everywhere by
  construction; a dense *prediction* is not an error. Sub-question 2 is about ground-truth
  masks.
- **Do not describe sklearn as uniformly wrong.** `confusion_matrix` with an explicit
  `labels=` behaves correctly; the sharper framing is the inconsistency across sibling
  functions in one namespace, where `jaccard_score(..., labels=[1])` silently returns the
  dense number.
- **Do not claim micro-sam inflates a pixel negative class in its headline numbers.** Its
  reported metrics are instance-level; the harm appears as precision deflation from unmatched
  predictions in unreviewed regions. The 2721× pixel figure does not transfer.
- **Do not claim unlabelled-as-background is always an error.** On an exhaustively annotated
  corpus it *is* the ground truth — which is exactly why these authors were never forced to
  think about it. The claim is conditional on sparse review and must say so, or it reads as
  gotcha journalism.
- **Do not present this as an execution audit.** It is a source audit. The only behaviour
  executed was the local sklearn/skimage signature probe.
- **Label the averaging.** Macro means of a ratio do not obey the ratio's algebra; an
  unlabelled table invites a referee to find an arithmetic error that is not there.

## Executed, not just read

`experiments/three_state_conformance.py` pushes one canonical three-state fixture — 512 crack,
128 adjudicated not-crack, 3456 UNREVIEWED px — through every library available in this
environment and records what comes back. It converts the source audit's most falsifiable
claims into a reproducible artefact, and **all of them survived execution**:

| probe | status | result |
|---|---|---|
| `sklearn.metrics` | EXECUTED | No `ignore_index` on `confusion_matrix`, `jaccard_score` or `precision_recall_fscore_support`. Pre-filtering changes IoU 0.5517 → 0.7273. The natural `jaccard_score(..., labels=[1])` idiom returns the **dense** number, while `confusion_matrix(labels=[1])` *does* drop the other class — the sibling inconsistency, confirmed by running it. |
| `skimage.metrics` | EXECUTED | `adapted_rand_error` ships `ignore_labels=(0,)` **as the default**; the result changes when label 0 is included. The capability exists, from the connectomics community. |
| `torch.nn.CrossEntropyLoss` | EXECUTED | `ignore_index` works — loss 0.3228 → 0.9182 when unreviewed pixels are excluded. But the default is `-100`, a sentinel no mask file contains, so it is only reachable if the caller first invents an encoding for UNREVIEWED. |
| `torchmetrics` | EXECUTED | `BinaryJaccardIndex` accepts `ignore_index` and the answer changes; **0 of 3** inspected classes in `torchmetrics.segmentation` expose it. The asymmetry is inside one library, which is why a practitioner can reasonably believe their stack handles this. |
| `segmentation_models_pytorch` | EXECUTED | `get_stats(..., mode="binary", ignore_index=255)` raises `ValueError: ``ignore_index`` parameter is not supported for 'binary' mode`. Verbatim. Binary mode is the object-versus-background case of a micrograph, so this is a **refusal**, not a bad default — and a refusal is a decision. |
| MONAI 1.6.0 | EXECUTED | No `ignore_index` on `DiceMetric`, `MeanIoU`, `DiceLoss` or `DiceCELoss`. The binarised one-hot contract leaves a third state nowhere to live, so the caller must pre-filter — moving Dice 0.7111 → 0.8421. `ignore_empty` is a **false friend**, and the distinction is now measured: on an all-empty ground truth it writes `nan` and counts **0** cases, versus `0.0` and **1** when off. It drops whole cases with no positive label — a real feature, and a different question from excluding unreviewed pixels *inside* a case. |
| datumaro 1.13.8 (CVAT's export path) | EXECUTED | A partially-annotated image exported to VOC writes annotated pixels `(128,0,0)` and **unannotated pixels `(0,0,0)` — class 0, background**. VOC's void colour `(224,224,192)` never appears. Yet the capability *is* in the library: `make_voc_categories()` carries `'ignored'` at palette index **21** with exactly that colour. The format supports void, the library knows about it, and this export path silently turns "nobody looked" into "background". |
| Label Studio converter | EXECUTED | `brush.py` thresholds with `np.array((np.array(image) > 128) * 255)`. A mask encoding unreviewed as 255 imports **all 3456 unreviewed pixels as foreground** while the 512 genuine crack pixels encoded as `1` are **dropped** — the result is almost exactly inverted. The two conventions do not merely disagree, they corrupt each other on round-trip. |
| elf 0.9.2 / micro-sam's scorer | EXECUTED | **Stronger than the source audit claimed.** `ignore_label=0` is the default and looks protective. It is not: a prediction lying entirely inside the ignored region drops precision 1.0000 → **0.5000**, which is *worse* than `ignore_label=None` (**0.6667**). It removes the ignored region from the ground-truth **objects** without exempting predictions that land there, so they are charged as unmatched false positives. `dice_score` has no mask argument at all and charges it too (0.8673 → 0.5537). |
| mask file round-trip | EXECUTED | uint8 PNG and TIFF both carry three states losslessly. **The formats are innocent; the loss is in the exporter.** And a mask encoding unreviewed as 255 (the PASCAL/Cityscapes convention) is read as **foreground** by a `> 128` import threshold — the two conventions actively corrupt each other. |

**Ten probes execute.** Two remain unexecuted and the report names both with the reason:

- **ilastik** — there is *no PyPI distribution*; it ships via conda or a binary installer, so
  its headless export could not be run in this environment. The source-level finding stands,
  unexecuted, and is labelled as such.
- **micro-sam's napari annotator** — the canvas and zarr fill value need an interactive
  session. Its **scorer** is covered by the elf probe, which is the half that affects
  reported numbers.

The elf result deserves emphasis because it is the only place the executed evidence came out
*stronger* than the source audit predicted. A parameter named `ignore_label`, on by default,
does not protect predictions in the ignored region — and passing it is worse than omitting
it. Anyone who read that signature and concluded their evaluation was safe would be wrong in
the unsafe direction, which is the most dangerous shape a false friend can take.

The datumaro result is the one worth dwelling on, because it is the export claim — the choke
point of the whole argument — moved from *read* to *run*. It also sharpens the framing: this
is not a library that lacks the concept. It ships the void colour, names it `ignored`, and
puts it at a known palette index. The concept is present and the export path does not reach
for it.

Two design choices worth stating, because they decide whether the artefact is trustworthy.
Probes assert on **behaviour**, not version strings, so the suite stays meaningful as these
libraries change. And the suite exits non-zero only when a probe *errors* — a library
treating unlabelled as background is the finding, so recording it must not masquerade as a
broken test run.

## Remaining holes

1. **Read, not run — now partly closed.** Six probes execute (see above), covering the
   sklearn sibling asymmetry, smp's refusal, the torchmetrics namespace split, and the
   round-trip corruption between the two conventions. What remains unexecuted is MONAI,
   Datumaro/CVAT, Label Studio's converter, ilastik and micro-sam. Those are dependency
   installs and a headless invocation, not research.
2. **Statistical fragility.** Specificity under exclusion rests on 145,441 adjudicated
   negatives, about three parts in ten thousand of the corpus. *Fixable without annotation:*
   bootstrap over images, and sub-sample the existing masks to derive the delta as a curve
   against adjudicated fraction rather than a single pair of numbers.
3. **Target selection.** Six targets, chosen by us. State the sampling frame and publish the
   rubric.
4. **Unresolved code paths**, to be marked UNVERIFIED rather than rounded up: ilastik's
   pixelwise/NN training path, OmniCrack30k's nnU-Net label handling, Label Studio's scoring,
   CVAT's other mask exporters.
5. **The deep one, and not fully fixable here.** Excluding unreviewed pixels is unbiased only
   if the reviewed region is representative. Annotators plausibly review preferentially — the
   ambiguous, feature-dense regions — so 0.4599 is itself computed on a non-random sample and
   the true full-image specificity is unknowable from this corpus. The honest deliverable is
   therefore an **interval**, not a point: the two conventions bracket the achievable
   estimate, the bracket is enormous (specificity 0.46–0.95), and reporting either endpoint
   without naming the convention is uninterpretable. That claim needs no new data.
