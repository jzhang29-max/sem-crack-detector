# Where this work stands against the newest crack-segmentation literature

*2026-09-19. Every DOI/arXiv id below was resolved live (Crossref REST or arXiv Atom) during
this session. Numbers from other papers are quoted with their protocol and are **not** compared
to ours — cross-dataset accuracy comparison is invalid and this document does not do it.*

## The one thing to get straight first

**You cannot rank methods by comparing our IoU to a published IoU.** Their numbers come from
other datasets, other label conventions, other operating points. A 0.79 IoU on CrackSeg9k and
a 0.19 IoU here are not commensurable — CrackSeg9k cracks are wide, high-contrast, on concrete,
with tight labels; ours are ~3 px, low-contrast, on textured metal, with labels that are region
assertions of median 59 px. The only valid comparison is **running their methods on our data**,
which is what `methods_bench.py` does, and **comparing protocols**, which is what this document
does.

## Verified reference points

| ref | what it establishes | verified |
|---|---|---|
| Arbeláez, Maire, Fowlkes & Malik, *Contour Detection and Hierarchical Image Segmentation*, IEEE TPAMI 33(5):898–916 | defines **ODS** and **OIS**; OIS is a per-image oracle threshold chosen using the label | `10.1109/TPAMI.2010.161` |
| Martin, Fowlkes & Malik, IEEE TPAMI 26(5):530–549, 2004 | the original oracle-threshold protocol | `10.1109/TPAMI.2004.1273918` |
| Zou et al., *DeepCrack*, IEEE TIP 28(3):1498–1512, 2019 | ODS/OIS as the standard crack-segmentation reporting pair | `10.1109/TIP.2018.2878966` |
| Benz & Rodehorst, *OmniCrack30k*, CVPRW 2024 | the large modern crack benchmark | `10.1109/CVPRW63382.2024.00392` |
| Maier-Hein, Reinke et al., *Metrics reloaded*, **Nature Methods**, Feb 2024, 73 authors | community recommendations for metric choice | `10.1038/s41592-023-02151-z` |
| Reinke, Maier-Hein et al., *Understanding metric-related pitfalls in image analysis validation*, **Nature Methods**, Feb 2024, 70 authors | the pitfalls, including small/thin structures under IoU | `10.1038/s41592-023-02150-0` |
| Kervadec, Dolz, Wang, Granger & Ben Ayed, MIDL 2020, PMLR 121:365–381 | the **tightness prior** for superset (box) labels | `arXiv:2004.06816` |
| Carion et al. (Meta), *SAM 3: Segment Anything with Concepts* | the presence head, in the abstract | `arXiv:2511.16719` |
| *Rapid-Deployment Crack Measurement Based on SAM3 Semantic-Edge Response Decoding* | SAM 3's final masks lose thin-crack evidence, six public crack datasets | `arXiv:2607.12292` |
| Dabaja & Celik, remote sensing SAM 3 evaluation | presence head repurposed as a standalone zero-shot classifier + 5-config prompt ablation | `arXiv:2607.09583` |
| CoCo-SAM3 | SAM 3 synonym inconsistency named | `arXiv:2604.19648` |
| *Prompt Sensitivity in Vision-Language Grounding* | prompt instability over 263 COCO images, six prompts | `arXiv:2604.17126` |

## What we are NOT claiming

- **Not** that our accuracy beats the published state of the art. It is not measured on the same
  data and, at n = 15 tiles from 9 frames, could not establish that if it were.
- **Not** that any method or metric here is novel. The tightness prior is Kervadec 2020;
  ODS/OIS is Arbeláez 2011; the presence head is Meta's; clDice is published.
- **Not** that SAM 3 is bad at cracks in general. On our data, at matched tuning budgets, it is
  statistically indistinguishable from thresholding, and `arXiv:2607.12292` reaches a
  compatible conclusion on six public datasets with far better power.

## The classifier arm: 0.85 accuracy is what you get by answering "crack" every time

`sem-crack-detector/models/crack_classifier_v3_metrics.json`, patch-level classification,
n = 7,505 patches (6,408 positive / 1,097 negative) over 45 images. **Majority-class prior =
0.8538.**

| model | pooled grouped-CV AUC | accuracy | specificity | reading |
|---|---|---|---|---|
| **LogisticRegression** | **0.7144 ± 0.0278** | 0.6547 | 0.6669 | the only model that discriminates |
| RandomForest | 0.5076 | 0.8337 | **0.0077** | predicts positive almost always |
| GradientBoosting | 0.5332 | 0.8486 | **0.0088** | predicts positive almost always |
| SVC (RBF) | **0.2756** | 0.8366 | **0.0047** | **below chance** — likely a sign-inverted score |

Three of the four models reach accuracy 0.83–0.85 **at or below the 0.8538 prior**, with
specificity under 0.01. They have learned to say "crack". The logistic model's accuracy of
0.6547 is *lower* precisely because it makes real decisions — which is why accuracy must not be
quoted for this task without the prior beside it. That is the class-imbalance pitfall catalogued
in *Understanding metric-related pitfalls in image analysis validation* (Nature Methods 2024,
`10.1038/s41592-023-02150-0`).

The SVC result is a defect, not a finding: an AUC of 0.2756 is *anti*-correlated with truth, and
the same model reports 0.9209 under leave-one-image-out. One of the two is computed with the
wrong sign or the wrong probability column. Flagged for a separate fix; do not quote either.

So on the classifier arm the honest answer to "is this the best model?" is: **LogisticRegression
at AUC 0.714 is the best of the four tried, and the other three are at or below chance.** 0.714
is a modest number and the corpus cannot currently support a better one — see the caveats.

## Segmentation on your tiles, at four tuning budgets

*(filled from `methods_bench_fast.json`; clDice and the extended-scale ridge results are
appended when those runs finish)*

| method | OIS (per-tile oracle) | ODS (one shared) | **LOFO** | LOFO 95% CI |
|---|---|---|---|---|
| global threshold | 0.4688 | 0.3217 | **0.2579** | [0.096, 0.597] |
| Sauvola local | 0.4593 | 0.2712 | 0.1117 | [0.067, 0.421] |
| Sato ridge | 0.2853 | 0.2155 | 0.1972 | [0.039, 0.273] |
| Meijering ridge | 0.2823 | 0.2350 | 0.1802 | [0.044, 0.353] |
| Frangi ridge | 0.1219 | 0.1007 | 0.0843 | [0.035, 0.197] |
| **nested LOFO** (method *and* params chosen on training frames) | — | — | **0.2526** | [0.017, 0.410] |
| SAM 3 union, τ=0.3 (zero-shot, no labels at all) | — | — | 0.1914 | [0.047, 0.559] |
| SAM 3 oracle instance | 0.3871 | — | — | |

**Nothing beats SAM 3 significantly.** Paired over the 15 tiles: nested LOFO wins 8/15,
median Δ +0.0098, **p = 0.4973**; the single best method (global threshold) wins 10/15,
median Δ +0.0828, **p = 0.2524**. And note the asymmetry runs *against* the classical arm:
it is allowed to fit parameters on 8 labelled frames, while SAM 3 sees no labels at all.

The ridge filters lost, but the first sweep used σ ∈ [1, 6] while the structures here are a
median 16 px wide, which needs σ ≈ 8. That was my scale error, not evidence about ridge
filtering; the extended sweep (σ up to 12) is reported below.

## The three results that actually matter

### 1. The metric decides the winner, and it reverses the ranking

Same predictions, same tiles, same LOFO protocol — only the metric changes:

| method | IoU (LOFO) | rank | clDice (LOFO) | rank |
|---|---|---|---|---|
| global threshold | **0.2579** | **1** | 0.1827 | 5 |
| nested LOFO | 0.2526 | 2 | 0.1739 | 6 |
| Sato ridge | 0.1972 | 3 | 0.2981 | 3 |
| SAM 3 union (zero-shot) | 0.1914 | 4 | **0.3030** | **2** |
| Meijering ridge | 0.1802 | 5 | **0.3382** | **1** |
| Sauvola local | 0.1117 | 6 | 0.2421 | 4 |
| Frangi ridge | 0.0843 | 7 | 0.0849 | 7 |

The method that wins on IoU comes **fifth** on clDice; the clDice winner comes **fifth** on IoU.
clDice is the metric designed for thin structures, and IoU is the one every crack paper leads
with. Anyone reporting a single number on this corpus is reporting their metric choice.

### 2. On clDice, nothing beats a zero-shot foundation model

Paired over 15 tiles against SAM 3's clDice: Meijering ridge wins 6/15 (median Δ −0.0004,
p = 0.4631), Sato 5/15 (p = 0.5417), global threshold 5/15 (p = 0.1909), nested LOFO 6/15
(p = 0.1726). **Every tuned classical method loses the median to SAM 3 on clDice**, and SAM 3
used no labels at all while they fitted on 8 labelled frames.

### 3. Honest method selection costs 95% of the apparent performance

Nested LOFO — method and parameters both chosen on the 8 training frames — scores clDice
**0.1739**. The best method chosen *with hindsight* scores **0.3382**. The gap, **+0.1643
(95% relative)**, is the premium for picking the winner after seeing the answer.

The reason is measurable: **no method dominates.** Under clDice, all five candidates win on at
least one tile (global threshold 3, Sato 4, Frangi 3, Meijering 1, Sauvola 4); under IoU, four
of five do. Method ranking is unstable across frames, so a choice made on 8 frames does not
transfer to the ninth. Any paper that reports "our method achieves X" after trying several
methods on one corpus is quoting the hindsight number unless it says otherwise.
