# 2026 verdict: one survivor — a direction, not a significance — and no new gap

*Title corrected 2026-09-22. It read "one survivor, and one genuinely new gap"; §2 withdrew
the gap on 2026-09-18 (it is owned: box supervision IS superset supervision), and §S1(b) now
records that the survivor's p = 0.0078 is the exact test's floor at n = 8 and falls to
p = 0.2500 under the per-specimen aggregation this document prescribes. What survives is a
unanimous direction on 8/8 pairs across 3 specimens, median +3.88 pp.*

Seven-angle sweep of 2026 / post-Nov-2025 literature against the six things that
survived the earlier audit. Your instinct to push on the dates was right — 2026 killed
or gutted **five of six**. All the killing DOIs below are **verified real via Crossref**,
not relayed on an agent's word.

## 1. Survivor table (as of 27 Aug 2026)

| | claim | status | decided by |
|---|---|---|---|
| **S1** | Paired CBS/ETD measurement on identical fields | **STANDS — only full survivor** | Clean on all 7 angles. No 2026 paper images one field with two detectors and reports a statistically tested difference in a *measured damage metric*. |
| S2 | Assembled uncertainty budget that inverts a ranking | **survives only as integration** | Every component now separately published — incl. *Materials Characterization* 235 (**May 2026**) `10.1016/j.matchar.2026.116319`, "Curvature-driven quantification of 2D SEM fracture morphology across metallic materials", which opens on your exact premise and reports published fractographic conclusions that fail scale normalisation. |
| S3 | Tortuosity null | **one subsection, not a paper** | Bias measurement is prior art (Legland, *F1000Research*, Aug 2026; Abdalla et al. Dec 2025). The *angle-matched null as an inferential device* is still unoccupied. |
| S4 | Weak-supervision reframing | **DEAD** | Unanimous across 5 angles. ACM TODAES `10.1145/3780101` (**28 Jan 2026**) — weakly-supervised high-precision defect segmentation **in real SEM**, fully-supervised accuracy at 10% labelling. Plus MSEA `10.1016/j.msea.2026.150170` (Jun 2026), *Mater. Charact.* `10.1016/j.matchar.2025.115718` (Dec 2025, cross-microscope round robin), TACoS (arXiv:2607.07169), SeSAM, MatStudio (Ni alloys, human loop), Helixnet. |
| S5 | Promotion gate | **DEAD as a claim** | Never scooped — it was never a contribution. Worse, its *context* collapsed: Teuber et al. (Mar 2026) and Convpaint (*Cell Reports Methods*, Mar 2026) benchmark your exact architecture (hand-crafted filters + shallow classifier + interactive paint) against pretrained-feature classifiers. |
| S6 | Scale-aware segmentation | **DEAD, all three legs** | CIMP arXiv:2604.24909 (Apr 2026) conditions on 7-D acquisition metadata **and shows all seven parameters are linearly recoverable from the frozen visual embedding** — networks already encode their own scale. Plus "One Model to Magnify Them All" (arXiv:2608.09403, Aug 2026), NanoPSD scale-bar OCR as a shipped feature. |

## 2. ~~The one genuinely new gap~~ — **WITHDRAWN 2026-09-18: this gap is owned**

> ⛔ The claim below is false, and it refutes itself two lines down: it lists *"boxes around it"*
> as a **subset** assumption, but a bounding box asserts the object is **inside** a region —
> a superset label by construction, and box-supervised segmentation is a mature literature.
>
> Prior art, verified against source 2026-09-18: the formal setting is **Superset Label
> Learning** (Liu & Dietterich, ICML 2014, PMLR 32:1629–1637); the segmentation machinery is
> **box supervision with a tightness prior** (Kervadec et al., MIDL 2020, PMLR 121:365–381,
> arXiv:2004.06816 — every line inside the annotated region must contain ≥1 foreground pixel,
> which is exactly what a 59 px brush over a 3 px crack asserts); and the crack domain has
> already done the over-inclusive case with a **shrink module** (*Unified weakly and
> semi-supervised crack segmentation framework using limited coarse labels*, Eng. Appl. Artif.
> Intell. 2024, 10.1016/j.engappai.2024.108497 — IoU 77.53%, +28.64 pp over
> fully-supervised-on-coarse-labels). Full writeup:
> `../../crack-depth-3d/docs/SUPERSET_CLAIM_CLOSED.md`.
>
> What survives is the **measurement**, not a methods gap: median stroke 59 px (max 413) against
> a ~3 px crack, 91% of 70,434,978 labelled px from strokes >80 px. That makes pixel-IoU against
> these labels close to meaningless, which is worth reporting as a corpus property.
>
> Original text follows.

Every 2026 weak-supervision method assumes the weak label is a **SUBSET** of the object:
scribbles *inside* it, points *on* it, boxes *around* it. Your labels are **SUPERSETS** —
a median 59 px (max 413 px) brush asserting a region that *contains* a ~3 px crack.

That inversion breaks the core assumption of the entire 2026 weak-supervision
literature. Exhaustive arXiv enumeration on label thickness / annotation width / label
dilation / dense-but-imprecise / region-level supervision returned **nothing** in the
window.

**This is the only live methods question in the project**, and it exists because I
measured your correction masks rather than trusting the 70.4 M headline. Sparse-but-
accurate → dense-but-wrong is a real problem statement, and nobody owns it.

## 3. Two threats that kill S1 in review unless addressed

**(a) Circularity — the biggest risk in the project.** The 2.04× extent difference is
measured *by a detector trained on 91 %-broad-brush labels*. If your segmenter is
contrast-sensitive and CBS renders cracks with more contrast, you have measured **your
segmenter's contrast response**, not crack extent. Fix: recompute the paired difference
with a segmentation-free or recall-matched estimator, and on fine hand-traced labels.

Mandatory citation and control: *Microsc. Microanal.* `10.1093/mam/ozag022`
(**30 Mar 2026**), a paired in-lens vs Everhart–Thornley study — verified as a
**contrast-mechanism** study on MoS₂ layers (polarity reverses with WD ≥ 15 mm) that
measures no material quantity. So it is not a kill, but it obliges you to rule out a
detector-response explanation. Your matched 6.0 mm WD helps — say so explicitly.

**(a2) THE 8/8 FILTER CONDITIONS ON THE OUTCOME — and the claim is better without it.**
*2026-09-23.* `paired_detector.py:70` computes Jaccard from `(A & B)`, the two masks whose
areas are being compared, and line 80 keeps `Jaccard >= 0.5` under the heading
"registration confirmed". It is not a registration test. It selected on agreement between
the quantities under comparison, and it removed **exactly the three pairs that disagree**
(AS_0001 −1.18 pp, AS_0006 −2.42 pp, HIP_0008 −0.27 pp).

Recomputed on all **16** index-matched pairs:

| | J>=0.5 subset (published) | all 16 pairs |
|---|---|---|
| same sign | 8/8 | 13/16, exact sign test p = 0.0213 |
| median delta | **+3.88 pp** | **+1.51 pp** |
| per specimen | "unanimous" | AS geomean **0.771x (reversed)**, Cast 1.646x, HIP 1.797x |

+1.51 pp is **below this repo's own 1.87 pp repeat-field repeatability floor**
(`CRACK_ANALYSIS.md`). So the percentage-point framing loses to its own noise, and one of
three specimens reverses. **Do not quote +3.88 pp, 8/8, or p = 0.0078 again.**

**(a3) THE EFFECT IS REAL, AND MUCH STRONGER, ONCE THE SEGMENTER AND THE OPERATOR'S GAIN
ARE REMOVED.** The failure above is of the *statistic*, not of the phenomenon. Crack area
fraction depends on the segmenter, on the labels it was trained on, and on the per-channel
brightness/contrast the operator set independently (CBS Contrast=45.5, ETD 73.5). Replace
it with a MAD-normalised dark-tail fraction — the share of pixels at z <= -2 about each
image's own median, which no gain setting can move — and score **every** pair, not a subset:

| | value |
|---|---|
| paired fields with both detectors | **56** across **7** specimen cells, not 16 across 3 |
| CBS dark-tail > ETD | **51 / 52** usable pairs, geometric mean **2.74x** |
| pooled sign test | p = 2.4e-14 |
| **specimen-level**, 7 cells, exact sign test | **7/7 positive, p = 0.0156** |
| robustness, pairs with <1% black clipping | 39/40, geomean 2.73x |

The specimen-level test is the one this document demands, and at 7 cells it can clear 0.05
— at n = 3 the floor was 0.25 and no result was reachable. The extra 40 pairs are the
2026-09-15 hydrogen batch, which is **simultaneous single-raster dual-channel**: all 40
carry identical `Date`, `Time`, `StageX`, `StageY` and `HFW` in their FEI TIFF tags for
both channels.

Two corrections that come with it. These are **BSE (CBS) vs SE (ETD)**, per
`[Detectors] Signal=BSE` / `Signal=SE` — not two SE detectors, which changes which prior
art applies. And the 16 originally analysed files were re-saved through `tifffile` with all
FEI metadata **stripped**, so the stated "10.00 kV / 1.6 nA / 6.0 mm WD" cannot be verified
from them and is contradicted by the sibling series, which records **HV = 30000, WD =
0.010** (30 kV, 10 mm). Re-source those conditions or drop them.

**(a4) THE MECHANISM: SE EDGE BRIGHTENING. Measured, and it corrects my own first answer.**
*2026-09-24.*

I first reported this as contrast-to-noise rather than darkness, on the grounds that the raw
crack/matrix intensity *ratio* was not significant while a MAD-normalised depth was. **That
was wrong, and the error was the noise proxy.** Whole-image MAD is not noise — it is total
spread, and it includes the crack population and large-scale shading. Measured properly, as
the standard deviation of a high-pass residual in matrix more than 60 px from any marking,
the two channels differ by only **1.09x** (5/7 cells, p = 0.45). Noise is not the story.

What actually differs is depth, against the bulk:

| quantity, 49 pairs | median CBS/ETD | pairs | cells | p |
|---|---|---|---|---|
| crack depth vs **far** matrix | **2.28x** | 46/49 | 7/7 | 0.0156 |
| matrix noise (high-pass, clean matrix) | 1.09x | 34/49 | 5/7 | 0.45 |
| contrast-to-noise | **2.42x** | 49/49 | 7/7 | 0.0156 |

Depth carries **93%** of the CNR log-ratio. So CBS resolves the crack against the bulk
better, and it is not because ETD is noisier.

**And there is a direct physical reason, which is the part worth publishing.** A
crack-normal profile using the *local* surround as its baseline shows no difference at all
(depth 1.04x, ISO50 width 1.00x, 3/7 and 4/7 cells, both p = 1.0). That null is not a
failure to find an effect — it locates it. The two baselines disagree because the region
immediately outside the crack is detector-dependent:

| halo 3-15 px outside the crack, vs far matrix, in each channel's own noise units |
|---|
| **CBS (BSE): -1.68** — the surround is *darker* than bulk. 2/51 pairs positive |
| **ETD (SE): +3.92** — a *bright lip*. 50/51 pairs positive |
| brighter in ETD than CBS: **51/51 pairs, 7/7 cells, p = 0.0156** |

Averaged crack-normal profiles show it directly (figure:
`~/Desktop/_Outputs/MAR_crack_identifications/figures/detector_mechanism_crack_profile.png`,
median over 51 field pairs, narrow cracks only, IQR shaded, matrix at zero): BSE reaches a
trough of **-7.54** noise units with a flat surround peaking at only +0.29, while SE reaches
**-5.32** and rises into a lip peaking at **+0.95** on the flanks. Shallower trough, brighter
shoulder -- the same ordering the annulus measurement gives, on a different statistic. (The
two sets of numbers are not interchangeable: the annulus pools pixels 3-15 px out over all
cracks, the profile is a median trace over narrow cracks with its baseline at +/-52-60 px.)

This is secondary-electron edge brightening, which is textbook SEM physics: an edge emits
more secondaries, so the crack lip glows in SE and does not in BSE. It explains the whole
pattern. The SE lip raises the local baseline, so any measure referenced to the immediate
surround sees the two channels as equivalent, while the crack-to-bulk contrast that a global
method actually uses is 2.3x better in BSE.

**The consequence was predicted, then tested, and it is WEAK.** The lip should make a
local-contrast method -- which references the surround -- see the two channels as more alike
than a global method does, because the lip raises the local baseline on SE only. Tested with
both methods at the SAME absolute sensitivity, 3 noise units below their own reference, so
detected area is free to differ:

| method, 46 pairs | median CBS/ETD area | pairs | cells |
|---|---|---|---|
| global, references the bulk median | **2.01x** | 45/46 | 7/7 |
| local, references a 51 px surround | **1.84x** | 52/55 | 7/7 |

The gap is in the predicted direction but does not clear significance: global exceeds local
in 30 of 46 pairs, **p = 0.054**. So both families favour CBS by about 2x, and the lip does
not buy a local method much immunity at this sensitivity. The mechanism is solid; the "it
predicts which methods will disagree" version of it is not supported, and an earlier draft
of this subsection asserted it. Do not claim methods will disagree.

**A trap worth naming, because it cost the first attempt at this test.** The obvious way to
compare methods -- threshold at a percentile of the response -- cannot show a difference in
detected AREA at all, because a percentile selects a fixed fraction of pixels by
construction. The first run gave CBS/ETD = 1.00 for both a 2nd-percentile global cut and a
98th-percentile ridge cut, which is not a null result, it is an identity. The same trap
produced a spurious 5.8% off-specimen reading elsewhere in this work. Any comparison of how
much a method finds has to fix an absolute criterion, never a quantile.

**(a4-prior) ⛔ SUPERSEDED THE SAME DAY BY (a4) ABOVE. Kept because the measurements are
real and the error is instructive; the CONCLUSION below is wrong.**

What is wrong with it: it treats the MAD of the whole image as "noise". MAD is total spread
-- it contains the crack population and any large-scale shading -- so ETD's 2.67x larger MAD
was read as ETD being noisier. Measured properly, on a high-pass residual in matrix >60 px
from any marking, the channels differ by 1.09x and noise explains nothing. The raw *ratio*
in the bottom row is also the wrong contrast statistic: crack depth is an additive quantity
against the bulk, and measured that way it is 2.28x at 7/7 cells.

So the sentence at the end of this subsection -- "not that the crack is absolutely darker",
"do not write the broader sentence" -- is the opposite of what the data support. Read (a4).

*2026-09-24, as originally written.* The control a referee will ask for: take the region BOTH channels' masks agree
is crack, and measure its depth **relative to each channel's own matrix**, so any global
contrast difference cancels. On 49 pairs with an agreed region:

| normaliser | pairs | specimen cells | p |
|---|---|---|---|
| MAD of the image | **49/49** | 7/7 | 0.0156 |
| IQR of the image | **49/49** | 7/7 | 0.0156 |
| **none — raw crack/matrix intensity ratio** | **32/49** | 6/7 | **0.1250** |

Median separation is 9.68 MAD, and it is crack-localised: the matrix term cancels by
construction. But the bottom row is why this has to be worded carefully. **Without a
normaliser the effect does not reach significance.** ETD carries 2.67x the spread of CBS
(median MAD 23.7 vs 8.9), so what is being measured is how far the crack sits from the
matrix *in units of the image's own variability* — contrast-to-noise — not that the crack is
absolutely darker.

That is the physically right quantity, because CNR is what decides whether a segmenter or a
human finds the feature at all, and it is the mechanism that connects to escape depth
(Taufique 2025, `10.1038/s41524-025-01801-4`). It is also a narrower claim than "CBS reads
the crack darker", and the narrower one is the one the data supports. Do not write the
broader sentence.

**(a5) THE CRACK INTERIOR IS SATURATED BLACK, on both channels, and nobody reports this.**
*2026-09-24.* Black clipping is real here, not an artefact of taking a minimum: the histogram
has a hard spike at zero, 280,245 pixels against 11,828 in the next bin on one frame, a 24x
step. Measured over 51 pairs inside the detected field of view:

| | whole frame | **inside the crack region** |
|---|---|---|
| CBS (BSE) | median 1.08%, mean 3.51%, max 30.9% | **median 59.6%**, max 99.6% |
| ETD (SE) | median 1.02%, mean 1.85%, max 20.7% | **median 45.6%**, max 98.5% |

**Roughly half to two thirds of the pixels inside a crack are pinned at the sensor floor.**
Every intensity-derived crack quantity in this corpus is therefore censored, not measured:
depth, contrast, any ISO50 width taken from a trough that has no bottom. It also means the
crack's true darkness is unknown and unknowable from these files -- only a lower bound on it
exists. This is a property of the acquisition, not of the analysis, and it applies to the
released 62-frame corpus as much as to the new batch.

**It does not manufacture the detector effect; it suppresses it.** Clipping correlates
*negatively* with the measured CBS advantage (Spearman rho = -0.26, p = 0.065), which is the
expected direction: a truncated trough understates depth, and CBS clips more inside the crack
(59.6% vs 45.6%), so CBS is the more truncated of the two. Restricting to the 20 pairs with
under 1% whole-frame clipping in both channels:

    CBS CNR higher in 20/20 pairs, median 2.48x, 6/6 specimen cells, exact p = 0.0312

So the effect is larger and cleaner on the unclipped subset than on the full set, and the
2.28x depth ratio reported in (a4) is a **lower bound**.

**Practical consequence, and the one concrete thing to change at the microscope.** Re-acquire
with the black level set so that no pixel reaches zero -- anchor the histogram at the 1st
percentile rather than at the floor. Until that is done, no absolute crack depth or width
from this corpus should be published, and any threshold-derived area metric inherits an
unquantified censoring that varies between channels.

**(a7) THE STRICT VERSION, and it is the one to quote.** *2026-09-24.*
Auditing for degenerate values -- prompted by the 4.4e9 in (a6) -- showed the problem was not
confined to geometry. **13 of 56 pairs cannot produce a valid statistic**, and the earlier
runs absorbed them with a `max(x, 1e-9)` clamp instead of rejecting them, which converts a
non-positive ETD contrast into a huge positive ratio. That is where 4.4e9 came from.

Rejected, with reasons:

| reason | pairs |
|---|---|
| the ETD "crack" is **brighter** than matrix (depth -3 to -71) | 4 |
| agreed crack region under 5,000 px (a near crack-free cell) | 6 |
| the two channels' masks are different sizes | 2 |
| too few usable pixels after excluding clipped ones | 1 |

Re-run rejecting all of them, with **no clamp anywhere**:

    CBS CNR higher in 43/43 retained pairs   median 2.41x   range 1.08 - 14.44
    7/7 specimen cells positive              exact p = 0.0156

Every retained pair has a genuine positive crack contrast in both channels, so the ratio is
well defined throughout and the range no longer contains anything impossible. This is
stricter than the 49/49 and 51/52 figures quoted above and gives the same answer, which is
the useful part: **the result does not depend on how the bad pairs are handled.**

Two things it does expose. `MAR_AmbB_HIP` falls from 10 pairs to **2**, because that cell is
nearly crack-free and the two channels barely agree on anything to measure -- so one of the
seven cells in the specimen-level test rests on two fields, and the test should be reported
with that stated. And the four pairs where the SE "crack" is brighter than its matrix are not
a measurement failure; they are the detector marking something in SE that is not a dark
feature at all, which belongs in the paper as a separate observation about false positives.

**(a6) MASK GEOMETRY: two pairs are ill-defined, and one of them returns 4.4e9.**
*2026-09-24.* The 142 exported masks come in **20 distinct sizes**. That sounds like an
inconsistent crop and it is not: the mask size equals the per-image detected field of view in
**20 of 20** of the non-standard cases. The originals themselves vary -- 3067x2044,
3068x2044, 3072x2045, 6139x4093 -- because they were exported at slightly different sizes.
The rule is consistent; the inputs are heterogeneous. For a data descriptor this is a
documentation item (state that mask dimensions follow the detected FOV), not a defect.

The real defect is narrower. **Two paired fields have masks of different size in the two
channels**, which makes a pixel-wise paired comparison undefined:

    MAR_Amb_AS_0001     CBS (2952, 2952)  vs  ETD (3036, 3037)
    MAR_Amb_Cast_0001   CBS (2970, 2971)  vs  ETD (3041, 3042)

Cropping both to the smaller extent, as every paired analysis here does, silently assumes
they are co-registered at the top-left corner. They are not. `MAR_Amb_AS_0001` then returns
a CNR ratio of **4.4e9** -- a division by an ETD contrast of essentially zero, produced by
comparing misaligned regions. It survived unnoticed because every statistic in this work is
a median, which absorbs it; a mean anywhere in the chain would have been destroyed by it.

Nothing rests on them: the headline is identical with and without. 49/49 pairs, median 2.45x,
7/7 cells, p = 0.0156 including them; **47/47 pairs, median 2.45x, 7/7 cells, p = 0.0156**
excluding them. But they should be excluded explicitly rather than left to the median, and a
released dataset must not ship a pair whose two channels cannot be overlaid.

One correction to an earlier review of this: `MAR_Amb_AS_0001` was **not** inside the
"registration-confirmed 8/8" -- its Jaccard is 0.479, below the 0.5 cut. `MAR_Amb_Cast_0001`
was. And the index-0010 overview pairs, flagged elsewhere as 2048-vs-2188 rows, agree exactly
at (3072, 2048) once the FOV detector has run; the raw TIFFs differ, the masks do not.

**(b) Pair arithmetic, stated cleanly before a reviewer asks.** There are **16**
index-matched CBS/ETD pairs. **15** have readable databars and all 15 share HFW
*exactly*. **8** have mask-overlap-confirmed registration (Jaccard ≥ 0.5), and those 8
carry the p = 0.0078 result; the other 8 were excluded rather than averaged in.

> ⛔ **Two corrections to this paragraph, 2026-09-22, and the second one costs the
> significance.**
>
> **The 8 confirmed pairs come from 3 specimens, not 8.** Grouped by the pipeline's own
> `aggregate.specimen_key()`: `MAR_Amb_AS` (2 pairs), `MAR_Amb_Cast` (4), `MAR_Amb_HIP` (2).
> The "8 specimens" figure belongs to the whole 62-frame corpus, not to these pairs.
>
> **p = 0.0078 is the floor of the test, not a measurement.** At n = 8 the smallest two-sided
> exact signed-rank p is 2/2⁸ = 0.0078125, which is exactly the reported value. It says
> "all 8 deltas had the same sign" and nothing more; no arrangement of these 8 numbers could
> have produced a smaller one.
>
> **And this sentence's own instruction destroys it.** "Aggregate per specimen" applied to
> these pairs gives 3 per-specimen medians (+1.453, +6.139, +2.957 pp) and an exact
> two-sided p of **0.2500** — again the floor, this time at n = 3. So the effect **does not
> reach significance under the aggregation this document tells the reader to use.**
>
> What survives is the direction and its consistency: CBS reads higher than ETD on
> **8 of 8** registration-confirmed pairs and on all three specimens, median **+3.88 pp**.
> That is a real, unanimous, same-field observation at n = 3 specimens. It is not a
> significance result, and §S1 should not be described as one.

## 4. A retraction of my own claim from yesterday

I listed "full-frame 25 MP inference, no tiling" as a win over every competitor.
**That is dead.** MegaSeg (*Medical Image Analysis*, online **10 Jan 2026**) does
end-to-end 8192×8192 = **67 MP** segmentation with no patch-wise inference and no
downsampling, and shows FROC rising 0.78 → 0.89 purely from global context. Remove it
from the advantages list.

## 5. Is what's left worth publishing?

**Yes — one paper, and it is metrology, not method.** Every method claim is dead. What
remains is legitimate and citable: *the detector channel is a measurement variable*,
evidenced on index-matched pairs, wrapped in a budget that flips a material conclusion.
Target *Materials Characterization*, *Microscopy and Microanalysis*, or *Ultramicroscopy*.
It will be rejected at a CV venue on novelty.

The field moved past your **methods**, not your **measurements**. And one thing 2026 made
*more* valuable: as foundation-model segmenters flood materials microscopy, "how much of
a reported crack quantity is instrument, grid and annotation rather than material?"
becomes more load-bearing, not less. There is also live **VAMAS** (Jan 2026 call for
participation) and **ASTM E04.14** activity — a standards-adjacent audience that a
novelty-hunting reviewer does not gatekeep. Cheapest visibility available.

## 6. Foundation models — **my 2026-09-15 "correction" was itself invalid; the original may have been right**

> ⛔ *2026-09-18.* On 2026-09-15 I overwrote this section's original claim (*"SAM-family models
> remain poor on thin low-contrast curvilinear structures"*) with a measurement that **leaked the
> label into the model input** (green channel of the annotated overlay; a bare threshold scores
> recall 1.000 on 16/16 — see `sam3/SAM3_ON_SEM_CRACKS.md`). The original statement is therefore
> **not** disproven, and stands unrefuted. Treat everything below as withdrawn.


> **Superseded.** This section originally read: *"Zero-shot: no. Nothing in 2026
> demonstrates usable zero-shot SEM crack segmentation; SAM-family models remain poor on
> thin low-contrast curvilinear structures."* **I then ran SAM 3 on 16 hand-labelled tiles
> from this corpus and that claim is false.** See `sam3/SAM3_ON_SEM_CRACKS.md`.

**Zero-shot: bimodal, and good when it engages.** Measured on 16 tiles from the 9
fine-stroke frames: **recall 0.969–1.000 on 10/16 tiles** (median 0.979), **0.000 on 6/16**.
Where the hand label is most complete (4.8–10.2 % of tile) SAM 3 scores **IoU 0.59–0.74**,
comparable to the 62.34 % native-SAM3 crack IoU that CoRe-SAM3 reports on infrastructure
data. SEM is not dramatically harder than concrete for SAM 3.

Two caveats that matter more than the headline:

1. **It fails silently.** 6 of 16 tiles are total misses and nothing in the output
   distinguishes them from the successes. So it is a strong baseline, not a trustworthy
   unsupervised measurement instrument.
2. **The prompt is brittle, and this is a new publishable finding.** Zero instances
   returned on 14/16 tiles for "a crack in metal" and "thin dark line", and on **16/16
   for "fracture"**. Only the bare noun `crack` works. Prompt wording is an unguessable
   researcher degree of freedom; any SAM-3 microscopy number reported without a prompt
   ablation is reporting luck with vocabulary.

Your classical detector is therefore **not** safe as infrastructure on the grounds
previously given. It is a baseline that a prompted foundation model already matches or
beats on the tiles where that model fires.

**From coarse prompts: effectively yes.** SeSAM takes coarse masks, decomposes them,
samples prompts along *skeletons*, and iteratively refines pseudo-labels — precisely the
move a 59-px stroke over a 3-px crack needs.

Consequence for the detector: **it is now a baseline you must defend, not a result.**
Swap the 8 hand features for frozen DINOv2/v3 features behind the *same* LogisticRegression
— about a day of work, and the experiment a reviewer will demand. Report it whichever way
it falls; the paper is not about the detector. Note also *Materials Characterization*
(2026) already does DINOv2-based SEM fractography classification at 96.3 % accuracy.

## 7. Revised plan

**P0 — de-risk S1 (~2–3 weeks). Nothing else until this is done.**
1. Hand-trace fine (≤5 px) crack outlines on **the 15 paired fields only** — ~2–4 h each.
   Small, and it serves the precise-label set, the circularity fix and the metrics at once.
2. Recompute the CBS/ETD difference segmentation-free or recall-matched, and on the fine
   labels, reported alongside the model-derived number.
3. Rule out contrast-mechanism; cite `10.1093/mam/ozag022`; state the matched WD.
4. Aggregate per specimen; fix the 16/15/8 reporting.

**P1** — assemble S2 as the uncertainty budget, with S3's null and µm/px matching as
*controls inside it*, not as separate contributions.

**P2** — run frozen DINOv2/v3 + same LogisticRegression as the demanded baseline.

**P3** — ~~if you want a methods paper, it is the **superset-label** problem in §2, not S4.~~
**Withdrawn 2026-09-18**: §2's gap is owned (see its banner). There is no methods paper here.

**Delete:** S4 as framing, S6 as contribution, the 25 MP differentiator, and any claim
that competitors merely "fail to standardise scale."

## 8. Verification note

I resolved all five killing DOIs through Crossref and all five are real, with titles and
dates as stated. The remaining unverified item is Legland, *F1000Research* 15:1438 — check
it yourself before deleting S3 on its authority.
