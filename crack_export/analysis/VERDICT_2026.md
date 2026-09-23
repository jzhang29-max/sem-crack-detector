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
