# 2026 verdict: one survivor, and one genuinely new gap

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

## 2. The one genuinely new gap — and it came out of measuring your labels

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

**(b) Pair arithmetic, stated cleanly before a reviewer asks.** There are **16**
index-matched CBS/ETD pairs. **15** have readable databars and all 15 share HFW
*exactly*. **8** have mask-overlap-confirmed registration (Jaccard ≥ 0.5), and those 8
carry the p = 0.0078 result; the other 8 were excluded rather than averaged in. Also:
16 pairs from 8 specimens is not 62 samples — aggregate per specimen.

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

## 6. Foundation models — **this section was wrong; corrected 2026-09-15 by measurement**

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

**P3** — if you want a methods paper, it is the **superset-label** problem in §2, not S4.

**Delete:** S4 as framing, S6 as contribution, the 25 MP differentiator, and any claim
that competitors merely "fail to standardise scale."

## 8. Verification note

I resolved all five killing DOIs through Crossref and all five are real, with titles and
dates as stated. The remaining unverified item is Legland, *F1000Research* 15:1438 — check
it yourself before deleting S3 on its authority.
