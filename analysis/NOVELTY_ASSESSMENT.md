# Novelty audit: all six claims failed. What is actually left.

> ### ⚠ SUPERSEDED by `VERDICT_2026.md`
> This audit searched 2023–2025 only. The claims it left standing (weak-supervision
> reframing, scale-aware segmentation, the promotion gate) were subsequently killed by
> 2026 work. Kept for the record.

I ran an adversarial prior-art audit on the six things this work might claim. Each
agent was told to hunt for literature that *kills* the claim and to default to the
less-novel rating when uncertain — so the exercise is biased toward killing, and that
should be kept in mind. But the citations came back specific, checkable, and in
several cases decisive. **Nothing survived as novel.**

This supersedes my earlier assessment, which leaned on the paired-detector dataset
being unprecedented. It is not.

## 1. The verdicts

| claim | verdict | what kills it |
|---|---|---|
| Paired same-field two-detector labelled crack dataset | **prior art** | Schmies et al. 2023; RODARE 2025 |
| Detector swap shifts crack area fraction (+3.88 pp) | **known, thinly documented** | Salvato et al. 2023; Mills 2011; Scrivener 2004 |
| Area fraction is really a width metric | **prior art + tautology** | it is an algebraic identity; OCTA/geotech/bone precedent |
| Human-in-the-loop correct-and-retrain tool | **prior art** | ilastik, Weka, RootPainter, Cellpose 2.0, micro_sam |
| Magnification pooling gives spurious ranking | **prior art** | Liu et al. 2005; Ortega et al. 2006; Bonnet et al. 2001 |
| Pixel-count skeleton length → tortuosity < 1 | **prior art** | Dorst & Smeulders 1987, §4.1 |

### The three that hurt most

**Schmies, Hemmleb & Bettge, *Eng. Fail. Anal.* 154:107814 (2023)** already annotated
**crack** features on the **same field** in SE + BSE + BSE-derived topography of
fatigue-cracked metal, and ablated which detector channels help. That is the exact
intellectual content, in the exact domain. **This paper was in my own first search
results and I listed it without recognising what it did.** That is my error, not the
audit's. Add `RODARE 10.14278/rodare.4124` (Nov 2025, public CC-BY: 13 same-field
two-detector labelled steel pairs) and *npj Comput. Mater.* 11 (2025), which does
same-region two-detector imaging with labels **on 316L** and runs an explicit
SE→BSE cross-detector generalisation test.

**Dorst & Smeulders (1987) §4.1** contains, verbatim: *"this is also the length
estimate one obtains when one simply counts the number of pixels on an 8-connected
contour. This estimate is consistently too low."* The cos(θ) relation I reported as
an empirical finding (ρ = −0.648) is the **closed-form definition** of that
estimator's bias, published in 1987. The 1/√2 = 0.707 floor is stated there too.

**Area fraction ≡ extent × width is an identity, not a finding.** For any binary
mask, `A_A = L_A · w̄` where `w̄ ≡ A/L`. I verified it on my own data: max absolute
error 5.6 × 10⁻¹⁷ over 62 frames — exact. And my distance-transform width proxy
ranks at Spearman +0.922 against `A/L`. So reporting ρ(area, width) = +0.862 >
ρ(area, extent) = +0.712 was **substantially tautological** — a statement about which
factor carried more variance in this corpus, not about the metric's nature. The
decomposition is already standard practice as OCTA's VAD/VSD/VDI triple, geotechnics'
CIF/length/aperture, and bone's BV/TV ≈ Tb.Th × Tb.N.

Also: **Liu et al., *Eng. Geol.* (2005)** ran a magnification ladder on SEM images and
found planar porosity *and* box-counting fractal dimension strongly
magnification-dependent — my §3 and §6 findings, twenty years early.

## 2. Two things I verified myself, and they matter

**(a) The paired frames are acquisition-matched.** The audit's sharpest methodological
objection was that my +3.88 pp "detector effect" might be a beam-settings effect,
since Salvato et al. showed kV/current/magnification changes move segmentation-derived
porosity by up to 30 %. I read the databars for all 16 paired frames: **every pair is
10.00 kV, 1.6 nA, 6.0 mm working distance, and matched HFW.** The detector is the only
printed parameter that differs. Brightness/contrast/gain/dwell are not printed, so the
defence is strong but not airtight — and on this instrument family multi-detector
channels are typically acquired in a single simultaneous raster, which would make even
dwell shared.

**(b) The non-tautological version of the decomposition survives, and it reverses my
earlier reading.** The identity says area fraction *must* factor into extent × width.
What is *not* implied by the identity is **which factor a given nuisance variable
moves.** Tested on the 8 registered pairs — same field, matched beam:

| | CBS / ETD ratio | direction | Wilcoxon p |
|---|---|---|---|
| crack **extent** (skeleton length per unit area) | **2.042** | CBS longer in 7/8 | **0.016** |
| crack **width** (A/L) | **0.605** | CBS wider in only 1/8 | 0.039 |

So CBS's higher area fraction comes from **finding roughly twice as much crack
length**, while its features are *narrower* — consistent with resolving finer detail,
not blurring. This also corrects my own earlier framing, where I treated high area
fraction as blobbiness. For the detector comparison the mechanism is extent, not
thickness.

## 3. So is it a useful contribution?

**Not as a novelty claim on any single result.** Every phenomenon is known. If you
submit any of the six as a discovery, you will be shown the citation above.

**Yes as a propagation audit, which is the one framing no prior art occupies.** Each
of these biases is documented *in isolation*, in six different literatures (digital
geometry 1987, cement microscopy 2004, geological fracture scaling 2001–2006, SEM
porosity 2023, ophthalmic imaging 2016, bioimage tools 2011–2025). Nobody has
assembled them into one uncertainty budget on one corpus and shown **what they do
together to a materials conclusion.** Here they inverted a process ranking — the
`Cast ≫ AS ≈ HIP` result I reported, which does not survive magnification matching.

The defensible sentence is: *"biases solved decades ago in digital geometry and
stereology are still being reintroduced in modern SEM crack-quantification pipelines;
assembled on one corpus they are large enough to invert a material ranking, and here
is the control for each."*

That is a real service and a citable methods paper. It is not a high-impact result,
and it will not be mistaken for one.

## 4. What would make it genuinely new — two cheap experiments

Both were independently recommended by more than one audit agent, and both convert
re-analysis-with-confounds into controlled measurement.

**(i) A magnification ladder on a fixed field.** Re-image the *same* field at 4–6 HFWs
spanning the corpus range, on both detectors. Fit area fraction against log(HFW) per
specimen and report **the scaling exponent**. This removes the field-selection
confound that currently makes my pooled-vs-matched comparison uninterpretable, and if
`d(area)/d(log HFW)` differs *between materials* that is a genuinely new, scale-aware
damage descriptor rather than a cautionary tale. Cost: hours of microscope time.

**(ii) Simultaneous dual-channel acquisition with independent per-channel
annotation.** One raster, both detectors, then annotate each channel *blind to the
other*. That is the only design that fully isolates the detector, and it produces
something the prior art genuinely lacks: RODARE merged its two channels into a single
label set, and Schmies used the channels as CNN inputs rather than annotating each
independently. Independent per-channel labels would let you separate
**detector-induced signal change** from **annotator response to detector**, which
nobody has done. Cost: a day of imaging plus labelling.

With (i) and (ii) this becomes a measurement paper with a controlled design. Without
them it is a well-executed cautionary re-analysis.

## 5. Revised recommendation

1. **Methods / measurement note** — *Materials Characterization* or *J. Microscopy*.
   Frame as propagation audit. Cite all prior art up front; claim the assembly, the
   numbers, and the controls. Ready now.
2. **Data release** — Zenodo DOI, plus *Data in Brief* if you want a citable article.
   Claim utility, **not** design novelty; cite RODARE and Schmies in the descriptor.
3. **Tool** — demote to a Methods subsection, or JOSS at most, explicitly as "we adopt
   the standard corrective-annotation pattern (ilastik, Trainable Weka, RootPainter,
   Cellpose 2.0, micro_sam)." Reviewers punish unacknowledged reinvention far harder
   than acknowledged reuse.
4. **Do experiments (i) and (ii) before aiming higher than a methods note.**

## 6. Retractions from my own earlier reports

- "No comparable public dataset exists" — **false** (RODARE; Schmies 2023).
- ρ(area, width) > ρ(area, extent) as evidence about the metric — **tautological**;
  the identity is exact to 5.6 × 10⁻¹⁷ on this data.
- The cos(θ) tortuosity relation as an empirical finding — **it is a 1987 identity**.
- The magnification-pooling inversion as novel — **Liu et al. 2005**.
- High area fraction implies blobby over-detection — **for the detector comparison the
  opposite holds**: CBS reads higher via 2.04× more extent, with *narrower* features.
