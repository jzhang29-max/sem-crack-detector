# Publishing this: what comparable work does, and where this fits

> ### ⚠ SUPERSEDED by `VERDICT_2026.md`
> Written before the prior-art audit and the 2026 sweep. Its two headline
> recommendations — a dataset/benchmark paper and the weak-supervision framing — are
> both dead. The venue list and the standards-body pointers remain useful. Kept for the record.

## 1. What the comparable literature actually does

Papers in this space fall into four distinct types, and it matters which one you are writing.

**(a) Deep-learning crack/defect segmentation on SEM.** The dominant genre. Typical
shape: a U-Net (very often UNet-ResNet34) or Mask R-CNN, trained on a few hundred
to a few thousand annotated tiles, reported as IoU / Dice / precision-recall against
a held-out set. Examples: [crack segmentation in SEM of metal AM (IEEE)](https://ieeexplore.ieee.org/document/9874171/),
[hybrid SEM crack detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC8371647/),
[PGI-CrackNet for Ti-6Al-4V fatigue microcracks ~15 µm, in-situ SEM (Comput. Mater. Sci. 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0927025625001387),
[semantic segmentation for morphological fractography (Eng. Fract. Mech.)](https://www.sciencedirect.com/science/article/abs/pii/S0013794424003126),
[input-data ablation for crack segmentation on SEM + topography](https://www.sciencedirect.com/science/article/abs/pii/S1350630723007689),
[corrosion segmentation in npj Materials Degradation 2025](https://www.nature.com/articles/s41529-025-00633-3).
Two things they almost always report that you currently do not: **a µm/px scale**,
and **fields imaged at a controlled magnification**.

**(b) Dataset / benchmark papers.** Growing fast and a good fit for what you have.
[Rock microstructure SEM benchmark in *Scientific Data* (2025)](https://www.nature.com/articles/s41597-025-05947-0)
is the closest template — note it explicitly performs **magnification
standardisation** as a preprocessing step, which is exactly the issue in
`CORRECTION_scale_and_magnification.md`. See also
[EM3M electron micrograph dataset](https://arxiv.org/html/2508.16239v2) and
[MicroNet / pretrained microscopy encoders in npj Comput. Mater.](https://www.nature.com/articles/s41524-022-00878-5).

**(c) Human-in-the-loop / active-learning tool papers.** The canonical model here is
**Cellpose 2.0**, which showed a human-in-the-loop pipeline needs only 100–200
corrected ROIs to match a fully-annotated model. That is precisely your app's thesis.
Also [PyTAGIT](https://link.springer.com/article/10.1007/s00521-025-11725-1) and the
[active-learning/HITL survey](https://www.sciencedirect.com/science/article/abs/pii/S1361841521001080).

**(d) Quantitative fractography / stereology.** Where the measurement conventions
live: [ASTM C1322](https://store.astm.org/c1322-15r19.html) (fractography practice),
[ASTM STP1085 *Quantitative Methods in Fractography*](https://store.astm.org/stp1085-eb.html),
and standard point-count stereology, which fixes **how many fields you need for a
target relative accuracy** (fields required scale inversely with the volume
fraction) — see [Buehler's quantitative metallography primer](https://www.buehler.com/assets/solutions/technotes/vol1_issue5.pdf)
and [Michigan's point-count module](https://mse.engin.umich.edu/internal/lab-modules/microscopy-and-microstructure-analysis/manual-point-count).
This is the literature that would reject the pooled process comparison, and it is
also the literature that tells you how to fix it.

## 2. The honest inventory of what you have

| asset | strength | publishable now? |
|---|---|---|
| 70.4 M hand-corrected pixels over 62 frames, 2 alloy systems, 4 detectors | strong and rare | **yes**, as a data paper |
| **15 same-field CBS/ETD pairs** (identical HFW, verified) | genuinely unusual | **yes** — few datasets have this |
| Paired detector bias: CBS +3.88 pp, 8/8 pairs, p=0.008 | clean, controlled result | **yes** |
| Repeatability floor 1.87 pp from a true repeat field | quantifies measurement noise | **yes** |
| Magnification confound (249× HFW range) as a cautionary result | methodologically valuable | **yes** |
| `Tortuosity` < 1 in 54.5 % of cracks (pixel-count length bug) | affects any pipeline doing this | **yes**, as a short note |
| Two-population region structure; censoring 42.6–96.1 % | good methods content | **yes** |
| Human-in-the-loop correction app + active learning loop | Cellpose-2.0-shaped story | **yes**, as a tool paper |
| **AS vs Cast vs HIP process ranking** | 1–2 fields per route at matched HFW | **no** — see the correction |
| Transgranular vs intergranular | no grain-boundary data, no etch | **no** |

## 3. Where to publish — ranked

**1. Data + benchmark paper — best fit.** Target
[*Scientific Data*](https://www.nature.com/articles/s41597-025-05947-0) or
*Integrating Materials and Manufacturing Innovation* (IMMI, the MGI/FAIR-oriented
venue); *Data in Brief* as the low-friction fallback. Lead with the **paired-detector
subset** — "the same field of view labelled under two detectors" is a benchmark
almost nobody publishes, and it lets others quantify how much of their model's
performance is detector-specific. Report per-frame HFW and µm/px, and stratify.

**2. Measurement-uncertainty note — highest novelty per page.** Target *Materials
Characterization*, *Journal of Microscopy*, or *Ultramicroscopy*. The content is
already done: paired detector bias, the 1.87 pp repeatability floor, area fraction
being predominantly a segmentation-thickness metric (ρ=+0.86 with feature width),
tortuosity's dependence on width (ρ=−0.556) and the pixel-count bug. This is a
genuine service to the field and needs no new experiments.

**3. Tool paper.** [JOSS](https://joss.theoj.org) if you want it fast and
citable (reviews the software, not the science); *SoftwareX* if you want a
conventional indexed article. Frame it Cellpose-style: how few corrections are
needed to reach usable accuracy.

**4. A materials-science process-comparison paper — not yet.** It needs a matched
campaign: one HFW, one detector, ≥5 randomly-sited fields per route, ≥2 specimens
per route, and a stated loading axis. That is a few days of microscope time and
would turn the weakest part of this work into its strongest.

## 4. What to fix before any submission

1. **Attach µm/px to every frame.** The pipeline already crops the databar — it
   should parse HFW from it instead of discarding it. That is the single
   highest-value code change, and it is small.
2. **Report every comparison within a magnification stratum**, and say how many
   *fields* (not frames) each stratum holds.
3. **Fix the tortuosity/mean-width pixel-count bug** (background task is queued).
4. **Finish the review debt** — 9 frames carry >15 % crack area with <20 % of it
   human-confirmed; 16 frames have zero review.
5. **Settle the orientation confound** by re-imaging one specimen rotated 90°.
6. State the loading axis and specimen preparation direction.

## 5. Most effective ways for people to actually use this

Ordered by leverage:

1. **A Zenodo DOI for data + code, versioned**, cross-linked from the GitHub repo.
   Journals will ask for it and it makes the dataset citable independently.
2. **Ship the paired CBS/ETD subset as a named benchmark** with a scoring script.
   That is the piece other groups will reuse, because it answers a question they
   cannot answer with their own single-detector data.
3. **One-command reproduction.** `pip install` or a container, plus a `make demo`
   that runs on 2–3 bundled frames. Reviewers and users both bounce off setup.
4. **Ship the correction UI, not just the model.** The model is specific to these
   alloys; the *workflow* (detect → correct → retrain, with the promotion gate that
   refuses degenerate operating points) generalises. That is the reusable
   contribution.
5. **Publish the pre-trained model with its provenance** — threshold, held-out AUC,
   the specimen list it was trained on, and the fingerprint. You already record all
   of it.
6. **Write the measurement caveats into the tool's output**, not just the paper.
   The provenance JSON already carries the censoring note; add the HFW/µm-px and a
   magnification-stratum warning so a future user cannot repeat my mistake.
