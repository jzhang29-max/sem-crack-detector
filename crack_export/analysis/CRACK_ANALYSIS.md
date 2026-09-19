# Crack analysis by set

> ### ⚠ CORRECTION — read `CORRECTION_scale_and_magnification.md` first
> The process ranking in §4 (`Cast ≫ AS ≈ HIP`) is **withdrawn**. The µm/px scale
> *is* recoverable from the SEM databar (still present in 47 originals), and the
> corpus spans a **249× magnification range** (HFW 10.4 µm – 2.59 mm). The three
> processing routes were sampled at systematically different magnifications, so the
> pooled area-fraction comparison compared observation scales, not materials. At
> matched magnification only 1–2 fields per route exist and no process comparison
> is supportable. `HIP is least cracked` survives; `AS ≈ HIP` does not — at matched
> 319 µm AS is the highest by area, not the lowest.
>
> Everything else in this report stands, including the paired detector effect, the
> censoring analysis, the two-population region structure and the review-coverage
> audit. Statements below that units are pixels and that no µm factor is
> recoverable apply to the *export*, not to the source data.

Built from `crack_export/` on 2026-08-26. Everything here is reproducible:
`tools/split_sets.py` → `tools/analyse_sets.py` → `tools/review_coverage.py` →
`tools/paired_detector.py` → `tools/report_sets.py`.

---

## 1. How the sets were defined

Not by hand. A "set" is the pipeline's own `aggregate.specimen_key()` — the same
grouping the trainer uses to hold out a whole specimen rather than a single frame.
So a set folder holds exactly the frames that are siblings from one block/session.

62 frames → **10 sets**, in `sets/<set>/`:

| set | family | frames | fields | detectors | reviewed |
|---|---|---|---|---|---|
| `260622_316_H_b2` | steel | 5 | 5 | CBS | 4/5 |
| `260622_316_H_b4` | steel | 2 | 2 | CBS | 1/2 |
| `260622_316_amb_b3` | steel | 2 | 2 | CBS | 0/2 |
| `260708_316_H_b2` | steel | 16 | **13** | CBS | 16/16 |
| `MAR_Amb_AS` | superalloy | 11 | **6** | CBS, ETD | 8/11 |
| `MAR_Amb_Cast` | superalloy | 10 | **5** | CBS, ETD | 3/10 |
| `MAR_Amb_HIP` | superalloy | 13 | **7** | CBS, ETD | 11/13 |
| `AS_24hr` | exposure | 1 | 1 | BSE | 1/1 |
| `Cast_24hr` | exposure | 1 | 1 | SE | 1/1 |
| `HIP_24hr` | exposure | 1 | 1 | SE | 1/1 |

Each folder contains `masks/`, `overlays/`, `regions/`, plus `set_summary.csv`
(that set's slice of the top-level summary), `regions_pooled.csv` (every region
row in the set, tagged with its frame) and `SET.txt` (the parsed tokens).

Files are **hard links**, not copies — the overlays alone are 860 MB and copying
would have doubled that for nothing. `sets/` costs ~0 extra bytes. The flip side:
editing a file under `sets/` edits the original, so treat it as read-only.

> **Frames are not fields.** In the MAR sets, `..._CBS_0003` and `..._ETD_0003`
> are *the same field of view imaged with two detectors*. Index-matched mask
> pairs overlap with median Jaccard **0.505**; mismatched indices from the same
> specimen give **0.024** (n=6 control). So `MAR_Amb_AS`'s 11 frames are 6
> fields, not 11 samples. This is measured, not assumed — and it is why the
> `fields` column above differs from `frames`.
>
> One caveat on the method: Jaccard proves fields are the *same* when it is high,
> but a *low* value is inconclusive, because it conflates "different field" with
> "same field, and the two detectors disagree about which pixels are crack".
> `HIP_0005` is the giveaway — CBS 21.35 % vs ETD 2.91 %, and the CBS version is
> 94 % human-confirmed, so that is almost certainly one field where ETD misses the
> damage, not two fields. Field counts here therefore use the index convention
> (which is itself strong evidence of pairing) and treat 43 as the best estimate,
> not a hard number.
>
> The steel sets were checked the same way, pairwise within each set.
> `260622_316_H_b2`, `_H_b4` and `_amb_b3` are clean (max pairwise Jaccard 0.44,
> 0.24, 0.33 — distinct fields). **`260708_316_H_b2` is not:** `CBS_003/004/005`
> and `CBS_006/007` overlap (Jaccard up to **0.752**), giving **13 distinct
> fields from 16 frames**. Section 5 explains why that specific overlap matters
> more than the others.

---

## 2. What this data is, and what it cannot tell you

Four constraints that shape every number below. They are limitations of the
files, not of the analysis.

1. **Units are pixels.** The export README is explicit: the scale bar lives in
   the SEM databar, which the pipeline crops before analysis, so no µm/px factor
   is recoverable. Areas are px², densities per megapixel. Multiply by the
   microscope's own µm/px to get physical units.

2. **These masks are detector output with human corrections merged in** — not
   validated ground truth. A difference between sets can be a contrast or
   detector-sensitivity difference rather than a material difference. Section 6
   measures that confound instead of assuming it away.

3. **Size measurements are right-censored.** Regions whose bounding box touches
   a frame edge continue outside the field of view, so their length and area are
   *lower bounds*. This is not a minor correction: at set level **42.6–96.1 % of
   crack area sits in edge-touching regions** for the seven multi-frame sets
   (19.9 % is the low end if the three single-frame exposure sets are included; fig 5). Consequently **no absolute
   crack length is quoted anywhere in this report.** Size and shape summaries use
   uncensored regions only; area *fractions* are unaffected, because a fraction
   measured inside the frame is a density and does not care that a network leaves.

4. **The specimen is the statistical unit.** Frames from one block are
   near-duplicates, and some are literally the same field twice (see above). Set-level
   `sd` is a within-specimen spread, **not** an error bar for the material. There
   is exactly **one specimen per processing route**, so no p-value here can
   separate "process" from "that particular specimen".

---

## 3. There are two populations of region, and only one of them is damage

Pooled over all 62 frames (10,565 regions):

- **66 % of regions are ≤500 px²** and together hold **0.73 %** of the crack area.
- **The largest 1 % of regions hold 92.5 %** of the crack area.

So *region count* is a fragmentation/noise metric, not a damage metric, and a
single "mean region size" would average a few frame-spanning networks with
thousands of specks and describe neither. Everything below splits at 500 px²
into **specks** and **networks** (fig 3).

Damage is overwhelmingly **one object**: a single region holds **41.2–96.1 %**
of all crack area in the seven multi-frame sets (area Gini 0.67–0.98). The
exception is `AS_24hr`, the least-cracked frame in the corpus, where the largest
region holds only 5.2 % — damage there is genuinely distributed rather than
concentrated in one network. 15 of 62 frames contain a network that spans the
full field of view edge-to-edge.

---

## 4. Headline result: Cast is the outlier, by ~6×

Crack area fraction per set. **Median, not mean** — every set is right-skewed by
a few very-high frames, and for `MAR_Amb_AS` the mean (14.25 %) describes no
frame in the set (median 2.90 %).

| set | median area % | mean | sd | max | max/median | pattern |
|---|---|---|---|---|---|---|
| `260622_316_H_b2` | 9.98 | 10.04 | 1.60 | 11.83 | 1.2 | pervasive |
| `260622_316_H_b4` | 10.56 | 10.56 | — | 12.92 | 1.2 | pervasive |
| `260622_316_amb_b3` | 10.60 | 10.60 | — | 13.92 | 1.3 | pervasive |
| `260708_316_H_b2` | 8.47 | 14.42 | 14.27 | 43.71 | 5.2 | localised¹ |
| `MAR_Amb_AS` | **2.90** | 14.25 | 19.41 | 51.17 | **17.6** | localised |
| `MAR_Amb_Cast` | **20.27** | 18.47 | 12.37 | 41.25 | 2.0 | pervasive-at-high-level |
| `MAR_Amb_HIP` | **4.03** | 6.54 | 6.18 | 21.33 | 5.3 | localised |
| `AS_24hr` | 4.04 | — | — | — | — | n=1 |
| `Cast_24hr` | **20.76** | — | — | — | — | n=1 |
| `HIP_24hr` | **2.74** | — | — | — | — | n=1 |

¹ `260708_316_H_b2` is inflated by re-imaging. Collapsed to its 13 distinct
fields the median drops **8.47 → 5.46 %**, the mean **14.42 → 10.36 %** and the
max **43.71 → 36.66 %**. Use the field-level numbers for this set.

**Cast ≫ AS ≈ HIP.** At field level: Cast 23.29 %, HIP 3.90 %, AS 2.43 %.

This ordering is reproduced **four independent ways**:

- in the MAR family under **CBS** (Cast 25.51 vs AS 2.52, HIP 4.40),
- in the MAR family under **ETD** (Cast 19.46 vs AS 3.46, HIP 3.47),
- in the entirely separate **24 hr exposure** family, different specimens and
  different detectors (SE/BSE): Cast 20.76 vs AS 4.04, HIP 2.74,
- and it survives every review-based subset (section 5).

**The two damage patterns differ, not just the amount.** Cast is *pervasively*
cracked — its worst field is only 2× its median. AS and HIP are mostly clean with
isolated catastrophic fields: AS's worst field is **17.6×** its median. An
average over fields hides that completely, and for AS it is the single most
important fact about the specimen.

---

## 5. Are the high numbers real? Cross-checking against human labels

A frame at 51 % crack area is either a genuinely shattered specimen or an
over-detection nobody has checked. The correction masks answer this: they record
which pixels a human explicitly marked. 47/62 frames have one; **16 frames have
zero human review**; 70,434,978 pixels are hand-marked in total.

Of the 17 frames above 18 % crack area:

| frame | area % | reviewed % | of crack area confirmed | verdict |
|---|---|---|---|---|
| `MAR_Amb_AS_CBS_0003` | 51.17 | 50.93 | **99.2 %** | human-confirmed |
| `MAR_Amb_AS_ETD_0003` | 49.71 | 49.44 | **99.0 %** | human-confirmed |
| `260708_316_H_b2_front_CBS_004` | 43.71 | 4.14 | 9.5 % | **UNVERIFIED** |
| `260708_316_H_b2_front_CBS_005` | 41.85 | 0.79 | 1.9 % | **UNVERIFIED** |
| `MAR_Amb_Cast_CBS_0005` | 41.25 | 12.16 | 29.5 % | partly |
| `MAR_Amb_HIP_CBS_0005` | 21.33 | 20.05 | 94.0 % | human-confirmed |
| `MAR_Amb_Cast_ETD_0004` | 21.08 | 18.83 | 89.4 % | human-confirmed |
| `MAR_Amb_AS_ETD_0002` | 20.19 | 19.74 | 97.5 % | human-confirmed |
| `MAR_Amb_Cast_CBS_0002`, `CBS_0004`, `ETD_0005`, `ETD_0002` | 19–27 | 0.00 | 0 % | **UNVERIFIED** |

So the ~50 % fields in AS are **real** — a human painted 12.8 M pixels of that
frame as crack and only 26 k as not-crack. The 43 % steel fields are **not
corroborated** and are the largest open question in the corpus.

**And they are one field, not two.** `CBS_004` and `CBS_005` overlap at Jaccard
0.752 — the same location imaged twice. So `260708_316_H_b2` does not have two
independent 43 % fields; it has one, counted twice, and that single unreviewed
field is what drags the set's frame-level mean from 8.47 % to 14.42 %.

**A useful by-product: detection repeatability.** Because `CBS_004`/`CBS_005` are
a genuine repeat of one field, the difference between them is measurement noise,
not material variation: **43.71 % vs 41.85 %, i.e. 1.87 pp apart.** That is the
noise floor for a single area-fraction measurement on this pipeline, and it is
what makes the ~16 pp Cast-vs-HIP gap credible and a 3 pp gap not.
(`CBS_003` overlaps both at Jaccard 0.52–0.54 but reads 24.42 % — an offset or
lower-magnification view, so not a repeat and not evidence of poor repeatability.)

**Sensitivity — does the ordering depend on which frames you trust?**

| frames included | AS | Cast | HIP |
|---|---|---|---|
| all | 2.90 (n=11) | 20.27 (n=10) | 4.03 (n=13) |
| ≥20 % of crack area confirmed | 11.83 (n=6) | 21.08 (n=3) | 4.40 (n=11) |
| any human review | 3.11 (n=8) | 21.08 (n=3) | 4.40 (n=11) |
| **zero** human review | 2.90 (n=3) | 19.46 (n=7) | 1.00 (n=2) |

**Cast ≈ 20 % and HIP ≈ 4 % under every filter** — that comparison is robust.

**AS is not rankable.** Its estimate moves 2.90 → 11.83 depending on inclusion,
and the inclusion is *not random*: this is an active-learning workflow, so the
labeller was steered toward the most crack-rich frames. Confirmed-only medians
are therefore **biased upward**, and AS is the set where that bias bites hardest.
Report AS as "typically ~3 %, with confirmed fields up to 51 %", not as a mean.

---

## 6. The detector is not neutral — and here it is measurable

Because the MAR frames pair CBS and ETD on the same field, this is a *paired*
comparison, not a between-groups one. On the 8 pairs whose masks confirm
registration (Jaccard ≥ 0.5):

**CBS reports more crack area than ETD on the identical field in 8 of 8 pairs.**
Median difference **+3.88 pp**; Wilcoxon signed-rank p = 0.0078, sign test
p = 0.0078 (fig 6). Per-pair differences run +0.77 to +19.09 pp.

ETD also produces **1.5× the speck density** of CBS (8.6 vs 5.7 per Mpx) — but
specks are ~0.1 % of area, so they inflate region *counts* without moving area.

**What this means practically:** any set-to-set comparison that mixes detectors
carries up to ~4 pp of pure instrument bias. The Cast ≫ HIP result is safe
because both were measured under both detectors and the gap (~16 pp) is four
times the detector effect. A 3-pp difference between two sets imaged on
different detectors would be meaningless.

8 further index-matched pairs did **not** register (Jaccard < 0.5) and were
excluded rather than averaged in. `HIP_0005` is the extreme case: CBS 21.35 % vs
ETD 2.91 % on nominally the same field, with the CBS version 94 % human-confirmed.
Either the field was renumbered or ETD misses most of this damage — worth a look.

---

## 7. Orientation: cracks run across the frame, but the cause is not settled

Convention first, because the naive reading is backwards. `Orientation_deg` is
`np.degrees(regionprops.orientation)`, measured from the **row** axis. Verified
with synthetic bars: a horizontal bar gives **+90.00**, a vertical bar **0.00**.
So **|angle| ≈ 90° means the crack runs horizontally**, across the frame.

Orientation is *axial* (period 180°): −89° and +89° are nearly the same
direction, so an arithmetic mean of `Orientation_deg` averages a perfectly
horizontal population to ≈ 0 and would report it as vertical. All pooling here
is circular on the doubled angle, area-weighted, uncensored networks only.

Result: **9 of 10 sets align horizontally**, mean angle −89° to +87°.
Within-frame anisotropy |R| runs 0.31–0.92 — a preference, not a strict
alignment. By family: exposure |R| = 0.82 (strongly aligned), steel 0.63,
superalloy 0.48 (closest to isotropic, fig 4). The one exception is
`260622_316_amb_b3` at −52.8°, |R| = 0.31 — weakly aligned and diagonal, but
n=2 frames and 0 reviewed.

**This is not a frame-shape artifact.** The frames are 3:2, so a frame-spanning
network would look horizontal for purely geometric reasons — but the preference
holds at every size band, including 500–5,000 px² regions (~70 px across, far
too small for the frame aspect to constrain them):

| region size band | n | mean angle | \|R\| |
|---|---|---|---|
| 500–5,000 px² | 2,549 | −89.9° | 0.45 |
| 5,000–50,000 px² | 432 | −87.5° | 0.66 |
| 50,000–500,000 px² | 23 | −90.0° | 0.55 |

It also holds **unweighted** (−89.6°, |R| = 0.38), and 75 % of small/mid
uncensored regions are wider than tall.

**But it cannot be attributed to the material from these files alone.** Specimen
polishing direction and SEM raster direction are also horizontal and are shared
across every set — and a darkness-based detector will happily pick up horizontal
polishing scratches. That the preference appears in three different families and
four detectors cuts both ways: a common loading axis, or a common preparation
artifact. **This is directly testable:** re-image one specimen rotated 90°. If
the preference follows the specimen it is real damage anisotropy; if it stays
horizontal in image coordinates it is preparation or imaging.

---

## 8. Steel: hydrogen vs ambient shows nothing, and could not have

| set | condition | n | median area % | reviewed |
|---|---|---|---|---|
| `260622_316_H_b2` | H | 5 | 9.98 | 4/5 |
| `260622_316_H_b4` | H | 2 | 10.56 | 1/2 |
| `260622_316_amb_b3` | amb | 2 | 10.60 | 0/2 |

Pooled H (n=7) 9.98 % vs ambient (n=2) 10.60 % — a **+0.62 pp** difference. With
two ambient frames from one block, neither of them reviewed, this is not a test
of anything, and the three 2026-06-22 blocks are strikingly uniform (max/median
1.2–1.3) regardless of condition. **Underpowered, reported as such.** Two more
ambient blocks would make this comparison real.

---

## 9. Where labelling effort would change the numbers most

Nine frames carry >15 % crack area with <20 % of it human-confirmed. These are
the frames whose numbers are currently load-bearing but unchecked:

```
260708_316_H_b2_front_CBS_004     43.71 % area,  9.5 % confirmed
260708_316_H_b2_front_CBS_005     41.85 % area,  1.9 % confirmed
260708_316_H_b2_front_CBS_006     29.90 % area, 17.3 % confirmed
MAR_Amb_Cast_CBS_0002             27.30 % area,  0.0 % confirmed
MAR_Amb_Cast_CBS_0004             25.51 % area,  0.0 % confirmed
MAR_Amb_Cast_ETD_0005             22.16 % area,  0.0 % confirmed
MAR_Amb_AS_CBS_0002               21.64 % area,  0.0 % confirmed
MAR_Amb_Cast_ETD_0002             19.46 % area,  0.0 % confirmed
MAR_Amb_Cast_CBS_0001             15.28 % area,  0.0 % confirmed
```

The two steel frames at the top are the highest-value labelling targets in the
corpus: they are 43 % and 42 % crack with essentially no review, and they alone
move `260708_316_H_b2`'s mean from 8.47 % to 14.42 %.

Note also that the review is almost entirely **crack** marks with ~0 not-crack
marks. So review coverage tells you how much detected crack was *accepted*; it
provides almost no evidence about false negatives, and none against
over-detection *outside* the marked area.

---

## 10. Bottom line

**Supported by the data:**
- Cast cracks ~5–6× more than AS or HIP by area fraction, replicated across two
  independent imaging campaigns, both MAR detectors, and every review subset.
- Cast damage is pervasive; AS and HIP damage is localised to a few catastrophic
  fields (AS worst/median = 17.6×). Different patterns, not just amounts.
- Damage is one dominant connected network per field (41–96 % of area in a
  single region in every multi-frame set), preferentially oriented across the frame.
- Area-fraction measurement repeats to within ~1.9 pp on a re-imaged field.
- CBS systematically over-reports crack area relative to ETD by ~3.9 pp on
  identical fields (8/8, p = 0.008).

**Not supported, and stated as such:**
- Any absolute crack length or crack count (censored and fragmentation-dependent).
- Any hydrogen-vs-ambient effect in the steel (n=2 ambient, 0 reviewed).
- A material interpretation of the horizontal orientation (confounded with
  polishing and raster direction).
- A ranking of AS against HIP (AS's estimate moves 4× with non-random inclusion).
- Frame counts as sample sizes. 62 frames are **~43 distinct fields** across
  **8 specimens**; `MAR_Amb_AS` is 6 fields and `260708_316_H_b2` is 13.
- **Any claim that generalises past these specimens** — there is one specimen per
  processing route, so process and specimen are inseparable here. Reproducing
  Cast ≫ HIP on a second Cast specimen is the single highest-value next experiment.

---

## Files

```
sets/<set>/masks|overlays|regions/   hard-linked frames, one folder per set
sets/<set>/set_summary.csv           that set's rows from summary.csv
sets/<set>/regions_pooled.csv        every region in the set, tagged by frame
sets/<set>/SET.txt                   parsed specimen tokens + frame list
sets/SETS.csv                        set -> family, frames, detectors

analysis/per_frame_metrics.csv              39 columns x 62 frames
analysis/per_frame_metrics_with_review.csv  + review coverage and confirmed share
analysis/per_set_metrics.csv                mean/median/sd per set
analysis/review_coverage.csv                hand-marked px per frame
analysis/paired_detector.csv                CBS vs ETD on the same field
analysis/figures/fig1..fig6.png
```

Figures: **1** area per set (points = frames, hollow = unverified) · **2** MAR
process × detector · **3** the two size populations · **4** orientation roses ·
**5** why no length is quoted · **6** paired detector effect.
