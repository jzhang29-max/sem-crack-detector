# Correction: the scale IS recoverable, and it changes the main result

**Status: this retracts the process ranking in `CRACK_ANALYSIS.md` §4.**

## 1. What I got wrong

I reported, following the export README, that "no µm/px factor is recoverable."
That is true of `crack_export/` but false of the source data:

- **47 of the 62 originals still contain the SEM databar.** They are 6144×**4376**
  (or 3072×2188) — the extra 280 (140) rows *are* the panel. The export crops it;
  the originals in `sem-crack-detector/original/` do not.
- Full-resolution databar crops already existed at
  `.../sem2txm/figures/databars/` (50 files). ⚠ **That is a different repository, and it is
  not shipped here — no reader of this repo can check this step.** The databars are still
  present in 47 of the originals under `original/`, so the measurement is reproducible from
  what ships; it is the crops used at the time that are unavailable.
- The Apreo **prints HFW directly**. I read it off the panels.

Validation on the one frame I checked arithmetically, `MAR_Amb_AS_CBS_0003`:
panel prints HFW 319 µm, so µm/px = 319/6144 = 0.05192. Its 100 µm bar should then
be drawn 1926 px long; `sem2txm/out/sem_scale.json` independently measured
**1927 px**. Agreement to 1 px.

**Do not use `sem_scale.json`'s `um_per_px` column.** It assumes every bar is
labelled 100 µm and says so in its own `assumption` field. The labels are actually
4, 5, 10, 20, 30, 50, 100, 400, 500 and 1000 µm, so that column is wrong by the
label ratio for most frames — e.g. `260622_316_H_b2_front_CBS_02` has a 50 µm bar,
so its true µm/px is 0.0282, not the 0.0562 recorded. Correct scale is
`HFW / image_width_px`; 45 frames are tabulated in `analysis/scale_hfw.csv`.

## 2. Why it matters: a 249× magnification range

**HFW spans 10.4 µm to 2.59 mm.** This is a multi-scale dataset, not one imaging
condition. Crack area fraction is only comparable between fields imaged at the
same HFW — at 10.4 µm you are inside a single crack, at 2.59 mm you see the
whole coupon.

My earlier "resolution-invariant" check was too weak. I verified `LineDensity` and
`RelWidth` against `CBS_004`/`CBS_005`, which are the same physical field
resampled 2×. Invariance to *resampling* says nothing about invariance to
*changing the physical field size*. Those metrics are not comparable across HFW.

One thing this independently confirms: **of the 16 index-matched CBS/ETD pairs, the 15
with a readable databar on both frames share
HFW exactly** (0 mismatches), which corroborates the same-field finding by a route
that has nothing to do with mask overlap.

## 3. The processes were sampled at different magnifications

| process | n | HFW values (µm) |
|---|---|---|
| AS | 11 | 41, 259, 259, 319, 319, 1040×4, 2070×2 |
| Cast | 8 | 319×4, 1040×4 |
| HIP | 13 | 10.4×2, 13.8×2, 41×2, 59×2, 319×2, 414×2, 1040 |

AS and Cast were shot mostly at 1040–2070 µm; **HIP mostly at 10–59 µm.** Pooling
across that compares observation scales, not materials.

## 4. The corrected comparison

Within the only stratum containing all three routes (**HFW = 319 µm**):

| process | fields | area % | crack length per mm² | crack width |
|---|---|---|---|---|
| AS | 1 | **50.44** | 49,119 µm | 19.4 µm |
| Cast | 2 | 23.83 | **82,924 µm** | 9.7 µm |
| HIP | 1 | **12.56** | 30,981 µm | 8.3 µm |

At HFW = 1040 µm: AS 11.55 %, Cast 10.00 % (n=4 frames each) — **tied**; HIP 0.07 %
on a single frame.

**What changes:**

- **`Cast ≫ AS ≈ HIP` is withdrawn.** AS's low pooled median (2.90 %) came from
  AS being sampled mostly at low magnification, where area fraction reads low. At
  matched 319 µm, AS is the *highest* by area (50.4 %), not the lowest.
- **AS vs Cast is not resolvable.** The ordering flips with the metric (AS higher
  by area, Cast higher by length density) and they tie at 1040 µm.
- **HIP being least cracked survives at matched magnification** — lowest at matched
  319 µm by area. ⛔ The clause "and lowest in the pooled data too" is **false and is
  withdrawn**: pooled medians of `CrackAreaPct` from `per_frame_metrics.csv` are AS 3.179 /
  Cast 20.764 / HIP 3.898 over all frames (AS 2.902 / Cast 20.266 / HIP 4.032 over MAR_Amb),
  so **AS** is lowest pooled, under either scope. The matched-magnification result is the
  one that stands; the pooled one never did, which is this document's own argument.
- Do **not** use the pooled physical length density (HIP appears *highest* at
  97,497 µm/mm²) — that is pure magnification confound, since HIP is the set shot
  at 10–59 µm HFW where fine cracks resolve.

**But the honest bottom line is deflationary.** With magnification controlled there
are **1–2 distinct fields per processing route**, all from one specimen each. No
process comparison in this corpus is supportable. The `Cast ≫ HIP` result I
reported was produced by pooling across a 249× magnification range.

## 5. What survives unchanged

- The **paired CBS/ETD detector effect** (+3.88 pp, 8/8). Pairs are
  magnification-matched by construction — now verified from the panels.
  ⛔ **p = 0.0078 is the exact test's floor at n = 8, not a measurement, and the 8 confirmed pairs are 3 specimens, not 8. Aggregating per specimen — which this repo instructs — gives exact p = 0.2500. The direction is unanimous (8/8 pairs, all 3 specimens, median +3.88 pp); the significance is withdrawn. See `VERDICT_2026.md` §S1(b).**
- The **1.87 pp repeatability floor** from `CBS_004`/`CBS_005`.
- The **two-population region structure** (66 % of regions ≤500 px² holding 0.73 %
  of area; top 1 % holding 92.5 %) — a within-frame structural result.
- **Censoring** (42.6–96.1 % of crack area edge-touching) and the refusal to quote
  absolute lengths.
- The **`Tortuosity` < 1 bug** (54.5 % of cracks) — independent of scale.
- The **tortuosity confounds** (length, and width at ρ=−0.556), and that there is no
  separation between sets. The *value* does not survive unchanged: ~1.09 is inside the
  estimator's own bias envelope, and the corrected excess over an angle-matched null is
  **1.030** (`LINEARITY_AND_FRACTURE_MODE.md` §7). Do not quote 1.09.
- **TG/IG remains not determinable**, for the reasons in
  `LINEARITY_AND_FRACTURE_MODE.md` §4.

## 6. What to do next

1. Read HFW for the remaining 17 frames (databars exist for 5 more; 12 originals
   were saved already cropped).
2. Re-express all measurements in µm using `HFW / image_width_px`, and **report
   every comparison within a magnification stratum**.
3. For a real process comparison, image a matched set: same HFW, same detector,
   ≥5 fields per route, ≥2 specimens per route.
