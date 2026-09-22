# Crack linearity and fracture mode

> ### ⚠ CORRECTION — read `CORRECTION_scale_and_magnification.md` first
> **Every process comparison in this file (`Cast ≫ HIP`, the AS-vs-HIP inversion, and the
> "Cast survives, and is now mechanistically explained" reading) inherits the withdrawal of
> the §4 process ranking in `CRACK_ANALYSIS.md`.** The three processing routes were sampled
> at systematically different magnifications over a 249× range, so any comparison pooled
> across them compares observation scales, not materials — and line density and width are
> themselves resolution-dependent, which is the confound this file set out to control for and
> did not. At matched magnification only 1–2 fields per route exist. The *methods* here (the
> resolution-invariant metrics, the tortuosity correction in §7, the hole analysis) stand; the
> process rankings drawn from them do not. This file carried no banner until 2026-09-21.

Follow-up to `CRACK_ANALYSIS.md`. Two questions: did the first analysis use the CSVs,
and can we graph linearity and transgranular/intergranular cracking?

Reproduce with: `tools/skeleton_metrics.py` → `tools/holes.py` →
`tools/linearity_figs.py` → `tools/csv_shape_figs.py`.

---

## 1. Yes, the CSVs — but they cannot answer "linearity"

Every number in `CRACK_ANALYSIS.md` came from `regions/*_regions.csv`: `Area_px`,
`Length_px`, `Width_px`, `AspectRatio`, `Eccentricity`, `Orientation_deg`,
`Solidity`, `Perimeter_px`. The masks were used only to test field-pairing, and the
correction masks for review coverage.

The problem for this question: **`Length_px` and `Width_px` are the axes of the
best-fit ellipse, not the crack's path.** `Solidity` is a convex-hull ratio and
`Perimeter_px` an outline length. None of them measures how straight a crack runs.
Fig 12 shows what the CSVs *can* say about shape, and it is real information —
`AS_24hr` stands out at aspect ratio 9.75 with solidity 0.540, i.e. long thin
ribbons — but a region's fitted ellipse cannot distinguish a straight crack from a
gently curved one of the same extent.

So path morphology was computed from the masks: **33,471 skeleton branches**.

Two things also turned up in the repo that belong here:

- `interior_active_learning/measurements/` already holds per-crack `Tortuosity`,
  `SkeletonLength_px`, `MeanWidth_px`, `MaxWidth_px`, `BranchPointCount` and
  `BoundaryRoughness` for **all 62 frames** (20,338 rows). Section 5 reports a bug
  in three of those columns.
- Every provenance file records `"calibrated": false, "um_per_px": null`. Confirmed:
  no physical scale.

### A correction to my own earlier run

**26 of 62 frames are not 4096×6144** — they run from 1490×1490 up, and many are
exactly half-scale. My first pass at the CSV shape metrics hardcoded 4096×6144, so
the edge-censoring test silently passed every region in a 2045-row frame (nothing
can reach `y1 ≥ 4096`). Fixed to read per-frame bounds; 228 edge-touching regions
were then correctly excluded and the medians moved by <0.02, so no conclusion
changed. `CRACK_ANALYSIS.md` was never affected — it read dimensions from
`summary.csv`.

**The half-size frames are 2× downsamples of the same optics, not crops.** Verified
on the known same-field pair `260708_316_H_b2_front_CBS_004` (2045×3072) and
`CBS_005` (4091×6139): linear size ratio 2.000, mean crack width ratio **2.023**.
That kills raw pixel lengths as a cross-frame quantity and forces every metric
below to be dimensionless or explicitly controlled:

| quantity | ratio across the 2× pair | comparable? |
|---|---|---|
| skeleton px per Mpx | 0.59 | **no** |
| mean crack width (px) | 2.02 | **no** |
| `LineDensity` = skeleton px / frame width | 1.19 | yes |
| `RelWidth` = mean width / frame width | 1.01 | yes |
| `CrackAreaPct` | 0.96 | yes |
| box-counting fractal dimension | 1.02 | yes |

---

## 2. Linearity: a clean negative, after two confounds are removed

Analysis unit is the **skeleton branch** (the arc between two nodes), not the
connected region. A region here can be 12 million px of branching network; its
end-to-end "tortuosity" would measure network topology, not how straight a crack
runs.

Tortuosity = path length / end-to-end chord, with diagonal steps counted as √2.
Raw per-set values look like a result — `MAR_Amb_Cast` 1.026 vs `HIP_24hr` 1.096 —
but they are an artifact of two confounds (fig 9):

1. **Length.** Tortuosity rises to 1.081 at 50–100 px then falls to 1.030 at
   250–1000 px and 1.003 above 1000 px. Short arcs cannot be straight (they are
   skeleton spurs, median turn 53°); long ones are genuinely straight.
2. **Width.** Spearman(tortuosity, branch width) = **−0.556**. It is flat at
   ~1.10 out to 64 px wide and then collapses to 1.015 above 128 px. The reason is
   geometric: **the medial axis of a wide blob is smooth no matter how ragged the
   real crack was.** `MAR_Amb_Cast` scored "straightest" because its detected
   features are the widest, not because its cracks are straight.

Controlling both — thin branches (≤16 px) of moderate length (100–250 px):

| set | n | tortuosity | median turn (deg) |
|---|---|---|---|
| `260622_316_H_b2` | 46 | 1.086 | 17.8 |
| `260622_316_H_b4` | 123 | 1.093 | 23.7 |
| `260708_316_H_b2` | 25 | 1.087 | 23.7 |
| `AS_24hr` | 120 | 1.093 | 21.9 |
| `Cast_24hr` | 46 | 1.086 | 21.4 |
| `MAR_Amb_AS` | 132 | 1.103 | 25.0 |
| `MAR_Amb_Cast` | 130 | 1.089 | 23.5 |
| `MAR_Amb_HIP` | 44 | **1.128** | 29.3 |

**Once length and width are controlled, crack path tortuosity is ~1.09 in every
set.** Paths are about 9 % longer than a straight line, and no set differs
meaningfully from any other. `MAR_Amb_HIP` is nominally highest at 1.128 on n=44
branches from one specimen — not a difference worth defending.

This is a negative result, and it is the honest one: the *apparent* linearity
differences between these materials are differences in detected feature width.

**Pipeline validated against known geometry.** A synthetic smooth sinusoid,
rasterized and put through the same skeleton→Douglas-Peucker→turn-angle path,
returns median turn 12.5° and **0 % of turns in 40–55°** at every width from 5 to
200 px. So the method does not manufacture kinks. Real thin branches measure
22–25°, i.e. genuinely rougher than a smooth curve.

---

## 3. What crack area fraction actually measures

This qualifies the headline of `CRACK_ANALYSIS.md`. Across the 62 frames:

- Spearman(area %, **relative feature width**) = **+0.862**
- Spearman(area %, **crack line density**) = **+0.712**
- Spearman(area %, fractal dimension) = **+0.894**; Spearman(width, D) = **+0.945**

So crack area fraction is substantially a *segmentation-thickness* metric (fig 8),
and box-counting fractal dimension is nearly a restatement of width, not an
independent measure of path complexity. The extreme cases make it concrete: the
widest frames have mean feature widths of 283–521 px with D ≈ 1.90 — those are
**blobs, not cracks** — while the thinnest are 5–8 px with D ≈ 0.89.

Decomposing area fraction into the two invariant factors (fig 7, medians):

| set | area % | line density | rel. width | reading |
|---|---|---|---|---|
| `Cast_24hr` | 20.76 | 15.46 | 0.0194 | many cracks, moderately wide |
| `MAR_Amb_Cast` | 20.27 | 10.98 | 0.0387 | many cracks **and** widest |
| `260622_316_H_b2` | 9.98 | 8.97 | 0.0170 | moderate both |
| `260708_316_H_b2` | 8.47 | **2.68** | **0.0400** | **fewest cracks, thickest detections** |
| `AS_24hr` | 4.04 | 10.84 | 0.0024 | **many very thin cracks** |
| `MAR_Amb_HIP` | 4.03 | 3.70 | 0.0078 | few, thin |
| `MAR_Amb_AS` | 2.90 | 8.01 | 0.0088 | **many thin cracks** |
| `HIP_24hr` | 2.74 | 4.94 | 0.0053 | few, thin |

Two findings follow.

**Cast survives, and is now mechanistically explained.** Cast is highest on *both*
factors (line density 10.98 vs HIP's 3.70; width 0.0387 vs 0.0078). Both point the
same way, so `Cast ≫ HIP` is not a thresholding artifact.

**The AS-vs-HIP ranking inverts.** By area fraction AS (2.90 %) looks *better* than
HIP (4.03 %). By crack line density AS is **2.2× worse** (8.01 vs 3.70) — AS has
many thin cracks, HIP has few wider ones. The same inversion appears independently
in the exposure family (`AS_24hr` 10.84 vs `HIP_24hr` 4.94, again 2.2×). In
`CRACK_ANALYSIS.md` I called AS ≈ HIP; on line density that is wrong, and AS is
the more cracked of the two.

**`260708_316_H_b2` is the set to distrust.** It has the *lowest* crack line
density of any steel set (2.68) but the *highest* relative width (0.0400). Its
8.47 % area comes from thick detections around a small amount of crack — and it is
the set holding the two unreviewed 43 % frames.

---

## 4. Transgranular vs intergranular: what I can and cannot give you

**Short answer: I can show you morphology, and for the 316 steel it looks
intergranular, but these files cannot support a quantitative TG/IG classification.**

### The visual evidence is genuinely informative — and set-dependent

I cropped native-resolution windows around real cracks (`analysis/crops/`).

- **316 steel, CBS detector: grain structure is visible**, and cracking looks
  intergranular. `260622_316_H_b4_CBS_01` shows distinct polygonal grains with
  internal parallel striations (deformation twins / slip traces, as expected in
  austenitic 316), and the detected network visibly traces the polygon boundaries
  with sharp turns at triple junctions. No etching needed — CBS is a backscatter
  detector and gives orientation/channeling contrast.
- **MAR superalloy, ETD detector: no grain structure.** ETD is topographic; the
  surface shows relief and machining marks, not grains.
- **`AS_24hr`, BSE: featureless matrix** with scattered dark specks (porosity or
  precipitates) and long, near-straight, near-parallel cracks.

So "is it intergranular?" has different answers per set, and only the steel sets
have the contrast to even ask.

### The quantitative test fails — and the reason matters

If a crack network runs along grain boundaries, the islands it encloses **are
grains**. So their size distribution should match the grain structure. Measured
(fig 11): **1–11 enclosed islands per frame, median equivalent diameter 10–100 px,
occupying 0.001–0.5 % of the frame.** 17 of 62 frames enclose nothing at all. The
grains visible in the steel crops are roughly 300–500 px across.

The islands are an order of magnitude too small to be grains. The reason is not
that the cracking isn't intergranular — the crops suggest it is — but that **the
segmentation is far too thick to preserve the network topology.** The detected
crack is a wide band covering the boundary *and* the adjacent grain interiors, so
it never closes a loop around a grain. The mask destroys exactly the structure the
test needs.

Turn angles agree (fig 10): they peak at **20–30°** and essentially vanish above
60°, with no population near the ~60° deviation you would expect at 120° triple
junctions. There is a spike at 45–50°, but it lives in the *wide* features and the
synthetic control shows the pipeline produces no such spike from smooth geometry —
so it is blob medial-axis geometry, not crack faceting.

### What would actually be needed

1. **EBSD** on one or two of these fields — the only direct method. Overlay the
   crack trace on the orientation map and score path-vs-boundary coincidence.
2. Failing that, **a thinner segmentation**: the current masks are 5–500 px wide
   where the real cracks are a few px. A tight segmentation of the steel CBS frames
   would let the enclosed-island test work, because grains are already visible.
3. **A µm/px factor** for any argument that compares crack facet length to a known
   grain size. Every provenance file says `um_per_px: null`.

What I would not do is report a "TG/IG index" from tortuosity or turn angles on
this data. Tortuosity is ~1.09 in every set with no separation, both modes can be
tortuous or straight depending on grain shape and loading, and the one structural
test that could discriminate is defeated by mask thickness.

---

## 5. A bug in the pipeline's own linearity metric

`extended_features.py:70` computes skeleton length as a **pixel count**:

```python
skel_len = int(skel.sum())
```

and `:136` divides it by a **Euclidean** endpoint distance:

```python
tortuosity = float(skel_len / straight_dist)
```

A diagonal step advances √2 in space but counts as one pixel, so the ratio is
biased down by up to 1/√2 for diagonal cracks. Verified on synthetic straight
cracks, where the true tortuosity is exactly 1.000:

| straight crack at | reported `Tortuosity` |
|---|---|
| 0° (horizontal) | 1.003 |
| 20° | 0.943 |
| **45°** | **0.710** |

And in the real measurements (1,503 uncensored cracks, skeleton ≥50 px):

- **819 of 1,503 (54.5 %) have `Tortuosity` < 1**, which is geometrically
  impossible — path length cannot be shorter than the straight line between its
  ends. Minimum reported: 0.759, close to the 0.707 floor.
- Spearman(`Tortuosity`, angular offset from the nearest pixel axis) = **−0.648**,
  and the per-band medians track `cos(offset)` — the value a *perfectly straight*
  crack would score:

| offset from axis | n | reported | cos(offset) |
|---|---|---|---|
| 0–10° | 473 | 1.012 | 0.996 |
| 10–20° | 474 | 0.991 | 0.967 |
| 20–30° | 302 | 0.951 | 0.914 |
| 30–40° | 177 | 0.889 | 0.830 |
| 40–45° | 77 | 0.902 | 0.739 |

So the column mostly reports **crack direction**, not crack tortuosity. This
compounds with the orientation finding in `CRACK_ANALYSIS.md`: sets whose cracks
sit at different angles get differently-biased tortuosity, so the bias is
set-dependent, not a constant offset.

**Two other columns share the bug**, both dividing by the same pixel count:

- `MeanWidth_px = area / skel_len` (`:108`) — **over**estimates width by up to √2
  for diagonal cracks.
- branch-point density `= branch_points / skel_len * 100` (`:92`).

**Fix** — sum Euclidean step lengths along the ordered skeleton path instead of
counting pixels (diagonal steps √2, orthogonal 1). That is what
`tools/skeleton_metrics.py` does, and its tortuosity is ≥1 by construction, median
1.08. Note `Tortuosity` is only emitted for unbranched 2-endpoint skeletons, so the
fix touches a well-defined subset.

I have not modified the repo.

---

## 6. Limitations

- **No physical scale.** Everything is px; `um_per_px` is null in every provenance
  file. Absolute lengths and any grain-size comparison are unavailable.
- **Resolution is not uniform.** 26 of 62 frames are 2× downsamples. Only the
  invariant metrics in §1 are compared across frames.
- **Pseudo-replication.** 33,471 branches come from ~43 fields in 8 specimens, one
  specimen per processing route. Thousands of branches from one specimen is n=1 for
  the material, not n=1000. No p-value on branch counts is quoted, and none should be.
- **Masks are detector output.** The crops show both over-detection (a large blob
  on a dark topographic region in `MAR_Amb_Cast_ETD_0001`) and clear misses (dark
  zig-zag cracks left unmarked in the same frame). Morphology inherits both.
- **Orientation confound stands.** The crops confirm horizontal polishing/machining
  striations across the surfaces, so the horizontal crack alignment reported in
  `CRACK_ANALYSIS.md` still cannot be separated from preparation direction.

---

## Files

```
analysis/LINEARITY_AND_FRACTURE_MODE.md   this report
analysis/branches.csv                     33,471 skeleton branches x 22 columns
analysis/skeleton_frames.csv              per-frame skeleton/fractal/width metrics
analysis/holes.csv                        enclosed-island size distributions
analysis/skeleton/*.json                  per-frame raw output
analysis/crops/*.png                      native-resolution windows around cracks
analysis/figures/fig7..fig12.png
tools/skeleton_metrics.py  holes.py  linearity_figs.py  csv_shape_figs.py
```

Figures: **7** line density vs feature width · **8** what area fraction measures ·
**9** tortuosity confounds and the controlled comparison · **10** turn angles and
the missing faceting signature · **11** the intergranular test · **12** CSV-only
region shape.

---

## 7. Addendum: my own tortuosity was biased too, by the same class of error

After writing §5 I checked my own estimator against the digital-geometry literature,
where this is a classic solved problem: naive pixel counting is the known-bad
baseline, and named corrections exist — Kulpa (`L = 0.948·nₑ + 1.343·n₀`),
Vossepoel–Smeulders, and the corner-count estimator. The repo's pixel count and my
√2-weighted sum are both just points on that spectrum.

Measured on perfectly straight 3-px lines, where **true tortuosity is exactly 1.000
at every angle**:

| angle | pixel count (repo) | √2 sum (my fix) | Kulpa |
|---|---|---|---|
| 0° | 1.001 | 1.000 | 0.948 |
| 20° | 0.941 | **1.081** | 1.026 |
| 45° | 0.708 | 1.000 | 0.950 |
| 75° | 0.967 | 1.073 | 1.018 |
| mean abs error | 0.083 | **0.045** | 0.030 |

So my fix cuts the error roughly in half and removes the impossible <1 values, but it
**over**estimates by up to 8.1 % at intermediate angles. That matters directly,
because my reported tortuosity was **1.09** — inside the estimator's own bias
envelope. As reported, it was indistinguishable from perfectly straight cracks.

### Corrected: excess tortuosity against an angle-matched null

For each branch I now compute what a *true* straight line at that branch's net angle
measures under the identical skeletonize → path-length code, and report
`excess = measured / null` (`analysis/straightline_null.json`). Thin branches,
100–250 px:

| set | n | raw | null | **excess** |
|---|---|---|---|---|
| `260622_316_H_b2` | 46 | 1.086 | 1.069 | **1.015** |
| `AS_24hr` | 120 | 1.093 | 1.076 | **1.016** |
| `MAR_Amb_Cast` | 130 | 1.089 | 1.056 | **1.031** |
| `260622_316_H_b4` | 123 | 1.093 | 1.059 | **1.033** |
| `Cast_24hr` | 46 | 1.086 | 1.051 | **1.034** |
| `260708_316_H_b2` | 25 | 1.087 | 1.042 | **1.043** |
| `MAR_Amb_AS` | 132 | 1.103 | 1.056 | **1.045** |
| `MAR_Amb_HIP` | 44 | 1.128 | 1.064 | **1.060** |
| **pooled** | 666 | 1.094 | 1.062 | **1.030** |

**Cracks deviate from straight by ~3 % in path length, not ~9 %.** Two-thirds of the
apparent waviness was digitization bias. 40.5 % of branches are straight to within
2 %. The between-set spread is 1.015–1.060, still from one specimen each, so it
remains uninterpretable as a material difference — but the measurement is now
honest.

**The generalisable point** — and the part worth publishing — is not that digital
length estimation is biased (that has been known since the 1970s). It is that **for
near-straight cracks the digitization bias is the same order as the signal**: 6.2 %
bias against 3.0 % real excess here. Any paper reporting crack tortuosity in the
1.0–1.2 range without an angle-matched null is reporting mostly its own pixel grid.
The null is trivial to generate — rasterize straight lines at matched angles through
the identical code path.
