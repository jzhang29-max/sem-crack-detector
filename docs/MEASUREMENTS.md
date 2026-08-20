# Measurement data dictionary

Every column in `interior_active_learning/measurements/<image>_crack_measurements.csv`,
what it means, and what it is *not*. Written because the CC-BY dataset shipped twelve
undocumented feature columns: redistributable but not reusable, which defeats the point of
publishing it.

One row is one **final crack** — a connected component of the final mask after both
passes, the MST merge, and any human correction. That is what a person looking at the
overlay would call one crack.

## Units

Columns ending `_px` are **pixels**. Columns ending `_um` / `_um2` are micrometres and
appear **only if the image has been calibrated** — see [Calibration](#calibration). An
uncalibrated image exports pixel columns only, and its `_provenance.json` says
`"calibrated": false`. There is deliberately no 1.0 fallback: a default scale factor is
indistinguishable from a real measurement once it is in a spreadsheet.

## Geometry

| Column | Definition | Not to be confused with |
|---|---|---|
| `Area_px` | Pixel count of the crack component. Converts as length², so `Area_um2`. | — |
| `AreaPct_of_image` | `Area_px` as a percentage of the cropped frame. Dimensionless — never scaled. | — |
| `SkeletonLength_px` | **Crack path length.** Length of the medial-axis skeleton, so it follows the crack's curve. This is the quantity the fatigue/creep literature calls crack length. | not the bounding-box diagonal |
| `EllipseMajorAxis_px` | Major axis of the ellipse with the same second moments as the region. A shape descriptor. | **not crack length** — for a curved crack it is substantially shorter than `SkeletonLength_px`. Renamed from `MajorAxisLength_px`, which read like a length and was routinely mistaken for one. |
| `EllipseMinorAxis_px` | Minor axis of the same ellipse. | not crack width — use `MeanWidth_px` |
| `MeanWidth_px` | Mean crack opening: twice the mean medial-axis distance-transform radius along the skeleton. | — |
| `MaxWidth_px` | Maximum crack opening by the same measure. Degenerate regions too small to skeletonise fall back to `sqrt(area)`. | — |
| `Tortuosity` | `SkeletonLength_px / straight-line end-to-end distance`. A **ratio**, dimensionless, never scaled. Defined only where the skeleton has exactly two endpoints; blank otherwise (blank, not `nan`, to keep the CSV clean). | — |
| `BranchPointCount` | Skeleton pixels with three or more neighbours. A **count** — tells a simple crack from a branching network. Never scaled. | — |
| `Orientation_deg` | Ellipse orientation in degrees. An **angle** — never scaled. | — |
| `BoundaryRoughness` | Perimeter of the region divided by the perimeter of its convex hull. Dimensionless. | — |
| `CentroidX_px`, `CentroidY_px` | Centroid in **cropped-frame** coordinates, not raw-file coordinates. The crop removes any burned-in info bar and, on some frames, an aperture vignette. | — |
| `SourceImage`, `CrackID` | Identifiers. `CrackID` is per-image and not stable across re-renders. | — |

### What scales and what does not

Lengths scale linearly, areas as the square, and ratios/angles/counts not at all. This is
enforced in one place — `calibration.LENGTH_POWERS` and `calibration.DIMENSIONLESS` — so
no call site has to remember it. Scaling a tortuosity or an angle is the obvious way to
get this wrong, and there is a test asserting it does not happen.

### How precise is the scale

A µm column is a point estimate, and until recently it was printed as though it were exact.
It is not: the scale bar is marked by hand, and each end lands within a pixel or two of the
tick. Two independent endpoint errors of 1.5 px add in quadrature over the marked span, so

| marked span | uncertainty on µm/px | on a length | on an area |
|---|---|---|---|
| 60 px | 3.54% | 3.54% | 7.07% |
| 200 px | 1.06% | 1.06% | 2.12% |
| 800 px | 0.27% | 0.27% | 0.53% |

A crack measured at `61.40 µm` off a 200 px bar is 61.4 ± 0.7 µm — two significant figures,
not four. `calibration.propagate(rel_sd, power)` gives the figure for any column, and the
provenance sidecar carries both `um_per_px_rel_sd` and a per-column table so nobody has to
rederive that area doubles it. The app shows it beside the scale, which is the only thing
that makes marking a longer bar feel worth the extra second.

Two things this is **not**:

- It is **instrument uncertainty only.** Segmentation error — whether the boundary the
  skeleton was measured from is the real crack edge — is larger and is not quantified
  anywhere in this repo. Reporting the small interval and staying quiet about the big one
  would be a worse misrepresentation than reporting no interval at all, so the sidecar says
  so in the same sentence.
- It is **absent, not zero, for the other routes.** A typed HFW and an instrument tag are
  numbers whose precision this program has no way to know, so they store `None` and the
  sidecar spells out that the scale was not characterised. A missing field reads as zero,
  and "we did not measure this" is a different claim from "this is exact".

## Calibration

Calibration is **per image**, explicit, and recorded with its provenance in
`interior_active_learning/paint/calibration.json`. Three routes:

- **`scale_bar`** — read the label off the burned-in bar (e.g. `400 µm`) and mark its two
  end ticks. `µm/px = label ÷ |x2 − x1|`.
- **`hfw`** — horizontal field width from the info panel ÷ image width in pixels.
- **`manual`** — typed directly.
- **`instrument_metadata`** — FEI/Thermo INI blocks or ZEISS `CZ_SEM` tags, read straight
  off the file (`--from-metadata`, or `calibration.read_instrument_metadata`). The unit is
  decided by the key's meaning, never by which magnitude looks plausible: FEI records
  metres, so a `HorizontalFieldWidth` of `2.048e-4` is 205 µm and reading it as µm is a
  factor-of-a-million error. If the image already has a hand-marked scale and the two
  disagree by more than 5%, this **refuses** rather than letting a machine value silently
  overwrite a person's.

It is **not automatic, on purpose.** Three automatic bar detectors gave three different
answers for `MAR_Amb_Cast_CBS_0002`:

| method | span | µm/px |
|---|---|---|
| longest contiguous bright run in the panel | 1000 px | 0.400 |
| widest bright extent in the panel's right half | 2816 px | 0.142 |
| tick-to-tick | ~2379 px | **0.168** |

Only the last agrees with the independent HFW route (1.04 mm ÷ 6144 px = 0.169 µm/px,
0.67% apart). The first overshot because the `400 µm` label *interrupts* the bar, so the
longest contiguous run is one segment of it; the second because the panel's right border is
also bright. A silently wrong calibration propagates into every exported length and nobody
notices, because the numbers still look plausible.

So `set_from_scale_bar()` takes an optional HFW and **refuses to store anything** when the
two readings disagree by more than 5%. A disagreement means one reading is wrong and the
program cannot tell which.

Vendor metadata cannot rescue **this** corpus, and the reason is worth stating plainly
because it is the same failure in miniature. The shipped TIFFs carry `Software: tifffile.py`,
`XResolution: (1, 1)` and an `ImageDescription` of exactly `{"shape": [h, w]}` — the corpus
was losslessly recompressed, pixels bit-identical, FEI tags not preserved — and the
pre-compression copies carry only print DPI (768 and 384, `ResolutionUnit=2`), which is not
specimen scale. The microscope recorded the field width; a re-save with a tool that does not
preserve private TIFF tags threw it away without saying so; and the consequence is that 0 of
62 images can be calibrated automatically and every scale here has to be marked by hand.

The reader exists anyway, for files that are still originals, and it is honest about
returning `None` on all 62 here rather than reporting a feature that appears to work. It is
tested against synthetic FEI and ZEISS files, not against this corpus, because this corpus
cannot test it.

## Provenance

Each CSV gets a `<image>_provenance.json` beside it recording the calibration and its
source, the model file and its mtime, the export timestamp, the crack count, and the column
list. Without it, once a CSV leaves the machine no number in it can be traced to a model, a
threshold, or a scale.

## Known limitations

**Fragment merging — fixed, recorded because the CSV changed.** This CSV used to re-label
the bare crack mask, so a main crack the MST step had merged from several large fragments
came out as one row per fragment: a count higher than a person reading the overlay would
give, and a `SkeletonLength_px` that understated the headline crack. `measure_stage` now
unions `stage["bridge_mask"]` before labelling, and drops any component containing no
originally-detected crack pixel so a connector cannot invent a region on its own. Each row
carries `NFragmentsMerged`, which is 1 when the pipeline found the crack whole — so the
merge is visible rather than something to take on trust. Counts from CSVs written before
this are not comparable with counts written after.

**Censored lengths.** A crack touching the frame edge continues outside it, so
`SkeletonLength_px` is a **lower bound**, flagged by `LengthIsCensored`. The bias is not
random: the longest cracks are the most likely to run off the edge, so the statistic most
affected is longest-crack-per-frame, which is exactly what a fatigue study reports.
`aggregate.py` refuses that figure where censoring is unknown rather than taking a maximum
over lower bounds. A survival-style estimator would be the proper treatment and is not
attempted here.

**Self-defined shape measures.** `Tortuosity` and `BoundaryRoughness` are computed as
defined in this file and follow no published standard, so they are comparable within this
corpus and not across studies.
