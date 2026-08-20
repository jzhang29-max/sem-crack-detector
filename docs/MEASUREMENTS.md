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

## Calibration

Calibration is **per image**, explicit, and recorded with its provenance in
`interior_active_learning/paint/calibration.json`. Three routes:

- **`scale_bar`** — read the label off the burned-in bar (e.g. `400 µm`) and mark its two
  end ticks. `µm/px = label ÷ |x2 − x1|`.
- **`hfw`** — horizontal field width from the info panel ÷ image width in pixels.
- **`manual`** — typed directly.

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

Vendor metadata is not available as a fallback: the shipped TIFFs carry
`Software: tifffile.py` and `XResolution: (1, 1)` — the corpus was losslessly recompressed,
pixels bit-identical, FEI tags not preserved — and the pre-compression copies carry only
print DPI (768 and 384, `ResolutionUnit=2`), which is not specimen scale.

## Provenance

Each CSV gets a `<image>_provenance.json` beside it recording the calibration and its
source, the model file and its mtime, the export timestamp, the crack count, and the column
list. Without it, once a CSV leaves the machine no number in it can be traced to a model, a
threshold, or a scale.

## Known limitation

`crack_measurements.py` re-labels the final crack mask with `measure.label`, so a main
crack that the MST step merged from several large fragments is still reported as one row per
fragment. `stage["bridge_mask"]` holds the connector geometry that would join them and is
currently unused. Until that is wired in, a crack count from this CSV can exceed the count
a person reading the overlay would give, and per-fragment `SkeletonLength_px` understates
the full crack's length.
