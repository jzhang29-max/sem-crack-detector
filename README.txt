Crack detection export
======================
masks/    crack = BLACK on white
overlays/ detected crack burned into the source image in red
regions/  one row per connected crack region, per image
summary.csv  one row per image

All lengths and areas are in PIXELS. The scale bar lives in the
SEM databar, which this pipeline crops off before analysis, so no
micron factor is recoverable here -- multiply by the microscope's
own um/px value.

Human corrections are already included: they override the model
wherever they exist.


sets/     one folder per specimen set (aggregate.specimen_key), hard-linked
analysis/ per-set crack analysis: CRACK_ANALYSIS.md, metrics CSVs, figures/
tools/    the scripts that built sets/ and analysis/, re-runnable
