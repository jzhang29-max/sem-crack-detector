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

all_overlays.pdf   every frame's hand-annotation overlay, one page each (62 pages)
all_masks_bw.pdf   the same frames as black-and-white masks (62 pages)

  These are the HAND-ANNOTATION exports. The DETECTOR's equivalent is
  analysis/sam3/detector_all_frames.pdf, built by analysis/sam3/make_results_pdf.py
  (gitignored -- regenerate it, or browse analysis/sam3/index.html instead).

  A second, 61-page copy of each of these lived in result/ until 2026-09-21. It was one
  frame short of the corpus and nothing referenced it, but nothing said which copy was
  current either, so it was removed. If you have a clone from before that date, the
  62-page copies here are the ones to use.
