# Postmortem: the input encoded the label

*2026-09-18. Self-reported. Nothing here was found by review; it was found by an agent
instructed to attack this corpus's strongest claims.*

## What happened

`make_tiles.py` built the model input from the **green channel of the annotated overlay**:

```python
red = (a[..., 0] > 150) & (a[..., 1] < 80) & (a[..., 2] < 80)
g = a[..., 1].copy()           # green channel is unaffected by the red overlay
```

The comment states the belief that made the bug invisible. The overlay burns **opaque red
(225, 25, 25)**, whose green channel is a constant **25** -- far below the 80 the threshold
used. (This paragraph said **0** until 2026-09-21; see "A third wrong diagnosis, made while
writing this retraction" below, which is where that error was caught and which this line had
gone on contradicting.) Every labelled pixel therefore reached the model as black. The `red` mask was even computed on the line above and used only for a metadata
field — the information needed to notice the bug was already in the function.

## Extent, measured

`leak_check.py` reruns the old input as a positive control:

| | invalid input | clean input |
|---|---|---|
| median within-label std | **0.000** | 33.890 |
| median modal share inside label | **1.0000** | 0.4121 |
| tiles with a written-in label | **14 / 16** | **0 / 16** |
| median best-threshold IoU | 0.3705 | 0.3843 |
| median max reachable recall | 1.0000 | 0.9689 |

The two non-contaminated tiles are from `260622_316_H_b4_CBS_02`, whose overlay red is
disjoint from its correction mask (0.00% overlap in both directions, 10,290 label px against
2,062,487 red px) — clean by accident, not by design.

## Two wrong diagnoses I made while fixing it

**1. I first reported the leak as 16/16.** That came from a bare threshold `green < 80`
reaching recall 1.000 on all 16 tiles. But these micrographs are **clipped at acquisition**:
`260622_316_H_b2_front_CBS_01` holds 9.4% of its pixels at exactly 0, `AS_24hr_BSE_Side_008`
holds 69.0% at exactly 65535, and one frame carries only 256 distinct values in a uint16
container. On a clean tile, 97.8% of labelled pixels are exactly 0 in the *raw* original, so a
threshold at 0 reaches recall 0.978 with no leak whatsoever. Threshold skill is not evidence of
contamination — cracks are genuinely the darkest pixels.

**2. I built a boundary-coincidence test and it does not work.** The idea: a burn-in puts an
image edge exactly on the label boundary. Measured against the known-contaminated input as a
positive control, the ratio of mean boundary gradient at the true label to displaced labels was
median **3.76** contaminated versus **3.05** clean, ranges 0.82–8.80 and 0.90–5.56 — fully
overlapping. Discarded rather than shipped. Cracks have edges where their labels are; that is
the point of the labels.

What discriminates is **exactness**: a burn-in writes one value, so within-label variance goes
to exactly zero even where the underlying micrograph was bright. That is the only criterion
`leak_check.py` applies, and its stated limit is that an alpha-blended annotation would evade
it — for which the boundary test would have been the natural catch, and it does not work here.

## The fix, and why it is structural rather than statistical

- `align_originals.py` registers every label frame to its raw `original/*.tif`. 9/9 at
  ncc ≥ 0.99, five at exactly **1.0000**. Two frames are square crops at non-obvious offsets
  — (278, 1016) and (997, 1974) — recovered, not assumed.
- `make_tiles.py` reads the grey from the registered original and writes the overlay only as
  `*_overlay_REFERENCE_DO_NOT_FEED.png`. No file a loader would reach for contains annotation.
- `leak_check.py` fails with exit 1 on any contaminated tile and carries the old input as a
  permanent positive control: if that arm ever passes, the guard has stopped working.

Registration also needed three attempts, recorded in `align_originals.py` because each failure
is a general lesson: a coarse **grid** over offsets scores ~0 at every candidate on SEM texture
(argmax of noise); whole-frame FFT correlation at **1/4 scale** put one frame 566 px from a true
offset that scores exactly 1.0000; what works is correlating a small red-free patch at **full**
resolution and letting full-frame ncc arbitrate.

## What it cost, and the one thing it bought

Retracted: `SAM3_ON_SEM_CRACKS.md` in full, `GAP_CONFOUND.md` in full, and — worse — a
*correction I had made to `VERDICT_2026.md` §6 on the strength of this result*. That section
originally said SAM-family models remain poor on thin low-contrast curvilinear structures. I
overwrote it as disproven. It was not disproven, and it now stands unrefuted.

The one thing gained: nobody had measured the **trivial baseline**. On the clean input, an
oracle-tuned single global threshold reaches median **IoU 0.384**. SAM 3 scored 0.119 (0.0775 over all 16 tiles) *with*
the answer visible. Any method proposed on this corpus has to beat 0.384, and that number had
never appeared in any of these documents.

## A third wrong diagnosis, made while writing this retraction

I wrote, in every file above and in the commit message, that the overlay burns **pure red
(255, 0, 0), whose green channel is 0**. It does not. Measured across all 16 tiles, the painted
region contains **exactly one** RGB triple — **(225, 25, 25)** — over 1,039,604 pixels, so the
green value written is **25**. The bug is unchanged (25 is still far below the 80 the threshold
used, and still a constant), but the number I published while retracting a number was wrong.

Two independent things caught it: an agent's own audit of this corpus had already recorded the
(225,25,25) triple in `crack-depth-3d/docs/PAPER_STRATEGY.md` §5b, and `verify_claims.py` now
carries both the triple count and the green value as registered claims, so the next drift fails
CI rather than being read.

The registry caught a fourth slip the same way: I wrote "four frames register at ncc exactly
1.0000" when the artefact says **five**.

## The general lesson

The old input passed every check I had, because every check I had scored the prediction. None
inspected the input. A benchmark harness needs a guard that asks whether the input already
contains the target — and a positive control, so the guard is known to be able to fail.

See also: `../LABEL_GRANULARITY.md` (the labels are region assertions, not outlines),
`../../tools/verify_claims.py` (the 33-claim recomputation registry this should have been part
of from the start).
