# CrackTrace — prototype

**Status: synthetic validation only. Not run on a real SEM frame. Not compared to the
shipped pipeline.** Every design claim is a hypothesis with a feasibility check.

## The idea

Every pixel method (U-Net, SAM) represents a crack as an **area** and recovers
connectivity afterwards. A crack is a 1-D curve in 2-D. The two pathologies measured in
this corpus are artefacts of that choice — fragmentation (the shipped pipeline needs a
Dijkstra merge as *cleanup*) and width confusion (area fraction tracks detected width at
ρ = +0.86; "crack" widths run 5–520 px). CrackTrace predicts the **centreline graph**
directly, so connectivity is a property of the output.

Four design claims:

1. **Curve output, not area.** Connectivity by construction.
2. **Global.** A minimum spanning forest over a geodesic cost field, solved on the whole
   frame. Cracks here run to the frame diagonal (~7,385 px); a tiled CNN cannot link
   fragments 5,000 px apart. (Note: MegaSeg, *Med. Image Anal.* Jan 2026, does 67 MP
   end-to-end, so "no tiling" alone is no longer a differentiator — the *curve output* is.)
3. **Scale-native.** Every filter is parameterised in **microns**, converted per frame
   via µm/px = HFW/width. σ = 0.4 µm means the same thing across the corpus's 249×
   magnification range; σ = 3 px does not.
4. **It consumes superset labels natively** — ~~the one gap the 2026 sweep found
   unoccupied~~. **That framing is withdrawn (2026-09-18): superset supervision is owned.**
   91% of this corpus's 70 M marked pixels are brush strokes of median 59 px (max 413 px)
   asserting a region containing a ~3 px crack — useless as a pixel target, near-exact as a
   **corridor constraint on a path**, and that remains a sound design choice. It is simply
   not novel: a bounding box is a superset label by construction, and the corridor constraint
   is the **tightness prior** of Kervadec et al. (MIDL 2020, PMLR 121:365–381,
   arXiv:2004.06816) restated for a curved region. The formal setting is Superset Label
   Learning (Liu & Dietterich, ICML 2014, PMLR 32:1629–1637), and the over-inclusive crack
   case already ships a **shrink module** (Eng. Appl. Artif. Intell. 2024,
   10.1016/j.engappai.2024.108497, IoU 77.53%). The multiple-instance framing — each corridor
   a positive bag scored by its top-quantile crackness — is retained on its merits.

## Measured baseline it must now beat (added 2026-09-15)

SAM 3 was run on 16 hand-labelled tiles from this corpus, and **that experiment is invalid** —
the model input was built from the green channel of the annotated overlay, and opaque red
(225, 25, 25) has green 0, so the label was written into the input as black pixels on 14 of 16
tiles. See `../analysis/sam3/LEAK_POSTMORTEM.md`. Every number that was quoted here — recall
0.969–1.000 on 10/16, IoU 0.59–0.74, and the "fails silently" reading — is withdrawn.

What survives as a design target is narrower and does not depend on that run:

- a single global grey threshold, oracle-tuned per tile, reaches **median IoU 0.384** on the
  clean input. That is the bar, and it was never stated before;
- SAM 3 rescales all 200 per-query scores by **one global presence scalar** per (image, prompt):
  `out_probs = sigmoid(pred_logits) * sigmoid(presence_logit_dec)`, then thresholds **per query**
  (`sam3_image_processor.py:195-200`). Whether the image returns *anything* is therefore decided
  by `s_i · max_j q_ij`, and when that is below τ nothing survives to inspect. Measured on this
  corpus, that scalar spans **112×** across four synonymous prompts (median 0.9102 for `crack`,
  0.0081 for `fracture`), with prompt identity explaining 81.7% of its variance against 10.6%
  for the image. The mechanism is architectural — readable in the source, not a finding; the
  112× span is the measurement. See `../analysis/sam3/CLEAN_RUN_RESULTS.md`.

That second point is structural, verifiable by reading the source, and is the one CrackTrace's
design actually addresses.

## What was measured (synthetic only)

| check | result |
|---|---|
| scale-native features at 0.05 µm/px | crack:background crackness ratio 14.0× |
| same at 0.20 µm/px (coarser) | 438×; the 0.15 µm scale correctly **dropped** as unresolvable |
| curve coverage, two sinusoidal cracks | 95.8 % and 98.0 % |
| bright decoy scratch (must be ignored) | 5 px picked up |
| runtime, 900×900, 60 terminals | 2.7 s, 44 edges |
| corridor as a **pixel** label | **96.8 % false-negative** |
| corridor as a **path** constraint | network stays inside it |
| paired metric, identical networks | F = 1.000 |
| paired metric, 0.5 µm shift at 2 µm tolerance | F = 1.000 |
| paired metric, perpendicular networks | F = 0.405 |

## To take this further

1. Fit `fit_weights()` on real corridors from `interior_active_learning/paint/`.
2. Run on the 8 registration-confirmed CBS/ETD pairs
   (`analysis/paired_detector.csv`, Jaccard ≥ 0.5) using per-frame HFW from
   `analysis/scale_hfw.csv`.
3. Compare `centreline_agreement()` against the **raw** (pre-correction) pipeline output —
   generate it with `SEMCRACK_PAINT_DIR` pointed at an empty directory, and pass frame
   names as **separate** shell arguments (zsh does not word-split an unquoted variable,
   which is why the first attempt silently produced nothing while exiting 0).

The paired metric is the point: it needs **no ground truth**, which this corpus does not
have. Two detectors on one field must contain the same cracks, so higher cross-detector
agreement means the method is measuring the material rather than the instrument.

## API

```python
import cracktrace as ct
upp = ct.um_per_px(hfw_um=319.0, width_px=6144)      # 0.0519 µm/px
r   = ct.trace(img_float01, upp, pool=4)             # -> net, length_um, edges, ...
ag  = ct.centreline_agreement(rA["net"], rA["upp_eff"],
                              rB["net"], rB["upp_eff"], tol_um=2.0)
```
