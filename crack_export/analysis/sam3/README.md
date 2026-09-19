# `analysis/sam3/` — what each file is

Thirty scripts accumulated here across a benchmark that changed shape several times. This is the
index. **Read `POSITION_VS_2026.md` first** — it is the result; everything else is how it was
obtained or why an earlier version of it was wrong.

## Read these

| file | what it says |
|---|---|
| `POSITION_VS_2026.md` | **the result.** Where this stands against the 2026 literature, what is stronger here, what is weaker |
| `SOTA_2026.md` | verified 2026 crack-segmentation literature, and the model that is the one to beat |
| `PRIOR_ART_KILL.md` | why the SAM 3 finding is not a contribution — it is in Meta's own abstract |
| `LEAK_POSTMORTEM.md` | how the first run measured its own annotation, and the guard that now prevents it |
| `CLEAN_RUN_RESULTS.md` | the leak-gated SAM 3 numbers, with a verdict table of what died |
| `GAP_CONFOUND.md`, `SAM3_ON_SEM_CRACKS.md` | **⛔ retracted**, kept unedited as the record |

## The detector

| file | what it does |
|---|---|
| `best_detector.py` | **the model.** `detect(grey) -> mask`. Meijering ridge, σ 1–4, top 2%, drop ≤32 px |
| `best_config.json` | its parameters, chosen by leave-one-frame-out |
| `detect_all_frames.py` | runs it over all 62 originals at full resolution → `detector_masks/`, `detector_overlays/` |
| `contact_sheet.py` | every frame on one page, ordered worst-last → `detector_contact_sheet.png` |
| `adaptive_threshold.py` | nine label-free per-frame threshold rules, to fix over-prediction |

## Building the evaluation corpus

Run in this order. Registration before tiling, tiling before the gate, the gate before any model.

| file | what it does |
|---|---|
| `align_originals.py` | registers the 9 fine label frames to their raw originals (ncc ≥ 0.99) |
| `expand_corpus.py` | the same for all 47 labelled frames, one tile each → `tiles_all/` |
| `make_tiles.py` | the 15 disjoint fine-subset tiles → `tiles/` |
| `leak_check.py` | **refuses the run** if any input encodes its label; keeps the known-bad input as a positive control |

## Scoring

| file | what it measures |
|---|---|
| `iou_ceiling.py` | **the key number**: a perfect 3 px trace scores pixel IoU 0.1662 here |
| `methods_bench.py` | 5 methods × OIS / ODS / LOFO / nested LOFO on the fine subset |
| `expanded_bench.py` | the same over 44 frames with width-insensitive metrics |
| `baseline_matched.py` | Otsu / ODS / OIS / information-free null at matched tuning budgets |
| `tolerance_sweep.py` | clIoU_τ; the leader changes with τ |
| `corridor_metric.py` | superset-label scoring — **always read its null panel** |
| `thin_metrics.py` | clDice, Tprec/Tsens, skeleton recall, Betti-0, with decoys |
| `power_analysis.py` | how many frames would be needed: ~40–60, we had 9 |
| `permutation_test.py` | nulls for the presence-magnitude claim |
| `pred_decomposition.py` | where every predicted pixel lands relative to both annotation layers |
| `width_probe.py` | tests whether IoU here just measures brush width (it does not) |

## The SAM 3 arm

| file | what it does |
|---|---|
| `run_real.py` | SAM 3 inference, leak-gated, capturing the presence scalar and pre-gate field |
| `prepare_checkpoint.py` | rebuilds the 3.45 GB checkpoint from the HF cache |
| `shim/` | three shims: triton pre-empt, CUDA redirect, `pkg_resources` |
| `analyse_clean_run.py` | per-prompt table, paired against the contaminated run |
| `serd_eval.py` | reads SAM 3's field **before** the presence gate |
| `trained_lofo.py`, `ensemble_lofo.py` | supervised and ensemble arms |
| `run_omnicrack.py`, `omnicrack_eval.py` | the published nnU-Net, run on our tiles |
| `make_figures.py` | regenerates every figure from committed artefacts |

## Generated, gitignored, regenerable

`tiles/` `tiles_all/` `masks/` `serd/` `omnicrack/` `detector_masks/` `detector_overlays/`
`runs/` — roughly 250 MB. `detect_all_frames.py` rebuilds the detector outputs in ~15 min; the
SAM 3 arms need the isolated venvs described in `../README.md`.

`sam3_examples_INVALID_contaminated_input.png` is kept **only** as the record of the retracted
run. It visualises the leaked input. Do not use it for anything.
