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
| `best_detector.py` | **the best-scoring arm of this evaluation** — *not* the app's detector, which is the two-pass pipeline in `interior_active_learning/` and does not import this. `detect(grey) -> mask`. Meijering ridge, σ 1–4, top 1% (quantile 99, the whole-frame deployment setting), drop ≤32 px |
| `best_config.json` | its parameters, chosen by leave-one-frame-out |
| `detect_all_frames.py` | runs it over all 62 originals at full resolution → `detector_masks/`, `detector_overlays/` |
| `contact_sheet.py` | every frame on one page, ordered worst-last → `detector_contact_sheet.png` |
| `adaptive_threshold.py` | nine label-free per-frame threshold rules, to fix over-prediction |
| `artefact_filter.py` | scratch / pit rejector. **Not shipped** — significant by rank, net-zero by mass; kept as the record of what orientation and width cannot separate |
| `make_results_pdf.py` | every frame's overlay beside its binary mask, one page each → `detector_all_frames.pdf` |

## Looking at the results

`index.html` is a browsable page for all 62 frames — filter by name, sort by score or by
predicted area, click any frame for the full-size overlay. It reads `detector_all.json`
(committed, 12 KB) for the scores and `detector_overlays/` for the images.

```bash
cd crack_export/analysis/sam3 && python3 -m http.server 8777   # then open http://localhost:8777/
```

It must be **served**, not opened by double-click: as a `file://` URL the browser blocks
it from reading its own JSON and the page stays blank. It says so on the page now, and it
also says so when the overlays are missing — on a fresh clone the scores and the frame
list render but the images do not, because `detector_overlays/` is gitignored. Run
`detect_all_frames.py` first if you want the pictures.

`detector_contact_sheet.png` (committed) is the same 62 frames on one page, worst last,
if you just want one look without serving anything.

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
| `shim/sam3_preload.py` | pre-registers `sam3.model.edt` so importing sam3 never reaches triton; raises loudly if anything calls in |
| `shim/cuda_redirect.py` | rewrites torch's hardcoded `device="cuda"` for the duration of the model build, so Apple Silicon can construct it |
| `shim/pkg_resources.py` | `resource_filename()` only, backed by importlib.resources; setuptools ≥ 81 dropped the real one and sam3 still imports it |
| `analyse_clean_run.py` | per-prompt table, paired against the contaminated run |
| `serd_eval.py` | reads SAM 3's field **before** the presence gate |
| `trained_lofo.py`, `ensemble_lofo.py` | supervised and ensemble arms |
| `run_omnicrack.py`, `omnicrack_eval.py` | the published nnU-Net, run on our tiles |
| `make_figures.py` | regenerates every figure from committed artefacts |

## Generated, gitignored, regenerable

`tiles/` `tiles_all/` `masks/` `serd/` `omnicrack/` `detector_masks/` `detector_overlays/`
`runs/` `detector_all_frames.pdf` — roughly 250 MB. `detect_all_frames.py` rebuilds the detector outputs in ~15 min; the
SAM 3 arms need the isolated venvs described in `../README.md`.

Deleted 2026-09-19 as part of a cleanup: `serd/`, `omnicrack/`, `masks/` and
`__pycache__/` (concluded experiments, all regenerable); `sam3_examples_…png` (it rendered the
LEAKED input — the retraction text in `LEAK_POSTMORTEM.md` is the surviving record, and keeping
a contaminated artefact around is a liability); and `fig_qualitative.png` (a SAM 3 tile view,
superseded by `detector_contact_sheet.png`).
