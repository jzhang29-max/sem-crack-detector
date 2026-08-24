# The experiment scripts

37 scripts plus `detector_config.py` (a helper, not a script). They produced the figures and
tables in `docs/MODEL_VALIDATION_BENCHMARK.md` and the results in `docs/PAPER_OUTLINE.md`.

**All 37 were run on 2026-08-23 against a clean checkout.** None crashes from a code defect.
21 of them refuse to run because an input they need is *derived and not shipped* — that is a
data prerequisite, not a bug, and this file says exactly what to generate.

## The prerequisite: interior candidate CSVs

Most scripts read `interior_active_learning/candidates/*_interior.csv`. Only **3** of those
ship, holding **34 rows, all of them crack, with no negatives at all**. That is enough to load
but not to train or cross-validate, which is why a stratified split over both classes fails.

Generate the full set first — hours, one pass per frame:

```bash
./.venv/bin/python3 interior_active_learning/code/run_all_candidates.py
```

Four scripts additionally need `candidates/original_ledger_unified_features.csv`, which is
produced by the extended-feature ingest and is also not shipped.

## What runs without it

These read masks, images or tracked result JSONs rather than the candidate pool, so they work
on a clean checkout:

| script | note |
|---|---|
| `fragmentation_check.py` | the SAM 2 fragmentation measurement behind the `--sam2` decision |
| `failure_mode_magnitudes.py` | detector-independent; sizes each failure mode |
| `three_state_conformance.py` | audits the 1/2/3/0 correction-mask states |
| `benchmark_confusion_matrix.py`, `benchmark_confusion_matrix_unified.py` | read the tracked benchmark JSONs |
| `benchmark_roc_curves.py`, `benchmark_roc_curves_unified.py` | same |

Long-running but working — each produces per-frame output as it goes, and each exceeded a
10-minute probe rather than failing:

`naive_baselines.py`, `scoring_convention_bias.py`, `sparsity_sensitivity.py`,
`threshold_sensitivity.py`, `oos_convention_gap.py`, `erased_state_sensitivity.py`,
`sam2_hybrid.py`, `cheap_proposers.py`, `proposal_harness.py`

`sam2_hybrid.py` and the `*.sam2_refine.*` variants also download a SAM 2 checkpoint on first
use and prompt the model once per accepted region, so budget minutes per frame, not seconds.

## What needs the candidate pool first

Grouped by what they actually complain about, so a failure is recognisable:

- **`n_splits=5 > number of groups: 3`** — needs candidates from at least 5 frames:
  `alternative_algorithms`, `feature_engineering`, `per_type_models`,
  `imbalance_and_calibration`, `hybrid_rule_plus_ml`, `benchmark_model_comparison`,
  `benchmark_learning_curve`, `round3_simple_untried_algorithms`
- **`needs samples of at least 2 classes`** — needs negatives, and the shipped 34 rows are all
  crack: `benchmark_decision_boundary`, `benchmark_feature_importance`,
  `refine_finer_grid_pareto`, `refine_soft_score_alternative`,
  `refine_stability_and_sensitivity`, `refine_third_gate_feature`, `round3_finer_grid_retry`,
  `round3_synthetic_negative_augmentation`
- **missing `original_ledger_unified_features.csv`** — `benchmark_extended_features`,
  `benchmark_feature_importance_unified`, `benchmark_learning_curve_unified`,
  `benchmark_model_comparison_unified`
- **`assumes exactly 2 known interior_fill negatives`** — `round3_margin_maximization`, which
  asserts a specific labelling state rather than discovering it

## Two things to know before quoting anything these print

Every script **pins its detector** through `detector_config.py` instead of inheriting the
pipeline default, because that default has already changed once: SAM 2 refinement was briefly
the default and was reverted after it was found to fragment the mask. A result that does not
say which detector produced it is not comparable to one that does.

Running an experiment **can overwrite a tracked result JSON**. The figures and the model caches
are gitignored, but the result JSONs are tracked on purpose so a number in the docs can be
traced to a file. Check `git diff` afterwards: if a measurement moved, that is a finding, not
noise.

Expect one line of harmless churn. Every artifact records the commit it was produced at, so
re-running at a later commit rewrites `git_commit` and nothing else. `failure_mode_magnitudes`
and `fragmentation_check` were both re-run at a clean checkout and came back byte-identical
apart from that one field — so if you see more than `git_commit` change, look at it.

## `--help` on an experiment script runs the experiment

30 of these have no argument handling, so `script.py --help` falls through and does the work
rather than printing anything. That is deliberate-by-neglect rather than dangerous — they write
result JSONs, not models — but it will surprise you. The scripts that could overwrite a deployed
model (`train_v3_weighted.py`, `train_interior_model.py`, `train_unified_model.py`,
`build_training_data.py`, `build_original_ledger_unified_features.py`) were guarded on
2026-08-23 and are covered by the suite's `--help` check; the experiment scripts are not.
