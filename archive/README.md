# Archive

Nothing here is used by the app. It is kept because each item is evidence for a
decision recorded in MODEL_VALIDATION_BENCHMARK.md, and deleting it would leave
those claims unsupported. Safe to delete entirely if you don't care about
reproducing the reasoning.

## superseded_code/
Earlier versions replaced by something better, not broken code:
- `pipeline_stages.py` — two-model version; superseded by `pipeline_stages_unified.py`
- `generate_full_workflow_diagram.py`, `generate_scientific_diagram.py` — superseded
  by the `_unified` diagram generator
- `bootstrap_from_ilastik.py` — one-time import from an external tool, never rerun
- `compare_training_strategies.py` — the experiment that chose the training scheme
- `seed_templates_with_steel_model.py.DEPRECATED` — reimplemented only Pass 1, so
  templates disagreed with what the app rendered; replaced by
  `regenerate_templates.py`, which calls the app's own pipeline
- `train_and_evaluate.py` — superseded by `train_v3_weighted.py`, which adds the
  per-image weighting that turned held-out AUC 0.729 into 0.925

## superseded_models/
- `crack_classifier_v2.joblib` — SVC. Best AUC of any model tried (0.946) and
  **unusable**: at threshold 0.5 it labels all 1,285 regions crack (specificity
  0.0%). Kept as the concrete counterexample to "pick the highest AUC".
- `mar_crack_classifier.joblib` — from when MAR was a separate project, before the
  datasets were merged
- `crack_classifier_v2_metrics.json` — its metrics

## analysis_scripts/
One-off measurements whose *results* are in the benchmark doc. Rerunnable if you
want to check a number:
- `label_bias_experiment.py` — found that 95% of not-crack labels come from one
  image, and that per-image weighting fixes the resulting bias
- `diagnose_cv.py` — showed why averaging per-fold AUC gives 0.65 here and why
  pooled out-of-fold is the right estimator
- `analyze_sam_additions.py`, `sam_addition_breakdown.py` — characterised what SAM
  adds (31.6% off-specimen background, 2.3% bright ridges)
- `sam_addition_contactsheet.py`, `marginal_calls_sheet.py` — built review sheets
- `sam_union_v2.py`, `sam_union_summary.py` — the filtered-union experiment
- `apply_v3_all_images.py`, `extract_extended_features_all25.py`,
  `build_original_ledger_unified_features.py` — superseded data-prep passes
