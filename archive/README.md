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
- `train_classifier.py`, `batch_apply.py` — the pre-app CLI pair, each referenced
  only by the other. Training is now `train_v3_weighted.py` and batch rendering is
  `regenerate_templates.py`
- `render_final_overlays.py`, `render_v3_overlays.py`, `sam_union_overlays.py` —
  three more overlay renderers. All superseded by `regenerate_templates.py`, which
  is the only one that calls the app's own pipeline. Four scripts whose docstrings
  all said "render the overlays" was the single most misleading thing in the tree
- `build_fusion_cache.py`, `eval_fusion.py` — the score-fusion experiment, which
  did not beat the plain union and was dropped

## superseded_models/
- `crack_classifier_v2.joblib` — SVC. Best AUC of any model tried (0.946) and
  **unusable**: at threshold 0.5 it labels all 1,285 regions crack (specificity
  0.0%). Kept as the concrete counterexample to "pick the highest AUC".
- `mar_crack_classifier.joblib` — from when MAR was a separate project, before the
  datasets were merged
- `crack_classifier_v2_metrics.json` — its metrics
- `crack_classifier_v3_weighted_REJECTED_BY_GATE.joblib` — the retrain candidate the
  deployment gate turned down: 4,128 training rows against production's 4,110, and
  held-out AUC 0.9153 against 0.9252. Kept because it is the concrete artifact behind
  the README's claim that retraining will not deploy a worse model. Threshold 0.628,
  which is also why a threshold must never be hardcoded at a call site.
- `crack_classifier_pre_per_image_weighting.joblib` — the fit from before per-image
  sample weighting, the change that moved held-out AUC from 0.729 to 0.925. It records
  neither a threshold nor a training-row count, which is itself the tell: those fields
  were added when the weighting was.

  Both of these were briefly deleted outright on the mistaken belief that they were
  byte-identical copies of the production model. They are 3,339 and 1,653 bytes against
  production's 4,159, and are three different fits.

## superseded_diagrams/
- `full_workflow_260708_316_H_b2_front_CBS_002.svg` — the pre-unified,
  two-model architecture. `docs/diagram/` keeps only the `_unified` one, which is
  what the code actually does now
- `command_guide.svg` — a guide to a multi-command workflow that no longer
  exists; setup is one command

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
