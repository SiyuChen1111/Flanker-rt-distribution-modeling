# VAM-studying Agent Guide

## Project Role

This is a research repository for LIM / Flanker reaction-time modeling. The current main line is a VGG/layer-to-time evidence pipeline feeding a four-choice Wong-Wang accumulator and R5 readout diagnostics for young and older participant groups.

## Current Result Spine

- Main retained model package: `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/`
- Supervisor follow-up diagnostics: `artifacts/results/r5_supervisor_followup/`
- Artifact documentation index: `artifacts/results/ARTIFACT_DOCS_INDEX.md`
- Reproducible follow-up script: `code/scripts/run_r5_supervisor_followup.py`
- Long-form follow-up note: `docs/r5-supervisor-followup.md`

## Common Commands

Run from the repository root unless a script says otherwise.

```bash
python code/scripts/run_r5_supervisor_followup.py
pytest tests/test_trial_variable_conflict_recovery.py
pytest tests/test_flanker_rt_bin_fitting.py
```

## Working Rules

- Treat `best_model_R5_combined_best` as a diagnostic model-development package, not a final full-cohort fit.
- Do not overwrite existing result bundles. Put new supervisor or mechanism follow-ups in a new directory under `artifacts/results/`.
- Before using old generated Markdown under `artifacts/results/`, check `artifacts/results/ARTIFACT_DOCS_INDEX.md` for its category and trust level.
- Validate derived CAF/CRF tables against raw trial-level rows before plotting or interpreting them.
- Use actual RT coordinates for CAF/CRF figures; do not relabel quantile indices as RT.
- Keep random seeds fixed and recorded for simulations.
- Do not claim the accumulator works perfectly. Use cautious terms such as "passes a minimal sanity check", "partially supported", or "unresolved".
- Keep user-facing summaries concise and in Chinese; keep technical work rigorous and verified.

## Key Code Paths

- Evidence cache and layer-to-time inputs: `code/scripts/run_natural_layer_to_time_var_ww_diagnostic.py`
- Wong-Wang state update and differentiable crossing: `code/scripts/vgg_wongwang_lim.py`
- R5 fitting and readout utilities: `code/scripts/run_representative_extreme_age_subset_fitting.py`, `code/scripts/optimize_natural_layer_to_time_rt_shape.py`
- R5 supervisor diagnostics: `code/scripts/run_r5_supervisor_followup.py`

## Documentation Map

- `README.md` gives the human-facing overview and current reading order.
- `docs/model_framework_summary.md` explains the model flow.
- `docs/current_results_and_limitations.md` summarizes the current interpretation and limitations.
- `docs/r5-supervisor-followup.md` records the latest supervisor-question diagnostic and where to find each output.
- `artifacts/results/ARTIFACT_DOCS_INDEX.md` classifies generated result Markdown into current, supporting, exploratory, and legacy tiers.
