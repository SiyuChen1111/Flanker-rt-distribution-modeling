# LIM / Flanker RT Distribution Modeling

This repository studies reaction-time behavior in the Lost in Migration (LIM) / Flanker task. The project connects visual evidence, conflict effects, and decision dynamics to explain human response time, accuracy, and age-related differences.

## Current focus

The current main line is the representative extreme-age subset workflow:

- young group: `young_20_29`, 5,000 representative trials
- older group: `older_80_89`, 5,000 representative trials
- current retained best candidate: `R5_combined_best`
- main result bundle: `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/`

This is a model-development and diagnostic result, not a final full-cohort age-group conclusion.

## Current conclusion

The current best candidate, `R5_combined_best`, combines group-specific timing variability with Wong-Wang/readout parameters. On the representative subset, it brings the model's mean reaction times close to human behavior:

- young 20-29: human/model mean RT is about 0.603 / 0.612 seconds
- older 80-89: human/model mean RT is about 0.941 / 0.919 seconds

The model still overestimates accuracy and does not fully match human correct/error patterns. The results should therefore be read as evidence about a promising mechanism, not as a completed human RT model.

The latest supervisor follow-up diagnostic is saved in `artifacts/results/r5_supervisor_followup/` and can be regenerated with `code/scripts/run_r5_supervisor_followup.py`. It recomputes CAF/CRF from raw trial-level rows using actual RT coordinates, audits the R5 state/readout path, and separates fixed-time `S(t)` distributions from first-passage-time behavior.

## Earlier mechanism branch

The earlier public mechanism branch is still retained because it explains the foundation of the project:

- variational evidence sampling in the encoding stage
- DMC-style conflict modulation
- Wong-Wang decision dynamics
- fast-error behavior as an intermediate mechanism result

That branch is useful for understanding the model logic, but it is no longer the only main description of the repository.

## How to read this repo

Start with the current best-model materials:

1. `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/README_best_model_R5_combined_best.md`
2. `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/summaries/representative_extreme_age_diagnostic_summary.md`
3. `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/summaries/representative_fitting_summary.md`
4. `docs/r5-supervisor-followup.md`
5. `docs/current_results_and_limitations.md`
6. `docs/model_framework_summary.md`

Then read the teaching and mechanism examples:

1. `examples/dmc_var_ww_minimal_pipeline.md`
2. `examples/dmc_var_ww_minimal_pipeline.ipynb`
3. `examples/toy_recurrent_ww_dmc_demo.ipynb`
4. `examples/single_trial_rt_variability_probe.ipynb`
5. `artifacts/results/rt_model_dmc_var_ww/summary_smoke.md`
6. `artifacts/results/rt_model_variational_ww_synthesis/var_ww_mechanism_memo.md`

## Key files

Current representative extreme-age workflow:

- `code/scripts/build_representative_extreme_age_subset.py`
- `code/scripts/build_representative_extreme_age_vgg_cache.py`
- `code/scripts/run_representative_extreme_age_subset_fitting.py`
- `code/scripts/make_representative_extreme_age_figures.py`
- `code/scripts/run_r5_supervisor_followup.py`

Earlier DMC + variational evidence + Wong-Wang workflow:

- `code/scripts/stage1_evidence_sampler.py`
- `code/scripts/vgg_wongwang_lim.py`
- `code/scripts/train_variational_ww_smoke.py`
- `code/scripts/train_dmc_var_ww_smoke.py`
- `code/scripts/run_subject_level_dmc_var_ww.py`
- `code/scripts/analyze_subject_level_dmc_var_ww.py`

Examples:

- `examples/dmc_var_ww_minimal_pipeline.ipynb`
- `examples/toy_recurrent_ww_dmc_demo.ipynb`
- `examples/single_trial_rt_variability_probe.ipynb`
- `examples/build_dmc_var_ww_minimal_pipeline.py`
- `examples/build_single_trial_rt_variability_probe.py`

## Key results

Results directory navigation:

- `artifacts/results/README.md`
- `artifacts/results/ARTIFACT_DOCS_INDEX.md`
- `artifacts/results/artifact_docs_inventory.csv`

Current best-model result bundle:

- `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/`
- `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/results/model_comparison_all_models.csv`
- `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/results/best_model_group_metrics.csv`
- `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/figures_publication/representative_extreme_age_diagnostic_figure.png`

Supervisor follow-up diagnostic bundle:

- `artifacts/results/r5_supervisor_followup/01_reproducibility_and_active_config_audit.md`
- `artifacts/results/r5_supervisor_followup/02_CAF_actual_RT.png`
- `artifacts/results/r5_supervisor_followup/03_CRF_actual_RT.png`
- `artifacts/results/r5_supervisor_followup/05_first_passage_distribution_summary.csv`
- `artifacts/results/r5_supervisor_followup/10_supervisor_response_summary_chinese.md`

Earlier retained mechanism-test results:

- `artifacts/results/rt_model_dmc_var_ww/summary_smoke.md`
- `artifacts/results/rt_model_dmc_var_ww/single_trial_rt_probe/`
- `artifacts/results/rt_model_variational_ww_synthesis/var_ww_mechanism_memo.md`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/`

## Repository layout

- `code/`
  Source code for model training, analysis, diagnostics, and figure generation.
- `code/scripts/`
  Main runnable scripts for the current workflow and earlier mechanism tests.
- `docs/`
  Short project notes explaining the model framework, current limitations, and public update status.
- `examples/`
  Teaching notebooks and minimal demonstrations of the DMC + variational evidence + Wong-Wang mechanism.
- `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/`
  Current representative extreme-age result bundle, including the retained `best_model_R5_combined_best` package.
- `artifacts/results/rt_model_dmc_var_ww/`
  Earlier retained DMC + variational evidence + Wong-Wang mechanism-test results.
- `artifacts/results/diagnostics/`
  Exploratory diagnostics and side analyses that are not the main result bundle.
- `artifacts/results/repro_legacy_interim/`
  Legacy and intermediate reproduction materials retained for comparison.
- `data/`
  Local data inputs and derived data folders.
- `tests/`
  Test files for checking selected code paths.

## Notes

Older experiments and archives remain in the tree, but the current README centers on the representative extreme-age `R5_combined_best` result and the mechanism work that supports it.

Do not interpret the current result bundle as a full-data final fit. It is the current best diagnostic model-development state, with clear remaining limitations in accuracy and correct/error behavior.
