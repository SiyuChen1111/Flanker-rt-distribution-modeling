# LIM / Flanker RT Distribution Modeling

This repository studies reaction-time behavior in the Lost in Migration (LIM) / Flanker task. The project connects visual evidence, conflict effects, and decision dynamics to explain human response time, accuracy, and age-related differences.

## Current focus

The current main line is an audited representative-subset extension covering all seven age groups:

- groups: `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70-79`, `80-89`
- 75 participants and 5,000 selected trials per group
- corrected choice/readout at the sustained-crossing step, with a shared decision-time scale and age-specific non-decision time
- result bundle: `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/`

This is a model-development and same-data diagnostic result, not a final held-out or hierarchical full-cohort fit.

## Current conclusion

The all-age update keeps the VGG evidence, Wong-Wang dynamics, choice rule, readout rule, and crossing definition fixed. A shared decision-time scale of `0.27` and age-specific `t0` values reduce the mean condition RT error from 95.6 ms to 3.2 ms. Choice and the formal readout remain aligned on every recorded model trial; one 70–79 trial did not cross and is treated as censored rather than as an observed RT.

The result supports a descriptive age-related timing pattern across the seven groups, with model accuracy close to the representative human subsets. It remains exploratory: the same trials were used for calibration and evaluation, the 80–89 group has only four participants, and model RT tails are still shorter than human tails.

The older two-group choice-coupled schedule result remains available for historical comparison in `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/`; it is not a replacement for the all-age diagnostic bundle.

The supervisor follow-up diagnostic is saved in `artifacts/results/r5_supervisor_followup/` and can be regenerated with `code/scripts/run_r5_supervisor_followup.py`. It recomputes CAF/CRF from raw trial-level rows using actual RT coordinates, audits the R5 state/readout path, and separates fixed-time `S(t)` distributions from first-passage-time behavior.

Two supporting checks extend that audit. `artifacts/results/ww_diffdecision_core_audit_20260802/` verifies the Wong-Wang/DiffDecision core in controlled two- and four-choice settings. `artifacts/results/r5_supervisor_round2_20260802/` provides separate human/model CAF and CRF panels with explicit RT ticks, a shape-only RT rescaling demonstration, and a synthetic speed-accuracy mechanism check. These are controlled sanity checks, not a new fit to human data.

The real-evidence follow-up in `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/` shows that the retained VGG evidence already contains early flanker capture followed by target recovery. The subsequent alignment and schedule analyses test whether that signal reaches a theoretically consistent choice/RT readout.

## Earlier mechanism branch

The earlier public mechanism branch is still retained because it explains the foundation of the project:

- variational evidence sampling in the encoding stage
- DMC-style conflict modulation
- Wong-Wang decision dynamics
- fast-error behavior as an intermediate mechanism result

That branch is useful for understanding the model logic, but it is no longer the only main description of the repository.

## How to read this repo

Start with the current interpretation:

1. `docs/current_results_and_limitations.md`
2. `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/summaries/updated_model_summary_chinese.md`
3. `docs/r5-supervisor-systematic-report-20260803.md`
4. `artifacts/results/r5_choice_rule_alignment_audit_20260803/summary.md`
5. `artifacts/results/r5_choice_coupled_refit_20260803/summary.md`
6. `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/summary.md`
7. `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/summary.md`
8. `docs/r5-supervisor-followup.md`
9. `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/README_best_model_R5_combined_best.md`
10. `docs/model_framework_summary.md`

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
- `code/scripts/run_ww_diffdecision_core_audit.py`
- `code/scripts/run_r5_supervisor_round2.py`
- `code/scripts/run_real_vgg_target_flanker_dynamics_audit.py`
- `code/scripts/run_r5_choice_rule_alignment_audit.py`
- `code/scripts/run_r5_choice_coupled_schedule_optimization.py`
- `code/scripts/plot_r5_rt_distribution_kde.py`
- `code/scripts/plot_r5_caf_and_delta_curves.py`
- `code/scripts/run_all_age_group_extension.py`
- `code/scripts/run_corrected_model_all_age_groups.py`
- `code/scripts/run_all_age_time_scale_refinement.py`
- `code/scripts/plot_all_age_group_rt_distributions.py`

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

Retained baseline result bundle:

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

Second-round supporting diagnostics:

- `artifacts/results/r5_supervisor_round2_20260802/summary.md`
- `artifacts/results/r5_supervisor_round2_20260802/01_CAF_explicit_quantile_RT_ticks.pdf`
- `artifacts/results/r5_supervisor_round2_20260802/02_CRF_explicit_quantile_RT_ticks.pdf`
- `artifacts/results/r5_supervisor_round2_20260802/03_time_scaling_preserves_shape.pdf`
- `artifacts/results/r5_supervisor_round2_20260802/04_improved_model_speed_accuracy.pdf`
- `artifacts/results/ww_diffdecision_core_audit_20260802/summary.md`
- `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/summary.md`

Choice-coupled and all-age results:

- `artifacts/results/r5_choice_rule_alignment_audit_20260803/summary.md`
- `artifacts/results/r5_choice_coupled_refit_20260803/summary.md`
- `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/summary.md`
- `artifacts/results/r5_rt_distribution_kde_20260803/observed_vs_model_rt_kde.pdf`
- `artifacts/results/r5_caf_delta_curves_20260803/current_model_caf_human_vs_model.pdf`
- `artifacts/results/r5_caf_delta_curves_20260803/current_model_delta_rt_human_vs_model.pdf`
- `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/figures_publication/all_age_caf_updated_model.pdf`
- `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/figures_publication/all_age_rt_distribution_updated_model.pdf`
- `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/results/updated_model_parameters_by_age.csv`
- `output/pdf/current_improved_r5_model_summary_tables.pdf`

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

Older experiments and archives remain for provenance. Use `artifacts/results/ARTIFACT_DOCS_INDEX.md` to distinguish the retained R5 baseline, current supporting diagnostics, and historical reports.

Do not interpret the choice-coupled schedule result or the all-age update as a full-data final fit or proof of a human conflict-control mechanism. They are exploratory representative-subset results with clear remaining limitations in RT shape, congruency effects, participant coverage, and out-of-sample validation.
