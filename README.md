# LIM / Flanker RT Distribution Modeling

This repository studies reaction-time behavior in the Lost in Migration (LIM) / Flanker task. The project connects visual evidence, conflict effects, and decision dynamics to explain human response time, accuracy, and age-related differences.

## Current focus

The current main line is the representative extreme-age subset workflow:

- young group: `young_20_29`, 5,000 representative trials
- older group: `older_80_89`, 5,000 representative trials
- retained baseline: `R5_combined_best`
- current exploratory extension: choice and RT are coupled at the sustained-crossing step, with the VGG layer-to-time schedule compressed before Wong-Wang accumulation
- current result bundle: `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/`

This is a model-development and diagnostic result, not a final full-cohort age-group conclusion.

## Current conclusion

The retained R5 baseline used the first sustained-crossing time for RT but the maximum state over the whole later trajectory for choice. The two rules disagree on 26.5% of all trials, entirely on incongruent trials, so the baseline accuracy partly used information arriving after its stated decision time.

Fresh analyses now bind choice to the same sustained-crossing step as RT. Compressing the existing VGG layer-to-time schedule then allows the real early-flanker/later-target signal to recover before readout while keeping all 10,000 model trials above threshold:

- young 20–29: human/model accuracy 0.949/0.961; mean RT 0.603/0.592 s; incongruent CAF RMSE 0.028
- older 80–89: human/model accuracy 0.976/0.979; mean RT 0.941/0.891 s; incongruent CAF RMSE 0.009

This is a strong in-sample diagnostic improvement, not a final fit. It uses 12 young and 4 older participants, age-specific schedules selected from the same representative trials, and no held-out participants or stimuli. The model also exaggerates the congruency RT cost and produces RT distributions with tails that are much shorter than the human distributions.

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
2. `docs/r5-supervisor-systematic-report-20260803.md`
3. `artifacts/results/r5_choice_rule_alignment_audit_20260803/summary.md`
4. `artifacts/results/r5_choice_coupled_refit_20260803/summary.md`
5. `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/summary.md`
6. `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/summary.md`
7. `docs/r5-supervisor-followup.md`
8. `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/README_best_model_R5_combined_best.md`
9. `docs/model_framework_summary.md`

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

Current choice-coupled result and figures:

- `artifacts/results/r5_choice_rule_alignment_audit_20260803/summary.md`
- `artifacts/results/r5_choice_coupled_refit_20260803/summary.md`
- `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/summary.md`
- `artifacts/results/r5_rt_distribution_kde_20260803/observed_vs_model_rt_kde.pdf`
- `artifacts/results/r5_caf_delta_curves_20260803/current_model_caf_human_vs_model.pdf`
- `artifacts/results/r5_caf_delta_curves_20260803/current_model_delta_rt_human_vs_model.pdf`
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

Do not interpret the choice-coupled schedule result as a full-data final fit or proof of a human conflict-control mechanism. It is an exploratory representative-subset result with clear remaining limitations in RT shape, congruency effects, participant coverage, and out-of-sample validation.
