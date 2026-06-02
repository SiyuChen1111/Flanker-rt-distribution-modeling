# Current Selected Model Visualization Summary

## Model shown

The main model shown is M3_time_gap with age-specific parameters. It is the top-ranked current readout/choice noise model.

## Input files

- metrics/readout_choice_model_ranking.csv
- metrics/readout_choice_model_selected_summary.csv
- metrics/readout_choice_model_bootstrap_ci.csv
- metrics/readout_choice_model_split_validation.csv
- metrics/human_reference_rt_error_metrics.csv
- fitting/representative_trial_level_predictions.csv
- congruent_ww_dynamics_diagnostic/metrics/threshold_lowering_counterfactual_trial_level.csv

## Figures generated

- Figure 1: current model overview against human.
- Figure 2: error rate by RT bin, split by group and congruency.
- Figure 3: congruent and incongruent error diagnostics.
- Figure 3b: deterministic/time-only/gap-only/time+gap mechanism context.
- Figure 4: RT distribution ECDF for human and current model.
- Figure 4b: RT distribution by correct/error.
- Figure 5: mechanism comparison context.
- Figure 6: split validation check.
- Figure 6b: bootstrap/seed-level intervals.
- Figure 7: target recovery preservation.

## What looks good

- Young congruent error rate is close to human: model 0.0186, human 0.0177.
- Older congruent error rate is close under age-specific M3: model 0.0081, human 0.0090.
- Congruent errors are fast in both groups: young -0.0051s, older -0.0087s.
- Incongruent fast-error pattern is preserved: young -0.2685s, older -0.4527s.
- Target recovery preservation is positive in both groups: young 0.0124, older 0.1025.

## Remaining problems

- Overall model accuracy remains far below human: young model 0.5170 vs human 0.9502; older model 0.6408 vs human 0.9768.
- Incongruent error rate is still much too high: young model 0.9509 vs human 0.0822; older model 0.7168 vs human 0.0377.
- The RT distribution is not changed by readout-choice noise, so good RT preservation here means the choice-noise step did not disrupt the existing RT timing, not that it solved all RT-fit issues.

## Direct answers

- It successfully produces congruent errors under the age-specific M3 setting.
- The congruent error RT difference is negative, supporting a fast-error pattern; Figure 2 gives the bin-level check.
- The incongruent fast-error pattern is preserved.
- The RT distribution is not directly altered by this noise mechanism.
- Overall accuracy is not close to human.
- Incongruent error rate remains too high.
- Older is much closer under age-specific M3 than under shared M3.
- These results support time+gap readout-choice uncertainty as an explanatory candidate, but still exploratory.

## More formal conclusions

- Deterministic readout fails to produce congruent errors.
- Time+gap readout-choice uncertainty can produce human-like congruent errors and fast congruent errors in the representative subset.
- The mechanism preserves target recovery and does not change RT timing.

## Cautious / exploratory conclusions

- The age-specific result suggests older group may require stronger readout-choice uncertainty, but this is not proof that older adults have more noise.
- The model still has major accuracy and incongruent error-rate mismatches.
- The current result should be validated on larger samples or more age groups before strong claims.

## Recommended 3-5 figures for advisor

1. Figure 2: error rate by RT bin.
2. Figure 1: current model overview.
3. Figure 3: congruent/incongruent error diagnostics.
4. Figure 5: mechanism comparison context.
5. Figure 7: target recovery preservation.

## Warnings / unavailable plots

- None.

## Files written

- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig1_current_model_overview.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig1_current_model_overview.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig1_current_model_overview.svg
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig2_error_rate_by_rt_bin_current_model.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig2_error_rate_by_rt_bin_current_model.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig2_error_rate_by_rt_bin_current_model.svg
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig3_congruency_error_diagnostics_current_model.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig3_congruency_error_diagnostics_current_model.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig3_congruency_error_diagnostics_current_model.svg
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig3b_mechanism_comparison_on_error_diagnostics.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig3b_mechanism_comparison_on_error_diagnostics.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig3b_mechanism_comparison_on_error_diagnostics.svg
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig4_rt_distribution_current_model.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig4_rt_distribution_current_model.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig4_rt_distribution_current_model.svg
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig4b_rt_distribution_by_accuracy_current_model.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig4b_rt_distribution_by_accuracy_current_model.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig4b_rt_distribution_by_accuracy_current_model.svg
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig5_mechanism_comparison_current_model_context.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig5_mechanism_comparison_current_model_context.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig5_mechanism_comparison_current_model_context.svg
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig6b_bootstrap_ci_current_model.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig6b_bootstrap_ci_current_model.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig6b_bootstrap_ci_current_model.svg
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig6_split_validation_current_model.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig6_split_validation_current_model.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig6_split_validation_current_model.svg
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/pdf/fig7_target_recovery_and_dynamics_current_model.pdf
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/png/fig7_target_recovery_and_dynamics_current_model.png
- /Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/visualization_current_selected_model/svg/fig7_target_recovery_and_dynamics_current_model.svg
