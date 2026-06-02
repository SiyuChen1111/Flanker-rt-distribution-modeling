# Mechanism redesign input inventory

## Inputs read
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/evidence_cache/representative_subset_layerwise_evidence.npz`: cached layerwise evidence.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/results/best_model_parameter_estimates.csv`: current group-specific WW and readout settings.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/fitting/representative_trial_level_predictions.csv`: trial metadata, human RT, human correctness.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/metrics/readout_choice_model_ranking.csv`: time+gap uncertainty ranking and selected parameters.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/metrics/human_reference_rt_error_metrics.csv`: human reference metrics.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/natural_evidence_dynamics_optimization/metrics/natural_dynamics_model_ranking.csv`: prior natural dynamics comparison.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_local_search_ranking_repaired.csv`: repaired global schedule ranking.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_pareto_front_repaired.csv`: repaired Pareto front.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_top_candidates_trial_level_repaired.csv`: repaired trial-level diagnostics.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_trajectory_diagnostics_repaired.csv`: repaired trajectory diagnostics.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/constraint_first_rescreen/metrics/constraint_first_rescreen_recomputed_metrics.csv`: rescreen metrics.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/constraint_first_rescreen/metrics/constraint_first_rescreen_representative_models.csv`: rescreen representative models.
- `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/constraint_first_rescreen/summaries/constraint_first_rescreen_summary.md`: current constraint-first conclusion.

## What each file contributes
- Evidence cache and best-model parameters let this round reconstruct trial-wise WW inputs and trajectories without retraining VGG or re-extracting evidence.
- Readout-choice ranking provides the current time+gap uncertainty baseline and human reference targets.
- Schedule-compression repaired outputs define the strongest existing global-schedule references and their failure pattern.
- Constraint-first rescreen confirms that global schedule compression plus retuned time-gap noise has zero survivors and mostly fails because older congruent errors disappear.

## Why test conflict-adaptive schedule
- Global compression improves incongruent repair but stabilizes congruent trials too much, especially in the older group.
- A more natural alternative is to accelerate high-level evidence only when early competition is high, instead of compressing every trial equally.
- This allows the mechanism to depend on current evidence conflict rather than congruency labels or future target crossing.

## Why bounded lapse is secondary
- Bounded lapse is treated as rare response-execution uncertainty only.
- It is included to test whether a very small downstream uncertainty source can recover a few older congruent errors without undoing the repaired incongruent behavior.
- It is not treated as the main explanation of the task behavior.

## What is not rerun
- VGG is not retrained.
- Image evidence is not re-extracted.
- No target-gated readout is reintroduced.
- No large schedule-compression fine search is run.
