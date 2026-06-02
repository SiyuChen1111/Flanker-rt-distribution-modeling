# Constraint-first rescreen input inventory

## trial_level
- Path: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_top_candidates_trial_level_repaired.csv`
- Shape: 310000 rows x 34 columns
- Missing required columns: none
## summary
- Path: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_local_search_summary_repaired.csv`
- Shape: 124 rows x 62 columns
- Missing required columns: none
## ranking
- Path: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_local_search_ranking_repaired.csv`
- Shape: 31 rows x 35 columns
- Missing required columns: none
## pareto
- Path: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_pareto_front_repaired.csv`
- Shape: 12 rows x 35 columns
- Missing required columns: none
## rt_bin
- Path: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_error_rate_by_rt_bin_repaired.csv`
- Shape: 1240 rows x 8 columns
- Missing required columns: none
## trajectory
- Path: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/metrics/schedule_compression_trajectory_diagnostics_repaired.csv`
- Shape: 36080 rows x 12 columns
- Missing required columns: none
## audit_notes
- Path: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/summaries/schedule_compression_coarse_metric_audit.md`
- Text length: 735 characters
## repaired_notes
- Path: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/schedule_compression_pareto_search/summaries/schedule_compression_pareto_search_repaired_summary.md`
- Text length: 840 characters
## human_ref
- Path: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/readout_choice_uncertainty_mechanism_comparison/metrics/human_reference_rt_error_metrics.csv`
- Shape: 2 rows x 26 columns
- Missing required columns: none

## Candidate pool
- Repaired trial-level candidate count: 31
- Repaired ranking candidate count: 31
- Repaired Pareto candidate count: 12

## Direct hard-constraint metrics
- Directly usable: model_config_id, schedule_config_id, noise_config_id, Pareto status, tradeoff region, repaired flags, trajectory columns, RT-bin profiles.
- Recomputed from trial-level: error rates, accuracy, RT quantiles, fast-error counts and RT differences, choice-type proportions, condition-level trajectory summaries.
- Recomputed from RT-bin profile: congruent/incongruent RMSE, fast-bin mismatch, slow-bin mismatch, CAF-like slope.