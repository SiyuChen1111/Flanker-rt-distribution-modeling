# Schedule compression Pareto input inventory

## Inputs used

- `natural_evidence_dynamics_optimization/metrics/natural_dynamics_model_ranking.csv`: previous round ranking and best schedule family.
- `natural_evidence_dynamics_optimization/metrics/natural_dynamics_model_comparison_summary.csv`: previous round group/condition diagnostics.
- `natural_evidence_dynamics_optimization/metrics/natural_dynamics_trial_level_predictions.csv`: previous round trial-level outputs.
- `natural_evidence_dynamics_optimization/metrics/natural_dynamics_trajectory_diagnostics.csv`: previous round trajectory summaries.
- `natural_evidence_dynamics_optimization/summaries/natural_evidence_dynamics_optimization_summary.md`: prior interpretation.
- `evidence_cache/representative_subset_layerwise_evidence.npz`: cached layerwise evidence.
- `best_model_R5_combined_best/results/best_model_parameter_estimates.csv`: group-specific WW parameters.
- `fitting/representative_trial_level_predictions.csv`: human trial metadata and RTs.
- `readout_choice_uncertainty_mechanism_comparison/metrics/readout_choice_model_ranking.csv`: prior time+gap uncertainty parameters.
- `readout_choice_uncertainty_mechanism_comparison/metrics/human_reference_rt_error_metrics.csv`: human reference metrics.

## Previous best schedule compression

- Previous best schedule model: `M1_schedule_compression / c0.4_ls-5_tw0.7_ep5`.
- Previous best parameters: {"dynamics": {}, "schedule": {"compression": 0.4, "early_shorten_steps": 5, "late_shift_steps": -5, "transition_scale": 0.7}}.
- Previous failure point: incongruent repair was strong, but congruent fast errors were lost or weakened.

## Why local Pareto search

- The current question is no longer whether schedule compression can repair incongruent flanker over-selection. It can.
- The open question is whether a less aggressive local region, combined with retuned time+gap choice noise, can recover congruent fast errors without undoing the incongruent repair.

## What is not retrained

- VGG is not retrained.
- Image evidence is not re-extracted.
- Earlier result folders are not overwritten.
