# Natural evidence-dynamics input inventory

## Inputs used

- `evidence_cache/representative_subset_layerwise_evidence.npz`: layerwise evidence arrays (`evidence_conv3`, `evidence_conv4`, `evidence_conv5`, `evidence_pooled`, `evidence_final`) and trial-to-stimulus ids.
- `best_model_R5_combined_best/results/best_model_parameter_estimates.csv`: current R5 group-specific WW and readout parameters.
- `fitting/representative_trial_level_predictions.csv`: trial metadata, human RT, human response, congruency, current model RT.
- `fitting/representative_best_model_mechanism_trial_level.csv`: readout-stage mechanism diagnostics and target recovery fields.
- `readout_choice_uncertainty_mechanism_comparison/metrics/*`: current time+gap uncertainty parameters and diagnostics.
- `readout_choice_uncertainty_mechanism_comparison/summaries/*`: prior diagnostic context for gating and trajectory viability.

## Key fields

- Trial metadata: `row_index`, `analysis_group`, `target_label`, `flanker_label`, `response_label`, `true_rt`, `congruency`.
- Layerwise evidence: one four-channel vector per layer per stimulus.
- Existing behavior outputs: `pred_rt`, `decision_time`, `human_correct`, `model_correct`.

## Reconstruction viability

- Trial-level WW trajectories can be reconstructed from cached layerwise evidence plus saved R5 parameters.
- Different layer-to-time schedules can be implemented by modifying the schedule weights before WW input is built.
- Attention gain, flanker decay, and online conflict control can all be applied to the time-varying WW input without retraining VGG or re-extracting image evidence.

## Limits

- There is no saved per-trial full trajectory file on disk; trajectories are reconstructed rather than loaded directly.
- Human correctness is available for evaluation only and is not used as model input.
- This round does not retrain VGG, does not re-extract image features, and does not overwrite earlier result directories.
