# Reproducibility and active config audit

Exact current R5 package verified:

- Model package: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best`
- Trial-level predictions: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/results/best_model_trial_level_predictions.csv`
- Parameter table: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/results/best_model_parameter_estimates.csv`
- Evidence cache: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/evidence_cache/representative_subset_layerwise_evidence.npz`
- Manifest used by reconstruction code: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/manifests/representative_subset_trial_to_stimulus_mapping.csv`

Checkpoint status:

- No standalone `R5` neural-network checkpoint file was found inside the retained R5 package.
- The reproducible active state for this diagnostic is therefore the archived R5 package: saved trial-level outputs, group-specific parameter table, evidence cache, manifest, and reconstruction code.

Active groups and trials:

- Young: `5000` trials, `12` participants.
- Older: `5000` trials, `4` participants.

R5 parameter rows:

| model_name       | analysis_group   |   t0_mean |   t0_sd | evidence_gain   | threshold      | sustained_k    | margin         | min_decision_time   | sigma_type   |   sigma_base |   sigma_conflict | parameter_details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |   score_total |   score_rt_quantile |   score_caf |   score_accuracy |   score_mechanism |
|:-----------------|:-----------------|----------:|--------:|:----------------|:---------------|:---------------|:---------------|:--------------------|:-------------|-------------:|-----------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|--------------------:|------------:|-----------------:|------------------:|
| R5_combined_best | older_80_89      |      0.75 |    0.2  | group_specific  | group_specific | group_specific | group_specific | group_specific      | none         |            0 |                0 | {"combined_from": "R5_combined_t0sd_ww", "evidence_gain": "group_specific", "group_params": "{\"older_80_89\": {\"evidence_gain\": 0.8, \"threshold\": 0.14, \"sustained_k\": 2, \"margin\": 0.02, \"min_decision_time\": 0.0}, \"young_20_29\": {\"evidence_gain\": 0.8, \"threshold\": 0.12, \"sustained_k\": 2, \"margin\": 0.0, \"min_decision_time\": 0.0}}", "margin": "group_specific", "min_decision_time": "group_specific", "sustained_k": "group_specific", "threshold": "group_specific"} |      0.403642 |            0.031558 |    0.137551 |           0.1342 |                 0 |
| R5_combined_best | young_20_29      |      0.55 |    0.12 | group_specific  | group_specific | group_specific | group_specific | group_specific      | none         |            0 |                0 | {"combined_from": "R5_combined_t0sd_ww", "evidence_gain": "group_specific", "group_params": "{\"older_80_89\": {\"evidence_gain\": 0.8, \"threshold\": 0.14, \"sustained_k\": 2, \"margin\": 0.02, \"min_decision_time\": 0.0}, \"young_20_29\": {\"evidence_gain\": 0.8, \"threshold\": 0.12, \"sustained_k\": 2, \"margin\": 0.0, \"min_decision_time\": 0.0}}", "margin": "group_specific", "min_decision_time": "group_specific", "sustained_k": "group_specific", "threshold": "group_specific"} |      0.403642 |            0.031558 |    0.137551 |           0.1342 |                 0 |

Model-selection scores:

| model_name           |   score_total |   score_rt_quantile |   score_caf |   score_accuracy |   score_mechanism |
|:---------------------|--------------:|--------------------:|------------:|-----------------:|------------------:|
| R5_combined_best     |      0.403642 |           0.031558  |    0.137551 |          0.1342  |                 0 |
| R2_group_t0_mean_sd  |      0.433355 |           0.04713   |    0.136597 |          0.1342  |                 0 |
| R3_group_ww_readout  |      0.571558 |           0.0926708 |    0.168011 |          0.1342  |                 0 |
| R4_variational_noise |      0.661644 |           0.141346  |    0.162275 |          0.13554 |                 0 |
| R1_group_t0_mean     |      0.663722 |           0.141762  |    0.163998 |          0.1342  |                 0 |
| R0_fixed_current     |      1.36969  |           0.494746  |    0.163998 |          0.1342  |                 0 |

Verified code paths:

- Visual evidence extraction and cache creation: `code/scripts/build_representative_extreme_age_vgg_cache.py`, summarized by `evidence_cache/extraction_metadata.json`.
- Layer-to-time mapping: `code/scripts/run_natural_layer_to_time_var_ww_diagnostic.py`, `natural_smooth_5stage`, `per_layer_gap_scale`.
- Wong-Wang state update and `DiffDecisionMultiClass`: `code/scripts/vgg_wongwang_lim.py`.
- R5 readout and t0 addition: `code/scripts/optimize_natural_layer_to_time_rt_shape.py`, `apply_readout`, plus group-specific `t0_mean` and `t0_sd`.

Training/loss audit:

- The R5 package is a finite model-selection result, not a full active training checkpoint with a saved optimizer/loss configuration.
- The active R5 selection score explicitly includes RT quantiles, CAF, accuracy, and mechanism terms (`score_rt_quantile`, `score_caf`, `score_accuracy`, `score_mechanism`).
- No active CRF loss, response NLL weight, `lambda_accuracy`, `lambda_rt_mse`, or full training-objective weights were found inside the R5 package itself. This remains unresolved rather than assumed.
- Because response NLL was not verified as active for R5, this diagnostic does not recommend simply adding a separate accuracy loss.
