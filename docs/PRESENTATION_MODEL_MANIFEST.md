# Presentation Model Manifest

## Identity

- Model name: `natural_layer_to_time_vgg16_ww_r5`.
- Presented result name: `best_model_R5_combined_best`.
- Main model class: `VGGWongWangLIM`.
- Accumulator class: `WongWangMultiClassDecision`.
- Decision class: `DiffDecisionMultiClass`.
- Primary implementation: `code/scripts/vgg_wongwang_lim.py`.
- Main corrected execution script: `code/scripts/run_r5_choice_coupled_schedule_optimization.py`.
- Original R5 reconstruction: `code/scripts/run_r5_supervisor_followup.py::reconstruct_r5`.
- Unified entry point: `code/scripts/reproduce_presentation_model.py`.
- Selected configuration: `configs/presentation_model.json`.

## Provenance fingerprint

The exact presentation mechanism figure is `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/05_natural_emergence_evidence_chain.pdf`. Its plotting script is `code/scripts/run_real_vgg_target_flanker_dynamics_audit.py`.

The traced chain is:

`mechanism figure` -> `run_real_vgg_target_flanker_dynamics_audit.py` -> `trial_level_target_flanker_dynamics.csv` -> original R5 predictions plus cached layer evidence -> `run_representative_extreme_age_subset_fitting.py` -> `vgg_wongwang_lim.py`.

This uniquely identifies the two-group natural layer-to-time R5 diagnostic, rather than the unrelated DMC, M3 noise, all-age, dual-route, or later experimental families.

## Visual evidence

- Frontend: VGG16.
- Layers: `conv3`, `conv4`, `conv5`, `pooled`, `final`.
- Evidence shape: one four-direction vector for each layer and stimulus.
- Target/flanker measure: evidence in the target channel minus evidence in the flanker channel.
- Normalization: `per_layer_gap_scale`.
- Cached input: `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/evidence_cache/representative_subset_layerwise_evidence.npz`.
- Trial/stimulus mapping: `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/manifests/`.

## Layer-to-time mapping

- Base mapping: `natural_smooth_5stage`.
- Grid: 80 steps at 10 ms per step.
- Presentation R5 uses the original mapping.
- The corrected-equivalent keeps the same five VGG arrays and normalization, then applies the selected group schedule recorded in `configs/presentation_model.json`.

## Wong-Wang and readout

- Four recurrent, mutually competitive decision channels.
- Evidence gain: `0.8` in both groups.
- AMPA noise: `0.0` in the retained R5 reconstruction.
- Young threshold/margin: `0.12 / 0.00`.
- Older threshold/margin: `0.14 / 0.02`.
- Sustained crossing: two consecutive steps with the same winner.
- RT: first sustained-crossing time plus group non-decision time.
- No crossing: final-step deadline sentinel plus an explicit crossing flag; the sentinel is censored and is never an observed RT.
- Presentation choice: historical maximum over the whole trajectory.
- Corrected-equivalent choice: winning channel at the sustained-crossing readout step.

The corrected choice rule repairs the presentation package's choice/RT timing mismatch. It does not change the VGG evidence definition or Wong-Wang equations. Schedule compression is documented separately because it changes when the same layer evidence reaches the accumulator.

## Parameters and seeds

All fixed parameters are in `configs/presentation_model.json`; the original source table is `best_model_parameter_estimates.csv` inside the R5 result package. Model reconstruction uses seed `20260530`. Plotting and bootstrap scripts use fixed seeds declared in their source. A reproduction is valid only when its output records the seed and preserves the trial manifest.

## Results retained

- Original young/older R5 package: `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/`.
- Corrected-equivalent young/older package: `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/`.
- Young and older trial predictions: `selected_trial_level_predictions.csv` in that corrected package.
- CAF and delta figures: `artifacts/results/r5_caf_delta_curves_20260803/`.
- RT distribution: `artifacts/results/r5_rt_distribution_kde_20260803/`.
- VGG target/flanker and Wong-Wang dynamics: `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/`.

## Plotting and tests

- CAF/delta: `code/scripts/plot_r5_caf_and_delta_curves.py`.
- RT distribution: `code/scripts/plot_r5_rt_distribution_kde.py`.
- Mechanism chain: `code/scripts/run_real_vgg_target_flanker_dynamics_audit.py`.
- Core tests: `tests/test_diffdecision_multiclass.py`.
- Choice/readout tests: `tests/test_r5_choice_rule_alignment_audit.py`.
- Schedule tests: `tests/test_r5_choice_coupled_schedule_optimization.py`.
- Mechanism tests: `tests/test_real_vgg_target_flanker_dynamics_audit.py`.

## Dependencies

Python dependencies are listed in `config/requirements.txt`. Full evidence extraction also requires the local LIM stimulus data and the original stage-1 checkpoint named in `evidence_cache/extraction_metadata.json`; these large local inputs are intentionally not Git-tracked. Analysis-only and smoke reproduction use the retained evidence cache and do not require rerunning VGG extraction.
