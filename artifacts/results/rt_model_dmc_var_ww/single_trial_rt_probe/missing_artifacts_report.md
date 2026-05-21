# Missing Artifacts Report

This notebook found enough artifacts for a diagnostic repeated-forward probe, but it did not find a formal end-to-end repeated-forward export from the training pipeline.

## Paths checked
- `data/age_groups_matched/20-29/test_data.csv`: found
- `artifacts/checkpoints/age_groups_matched/20-29/stage2/test_logits.npz`: found
- `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt`: found
- `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/predictions_neg_drt.npz`: found
- `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/predictions_smoke.npz`: found
- `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/best_model_params.npz`: found
- `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/config.json`: found
- `artifacts/results/rt_model_dmc_var_ww/summary_smoke.md`: found
- `artifacts/checkpoints/age_groups_matched/20-29/stage2/best_model_params.npz`: found

## Missing or incomplete items
- No saved Stage-1 variational-head checkpoint tied directly to `smoke_a5_s3_neg_drt` was found.
- No training-pipeline export with trial-level `mu`, `sigma`, sampled evidence, DMC-modulated evidence, WW trajectories, and repeated-forward seeds was found.
- `predictions_neg_drt.npz` contains RT/choice/labels but not subject id, image id, logits, evidence, or trajectory. This notebook reconstructs metadata from the matched test CSV and cached logits.

## What can be reproduced from current files
- Real 20-29 test trial metadata from `data/age_groups_matched/20-29/test_data.csv`.
- Cached Stage-1 logits from `artifacts/checkpoints/age_groups_matched/20-29/stage2/test_logits.npz`.
- DMC + Wong-Wang config, parameters, and evaluation predictions from `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/`.

## Diagnostic status
This is a diagnostic repeated-forward probe, not a formal trained-model result.

## To make this formal
- Extend the main pipeline to save selected trial ids, logits/features, `mu`, `sigma`, sampled evidence, DMC-modulated evidence, WW trajectory, decision times, choice probabilities, and random seeds for repeated forward passes.