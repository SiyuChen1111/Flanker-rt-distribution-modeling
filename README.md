# Flaner-rt-distribution-modeling

This repository studies reaction-time behavior in the LIM / Flanker task with a visual encoder and decision-dynamics models.

## Public line

The public story is limited to:

- variational evidence sampling in the encoding stage
- DMC-style conflict modulation
- Wong-Wang decision dynamics
- fast-error mechanism as an intermediate result

## Current conclusion

This is a **mechanism test**, not a final human RT fit.

It shows that the DMC + variational evidence branch can produce the right kind of fast-error direction and a long-tailed RT shape, but the model is still too fast and not yet fully calibrated to human data.

## Key files

- `examples/dmc_var_ww_minimal_pipeline.ipynb`
  Public teaching example for the current DMC + variational evidence + Wong-Wang pipeline.
- `examples/toy_recurrent_ww_dmc_demo.ipynb`
  Small toy notebook showing how recurrent Wong-Wang competition can amplify early DMC flanker capture into a fast error.
- `examples/single_trial_rt_variability_probe.ipynb`
  Diagnostic notebook for single-trial repeated stochastic forward probes of RT and choice variability.
- `examples/build_single_trial_rt_variability_probe.py`
  Generator used to recreate the single-trial diagnostic notebook and outputs.
- `code/scripts/train_variational_ww_smoke.py`
- `code/scripts/run_subject_level_dmc_var_ww.py`
- `code/scripts/vgg_wongwang_lim.py`
- `code/scripts/stage1_evidence_sampler.py`
  This module provides Stage-1 deterministic, variational, and MC-dropout evidence sampling.
- `code/scripts/analyze_subject_level_dmc_var_ww.py`
- `code/scripts/train_dmc_var_ww_smoke.py`

## Key results

- `artifacts/results/rt_model_dmc_var_ww/summary_smoke.md`
- `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/`
- `artifacts/results/rt_model_dmc_var_ww/single_trial_rt_probe/`
- `artifacts/results/rt_model_variational_ww_synthesis/var_ww_mechanism_memo.md`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/`
- `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/`
  Current internal results bundle for the representative extreme-age R5 update. This path is kept under `artifacts/results/` because it is now treated as a model-progress result bundle rather than a diagnostics-only folder.

These are the only result paths intentionally retained as the public evidence spine for the current mechanism-test release.

## How to read this repo

Start with:

1. `docs/model_framework_summary.md`
2. `docs/current_results_and_limitations.md`
3. `docs/public_update_notes.md`
4. `examples/dmc_var_ww_minimal_pipeline.md`
5. `examples/toy_recurrent_ww_dmc_demo.ipynb`
6. `examples/single_trial_rt_variability_probe.ipynb`
7. `artifacts/results/rt_model_dmc_var_ww/summary_smoke.md`
8. `artifacts/results/rt_model_variational_ww_synthesis/var_ww_mechanism_memo.md`

## Repository layout

- `code/scripts/`
  The retained script set for the current mechanism-test branch. This directory now contains the public-core training and analysis scripts, plus a small number of supporting utility scripts that are still required by the retained workflow.
- `docs/`
  Short explanatory documents for the current public narrative.
- `examples/`
  Public teaching example for the current DMC + variational evidence + Wong-Wang pipeline.
  - `examples/dmc_var_ww_minimal_pipeline.ipynb`
    Executable notebook example.
  - `examples/build_dmc_var_ww_minimal_pipeline.py`
    Generator used to recreate the notebook and outputs.
  - `examples/dmc_var_ww_minimal_pipeline.md`
    Short run note for the example.
  - `examples/toy_recurrent_ww_dmc_demo.ipynb`
    Small teaching notebook for the recurrent competition fast-error mechanism.
  - `examples/outputs/dmc_var_ww_minimal/`
    Saved predictions, metrics, and figures from the example run.
  - `examples/single_trial_rt_variability_probe.ipynb`
    Diagnostic notebook that compares condition-level RT distributions with repeated stochastic forward passes on fixed real trials.
  - `examples/build_single_trial_rt_variability_probe.py`
    Generator used to recreate the single-trial diagnostic notebook, tables, and figures.
- `artifacts/results/rt_model_dmc_var_ww/`
  Retained DMC + variational evidence + Wong-Wang result bundle for the public release.
  - `artifacts/results/rt_model_dmc_var_ww/single_trial_rt_probe/`
    Saved tables and figures from the single-trial RT variability probe. This is a diagnostic mechanism probe, not a final trained-model result.
- `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/`
  Current internal result bundle for the representative extreme-age subset workflow, including the retained `best_model_R5_combined_best` package and its follow-up analyses.
- `artifacts/results/diagnostics/`
  Diagnostic and exploratory side-results that are not the main retained result bundle.
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/`
  Retained aggregated single-subject evidence tables for the public release.
- `artifacts/archive_legacy_not_for_public/`
  Archived material kept out of the public release path, including older result branches and non-core legacy scripts moved out of `code/scripts/`.

## Notes

Older experiments and archives remain in the tree, but the public update is centered on the DMC + variational evidence + Wong-Wang mechanism test.
Supporting utility code may remain for reproducibility, but it is not part of the public narrative unless it directly supports the retained evidence above.
