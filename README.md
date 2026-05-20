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
- `code/scripts/train_variational_ww_smoke.py`
- `code/scripts/run_subject_level_dmc_var_ww.py`
- `code/scripts/vgg_wongwang_lim.py`
- `code/scripts/stage1_semisup_evidence_sampler.py`
- `code/scripts/analyze_subject_level_dmc_var_ww.py`
- `code/scripts/train_dmc_var_ww_smoke.py`

## Key results

- `artifacts/results/rt_model_dmc_var_ww/summary_smoke.md`
- `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/`
- `artifacts/results/rt_model_variational_ww_synthesis/var_ww_mechanism_memo.md`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/`

These are the only result paths intentionally retained as the public evidence spine for the current mechanism-test release.

## How to read this repo

Start with:

1. `docs/model_framework_summary.md`
2. `docs/current_results_and_limitations.md`
3. `docs/public_update_notes.md`
4. `examples/dmc_var_ww_minimal_pipeline.md`
5. `artifacts/results/rt_model_dmc_var_ww/summary_smoke.md`
6. `artifacts/results/rt_model_variational_ww_synthesis/var_ww_mechanism_memo.md`

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
  - `examples/outputs/dmc_var_ww_minimal/`
    Saved predictions, metrics, and figures from the example run.
- `artifacts/results/rt_model_dmc_var_ww/`
  Retained DMC + variational evidence + Wong-Wang result bundle for the public release.
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/`
  Retained aggregated single-subject evidence tables for the public release.
- `artifacts/archive_legacy_not_for_public/`
  Archived material kept out of the public release path, including older result branches and non-core legacy scripts moved out of `code/scripts/`.

## Notes

Older experiments and archives remain in the tree, but the public update is centered on the DMC + variational evidence + Wong-Wang mechanism test.
Supporting utility code may remain for reproducibility, but it is not part of the public narrative unless it directly supports the retained evidence above.
