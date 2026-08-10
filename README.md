# VGG16 - Wong-Wang Flanker Model

This repository contains the implementation and evaluation of the VGG16 -> time-varying visual evidence -> multiclass Wong-Wang model used for the current Flanker-task analysis.

## Research question

The model tests whether a visual network can supply an early flanker advantage followed by later target recovery, and whether a recurrent four-choice accumulator can turn that changing evidence into both choices and reaction times.

The retained presentation analysis covers two representative groups:

- young adults: `20-29`, 5,000 trials;
- older adults: `80-89`, 5,000 trials.

The current diagnostic extension additionally covers all seven age groups:

- `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70-79`, and `80-89`;
- 75 participants in total;
- 5,000 deterministic representative trials per age group, 35,000 trials in total.

Both the two-group presentation analysis and the seven-group extension are diagnostic representative subsets, not held-out full-cohort fits.

## Model

1. VGG16 supplies four-direction evidence at `conv3`, `conv4`, `conv5`, `pooled`, and `final`.
2. `per_layer_gap_scale` normalizes the layerwise evidence.
3. `natural_smooth_5stage` maps the five layers to an 80-step evidence sequence.
4. A four-choice recurrent Wong-Wang accumulator integrates the sequence.
5. The presentation R5 result derives RT from the first sustained crossing. Its historical choice rule used the maximum over the complete trajectory.
6. The retained corrected-equivalent result chooses the winner at the same sustained-crossing readout step and compresses the existing schedule so late target evidence can affect the decision.
7. The all-age update keeps VGG evidence, Wong-Wang dynamics, choice, readout, and crossing fixed; it uses a shared decision-time scale of `0.27` and age-specific non-decision time.

The exact fingerprint, parameters, inputs, outputs, and correction boundary are recorded in [docs/PRESENTATION_MODEL_MANIFEST.md](docs/PRESENTATION_MODEL_MANIFEST.md).

## Quick start

Install the Python dependencies:

```bash
python -m pip install -r config/requirements.txt
```

Run the fast integrity and model smoke checks:

```bash
python code/scripts/reproduce_presentation_model.py --group all --smoke-test
```

Validate the retained young and older presentation tables without rerunning the full model:

```bash
python code/scripts/reproduce_presentation_model.py --group all --analysis-only
```

Regenerate the current CAF, RT-distribution, and mechanism figures into a new directory:

```bash
python code/scripts/reproduce_presentation_model.py --plot-only --output-dir artifacts/results/presentation_model_reproduction
```

The full two-group schedule search remains available through:

```bash
python code/scripts/run_r5_choice_coupled_schedule_optimization.py --output-dir <new-output-directory>
```

Run or validate the seven-group diagnostic extension:

```bash
python code/scripts/run_all_age_group_extension.py
python code/scripts/run_all_age_time_scale_refinement.py
pytest tests/test_all_age_group_extension.py tests/test_all_age_time_scale_refinement.py
```

## Seven-group diagnostic results

The timing update reduces the mean absolute congruent/incongruent RT error from 95.6 ms to 3.2 ms. One 70-79 trial did not cross the decision criterion and is censored rather than treated as an observed RT.

| Age group | Participants | Human / model accuracy | Human / model mean RT | `t0_mean` |
|---|---:|---:|---:|---:|
| 20-29 | 12 | 0.949 / 0.961 | 0.603 / 0.595 s | 0.556 s |
| 30-39 | 7 | 0.946 / 0.955 | 0.646 / 0.645 s | 0.603 s |
| 40-49 | 4 | 0.947 / 0.964 | 0.629 / 0.624 s | 0.581 s |
| 50-59 | 11 | 0.965 / 0.978 | 0.664 / 0.662 s | 0.616 s |
| 60-69 | 21 | 0.950 / 0.963 | 0.717 / 0.715 s | 0.664 s |
| 70-79 | 16 | 0.944 / 0.966 | 0.785 / 0.784 s | 0.727 s |
| 80-89 | 4 | 0.976 / 0.979 | 0.941 / 0.940 s | 0.882 s |

The mean RT alignment is strong, but the result is still in-sample. Model RT distributions remain narrower than human distributions, and the model makes no congruent errors in any age group.

## Repository map

- `configs/presentation_model.json`: selected model fingerprint and fixed parameters.
- `code/scripts/`: retained evidence, accumulator, readout, evaluation, and plotting pipeline.
- `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/`: canonical two-group inputs and original R5 package.
- `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/`: corrected-equivalent trial predictions and selected schedule.
- `artifacts/results/r5_caf_delta_curves_20260803/`: CAF and delta-curve outputs.
- `artifacts/results/r5_rt_distribution_kde_20260803/`: RT-distribution outputs.
- `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/`: layer, temporal-evidence, and Wong-Wang mechanism outputs.
- `artifacts/results/all_age_groups_20260806/`: seven-group audit, manifests, evidence caches, and base predictions.
- `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/`: current seven-group timing update, fit tables, CAF, and RT-distribution figures.
- `docs/`: model manifest, framework, limitations, reproduction, and cleanup record.
- `tests/`: focused tests for the retained model and plotting inputs.

## Key figures

- Presentation mechanism chain: `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/05_natural_emergence_evidence_chain.pdf`
- Corrected-equivalent CAF: `artifacts/results/r5_caf_delta_curves_20260803/current_model_caf_human_vs_model.pdf`
- Corrected-equivalent RT distribution: `artifacts/results/r5_rt_distribution_kde_20260803/observed_vs_model_rt_kde.pdf`
- Seven-group CAF: `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/figures_publication/all_age_caf_updated_model.pdf`
- Seven-group RT distributions: `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/figures_publication/all_age_rt_distribution_updated_model.pdf`
- Original R5 CAF: `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/figures/representative_caf_human_vs_model.pdf`

## Limitations

The retained results are in-sample diagnostics on representative subsets. Timing parameters were selected on the same trials used for evaluation, human RT tails remain broader than model tails, and participant coverage is uneven: the 40-49 and 80-89 groups each contain only four participants. The model produces no congruent errors in any age group, so overall accuracy is slightly inflated and congruent-error behavior is not explained. The evidence and accumulator pass focused consistency checks, but the model should not be described as a final validated account of human conflict control or causal aging.

See [docs/REPRODUCING_RESULTS.md](docs/REPRODUCING_RESULTS.md) and [docs/current_results_and_limitations.md](docs/current_results_and_limitations.md) for the exact checks and interpretation boundary.
