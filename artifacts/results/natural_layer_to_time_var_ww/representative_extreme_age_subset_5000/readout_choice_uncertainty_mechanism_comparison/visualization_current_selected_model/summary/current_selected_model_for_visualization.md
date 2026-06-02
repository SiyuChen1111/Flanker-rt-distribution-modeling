# Current Selected Model for Visualization

## Selected model

- Main display model: M3_time_gap, age-specific parameters.
- Ranking position: 1 of 154 rows.
- Composite score: 0.2592.
- Parameterization:
  - young_20_29: sigma_base=0.002, sigma_time=0.004, sigma_gap=0.008, gap_scale=0.08
  - older_80_89: sigma_base=0.001, sigma_time=0.012, sigma_gap=0.012, gap_scale=0.08

## Why this model

- It is the top-ranked current readout/choice noise model.
- It is the only selected configuration that brings older congruent errors close to the human level while also preserving fast-error patterns.
- Shared M3 is useful as a comparison, but it still produces too few older congruent errors.

## Advantage over alternatives

- Deterministic shared score: 0.3888.
- Time-only shared score: 0.3051.
- Gap-only shared score: 0.2841.
- Time+gap shared score: 0.2824.

## Data files available for plotting

- metrics/human_error_rate_by_rt_bin.csv
- metrics/human_reference_rt_error_metrics.csv
- metrics/readout_choice_model_bootstrap_ci.csv
- metrics/readout_choice_model_comparison_summary.csv
- metrics/readout_choice_model_error_rate_by_rt_bin.csv
- metrics/readout_choice_model_ranking.csv
- metrics/readout_choice_model_seed_level_metrics.csv
- metrics/readout_choice_model_selected_summary.csv
- metrics/readout_choice_model_split_validation.csv
