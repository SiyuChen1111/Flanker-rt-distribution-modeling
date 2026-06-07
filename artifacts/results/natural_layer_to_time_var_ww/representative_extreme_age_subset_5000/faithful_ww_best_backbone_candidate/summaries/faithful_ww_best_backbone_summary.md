# Faithful WW Best Backbone Candidate

## Model

- `model_config_id`: `S1_MAP1_4_cg1.00_dg0.50_mean_abs_clip2_off0.05_eg1.25_n0.010_th0.95`
- `status`: `near_balanced_ww_candidate`
- `source`: `faithful_ww_hvenet_core_fit_stage2_stage3_completion`

## Why this folder exists

This directory packages the current report-worthy faithful WW backbone candidate on its own, instead of mixing it into the historical `best_model_R5_combined_best` folder.

## Main takeaways

1. This model is the cleanest faithful WW backbone candidate for reporting.
2. It keeps incongruent error at a manageable level for both young and older groups.
3. It improves RT/CAF-related behavior relative to direct mapping.
4. It still does **not** produce nonzero congruent fast errors.
5. So it is suitable as the abstract backbone candidate, but not as a full final explanation of all human error signatures.

## Human-signature figures included

- RT distribution (KDE)
- Congruent / incongruent RT distribution (KDE)
- CAF
- SAT-related evidence-strength signature
- RT quantile profile
- RT skewness / tail summary
- WW internal `S_traj`
- target-minus-flanker trajectory
- example trial WW trajectories
- example trial evidence -> WW state -> readout
- single representative mechanistic trial figure

## Figure style

- White background
- No horizontal background guide lines
- Significant human-vs-model differences are annotated where they are well-defined in the grouped bar panels

## Core metrics

- young overall accuracy: `0.9373`
- older overall accuracy: `0.9345`
- young congruent error rate: `0.0000`
- older congruent error rate: `0.0000`
- young incongruent error rate: `0.1254`
- older incongruent error rate: `0.1310`
- young RT quantile RMSE: `0.0288`
- older RT quantile RMSE: `0.0507`
- young CAF slope sign match: `0.925`
- older CAF slope sign match: `0.975`
