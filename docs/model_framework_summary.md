# Model Framework Summary

## Current modules

- Visual evidence extraction from LIM / Flanker stimuli
- VGG layer-to-time evidence scheduling
- Four-choice Wong-Wang state dynamics
- R5 sustained crossing / readout rule
- Group-specific non-decision time (`t0_mean`, `t0_sd`)
- CAF, CRF, first-passage, state-trajectory, and incongruent-error diagnostics

## Current data flow

1. A trial stimulus is mapped to four response-direction evidence channels.
2. VGG layer outputs are cached at the stimulus level.
3. The cached layerwise evidence is normalized with `per_layer_gap_scale`.
4. The `natural_smooth_5stage` schedule maps earlier and later VGG layers to an 80-step decision sequence.
5. The four-channel sequence enters the Wong-Wang accumulator.
6. The R5 readout applies a sustained crossing rule with group-specific threshold and margin.
7. Group-specific non-decision time is added to produce final RT.
8. Diagnostics compare human and model behavior by age group, congruency, RT quantile, response category, and trajectory source.

## Key implementation points

- `code/scripts/run_natural_layer_to_time_var_ww_diagnostic.py` builds the natural layer-to-time schedule and normalized evidence inputs.
- `code/scripts/vgg_wongwang_lim.py` implements the Wong-Wang update and `DiffDecisionMultiClass`.
- `code/scripts/run_representative_extreme_age_subset_fitting.py` builds the representative young/older subset, fits finite R0-R5 candidates, and writes the retained R5 package.
- `code/scripts/optimize_natural_layer_to_time_rt_shape.py` contains the readout utilities used by the R5 workflow.
- `code/scripts/run_r5_supervisor_followup.py` reruns the supervisor follow-up diagnostics without retraining the full VGG model.

## Current R5 parameters

The retained R5 package uses group-specific timing and readout settings:

- young 20-29: `t0_mean=0.55s`, `t0_sd=0.12s`, `evidence_gain=0.8`, `threshold=0.12`, `sustained_k=2`, `margin=0.0`
- older 80-89: `t0_mean=0.75s`, `t0_sd=0.20s`, `evidence_gain=0.8`, `threshold=0.14`, `sustained_k=2`, `margin=0.02`

The current diagnostic reconstruction treats the R5 accumulator as deterministic given evidence, because the retained R5 path uses `noise_ampa=0.0`.

## Main difference from the ideal target

R5 improves RT scale on the representative subset, but it still does not fully match human choice/error structure. In particular, the hard first-crossing decision-time distribution is more compressed than final RT, and final RT spread depends substantially on t0 variability.

The current framework should therefore be described as a targeted diagnostic model, not as proof that the full human evidence-accumulation process has been captured.
