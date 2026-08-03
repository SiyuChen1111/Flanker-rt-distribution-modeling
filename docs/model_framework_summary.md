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
4. The `natural_smooth_5stage` schedule maps earlier and later VGG layers to an 80-step decision sequence. The current exploratory model compresses this schedule so that late target evidence arrives with enough time to affect a coupled decision.
5. The four-channel sequence enters the Wong-Wang accumulator.
6. The R5 readout applies a sustained crossing rule with group-specific threshold and margin; current fits choose the winning channel at this same readout step.
7. Group-specific non-decision time is added to produce final RT.
8. Diagnostics compare human and model behavior by age group, congruency, RT quantile, response category, and trajectory source.

## Key implementation points

- `code/scripts/run_natural_layer_to_time_var_ww_diagnostic.py` builds the natural layer-to-time schedule and normalized evidence inputs.
- `code/scripts/vgg_wongwang_lim.py` implements the Wong-Wang update and `DiffDecisionMultiClass`.
- `code/scripts/run_representative_extreme_age_subset_fitting.py` builds the representative young/older subset, fits finite R0-R5 candidates, and writes the retained R5 package.
- `code/scripts/optimize_natural_layer_to_time_rt_shape.py` contains the readout utilities used by the R5 workflow.
- `code/scripts/run_r5_supervisor_followup.py` reruns the supervisor follow-up diagnostics without retraining the full VGG model.
- `code/scripts/run_ww_diffdecision_core_audit.py` isolates two- and four-choice Wong-Wang/DiffDecision behavior from the VGG and R5 layers.
- `code/scripts/run_r5_supervisor_round2.py` produces explicit-tick CAF/CRF figures and controlled RT-shape and speed-accuracy checks.
- `code/scripts/run_r5_choice_rule_alignment_audit.py` compares legacy whole-trajectory choice with choice at the RT readout step.
- `code/scripts/run_r5_choice_coupled_schedule_optimization.py` searches schedule timing while keeping VGG logits and retained Wong-Wang parameters fixed.
- `code/scripts/plot_r5_rt_distribution_kde.py` and `code/scripts/plot_r5_caf_and_delta_curves.py` produce current distribution and congruency diagnostics.

`DiffDecisionMultiClass` records whether each channel actually crossed. A channel that does not cross receives the final simulation step as a deadline sentinel rather than `0`; downstream analyses must use the crossing flag and must not count the sentinel as a genuine RT. The optional normalized-competition variant is retained only for ablation and is not active in the retained R5 model.

## Retained baseline and current exploratory timing

The retained R5 package uses group-specific timing and readout settings:

- young 20-29: `t0_mean=0.55s`, `t0_sd=0.12s`, `evidence_gain=0.8`, `threshold=0.12`, `sustained_k=2`, `margin=0.0`
- older 80-89: `t0_mean=0.75s`, `t0_sd=0.20s`, `evidence_gain=0.8`, `threshold=0.14`, `sustained_k=2`, `margin=0.02`

The diagnostic reconstruction treats the accumulator as deterministic given evidence because the retained R5 path uses `noise_ampa=0.0`.

The current exploratory result keeps the retained VGG logits and Wong-Wang parameters but changes the evidence schedule and refits non-decision time:

- young 20–29: `compression=0.275`, `late_shift_s=0.04`, `width_scale=0.8`, `t0_mean=0.447s`, `t0_sd=0.12s`
- older 80–89: `compression=0.475`, `late_shift_s=-0.04`, `width_scale=0.8`, `t0_mean=0.670s`, `t0_sd=0.18s`

Whole-trajectory maximum choice is retained only to reproduce historical R5 artifacts. It must not be described as a first-passage choice rule.

## Main difference from the ideal target

The choice-coupled schedule result aligns mean RT, overall accuracy, incongruent accuracy, and incongruent CAF on the representative subset. It still does not reproduce human RT tails, within-condition skew, congruency RT costs, or congruent errors, and final RT spread still depends on t0 variability.

The current framework should therefore be described as a targeted diagnostic model, not as proof that the full human evidence-accumulation process has been captured.

Controlled core simulations show that the four-choice accumulator can produce stable, right-skewed first-passage times in a calibrated operating regime. Real VGG evidence supplies the early-flanker/later-target structure without an added handcrafted conflict controller, and schedule compression can transmit it to the corrected readout. These are partial mechanism results, not held-out evidence for a human cognitive mechanism.
