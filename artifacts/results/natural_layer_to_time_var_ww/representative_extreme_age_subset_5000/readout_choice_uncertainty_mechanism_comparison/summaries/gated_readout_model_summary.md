# Gated readout model summary

## Data status

- Full per-trial trajectory CSV files were not found.
- Full per-trial trajectories were reconstructed from the saved evidence cache and saved R5 parameter table. No VGG or main WW model was retrained.
- The resulting simulation performs real delayed readout over future time points.

## Best model

- Best overall ranked model: `original_time_gap_no_gating`.
- Best gated candidate by the balanced ranking: `soft_gated_theta0.000_temp0.020_maxdelay_0.20`.
- Strongest incongruent-error reduction: `margin_gated_m0.020_maxdelay_0.20` with mean incongruent error rate 0.7930, but mean forced readout 0.4141 and mean delay 0.0835s.
- The gate is interpreted as a response commitment-stage task-relevant evidence gate, not as a mechanism that knows the correct answer.
- Time+gap choice uncertainty is retained after the gated readout point, so congruent fast errors are still explained by early/low-gap readout uncertainty.

## Group metrics

### Young 20-29
- overall_accuracy: best=0.5142; original=0.5184
- congruent_error_rate: best=0.0152; original=0.0156
- incongruent_error_rate: best=0.9572; original=0.9484
- congruent_error_rt_minus_correct_rt: best=-0.0518; original=-0.0087
- mean_gating_delay: best=0.0419; original=0.0000
- proportion_forced_readout: best=0.1302; original=0.0000
- choice_type_proportion_flanker: best=0.4780; original=0.4734

### Older 80-89
- overall_accuracy: best=0.6434; original=0.6384
- congruent_error_rate: best=0.0084; original=0.0088
- incongruent_error_rate: best=0.7073; original=0.7170
- congruent_error_rt_minus_correct_rt: best=-0.0050; original=-0.0068
- mean_gating_delay: best=0.0496; original=0.0000
- proportion_forced_readout: best=0.1728; original=0.0000
- choice_type_proportion_flanker: best=0.3524; original=0.3570

## Interpretation

- The tested gates did not fully fix the incongruent flanker-over-selection problem. The balanced score still favors the no-gating baseline because gated variants either leave incongruent errors high or introduce substantial forced readout/delay.
- Rank and margin gates directly test whether delaying commitment until task-relevant evidence has recovered reduces premature flanker commitments.
- Soft gates are more psychologically graded, but their stochastic commitment can preserve more early errors and may leave more incongruent failures.
- Incongruent-only gates are diagnostic controls. They show how much of the problem is specifically due to incongruent premature commitment and should be treated as exploratory.
- Results suitable for reporting are the reconstructed-trajectory gated simulations, the ranking, and the human/model condition comparisons. The exact best parameter should still be treated as exploratory until formal parameter fitting is added.
- A formal next step would fit the commitment gate parameters jointly with the time+gap uncertainty parameters instead of selecting from this diagnostic grid.
