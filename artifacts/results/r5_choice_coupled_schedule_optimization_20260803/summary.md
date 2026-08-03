# Choice-coupled schedule compression optimization

## Design

- Tested `171` schedule configurations per age group while keeping VGG logits and retained R5 Wong-Wang parameters fixed.
- Choice was always the winner at the sustained-crossing RT step.
- Candidates below `95%` real crossing coverage were ineligible.
- Non-decision mean and spread were refit for every candidate.

## Selected result

- Shared optimum: `{'compression': 0.275, 'late_shift_s': 0.04, 'width_scale': 1.2}` with score `0.2173`.
- Age-specific optimum score: `0.1678`; age-specific schedules selected: `True`.
- Young 20–29: schedule `{'compression': 0.275, 'late_shift_s': 0.04, 'width_scale': 0.8}`, crossing `1.000`, accuracy `0.961` vs human `0.949`, incongruent accuracy `0.922` vs human `0.917`, RT quantile MAE `0.061` s, incongruent CAF RMSE `0.028`.
  Mean input reversal `0.077` s, mean WW-state reversal `0.139` s, target state recovered before readout on `86.0%` of incongruent trials (`93.4%` of correct; `0.0%` of errors).
- Older 80–89: schedule `{'compression': 0.475, 'late_shift_s': -0.04, 'width_scale': 0.8}`, crossing `1.000`, accuracy `0.979` vs human `0.976`, incongruent accuracy `0.959` vs human `0.961`, RT quantile MAE `0.038` s, incongruent CAF RMSE `0.009`.
  Mean input reversal `0.116` s, mean WW-state reversal `0.217` s, target state recovered before readout on `95.8%` of incongruent trials (`99.9%` of correct; `0.0%` of errors).

## Interpretation boundary

This is an exploratory representative-subset schedule optimization, not a held-out or full-cohort fit. It tests whether the existing real-VGG target-recovery signal can support a theoretically coupled decision after its timing is corrected; it does not by itself validate a human conflict-control mechanism.
