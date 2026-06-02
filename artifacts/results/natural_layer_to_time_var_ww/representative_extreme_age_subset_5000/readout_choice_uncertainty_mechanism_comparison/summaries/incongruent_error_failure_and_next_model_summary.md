# Incongruent Error Failure and Next Model Summary

## 1. Why the current model produces many incongruent errors

The failure is already present at deterministic readout: in many incongruent trials, the target is not the top-ranked channel at the readout time. Readout-choice noise then operates on an already wrong or unstable evidence state, so it cannot rescue incongruent accuracy.

## 2. Error type

Noisy incongruent choices by group:
- older_80_89 flanker: 0.709
- older_80_89 other: 0.009
- older_80_89 target: 0.282
- young_20_29 flanker: 0.937
- young_20_29 other: 0.010
- young_20_29 target: 0.053

Deterministic incongruent choices by group:
- older_80_89 flanker: 0.704
- older_80_89 other: 0.008
- older_80_89 target: 0.288
- young_20_29 flanker: 0.954
- young_20_29 other: 0.008
- young_20_29 target: 0.038

## 3. Target recovery at readout

- older_80_89: target recovered before readout=0.219; readout before target recovery=0.781; deterministic accuracy=0.288; noisy accuracy=0.282; mean target rank=1.72; mean signed margin=-0.0295.
- young_20_29: target recovered before readout=0.037; readout before target recovery=0.963; deterministic accuracy=0.038; noisy accuracy=0.053; mean target rank=1.97; mean signed margin=-0.0317.

## 4. Scoring function diagnosis

Old best model: age_specific M3_time_gap with score 0.2592. The old score over-weighted congruent fast-error recovery and did not sufficiently penalize massive incongruent error rates.
New constrained best model: age_specific M2_gap_only with score 97.5755; passes all hard constraints=False.

## 5. Does the old time+gap age-specific model remain first?

No if strict incongruent/accuracy constraints are prioritized: the new ranking is led by age_specific M2_gap_only. However, none of the existing ungated readout-noise parameter families passes all hard constraints in this subset.

## 6. What still holds

- Time+gap readout-choice uncertainty can recover congruent fast errors.
- The current evidence-to-choice noise mechanism does not alter RT timing directly.

## 7. What needs caution

- The current selected time+gap age-specific model does not fit overall human behavior because incongruent error rate is far too high.
- Any statement that older requires stronger uncertainty remains exploratory.

## 8. Gating diagnostic

Best gating counterfactual: G1_target_rank_1 with mean absolute error-rate deviation 0.0262.
Because full trial-level future trajectories are not available, this is a counterfactual diagnostic rather than a fully re-read-out model.

## 9. Recommended next model

The next candidate should be a target-recovery-gated readout: keep time+gap choice uncertainty, but prevent commitment when the target is not yet the top-ranked channel or the signed target margin is still negative. This directly addresses the incongruent failure while preserving the congruent fast-error mechanism.
