# Trajectory viability diagnostic summary

## Core answer

- Late readout does not fully repair the incongruent failure. Relative to original readout, +200 ms improves deterministic incongruent accuracy from 0.1635 to 0.2028, but the failure remains large.
- Even at the final available time point, deterministic incongruent accuracy is 0.9216.
- The best possible post-readout upper bound reaches 0.9218 deterministic incongruent accuracy. This is the ceiling available inside the current trajectory family if commitment timing were idealized.

## Recovery viability

- Mean proportion of incongruent trials where target ever becomes rank 1 after original readout: 0.9218.
- Mean proportion of incongruent trials where target ever exceeds flanker after original readout: 0.9218.
- Mean proportion of incongruent trials where target ever exceeds max other after original readout: 0.9218.

## Human alignment

- In incongruent trials, target-crossing status predicts human correctness by 0.0085 on average.
- Correlation between target crossing time and human RT in incongruent trials: 0.0028.
- In human-correct incongruent trials, target crossing occurs before human RT with mean proportion 0.7657.

## Interpretation

- If late readout had strongly repaired incongruent accuracy, commitment timing would still be the main lever. That is not what the current trajectories show.
- Because even final-time and best-post-readout analyses leave a large residual failure, the bottleneck is not just premature commitment. The underlying task-relevant evidence recovery is often too weak or too late.
- This points more toward evidence mapping / WW target-recovery dynamics than toward a pure gating-fit next step.
- The most defensible next model direction is to modify evidence dynamics or the target-recovery mechanism first, then revisit commitment timing after the trajectories themselves are more viable.
