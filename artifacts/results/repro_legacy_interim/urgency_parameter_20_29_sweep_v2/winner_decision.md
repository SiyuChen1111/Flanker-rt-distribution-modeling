# Urgency winner decision

## Verdict

No unique winner is promoted from `urgency_parameter_20_29_sweep_v2`.

## Why

The top-ranked candidates under the primary metric hierarchy are:

1. `additive_urgency`, start `0.60`, slope `0.25`, floor `0.00`
2. `collapsing_bound`, start `0.60`, slope `0.25`, floor `0.00`

They are numerically identical across the saved ranking columns used for decision-making:

- `score = 0.6033560038`
- `frac_at_ceiling = 0.0`
- `model_congruency_rt_gap = 0.0342101455`
- `pred_q99 = 1.7400000095`
- `coverage_score = 0.0938783437`
- `pred_skewness = -0.4117823839`
- `error_minus_correct_rt = -0.7217578888`

Because the primary ranking did not separate the two urgency types, there is no single unambiguous winner to promote.

## Interpretation

- The branch is improved enough to remove ceiling artifacts and restore a positive congruency gap.
- However, the current sweep does not provide evidence that `urgency_type` itself is functionally resolved as a meaningful choice at the top rank.
- Under the one-winner-or-none policy, this should be treated as **no unique winner** rather than forcing an arbitrary promotion.

## Recommended next move

Before fuller-data promotion, either:

1. introduce an additional primary discriminator that separates the top tied settings, or
2. audit whether the current implementation makes `additive_urgency` and `collapsing_bound` effectively equivalent under the tested configuration.
