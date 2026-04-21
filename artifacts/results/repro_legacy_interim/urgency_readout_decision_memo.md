# Urgency readout decision memo

## Bottom line

The urgency branch is now reproducible enough to evaluate under the canonical deterministic path, but the current constrained sweep does **not** produce a single unique winner to promote.

The branch should remain the active mechanism line, but promotion to fuller-data validation is paused because the top-ranked `additive_urgency` and `collapsing_bound` settings are numerically indistinguishable under the primary decision metrics.

## What was fixed before this memo

- Determinism patching and urgency canonicalization were completed first.
- The fixed-horizon control recheck stayed in the same qualitative urgency regime after patching.
- The urgency branch no longer exhibits the earlier sign-flip instability within the patched canonical path.

Relevant evidence bundles:

- `artifacts/results/repro_legacy_interim/repro_drift/`
- `artifacts/results/repro_legacy_interim/readout_mode_20_29_sweep_recheck_postpatch/`
- `artifacts/results/repro_legacy_interim/urgency_parameter_20_29_sweep_v2/`

## Primary metric decision rule used

The candidate ranking followed the repo-local mechanism hierarchy:

1. ceiling control
2. RT tail / support
3. conflict-conditioned structure
4. only then secondary diagnostics such as error RT structure

Global slow-error was not used as the main target.

## Best-ranked candidates

Top two candidates from `urgency_parameter_20_29_sweep_v2`:

1. `additive_urgency`, start `0.60`, slope `0.25`, floor `0.00`
2. `collapsing_bound`, start `0.60`, slope `0.25`, floor `0.00`

They are tied across the saved ranking columns used for promotion:

- `score = 0.6033560038`
- `frac_at_ceiling = 0.0`
- `model_congruency_rt_gap = 0.0342101455`
- `pred_q95 = 1.6200000048`
- `pred_q99 = 1.7400000095`
- `coverage_score = 0.0938783437`
- `pred_skewness = -0.4117823839`
- `error_minus_correct_rt = -0.7217578888`

Because no primary metric separates them, this is a **no-unique-winner** outcome.

## q95 / q99 summary

The best urgency candidates restore a plausible upper tail without reintroducing ceiling artifacts.

- Best tied urgency settings:
  - `pred_q95 = 1.62`
  - `pred_q99 = 1.74`
- Patched canonical default urgency recheck:
  - `pred_q95 = 1.70`
  - `pred_q99 = 1.82`

Interpretation:

- The constrained urgency sweep is capable of producing a cleaner, slightly less inflated upper tail than the patched default urgency setting.
- Tail behavior is therefore improved enough to keep urgency active as the main mechanism branch.

## Congruent vs incongruent RT conclusion

The top tied urgency settings maintain the correct sign of the conflict RT effect:

- `model_congruency_rt_gap = +0.0342`

This is smaller than the human reference gap used elsewhere in the repository, but it stays positive and therefore respects the mechanism-level direction constraint that previously blocked promotion.

Interpretation:

- urgency no longer fails the branch on conflict-gap sign;
- however, the top settings are still not clearly distinct enough to justify a single promoted winner.

## Error vs correct RT conclusion

The top tied urgency settings still show:

- `pred_error_rt = 0.2720`
- `pred_correct_rt = 0.9938`
- `error_minus_correct_rt = -0.7218`

Interpretation:

- error RT structure remains secondary and still does not align with a global slow-error pattern;
- this does **not** block urgency as the active mechanism line, but it does mean urgency is not yet behaviorally complete.

## Decision

### What improves enough to keep urgency active

- zero ceiling mass
- plausible q95 / q99 tail support
- positive congruency RT gap
- stable deterministic canonical evaluation path

### What prevents promotion right now

- no unique winner: top `additive_urgency` and `collapsing_bound` settings are tied on the primary decision surface
- negative `error_minus_correct_rt` remains unresolved, even though it is secondary

## Recommended next step

Continue the urgency line, but do **not** promote a fuller-data winner yet.

The next move should be one of:

1. add one additional primary discriminator that can separate the tied top settings, or
2. audit whether `additive_urgency` and `collapsing_bound` are effectively equivalent under the current implementation/parameterization.

Until that tie is resolved, fuller-data promotion would be arbitrary rather than evidence-based.
