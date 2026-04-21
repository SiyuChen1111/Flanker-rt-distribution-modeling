# Urgency mechanism decision memo

## Bottom line

No unique winner remains under the richer diagnostics.

The two urgency candidates remain unresolved because they are operationally equivalent under the current canonical implementation and fixed regime, not merely close in score.

## Fixed regime used

- `time_steps_factor = 2.0`
- `lambda_pileup = 1.0`
- `scale = 0.1`
- no new readout family
- no broad urgency sweep

## Candidates compared

1. `additive_urgency`, start `0.60`, slope `0.25`, floor `0.00`
2. `collapsing_bound`, start `0.60`, slope `0.25`, floor `0.00`

## What the new diagnostics show

### CAF

- In the earliest incongruent RT bin, human accuracy is `0.7325`.
- Both urgency candidates produce `0.9544` in that same diagnostic slot.
- Because both candidates are identical on this analysis, CAF realism does not break the tie.

### Delta plot

- Human early-quantile delta is `0.0220` s.
- Both urgency candidates produce `-0.0843` s at the first quantile.
- The full delta curves overlap exactly for the two urgency candidates.

### Conditional error RT structure

- Human incongruent error-minus-correct RT is `-0.1204` s.
- Both urgency candidates produce `-0.7098` s for the same conditional diagnostic.
- This means the richer error-RT analysis still does not separate the two mechanisms.

### Conditional tail summary

- The conditional q90/q95/q99 and skewness summaries are identical across the two candidates.
- Therefore the slow-tail structure does not provide a tie-break under the current implementation.

## Equivalence audit

- Saved prediction bundles identical across all arrays: `True`
- Differing saved-array keys: `[]`
- Decision-variable range: `0.000000` to `0.695936`
- Additive vs collapsing commit masks identical when evaluated directly from the saved arrays: `True`
- Commit-mask differences found: `0`

Interpretation:

- Under the current implementation, `additive_urgency` and `collapsing_bound` collapse to the same effective commitment rule for this candidate setting.
- That means the present stage cannot legitimately promote one mechanism over the other.

## Decision

**No unique winner remains.**

The most defensible reading is not simply that the two candidates are "still tied," but that they are **mechanistically unresolved because the current parameterization makes them empirically equivalent on the canonical deterministic path**.

## Recommended next branch

Because the tie persists under mechanism-sensitive diagnostics, the next useful change should be one of:

1. a stronger urgency parameterization that is not algebraically equivalent across urgency types under `floor = 0.00`, or
2. a mechanism pivot toward dynamic attentional selection / DMC-like conflict control if the goal is to improve early capture and conditional slow-tail structure.

Fuller-data winner promotion should remain paused until the branch actually produces distinguishable mechanism-level predictions.
