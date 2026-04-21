# Flanker Mechanism-Oriented Evaluation Framework

## Why this framework exists

Recent model exploration in this repository showed a recurring pattern:

- some variants preserve coarse behavioral structure but produce strong RT ceiling artifacts;
- some variants remove ceiling artifacts but collapse RT distributions into unrealistically narrow or short-latency regimes;
- some variants improve overall score and RT scale, but invert conflict or error-RT structure.

This means the project should not treat a single aggregate score, a single mean RT match, or the presence/absence of global slow-error as the main success criterion. The evaluation should be reorganized around **psychological mechanism plausibility**.

## Literature-backed interpretation

The current literature review supports four working conclusions.

1. In standard flanker tasks, the most stable empirical signature is often **fast errors under incongruent conflict**, not a universal global slow-error effect.
2. Human right-tail structure is more plausibly tied to **dynamic attentional selection, suppression, time-varying evidence, or decision commitment/readout mechanisms** than to conflict alone.
3. Because of that, global `error_minus_correct_rt > 0` should be treated as a **secondary** rather than primary target.
4. Mechanism evaluation should prioritize whether the model reproduces the **shape and conditional structure** of conflict behavior, not just central moments.

## Primary metrics

These should be the first-pass filters when deciding whether a model variant is worth keeping.

### P1. Ceiling-artifact control

Purpose: ensure RT behavior is not being dominated by hard truncation.

Use:

- `frac_at_ceiling`
- `n_at_ceiling`
- `pred_q95`
- `pred_q99`
- `coverage_score`

Interpretation:

- If ceiling mass is high, downstream conclusions about skew, tails, or slow/fast error are unreliable.
- This is a gating criterion, not just another metric in the average.

### P2. RT distribution shape

Purpose: test whether the model produces a human-like RT distribution, especially in the slow tail.

Use:

- `pred_skewness`
- `quantile_score`
- `rt_shape_score`
- conditional RT distribution losses when available

Interpretation:

- Mean/median agreement is insufficient.
- A usable model should get the tail and asymmetry into the right regime.

### P3. Conflict-conditioned structure

Purpose: test whether conflict changes RT in the right direction and roughly the right magnitude.

Use:

- `model_congruency_rt_gap`
- congruent vs incongruent RT distribution comparisons
- conditional RT distribution metrics by congruency

Interpretation:

- Flanker is fundamentally a conflict task.
- A model with plausible global RTs but wrong conflict structure should not be treated as psychologically adequate.

## Secondary metrics

These are still useful, but should not dominate model selection.

### S1. Error-vs-correct RT structure

Use:

- `pred_error_rt`
- `pred_correct_rt`
- `error_minus_correct_rt`
- human analogs of the same quantities

Interpretation:

- This should be interpreted conditionally when possible.
- Global slow-error is not a necessary signature for a good flanker model.
- It is better used as a structured diagnostic after the primary metrics are already acceptable.

### S2. Overall central-moment fit

Use:

- mean RT
- median RT
- overall accuracy
- response agreement

Interpretation:

- These matter, but they are not enough to establish psychological plausibility.

## Recommended plots to add or emphasize

1. **Conditional Accuracy Function (CAF)**
   - especially incongruent fast-bin accuracy
2. **Delta plot**
   - congruency effect as a function of RT quantile
3. **Conditional error RT plots**
   - correct vs error, separated by congruent/incongruent
4. **Tail-focused RT comparison**
   - q90/q95/q99 or CDF overlays

## Suggested model-selection procedure

### Tier 1: reject obvious failures

Reject or deprioritize a run if any of the following is true:

- high `frac_at_ceiling`
- very poor RT coverage / compressed support
- implausible tail quantiles

### Tier 2: mechanism plausibility

Among surviving runs, prioritize models that:

- get skew/tail into the right regime;
- preserve the correct sign and rough size of the congruency RT gap;
- maintain interpretable conditional RT structure.

### Tier 3: secondary behavioral refinements

Only after Tier 1 and Tier 2 are satisfactory should the project optimize:

- global `error_minus_correct_rt`;
- small mean/median RT mismatches;
- small overall agreement improvements.

## Current practical recommendation for this repository

Based on the recent local experiments:

- plain baseline readout preserves some behavioral structure but still suffers from ceiling-heavy RT generation;
- soft-hazard removes ceiling mass but currently collapses RTs too aggressively;
- urgency-style readout appears to be the most promising direction because it removes ceiling artifacts and restores a more plausible RT scale, even though some conflict/error directions still need tuning.

Therefore, the next optimization cycle should focus more on **readout / commitment mechanisms** than on only adjusting static WW parameters.

## Working hypothesis for the next modeling cycle

The model should be optimized toward this target profile:

1. negligible ceiling mass;
2. human-like right tail;
3. correct conflict-conditioned RT structure;
4. then, conditional error-RT structure.

This keeps the project aligned with a psychological-mechanism interpretation of flanker behavior rather than a purely descriptive fit to global averages.
