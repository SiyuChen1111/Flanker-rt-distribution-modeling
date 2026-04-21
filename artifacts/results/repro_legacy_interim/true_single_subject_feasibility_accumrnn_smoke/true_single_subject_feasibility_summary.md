# True single-subject feasibility summary (VGG + recurrent accumulator)

This workflow tested a **bounded true single-subject feasibility screen** for `VGG + AccumRNN`: each fit was learned from one subject's own trials, using an **internal noisy recurrent accumulator** to produce choice + RT via **differentiable boundary crossing**.

## Why this branch exists
- Previous heterogeneity result: `HETEROGENEITY-NOT-SUPPORTED`
- Previous minimal mechanism result: `CAPTURE-PROBE-NOT-SUPPORTED`
- Previous VGG + WW single-subject verdict: `VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE`
- Those results motivate changing the accumulation mechanism itself rather than patching WW or sampling RT/choice post-hoc from predicted parameters.

## Panel-wide result
- final verdict: `VGGACCUM-SINGLE-SUBJECT-NOT-FEASIBLE`
- recommendation: `deprioritize_accumrnn`
- scope: `bounded_panel_bounded_trial_budget_screen`
- feasible subjects: `0/2`
- age-group feasible counts: `{'20-29': 0, '80-89': 0}`

## Subject-level feasibility rule
A subject counts as feasible only if all of the following are true:
- predicted RTs remain right-skewed (`pred_skewness > 0.5`)
- the fitted model actually produces errors on held-out trials
- model error-vs-correct RT direction matches the subject's own direction
- model accuracy is not degenerate relative to the subject (`|model_accuracy - human_accuracy| <= 0.05`)

## Scorecard highlights
| age_group | user_id | selected_scale | pred_skewness | true_skewness | human_error_minus_correct_rt | model_error_minus_correct_rt | human_slow_error | model_slow_error | subject_feasible |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 20-29 | 3875 | 0.100 | 1.123 | 0.197 | -0.00621957 | -0.060187 | no | no | no |
| 80-89 | 6609 | 0.100 | 1.555 | 2.556 | -0.151992 | 0.0310725 | no | yes | no |

## Interpretation
This is a framework screen under bounded compute. A smoke/verification run that produces these outputs does **not** establish scientific feasibility; it only establishes that the recurrent-accumulator workflow is runnable end-to-end and comparable to the WW single-subject screen.

**Panel diagnostics (counts over the bounded panel):**
- `skew_present (pred_skewness > 0.5)`: 2/2
- `model_has_errors` (non-empty predicted error regime on held-out): 2/2
- `error_direction_match` (sign(model_gap) == sign(human_gap)): 1/2
- `nondegenerate_accuracy` (|model_acc - human_acc| <= 0.05): 0/2

**Most common failure modes (not mutually exclusive):**
- missing model errors on held-out: 0/2
- wrong error-vs-correct RT direction given errors: 1/2
- degenerate accuracy relative to subject: 2/2
- insufficient predicted RT skew: 0/2

**Slow-error direction:** we define slow-error as `error_minus_correct_rt > 0`. The scorecard includes both the human and model slow-error booleans explicitly.