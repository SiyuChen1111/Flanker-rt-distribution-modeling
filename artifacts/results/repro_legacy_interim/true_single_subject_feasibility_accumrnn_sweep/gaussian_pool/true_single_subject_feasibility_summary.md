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
- feasible subjects: `0/6`
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
| 20-29 | 3875 | 0.300 | 1.838 | 23.786 | -0.0482132 | nan | no | no | no |
| 20-29 | 5675 | 0.100 | 1.836 | 0.923 | -0.0165365 | -0.00708513 | no | no | no |
| 20-29 | 677 | 0.500 | 0.127 | 6.637 | -0.0704782 | nan | no | no | no |
| 80-89 | 3403 | 0.500 | 0.548 | 7.467 | 0.159388 | nan | yes | no | no |
| 80-89 | 6609 | 0.500 | -5.160 | 13.892 | -0.159868 | nan | no | no | no |
| 80-89 | 984 | 0.300 | -0.576 | 2.810 | -0.245767 | -0.295424 | no | no | no |

## Interpretation
This is a framework screen under bounded compute. A smoke/verification run that produces these outputs does **not** establish scientific feasibility; it only establishes that the recurrent-accumulator workflow is runnable end-to-end and comparable to the WW single-subject screen.

**Panel diagnostics (counts over the bounded panel):**
- `skew_present (pred_skewness > 0.5)`: 3/6
- `model_has_errors` (non-empty predicted error regime on held-out): 2/6
- `error_direction_match` (sign(model_gap) == sign(human_gap)): 2/6
- `nondegenerate_accuracy` (|model_acc - human_acc| <= 0.05): 0/6

**Most common failure modes (not mutually exclusive):**
- missing model errors on held-out: 4/6
- wrong error-vs-correct RT direction given errors: 0/6
- degenerate accuracy relative to subject: 6/6
- insufficient predicted RT skew: 3/6

**Slow-error direction:** we define slow-error as `error_minus_correct_rt > 0`. The scorecard includes both the human and model slow-error booleans explicitly.