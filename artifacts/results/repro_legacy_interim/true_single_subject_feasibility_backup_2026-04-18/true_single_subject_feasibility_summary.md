# True single-subject feasibility summary

This workflow tested **true single-subject** fitting for `VGG + WW`: each fit was learned from one subject's own trials rather than from group parameters plus a small tweak.

## Why this branch exists
- Previous heterogeneity result: `HETEROGENEITY-NOT-SUPPORTED`
- Previous minimal mechanism result: `CAPTURE-PROBE-NOT-SUPPORTED`
- Those results ruled out aggregation artifacts and tiny mechanism patches as the main explanation.
- This branch asks the framework question directly: is `VGG + WW` viable at the individual-subject level?

## Panel-wide result
- final verdict: `VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE`
- recommendation: `deprioritize_vgg_ww`
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
| 20-29 | 3875 | 0.500 | 0.504 | 23.786 | -0.0482132 | nan | no | no | no |
| 20-29 | 5675 | 0.500 | 0.594 | 0.923 | -0.0165365 | nan | no | no | no |
| 20-29 | 677 | 0.500 | 0.609 | 6.637 | -0.0704782 | nan | no | no | no |
| 80-89 | 3403 | 0.300 | 0.686 | 7.467 | 0.159388 | nan | yes | no | no |
| 80-89 | 6609 | 0.100 | -0.996 | 13.892 | -0.159868 | 0.422134 | no | yes | no |
| 80-89 | 984 | 0.100 | -1.279 | 2.810 | -0.245767 | 0.212496 | no | yes | no |

## Interpretation
This is the direct framework-retention test. If the verdict is `VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE`, the current evidence supports deprioritizing `VGG + WW` rather than continuing to patch it indirectly.

**Panel diagnostics (counts over the bounded panel):**
- `skew_present (pred_skewness > 0.5)`: 4/6
- `model_has_errors` (non-empty predicted error regime on held-out): 2/6
- `error_direction_match` (sign(model_gap) == sign(human_gap)): 0/6
- `nondegenerate_accuracy` (|model_acc - human_acc| <= 0.05): 4/6

**Most common failure modes (not mutually exclusive):**
- missing model errors on held-out: 4/6
- wrong error-vs-correct RT direction given errors: 2/6
- degenerate accuracy relative to subject: 2/6
- insufficient predicted RT skew: 2/6

**Slow-error direction:** we define slow-error as `error_minus_correct_rt > 0`. The scorecard includes both the human and model slow-error booleans explicitly.