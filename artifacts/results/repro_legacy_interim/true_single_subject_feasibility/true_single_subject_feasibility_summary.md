# True single-subject feasibility summary

This workflow tested a **bounded true single-subject feasibility screen** for `VGG + WW`: each fit was learned from one subject's own trials rather than from group parameters plus a small tweak, but on a bounded representative panel and bounded per-subject trial budgets.

## Why this branch exists
- Previous heterogeneity result: `HETEROGENEITY-NOT-SUPPORTED`
- Previous minimal mechanism result: `CAPTURE-PROBE-NOT-SUPPORTED`
- Those results ruled out aggregation artifacts and tiny mechanism patches as the main explanation.
- This branch asks the framework question directly: is `VGG + WW` viable at the individual-subject level?

## Panel-wide result
- final verdict: `VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE`
- recommendation: `deprioritize_vgg_ww`
- scope: `bounded_panel_bounded_trial_budget_screen`
- feasible subjects: `0/6`
- age-group feasible counts: `{'20-29': 0, '80-89': 0}`
- interpretation rule: use this as a screen for whether `VGG + WW` still looks worth pursuing, not as a full uncapped proof over every subject trial.

## Subject-level feasibility rule
A subject counts as feasible only if all of the following are true:
- predicted RTs remain right-skewed (`pred_skewness > 0.5`)
- the fitted model actually produces errors on held-out trials
- model error-vs-correct RT direction matches the subject's own direction
- model accuracy is not degenerate relative to the subject (`|model_accuracy - human_accuracy| <= 0.05`)

## Scorecard highlights
| age_group | user_id | selected_scale | pred_skewness | true_skewness | human_error_minus_correct_rt | model_error_minus_correct_rt | human_slow_error | model_slow_error | subject_feasible |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 20-29 | 3875 | 0.500 | 0.565 | 0.849 | -0.0414681 | nan | no | no | no |
| 20-29 | 5675 | 0.500 | -0.645 | 2.157 | -0.0223074 | nan | no | no | no |
| 20-29 | 677 | 0.300 | 0.635 | 21.774 | -0.0493994 | nan | no | no | no |
| 80-89 | 3403 | 0.100 | -2.546 | 2.238 | -0.0205762 | 0.0803878 | no | yes | no |
| 80-89 | 6609 | 0.100 | 0.422 | 7.349 | -0.165784 | 1.03385 | no | yes | no |
| 80-89 | 984 | 0.300 | 0.623 | 4.087 | -0.182295 | nan | no | no | no |

## Interpretation
This is the direct framework-retention screen under bounded compute. If the verdict is `VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE`, the current evidence supports deprioritizing `VGG + WW` rather than continuing to patch it indirectly, while still acknowledging that this was a bounded panel / bounded trial-budget screen rather than an uncapped final proof.

**Panel diagnostics (counts over the bounded panel):**
- `skew_present (pred_skewness > 0.5)`: 3/6
- `model_has_errors` (non-empty predicted error regime on held-out): 2/6
- `error_direction_match` (sign(model_gap) == sign(human_gap)): 0/6
- `nondegenerate_accuracy` (|model_acc - human_acc| <= 0.05): 4/6

**Most common failure modes (not mutually exclusive):**
- missing model errors on held-out: 4/6
- wrong error-vs-correct RT direction given errors: 2/6
- degenerate accuracy relative to subject: 2/6
- insufficient predicted RT skew: 3/6

**Slow-error direction:** we define slow-error as `error_minus_correct_rt > 0`. The scorecard includes both the human and model slow-error booleans explicitly.