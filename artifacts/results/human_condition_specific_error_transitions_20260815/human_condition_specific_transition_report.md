# Human condition-specific error-transition audit

## Scope and adjacency audit

This is a human-only audit using the frozen LIM preprocessing. It retained **3,143,319 true adjacent trial pairs** from 75 participants. Every pair is within the same participant and `nth_play` session and has original trial-index difference exactly 1. Session starts, cleaning gaps, and nonconsecutive pairs are excluded. C0v2 and all VGG/Wong–Wang or cognitive-model files were not read, modified, or refitted.

## Four primary transition effects

Absolute risks are participant-level empirical-Bayes estimates; intervals bootstrap participants. Positive point estimates are not individual significance declarations.

| Transition   |   After previous correct |   After previous error |   Risk difference | 95% interval     | Participants positive   | Raw-stable participants   |   Odds ratio |
|:-------------|-------------------------:|-----------------------:|------------------:|:-----------------|:------------------------|:--------------------------|-------------:|
| C→C          |                   0.0170 |                 0.0572 |            0.0401 | [0.0328, 0.0478] | 72/75 (96.0%)           | 72/75                     |       3.6941 |
| I→C          |                   0.0162 |                 0.0522 |            0.0360 | [0.0301, 0.0424] | 72/75 (96.0%)           | 75/75                     |       3.6712 |
| C→I          |                   0.0760 |                 0.1247 |            0.0486 | [0.0360, 0.0618] | 60/75 (80.0%)           | 72/75                     |       1.8197 |
| I→I          |                   0.0732 |                 0.1210 |            0.0477 | [0.0374, 0.0583] | 65/75 (86.7%)           | 75/75                     |       1.8427 |

## Interaction model

The participant-fixed-effect grouped-binomial model is likelihood-equivalent to the requested trial-level categorical logistic model. The previous-error × previous-condition coefficient is -0.4537, 95% CI [-0.5282, -0.3793]. The previous-error × current-condition coefficient is -1.0636, 95% CI [-1.1435, -0.9837]. The three-way coefficient is 0.4265, 95% CI [0.3332, 0.5198]. Thus previous and current congruency modulate the effect on the log-odds scale, although all four probability-scale effects remain positive and similar in absolute magnitude.

## Sensitivity controls

The pretrial sensitivity model controls previous RT, target repetition, previous target, previous response, and participant identity. A separate extended diagnostic additionally includes current RT percentile and response repetition; these can be downstream of or jointly determined with the current response, so they are not treated as clean pretrial controls. Age is not added because it is fixed within participant and therefore redundant with participant effects.

| sensitivity_set                | transition_label   |   risk_after_previous_correct |   risk_after_previous_error |   risk_difference | optimizer_converged   |
|:-------------------------------|:-------------------|------------------------------:|----------------------------:|------------------:|:----------------------|
| pretrial_controls              | C→C                |                        0.0169 |                      0.0690 |            0.0521 | True                  |
| pretrial_controls              | I→C                |                        0.0168 |                      0.0432 |            0.0264 | True                  |
| pretrial_controls              | C→I                |                        0.0779 |                      0.1123 |            0.0344 | True                  |
| pretrial_controls              | I→I                |                        0.0749 |                      0.1019 |            0.0271 | True                  |
| extended_downstream_diagnostic | C→C                |                        0.0167 |                      0.1156 |            0.0989 | True                  |
| extended_downstream_diagnostic | I→C                |                        0.0160 |                      0.0779 |            0.0619 | True                  |
| extended_downstream_diagnostic | C→I                |                        0.0775 |                      0.1533 |            0.0758 | True                  |
| extended_downstream_diagnostic | I→I                |                        0.0724 |                      0.1503 |            0.0779 | True                  |

All four pretrial-adjusted effects remain positive; the largest absolute change from the primary shrunk effect is 0.0207. The extended diagnostic is reported transparently but does not replace the primary estimand.

## Lag decay

|    lag |   population_mean_risk_difference |   risk_difference_ci_low |   risk_difference_ci_high |   n_valid_pairs |
|-------:|----------------------------------:|-------------------------:|--------------------------:|----------------:|
| 1.0000 |                            0.0474 |                   0.0388 |                    0.0563 |    3143319.0000 |
| 2.0000 |                            0.0301 |                   0.0245 |                    0.0360 |    3079703.0000 |
| 3.0000 |                            0.0263 |                   0.0216 |                    0.0313 |    3017116.0000 |
| 4.0000 |                            0.0250 |                   0.0209 |                    0.0295 |    2955436.0000 |
| 5.0000 |                            0.0132 |                   0.0088 |                    0.0179 |    2894600.0000 |

For the overall series, the descriptive exponential fit gives A=0.0453, τ=3.70 trials, R²=0.913. τ is only a behavioral history timescale, not a neural time constant. Strict monotonic decrease was True.

## Blocked held-out prediction

| model                              |   log_loss |   delta_log_loss_vs_m0 |   brier_score |   delta_brier_vs_m0 |   expected_calibration_error |
|:-----------------------------------|-----------:|-----------------------:|--------------:|--------------------:|-----------------------------:|
| M0_participant_current_condition   |  0.1771551 |              0.0000000 |     0.0443526 |           0.0000000 |                    0.0033826 |
| M1_plus_generic_previous_error     |  0.1766356 |             -0.0005194 |     0.0443218 |          -0.0000307 |                    0.0037160 |
| M2_plus_previous_condition_history |  0.1766133 |             -0.0005417 |     0.0443203 |          -0.0000322 |                    0.0033999 |
| M3_full_three_way_interaction      |  0.1764111 |             -0.0007440 |     0.0442920 |          -0.0000606 |                    0.0031366 |

The five folds hold out contiguous temporal blocks within every participant. Improvements are absolute per-trial probability-score changes; p-values are not used as the main evidence.

## Null models

The stable participant/current-condition null and the generic previous-error null each used 300 fixed-seed grouped-binomial simulations. Under the generic-history null, the largest standardized mismatch is C_to_C: observed 0.0401 versus null mean 0.0096, z=32.52. This asks whether one condition-invariant history effect can reproduce the four-cell matrix.

## Classification and interpretation

**T1 — GENERAL ERROR-PRONE STATE.** All four transitions are positive; magnitudes vary, with a stronger effect after congruent errors, but no transition is isolated or reversed.

The most plausible next mechanistic hypothesis is therefore a general, short-lived error-prone state whose strength is modulated by trial condition, especially the condition of the preceding error. This is a hypothesis for a later model-comparison task, not a mechanism implemented here.

## What this does not establish

- The associations do not establish that an error causally creates the later state; an unmeasured state may produce both errors.
- The decay parameter is descriptive and is not a neural time constant.
- Positive participant estimates do not imply individually significant effects.
- Sensitivity adjustment cannot eliminate every sequential confound and current RT may be downstream.
- No incomplete reset, state carryover, starting-state variability, lapse, sensory-noise, or history-dependent cognitive parameter was created.

## Integrity confirmation

**C0v2 was not read, modified, or refitted. No cognitive model was created or tuned.**
