# Current Results and Limitations

## Current result status

The repository now distinguishes the retained two-group diagnostics from the audited seven-group update:

- **Retained R5 baseline:** `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/`
- **Earlier two-group choice-coupled result:** `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/`
- **Audited all-age update:** `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/`

The all-age update covers `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70-79`, and `80-89`: 75 participants and 5,000 selected trials per group. It is a same-data representative-subset diagnostic, not a final full-cohort, held-out, or hierarchical fit.

## Problems found and corrections made

1. **CAF and CRF coordinates:** older figures used quantile positions that obscured the actual time windows. Current figures use each bin's observed median RT and validate summaries against trial-level rows.
2. **No-crossing handling:** a channel with no threshold crossing could be returned at step zero by the Boolean first-crossing helper. It now receives the simulation deadline as a censoring sentinel and is interpreted only together with its crossing flag.
3. **Choice–RT mismatch:** the retained baseline derived RT from the first sustained crossing but choice from the largest state anywhere in the complete trajectory. The two rules disagree on 2,653 of 10,000 trials, all incongruent. Fresh fits now choose the winner at the same readout step used for RT; the old rule remains only for legacy reproduction.
4. **Late target evidence arrived too close to the deadline:** directly imposing the corrected choice rule exposed very poor incongruent accuracy. Compressing the existing VGG layer-to-time schedule gives Wong-Wang enough time to express the naturally present target recovery before readout.

## What the all-age update achieves

The shared decision-time scale is `0.27`; age-specific `t0` values were then selected for the seven groups. Choice is fixed at the sustained-crossing readout step. The mean absolute condition RT error falls from 95.6 ms to 3.2 ms.

| Group | Participants | Human / model accuracy | Human / model mean RT | Condition RT MAE | CAF RMSE |
|---|---:|---:|---:|---:|---:|
| 20–29 | 12 | 0.949 / 0.961 | 0.603 / 0.595 s | 7.8 ms | 0.018 |
| 30–39 | 7 | 0.946 / 0.955 | 0.646 / 0.645 s | 1.9 ms | 0.027 |
| 40–49 | 4 | 0.947 / 0.964 | 0.629 / 0.624 s | 4.1 ms | 0.022 |
| 50–59 | 11 | 0.965 / 0.978 | 0.664 / 0.662 s | 1.9 ms | 0.017 |
| 60–69 | 21 | 0.950 / 0.963 | 0.717 / 0.715 s | 2.2 ms | 0.026 |
| 70–79 | 16 | 0.944 / 0.966 | 0.785 / 0.784 s | 3.5 ms | 0.032 |
| 80–89 | 4 | 0.976 / 0.979 | 0.941 / 0.940 s | 0.9 ms | 0.018 |

The model has 35,000 trial-level rows. Choice/readout consistency is 1.0; 34,999 model RTs cross the criterion, while one 70–79 trial is recorded as no-crossing and excluded from model RT summaries.

The update preserves the theoretically interpretable early-flanker/later-target chain on the representative trials. This supports computational sufficiency of the existing VGG evidence after timing adjustment; it does not establish that people use the same mechanism.

The previous two-group diagnostic also preserves a more detailed chain on incongruent trials:

- young: mean input reversal 0.077 s, mean WW-state reversal 0.139 s; target recovery precedes readout on 93.4% of correct trials and 0% of errors;
- older: mean input reversal 0.116 s, mean WW-state reversal 0.217 s; target recovery precedes readout on 99.9% of correct trials and 0% of errors.

## RT distribution and congruency checks

The model's combined RT distribution is mildly right-skewed, but much less so than the human distribution:

| Group | Human skew | Model skew |
|---|---:|---:|
| 20–29 | 4.162 | 0.283 |
| 30–39 | 2.712 | 0.277 |
| 40–49 | 2.771 | 0.471 |
| 50–59 | 1.599 | 0.428 |
| 60–69 | 1.806 | 0.278 |
| 70–79 | 2.930 | 0.281 |
| 80–89 | 6.855 | 0.163 |

Within congruent and incongruent conditions, model skew is close to zero. The modest overall right skew mainly comes from mixing two conditions with different locations. KDE plots show that the model's long tail remains too short.

The model also exaggerates the correct-trial incongruent-minus-congruent RT cost in every RT bin. Human delta values rise from 33 to 83 ms in young adults and 54 to 131 ms in older adults; model values are about 188–279 ms and 299–353 ms respectively. CAF shape is close, but matching CAF alone therefore does not mean the full RT structure is correct.

## Main limitations

- The shared decision-time scale and age-specific `t0` values were selected on the same representative trials used for evaluation; there are no held-out participants or stimuli.
- The older result is based on only four participants, so participant-level uncertainty is poorly estimated.
- The middle age groups have unequal participant counts, and 70–79 has one no-crossing model trial.
- Trial contributions are unequal across participants, and the optimization is trial-weighted rather than a full hierarchical participant fit.
- The model has no congruent errors, whereas humans do. This limits error-RT and response-category interpretation.
- Human-like long RT tails and within-condition right skew are not reproduced.
- The congruency RT cost is much too large despite good incongruent CAF values.
- Age-specific timing improves the score relative to a shared schedule, but may reflect overfitting rather than a stable age mechanism.
- Non-decision-time variability still contributes materially to the final RT distribution.
- Controlled Wong-Wang checks pass only minimal sanity tests; they do not prove that the accumulator or the full behavioral model works perfectly.
- Historical reports under `artifacts/results/` may describe superseded readout rules. Use `artifacts/results/ARTIFACT_DOCS_INDEX.md` before treating them as current evidence.

## Interpretation guardrails and next step

Do not treat normal RT as the target: first-passage RT is bounded below and often right-skewed. Also do not infer a first-passage distribution from the distribution of a fixed-time state `S(t)`.

The next decisive test is out-of-sample rather than another in-sample schedule adjustment: freeze the choice-coupled rule and schedule-selection procedure, fit on training participants or stimuli, and test accuracy, crossing coverage, CAF, KDE/tail shape, delta curves, and four-direction response errors on held-out data. A participant-balanced or hierarchical objective is needed before making age-group claims.
