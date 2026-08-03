# Current Results and Limitations

## Current result status

The repository now distinguishes two result layers:

- **Retained R5 baseline:** `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/`
- **Current exploratory choice-coupled result:** `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/`

Both use 5,000 representative trials for young adults and 5,000 for older adults. The sample contains 12 young participants and 4 older participants, with unequal trial contributions across participants. Neither result is a final full-cohort or held-out fit.

## Problems found and corrections made

1. **CAF and CRF coordinates:** older figures used quantile positions that obscured the actual time windows. Current figures use each bin's observed median RT and validate summaries against trial-level rows.
2. **No-crossing handling:** a channel with no threshold crossing could be returned at step zero by the Boolean first-crossing helper. It now receives the simulation deadline as a censoring sentinel and is interpreted only together with its crossing flag.
3. **Choice–RT mismatch:** the retained baseline derived RT from the first sustained crossing but choice from the largest state anywhere in the complete trajectory. The two rules disagree on 2,653 of 10,000 trials, all incongruent. Fresh fits now choose the winner at the same readout step used for RT; the old rule remains only for legacy reproduction.
4. **Late target evidence arrived too close to the deadline:** directly imposing the corrected choice rule exposed very poor incongruent accuracy. Compressing the existing VGG layer-to-time schedule gives Wong-Wang enough time to express the naturally present target recovery before readout.

## What the current exploratory model achieves

All 10,000 model trials cross the decision criterion. Choice is fixed at the sustained-crossing readout step.

| Group | Human / model accuracy | Human / model incongruent accuracy | Human / model mean RT | RT quantile MAE | Incongruent CAF RMSE |
|---|---:|---:|---:|---:|---:|
| Young 20–29 | 0.949 / 0.961 | 0.917 / 0.922 | 0.603 / 0.592 s | 0.061 s | 0.028 |
| Older 80–89 | 0.976 / 0.979 | 0.961 / 0.959 | 0.941 / 0.891 s | 0.038 s | 0.009 |

The current result also preserves a theoretically interpretable chain on incongruent trials:

- young: mean input reversal 0.077 s, mean WW-state reversal 0.139 s; target recovery precedes readout on 93.4% of correct trials and 0% of errors;
- older: mean input reversal 0.116 s, mean WW-state reversal 0.217 s; target recovery precedes readout on 99.9% of correct trials and 0% of errors.

This supports the claim that the existing VGG evidence can drive a choice-coupled decision after its timing is adjusted. It does not establish that people use the same mechanism.

## RT distribution and congruency checks

The model's combined RT distribution is mildly right-skewed, but much less so than the human distribution:

| Group | Human skew | Model skew |
|---|---:|---:|
| Young 20–29 | 4.162 | 0.283 |
| Older 80–89 | 6.855 | 0.163 |

Within congruent and incongruent conditions, model skew is close to zero. The modest overall right skew mainly comes from mixing two conditions with different locations. KDE plots show that the model's long tail remains too short.

The model also exaggerates the correct-trial incongruent-minus-congruent RT cost in every RT bin. Human delta values rise from 33 to 83 ms in young adults and 54 to 131 ms in older adults; model values are about 188–279 ms and 299–353 ms respectively. CAF shape is close, but matching CAF alone therefore does not mean the full RT structure is correct.

## Main limitations

- The age-specific schedule was selected from 171 candidates per group on the same representative trials used for evaluation; there are no held-out participants or stimuli.
- The older result is based on only four participants, so participant-level uncertainty is poorly estimated.
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
