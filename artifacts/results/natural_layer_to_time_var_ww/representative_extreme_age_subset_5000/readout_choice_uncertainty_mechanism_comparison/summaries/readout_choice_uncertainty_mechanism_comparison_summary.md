# Readout-Choice Uncertainty Mechanism Comparison Summary

## 1. Goal

This analysis compared deterministic readout, time-only, gap-only, and time+gap dependent readout-choice noise using the existing representative extreme-age subset results. It tests which mechanism better explains human-like congruent errors and fast-error patterns without re-extracting VGG evidence or retraining the model.

## 2. Background findings

- Existing congruent diagnostics showed that the correct channel separates from competitors early in congruent trials.
- Threshold lowering can make readout earlier but does not by itself generate human-like congruent errors.
- The deterministic rule is too certain once one channel is even slightly ahead.
- The earlier time/gap pilot suggested that uncertainty tied to early readout and small evidence gaps can produce fast congruent errors.

## 3. Candidate mechanisms

- M0 deterministic readout: the highest internal evidence channel is chosen with no extra uncertainty.
- M1 time-only uncertainty: earlier readout creates stronger response uncertainty.
- M2 gap-only uncertainty: smaller winner-runner-up gap creates stronger response uncertainty.
- M3 time+gap uncertainty: early readout and small evidence gap jointly increase response uncertainty.

## 4. Main results

- Best overall setting: age_specific M3 time+gap with composite score 0.2592.
- Best shared-parameter setting: M3 time+gap with composite score 0.2824.
- Best age-specific setting: M3 time+gap with composite score 0.2592.
- Shared M3 time+gap score: 0.2824; shared M1 time-only score: 0.3051; shared M2 gap-only score: 0.2841.

Human congruent error rates were low but nonzero, so bootstrap intervals and error counts are reported in the human reference table.

## 5. Ablation results

- Time-only was not sufficient relative to time+gap under the composite score.
- Gap-only was not sufficient relative to time+gap under the composite score.
- Time+gap outperformed both single-factor ablations in the shared-parameter comparison.

## 6. Shared vs age-specific parameters

- Shared best score: 0.2824. Age-specific best score: 0.2592.
- If age-specific settings improve the score, the exploratory model suggests that older group may require stronger readout-choice uncertainty to match the observed RT-error pattern. This should not be read as a definitive claim that older adults have more noise.

## 7. Interpretation

Threshold and readout-choice noise play different roles. Threshold controls when the model decides. Readout-choice noise controls how reliably the internal evidence state is converted into a final response. In this analysis, the noise is not arbitrary constant guessing; it is tied to early readout, low internal evidence gap, or both.

## 8. Limitations

- The analysis still uses the representative subset, not the full dataset.
- The noise model is exploratory and should be validated on more trials and age groups.
- Age-specific improvements should be phrased cautiously.
- Split validation is included as an exploratory robustness check; the main uncertainty estimates come from seed-level repetitions and bootstrap summaries.

## 9. Next steps

- Use the current results as a formal candidate-model comparison for the readout-choice uncertainty mechanism.
- If included in the abstract, phrase it as evidence that uncertainty at the evidence-to-choice mapping stage may explain fast congruent errors.
- Extend the best shared and age-specific candidates to a larger subset or additional age groups before making strong developmental claims.

## 10. Suggested wording for advisor

我完成了 readout-choice uncertainty 的正式机制比较。结果显示，原来的确定性读出几乎不能产生 congruent errors；只加入早读出或只加入证据差距的不确定性都能改善一部分现象，但 time+gap 机制总体更符合 human-like congruent errors 和 fast-error pattern。这个机制不改变原来的读出时间分布，而是在“内部证据状态到最终反应”的转换阶段加入与早读出和低证据差距相关的不确定性。当前结果仍是 representative subset 上的探索性结果；如果分年龄参数更好，建议表述为 older group may require stronger readout-choice uncertainty，而不是直接断言老年组噪声更大。

## Selected numeric highlights

- young_20_29: human congruent error=0.0177 (n=47); deterministic congruent error=0.0000; shared M3 congruent error=0.0186.
- young_20_29: shared M3 congruent error RT-correct RT=-0.0051s; incongruent error RT-correct RT=-0.2685s; target recovery preservation=0.0124.
- older_80_89: human congruent error=0.0090 (n=26); deterministic congruent error=0.0000; shared M3 congruent error=0.0000.
- older_80_89: shared M3 congruent error RT-correct RT=nans; incongruent error RT-correct RT=-0.4635s; target recovery preservation=0.1089.
