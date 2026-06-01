# Error-RT Coupling Diagnostic Summary

## 1. Goal

This analysis tests whether the model makes errors in a human-like way, not just whether it can make errors.

## 2. Error count and sample size

- The smallest human error count in any group-by-congruency-by-RT-bin cell is 0.
- Human cells with fewer than 10 errors: 8.
- Sparse bins matter most for the older group and for slow congruent trials, so local uncertainty is wider there.
- If mismatch appears in well-populated bins too, sample size alone cannot explain it.

## 3. Error RT pattern

- Human young all-trial error-minus-correct mean RT: -0.039s, 95% CI [-0.053, -0.022].
- Model young all-trial error-minus-correct mean RT: 0.001s, 95% CI [-0.009, 0.010].
- Human older all-trial error-minus-correct mean RT: -0.200s, 95% CI [-0.244, -0.152].
- Negative values mean errors are faster than correct responses.

## 4. P(error | RT bin)

- Human all-trial error-rate slope: -0.015 per RT bin.
- Model all-trial error-rate slope: 0.005 per RT bin.
- Human-model CAF RMSE: 0.138.
- Fast-bin error gap (model minus human): 0.080; slow-bin gap: 0.168.
- Matching negative slopes would mean both systems make more errors in faster bins.

## 5. Trial/stimulus-level correspondence

- Trial-level Jaccard overlap: young 0.071, older 0.031.
- Stimulus-level Pearson correlation: young 0.073, older 0.028.
- Condition/RT-bin Pearson correlation: young 0.778, older 0.402.
- High aggregate error without overlap would imply the model is failing on different trials than humans.

## 6. Readout-state explanation

- Delta_s_at_readout predicts model correctness with coefficient 13.557 and odds ratio 772106.086 (p = 6.19e-60).
- Delta_s_at_readout predicts human correctness with coefficient -1.679 and odds ratio 0.187 (p = 0.145).
- Among model fast errors, the incongruent share is 1.000.
- Mean delta_s_at_readout for fast model errors: -0.006; for slow model correct trials: 0.011.
- Share of fast model errors that occur before target recovery is high: 0.833.

## 7. Epoch or parameter sensitivity

- Existing comparison covers 6 already-fitted model variants.
- Accuracy score range across those variants is 0.134 to 0.136.
- Checkpoint note: No trainable epoch checkpoint found; ACC mismatch is more likely related to WW/readout/noise parameters than training epoch.
- If many parameterized variants move RT/CAF metrics while leaving accuracy mismatch almost unchanged, the bottleneck is more likely the readout-choice coupling than training length.

## 8. Interpretation

- The model can generate errors.
- The key question is whether those errors are concentrated in the same fast, conflict-heavy, early-readout regime as human errors.
- Agreement in error speed without agreement in which trials are hard would still be only a partial match.

## 9. Recommended next steps

- If delta_s_at_readout strongly predicts model correctness but not human correctness, revise the readout-time choice rule first.
- If both systems show fast-error coupling but the model slope is too flat or reversed, add P(error|RT) directly into the fitting objective.
- If accuracy barely changes across existing model variants, do not prioritize longer training.
- If model errors remain too late or too target-favoring, next targets are threshold, noise, and readout coupling rather than sample size expansion.

## 10. Short Chinese Summary for Discussion

这次检查的重点不是模型会不会出错，而是模型是不是像人一样在快反应、冲突更强、较早读出的试次上更容易出错。如果人类表现出“错误反应更快”，但模型没有，或者模型犯错的试次和人类困难试次对不上，那就说明问题更像是读出规则和内部选择耦合不对，而不是单纯训练不够久。若不同现有模型之间准确率差不多、但 RT 和 CAF 指标会动，也更支持这个判断。
