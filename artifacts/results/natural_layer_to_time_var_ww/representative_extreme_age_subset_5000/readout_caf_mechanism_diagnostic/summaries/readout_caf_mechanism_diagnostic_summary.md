# Readout-CAF Mechanism Diagnostic Summary

## 1. Goal

This analysis stays within the existing representative extreme-age subset and asks why RT marginal distributions fit reasonably well while CAF and correct/error RT structure still mismatch.

## 2. Current best model audit

- Best model: `R5_combined_best`
- Best score: `0.403642`
- Young parameters: t0_mean=`0.55`, t0_sd=`0.12`, threshold=`0.12`, margin=`0.00`
- Older parameters: t0_mean=`0.75`, t0_sd=`0.20`, threshold=`0.14`, margin=`0.02`

## 3. Do group-specific parameters affect WW dynamics?

- young_20_29: trajectory delta=`0.000000`, crossing change=`0.1037`s, recovery change=`0.0000`s, flanker-dominance change=`0.0358`s, WW dynamics changed=`False`
- older_80_89: trajectory delta=`0.000000`, crossing change=`-0.1095`s, recovery change=`0.0000`s, flanker-dominance change=`-0.0373`s, WW dynamics changed=`False`

## 4. Does readout-time evidence predict correctness?

- Correct trials have mean delta_s_at_readout `-0.0060`; error trials have `-0.0388`.
- Fast-bin delta_s_at_readout is `-0.0024`; slow-bin delta_s_at_readout is `0.0001`.
- Readout-time choice and final choice mismatch on `0.271` of trials.

## 5. Is choice coupled to readout time?

- Current rule CAF RMSE: `0.1360`
- Readout-time argmax CAF RMSE: `0.3848`
- Current rule incongruent error-minus-correct RT: `-0.0721`
- Readout-time argmax incongruent error-minus-correct RT: `-0.5119`

## 6. CAF decomposition

- Current all-trial incongruent proportions by RT bin: `[0.357, 0.408, 0.459, 0.548, 0.719]`
- Current all-trial delta_s_at_readout by RT bin: `[-0.0024, -0.0057, -0.0195, -0.0307, 0.0001]`

## 7. RT spread mechanism attribution

- Best RT quantile score model: `R5_combined_best` with `0.0316`
- R2 t0_mean+t0_sd RT quantile score: `0.0471`
- R3 WW/readout RT quantile score: `0.0927`
- R4 variational sigma RT quantile score: `0.1413`

## 8. Psychological/neuroscientific interpretation

- t0_mean: interpretable as non-decision time shift.
- t0_sd: interpretable as non-decision-time variability and likely the main driver of RT spread here.
- threshold: can reflect readout/commit criterion, but current evidence suggests weak leverage on age-specific internal separation.
- sustained_k / margin: interpret as readout stability constraints rather than strong latent neural differences.
- evidence_gain: controls effective input scaling into WW competition; some mechanism value, but currently mostly calibration.
- variational sigma: broadens RT distribution more than it fixes CAF, so its mechanism value remains limited.
- WW S trajectory: still mechanistically useful because target recovery and flanker dominance patterns are visible, but group separation is weak.

## 9. What scientific questions can be answered now?

- Already answerable: the model can fit RT marginal shape mainly through timing terms while preserving a target-recovery mechanism.
- Not yet answerable: whether age-group WW internal dynamics meaningfully differ in the current setup.
- Conference abstract readiness: yes for a cautious mechanism-diagnostic abstract, not for a strong age-mechanism claim.
- Avoid overclaim: describe this as evidence that RT fit and CAF fit dissociate because choice/readout coupling and conflict dynamics remain mismatched.

## 10. Recommended next steps

- Modify the choice/readout rule before expanding to more age groups.
- Keep variational sigma only if it helps after the choice/readout coupling is fixed.
- Treat CAF as a core objective, not a secondary figure.
- Do not expand to more age bins until the readout-choice mechanism is clarified.
- A draft abstract can start now if framed as a focused mechanism diagnosis.

## 11. Short Chinese Summary for Discussion

APA visualization skills loaded: True

这次分析没有扩大样本，也没有恢复完整提取，而是直接用现有 representative subset 检查为什么模型能拟合 RT 分布，却拟合不好 CAF 和 correct/error RT。结果显示：年轻组和高龄组的参数确实不同，也确实会进入 WW，但它们对内部轨迹的影响很小；更大的年龄差主要还是来自 t0_mean 和 t0_sd。readout 时刻的 target-minus-flanker 差值和正确率有关，但 slow bin 并没有稳定变得更偏向 target，因此 CAF 斜率没有自然形成。更关键的是，当前 choice 不是在 readout 时刻直接决定，而是沿用整段轨迹后的选择，所以 RT 和 choice 有一定解耦。这可以解释为什么 RT marginal distribution 能拟合，但 p(correct | RT) 还是不对。下一步最应该做的是修改 readout/choice coupling，并把 CAF 直接放进优化目标，再决定是否保留 variational sigma 或扩展更多年龄段。
