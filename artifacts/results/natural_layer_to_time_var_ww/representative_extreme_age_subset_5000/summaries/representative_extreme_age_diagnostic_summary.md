# Representative Extreme-age Subset Diagnostic Summary

## 1. Goal

This analysis replaces the forced-balanced correct/error subset with a representative 5,000-trial-per-group diagnostic subset. The goal is to preserve observed human accuracy and RT distributions before testing model mechanisms. All results are representative subset diagnostic / exploratory, not full age-group conclusions.

## 2. Sampling strategy

Young group: young_20_29. Older group: older_80_89. Used preferred older group 80-89. Sampling preserved subject, congruency, correctness, and age-group-specific RT quantile-bin proportions instead of forcing equal correct/error counts.

## 3. Representativeness audit

- older_80_89: accuracy diff 0.0005, mean RT diff 0.0034s, median RT diff 0.0010s, q90-q10 diff 0.0009s, q95-median diff 0.0120s, KS 0.0068, subject coverage 1.00.
- young_20_29: accuracy diff 0.0006, mean RT diff 0.0016s, median RT diff 0.0000s, q90-q10 diff 0.0010s, q95-median diff 0.0030s, KS 0.0046, subject coverage 1.00.

Representativeness gate passed: True.

## 4. Evidence cache

Unique stimuli: 9990. Evidence coverage: 1.0. Failed stimuli: 0. Evidence is cached once per stimulus, not by subject or age group.

## 5. Model comparison

Composite score ranking:

- R5_combined_best: total 0.4036, RT quantile 0.0316, CAF 0.1376, accuracy 0.1342.
- R2_group_t0_mean_sd: total 0.4334, RT quantile 0.0471, CAF 0.1366, accuracy 0.1342.
- R3_group_ww_readout: total 0.5716, RT quantile 0.0927, CAF 0.1680, accuracy 0.1342.
- R4_variational_noise: total 0.6616, RT quantile 0.1413, CAF 0.1623, accuracy 0.1355.
- R1_group_t0_mean: total 0.6637, RT quantile 0.1418, CAF 0.1640, accuracy 0.1342.
- R0_fixed_current: total 1.3697, RT quantile 0.4947, CAF 0.1640, accuracy 0.1342.

Best model: R5_combined_best.

## 6. t0 findings

- older_80_89: best t0_mean 0.75s, t0_sd 0.20s.
- young_20_29: best t0_mean 0.55s, t0_sd 0.12s.

Group-specific t0_mean improved RT location relative to R0, but t0_mean alone did not repair RT spread. R2 improved spread by adding t0 variability; R5 retained this contribution.

## 7. RT variability findings

Older 80-89 required larger t0_sd (0.20s) than young 20-29 (0.12s) in the best combined model. WW/readout parameters improved the combined score when paired with t0 variability. The best standalone variational model selected conflict-dependent sigma, but it did not outperform t0 variability or the combined t0_sd + WW/readout candidate for this subset.

## 8. Behavior fit

- older_80_89: human/model mean RT 0.941/0.919s, median RT 0.879/0.886s, q90-q10 0.550/0.693s, CAF RMSE 0.149.
- young_20_29: human/model mean RT 0.603/0.612s, median RT 0.580/0.605s, q90-q10 0.275/0.321s, CAF RMSE 0.126.

The best model improves RT distribution shape but still underfits human choice/error rates because model accuracy remains higher than human accuracy.

## 9. WW internal dynamics

- older_80_89: target recovery correct 0.472s, error 0.631s, error-correct gap 0.159s, late target recovery 0.971.
- young_20_29: target recovery correct 0.469s, error 0.628s, error-correct gap 0.159s, late target recovery 0.966.

Incongruent errors retain later target recovery than incongruent correct trials. Older and young groups show very similar recovery timing in the best model, so this subset does not support a strong model-internal older-specific recovery delay yet.

## 10. Interpretation

The model can express age-related timing differences through decision-layer parameters, especially t0 variability plus WW/readout. The current fitting still does not fully express age-related choice/error differences, so these results should be treated as mechanism diagnostics rather than age-group conclusions.

## 11. Recommended next steps

- Extend to all age groups with representative 5,000-trial subsets only after keeping the same gate.
- Keep t0 variability in the candidate set.
- Keep WW/readout optimization, but refine the score to penalize over-high model accuracy.
- Do not prioritize variational evidence noise alone unless paired with readout changes.
- Use the current figures for discussion, not as final full-cohort evidence.

## 12. Short Chinese Summary for Discussion

APA visualization skills loaded: True. 这次分析没有再强行平衡正确和错误试次，而是从年轻组和80-89岁组各抽取5000个更接近真实行为分布的试次。抽样审计通过，说明子集基本保留了原始组的正确率和反应时分布。VGG证据按刺激缓存，9990个独立刺激全部成功。模型比较中，最佳结果来自“t0均值+t0波动+WW/readout”的组合。t0均值主要修正反应时位置，t0波动明显帮助反应时分布宽度；WW/readout进一步改善整体拟合。内部轨迹仍显示错误试次的目标恢复更晚，但当前最佳模型中高龄组并没有明显比年轻组更晚恢复。因此，这是一份有用的机制诊断，但还不能作为完整年龄组结论。
