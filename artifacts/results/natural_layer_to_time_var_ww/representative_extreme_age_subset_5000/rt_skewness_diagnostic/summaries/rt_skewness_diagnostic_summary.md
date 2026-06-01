# RT Skewness Diagnostic Summary

## 1. Goal

This analysis tests whether the model reproduces the right-skewed shape of the human RT distribution, not just its mean or coarse spread.

- Trial-level RT fields used: analysis_group=`analysis_group`, human_rt=`true_rt`, model_rt=`pred_rt`, subject_id=`user_id`, trial_id=`row_index`.

## 2. Main skewness results

- Young human Fisher-Pearson skewness = 4.162; model = 0.846.
- Young human 1% / 5% trimmed Pearson skewness = 1.060 / 0.526.
- Young model 1% / 5% trimmed Pearson skewness = 0.143 / 0.081.
- Young bootstrap 95% CI: human [2.154, 6.377], model [0.635, 1.040].
- Older human Fisher-Pearson skewness = 6.855; model = 0.786.
- Older human 1% / 5% trimmed Pearson skewness = 1.057 / 0.593.
- Older model 1% / 5% trimmed Pearson skewness = 0.677 / 0.481.
- Older bootstrap 95% CI: human [4.173, 8.670], model [0.720, 0.852].

## 3. Skewness correlation

- Correlation was not computed from only two age-group summary points.
- Young bootstrap paired skewness correlation: Pearson r = 0.002, Spearman r = 0.004.
- Young subject-level correlation: Pearson r = 0.125, Spearman r = 0.441; warning = none.
- Older bootstrap paired skewness correlation: Pearson r = -0.013, Spearman r = -0.012.
- Older subject-level correlation: Pearson r = -0.477, Spearman r = -0.400; warning = unstable_n_lt_5.

## 4. Distribution-shape similarity

- Young density correlation: Pearson r = 0.963, Spearman r = 0.875, Wasserstein = 0.027, JS divergence = 0.032.
- Young quantile-profile correlation: Pearson r = 0.987, Spearman r = 1.000, RMSE = 0.026s.
- Older density correlation: Pearson r = 0.990, Spearman r = 0.769, Wasserstein = 0.043, JS divergence = 0.033.
- Older quantile-profile correlation: Pearson r = 0.996, Spearman r = 1.000, RMSE = 0.055s.

## 5. Condition-specific skewness

- Young human skewness is especially visible in incongruent trials (4.951) and error trials (2.935, if stable).
- Older human skewness is especially visible in incongruent trials (6.079) and error trials (0.696, if stable).

## 6. Interpretation

1. Human RT is right-skewed in both groups: yes.
2. Model RT is right-skewed in both groups: yes.
3. The model reproduces young right-skewness: poorly.
4. The model reproduces older right-skewness: poorly.
5. The model is not strongly over-skewed in the tail relative to humans.
6. Shape similarity can come from mean and spread alone, so skewness adds a stricter test of the tail.
7. Skewness can be mentioned in the abstract if it is paired with CI and a clear statement that the model only partially reproduced the right tail.

## 7. Recommended wording for abstract

可写为：‘在人类数据中，年轻组与老年组的反应时分布均呈明显右偏；模型也产生了右偏分布，但对右尾形态的复现并不完全。基于 trial-level bootstrap 的偏度估计显示，模型在分位数轮廓上与人类较为接近，但在偏度和尾部分布上仍存在系统差异。’
