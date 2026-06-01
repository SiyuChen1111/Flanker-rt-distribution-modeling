# README: best_model_R5_combined_best

## 1. 这个模型是什么
- 当前保留的主线模型是 `best_model_R5_combined_best`。
- 它对应的候选集合来自同一条 representative extreme-age 5000-trial 诊断路线中的 R0-R5 比较。
- 从现有参数文件可读到，它把组别特异的 `t0_mean + t0_sd` 和组别特异的 WW/readout 组合在一起。

## 2. 使用的数据是什么
- 数据路径：`artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/`。
- 最终保留的 trial 清单来自 `manifests/representative_subset_trial_manifest.csv`。
- 现有最终文件显示的年龄组是 `young_20_29` 和 `older_80_89`。
- 如果按用户最初口径写成 18-29 和 70-89，会和当前保留文件不一致，因此这里按实际文件记录。

## 3. 年轻组和老年组各用了多少 trial
- 年轻组 `young_20_29`: 5000 trial。
- 老年组 `older_80_89`: 5000 trial。
- 总 trial 数：10000。
- 独特刺激数：9990。

## 4. 核心架构
- 视觉证据来自 layer-to-time 路线，并进入 WW 决策/读出阶段。
- 最终被保留的组合包含：
  - 年轻组 `t0_mean = 0.55`, `t0_sd = 0.12`；
  - 老年组 `t0_mean = 0.75`, `t0_sd = 0.20`；
  - 组别特异 WW/readout 参数也被保留。
- 详细参数见 `results/best_model_parameter_estimates.csv`。

## 5. 为什么选择它作为当前 best model
- 在 `results/model_comparison_all_models.csv` 和 `model_selection_evidence/representative_model_comparison.csv` 中，R5 的综合分数最低，为 `0.403642`。
- 同一批候选里的其他分数分别为：R2 `0.433355`，R3 `0.571558`，R4 `0.661644`，R1 `0.663722`，R0 `1.369689`。
- 现有总结文件说明：R2 主要改善 RT 宽度，R3 主要改善 WW/readout 部分，R5 把这两部分合在一起后成为当前最优。

## 6. 主要结果摘要
- 年轻组 mean RT: human/model = `0.603013 / 0.611934`。
- 老年组 mean RT: human/model = `0.940945 / 0.919169`。
- 年轻组 RT spread(q90-q10): human/model = `0.2750 / 0.320900`。
- 老年组 RT spread(q90-q10): human/model = `0.5501 / 0.693274`。
- 年轻组 CAF RMSE = `0.125803`；老年组 CAF RMSE = `0.149298`。
- target recovery error-correct gap:
  - 年轻组：`0.158873`。
  - 老年组：`0.158863`。

## 7. 当前模型仍存在的不足
- 现有 summary 明确写到：这些结果是 model-development / diagnostic subset，不是 full-data final fitting。
- 模型 accuracy 仍高于人类，说明 correct/error 模式还没有完全对上。
- 年轻组和老年组的 target recovery 时间仍然很接近，不能据此说模型已经解释了更老组更晚恢复。
- 现有文档还指出 RT-error pattern 仍没有完全达到 human-like。

## 8. 如何复现主要结果和图
- 先保留并使用当前代表性 subset：`code/scripts/build_representative_extreme_age_subset.py`。
- 再运行拟合：`code/scripts/run_representative_extreme_age_subset_fitting.py`。
- 再生成主图：`code/scripts/make_representative_extreme_age_figures.py`。
- 如果需要补充诊断：
  - `code/scripts/run_representative_rt_skewness_diagnostic.py`
  - `code/scripts/run_representative_error_rt_coupling_diagnostic.py`
  - `code/scripts/run_representative_error_rt_length_decomposition.py`
  - `code/scripts/run_representative_readout_caf_mechanism_diagnostic.py`

## 9. 相关脚本路径
- `code/scripts/build_representative_extreme_age_subset.py`
- `code/scripts/build_representative_extreme_age_vgg_cache.py`
- `code/scripts/run_representative_extreme_age_subset_fitting.py`
- `code/scripts/make_representative_extreme_age_figures.py`
- `code/scripts/run_representative_rt_skewness_diagnostic.py`
- `code/scripts/run_representative_error_rt_coupling_diagnostic.py`
- `code/scripts/run_representative_error_rt_length_decomposition.py`
- `code/scripts/run_representative_readout_caf_mechanism_diagnostic.py`

## 10. not found in current files
- 没有发现把当前最终 subset 定义写成 18-29 / 70-89 的最终主线文件。
- 没有发现单独的 full-data final fitting 说明文件。
