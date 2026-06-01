# 当前模型流程总结：从训练数据到结果结论

本文档总结当前代表性极端年龄组诊断分析中使用的模型流程。重点是说明：数据从哪里来，刺激如何进入模型，模型如何把图像证据变成反应和反应时，如何测试，以及当前结果说明了什么。

重要说明：本结果是 representative subset diagnostic / exploratory，不是完整年龄组结论，也不是全量年龄组拟合。

## 1. 研究目标

当前目标是比较年轻组和高龄组在 Flanker/LIM 任务中的行为差异，并测试模型是否能用决策层机制解释这些差异。这里没有继续做全量提取，也没有使用强行平衡 correct/error 的子集；这次使用的是更接近真实行为分布的代表性子集。

本轮重点问题包括：

- 年轻组和高龄组的反应时分布是否能被模型复现。
- t0，即非决策时间，是否能解释平均反应时差异。
- t0 的波动是否能解释反应时分布宽度。
- Wong-Wang 决策读出参数是否能改善 CAF 和 error pattern。
- 模型内部 target/flanker 状态轨迹是否保留“错误试次目标恢复更晚”的机制。

## 2. 训练数据和行为数据

原始行为数据来自 `data/vam_data/` 下的被试 trial 文件，以及 `data/vam_data/metadata.csv` 中的年龄组信息。每个 trial 包含目标方向、干扰方向、被试反应、反应时、刺激布局和目标位置。

本次诊断使用两个年龄段：

- young_20_29：20-29 岁。
- older_80_89: 使用优先高龄组 80-89 岁。

代表性抽样结果：

- young_20_29：5000 trials。
- older_80_89：5000 trials。
- unique stimuli：9990。

抽样不是把 correct/error 强行拉平，而是尽量保留原始年龄组的正确率、反应时分布、一致/不一致比例、RT 分位数和被试覆盖。审计通过后才继续后续模型步骤。

代表性审计摘要：

- older_80_89: accuracy diff = 0.0005; mean RT diff = 0.0034s; median RT diff = 0.0010s; RT KS distance = 0.0068; subject coverage = 1.00。
- young_20_29: accuracy diff = 0.0006; mean RT diff = 0.0016s; median RT diff = 0.0000s; RT KS distance = 0.0046; subject coverage = 1.00。

## 3. 模型接受的实验刺激

模型输入不是抽象表格，而是根据每个 trial 的刺激信息重建出的图像刺激。每个刺激由以下信息决定：

- 背景图：`code/vam/bkgrnd.png`。
- 四个方向的目标/干扰图像：`code/vam/bird0.png` 到 `bird3.png`。
- 目标方向：L/R/U/D。
- 干扰方向：L/R/U/D。
- 目标位置：`xpos`, `ypos`。
- 刺激布局：`stimulus_layout`，决定干扰刺激围绕目标如何排列。

同一个 `global_stimulus_key` 只提取一次视觉证据。因此 VGG evidence 是 stimulus-level cache，不随年龄组或被试改变。年龄差异只允许出现在后面的决策层，例如 t0、t0 波动、WW/readout 参数等。

本次证据缓存结果：

- evidence coverage = 1.0。
- failed stimuli = 0。
- 提取层级包括 conv3、conv4、conv5、pooled 和 final evidence。

## 4. 模型架构

当前模型可以分成三个部分：视觉编码、时间化证据、决策读出。

### 4.1 视觉编码

视觉编码部分基于 VGG16。图像进入 VGG 后，模型在多个层级提取证据，包括中层和高层视觉特征。每一层会产生对四个反应方向 L/R/U/D 的证据。

这里的关键点是：视觉证据只由刺激图像决定，不由年龄组决定。也就是说，同一张刺激图给 young 和 older 的视觉 evidence 是同一份。

### 4.2 自然 layer-to-time evidence

模型不会只用最后一层输出，而是把不同 VGG 层的 evidence 放到不同时间阶段中。较早的视觉层影响早期证据，较晚的视觉层影响后期证据。这样可以让模型表达早期干扰捕获和后期目标恢复。

### 4.3 Wong-Wang 决策模块

决策模块是一个四选一竞争模型，对应四个反应方向。每个方向都有一个内部状态 S。模型随时间积累证据，某个方向的状态达到读出条件后产生模型反应和决策时间。

本次最佳模型保留的关键决策层参数包括：

- older_80_89: t0_mean = 0.75s, t0_sd = 0.20s。
- young_20_29: t0_mean = 0.55s, t0_sd = 0.12s。
另外，最佳模型使用 group-specific WW/readout 参数。年轻组和高龄组使用相同 evidence_gain，但高龄组使用更高 threshold 和更严格 margin，用来表达较慢或更谨慎的读出。

## 5. 训练和拟合流程

当前流程不是一个简单的端到端一次性训练，而是分阶段完成：

1. 读取 human trial 数据和年龄组 metadata。
2. 构建 representative 5000-trial young-vs-older subset。
3. 审计 subset 是否保留原始年龄组行为分布。只有 gate 通过才继续。
4. 对 subset 中的 unique stimuli 提取 stimulus-level VGG/layerwise evidence cache。
5. evidence coverage 必须等于 1.0，失败刺激必须为 0，才进入拟合。
6. 在固定的有限模型集合中比较 R0-R5，不做无穷扫参。
7. 用综合指标选择最佳模型：RT 分位数、CAF、accuracy、机制保留等。
8. 对最佳模型生成 trial-level predictions、行为图、WW 内部轨迹图和 summary。

比较的候选模型：

- R0：固定当前模型，不做组特异参数。
- R1：只允许每个年龄组有不同 t0_mean。
- R2：允许每个年龄组有不同 t0_mean 和 t0_sd。
- R3：允许每个年龄组有不同 WW/readout 参数。
- R4：测试 variational evidence noise。
- R5：只组合前面有贡献的机制，不做全组合爆炸搜索。

## 6. 测试和评分方式

模型不是只看 accuracy，也不是只看 mean RT。评分包括：

- human vs model accuracy。
- human choice agreement。
- incongruent error rate。
- mean RT 和 median RT。
- RT quantiles: q10, q25, q50, q75, q90, q95。
- RT spread: q90-q10。
- right tail: q95-median。
- CAF bin-wise RMSE。
- correct vs error RT。
- incongruent correct vs incongruent error RT。
- target recovery time 和 WW 内部 S trajectory。

t0 明确进入 predicted RT 和模型评分，不是画图时事后平移。

## 7. 当前结果

模型比较结果：

- R5_combined_best: total score = 0.4036; RT quantile score = 0.0316; CAF score = 0.1376; accuracy score = 0.1342。
- R2_group_t0_mean_sd: total score = 0.4334; RT quantile score = 0.0471; CAF score = 0.1366; accuracy score = 0.1342。
- R3_group_ww_readout: total score = 0.5716; RT quantile score = 0.0927; CAF score = 0.1680; accuracy score = 0.1342。
- R4_variational_noise: total score = 0.6616; RT quantile score = 0.1413; CAF score = 0.1623; accuracy score = 0.1355。
- R1_group_t0_mean: total score = 0.6637; RT quantile score = 0.1418; CAF score = 0.1640; accuracy score = 0.1342。
- R0_fixed_current: total score = 1.3697; RT quantile score = 0.4947; CAF score = 0.1640; accuracy score = 0.1342。

最佳模型是 `R5_combined_best`。它的核心形式是：t0_mean + t0_sd + group-specific WW/readout。

最佳模型行为结果：

- older_80_89: human/model mean RT = 0.941/0.919s；human/model median RT = 0.879/0.886s；human/model q90-q10 = 0.550/0.693s；CAF RMSE = 0.149；human/model accuracy = 0.976/0.830。
- young_20_29: human/model mean RT = 0.603/0.612s；human/model median RT = 0.580/0.605s；human/model q90-q10 = 0.275/0.321s；CAF RMSE = 0.126；human/model accuracy = 0.949/0.827。

机制结果：

- older_80_89: target recovery correct/error = 0.472/0.631s；error-correct gap = 0.159s；late target recovery = 0.971。
- young_20_29: target recovery correct/error = 0.469/0.628s；error-correct gap = 0.159s；late target recovery = 0.966。

## 8. 主要结论

1. 从 forced-balanced subset 改为 representative subset 是必要的。新的 subset 更好保留了人类真实 accuracy 和 RT 分布。
2. t0_mean 能修正平均反应时位置，但不能单独解释反应时分布宽度。
3. t0_sd 对 RT spread 有明显帮助。最佳模型中 older 的 t0_sd 大于 young，说明高龄组需要更大的非决策时间波动。
4. WW/readout 参数和 t0 variability 结合后得到最佳模型，说明年龄差异更可能出现在决策读出层，而不是视觉 evidence 层。
5. variational evidence noise 单独没有超过 t0_sd 或 t0_sd + WW/readout 的组合。
6. 模型内部机制仍保留：incongruent error trials 的 target recovery 晚于 incongruent correct trials。
7. 当前最佳模型中 older 和 young 的 target recovery 时间很接近，因此还不能说模型已经解释了高龄组更晚 target recovery。
8. 模型目前仍高估 accuracy，说明 choice/error pattern 还没有完全拟合好。

## 9. 当前限制

- 这是代表性子集诊断，不是全数据年龄组结论。
- older_80_89 只有 4 名被试，虽然 trial 数足够，但被试覆盖有限。
- 当前最佳模型对 RT 分布已有改善，但 accuracy 和 error rate 仍不够贴近人类。
- 高龄组更晚 target recovery 在当前最佳模型里没有明显出现。
- 当前流程是有限模型比较，不是大规模参数搜索。

## 10. 建议下一步

- 保留 representative subset gate，不回到 forced-balanced correct/error。
- 扩展到所有年龄段的 representative 5000-trial subset。
- 保留 t0 variability 作为核心候选机制。
- 继续优化 WW/readout，但评分中应更强地惩罚过高 accuracy。
- 暂时不要把 variational evidence noise 作为单独主解释，更适合作为辅助机制。
- 可以把当前结果作为会议摘要或导师讨论中的机制诊断材料，但不要表述为最终人类行为拟合。

## 11. 一句话总结

当前最佳模型说明：视觉证据可以保持 stimulus-level 共用，年龄差异主要需要放在 t0 波动和决策读出层；模型已经能较好改善 RT 分布并保留错误试次目标恢复更晚的内部机制，但还没有完全拟合人类错误率，也还不能得出完整年龄组结论。
