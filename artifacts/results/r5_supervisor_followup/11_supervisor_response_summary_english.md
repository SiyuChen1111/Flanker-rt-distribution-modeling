# Concise Supervisor Response

## A. CAF/CRF x-axis

CAF and CRF were recomputed from raw trial-level outputs. The x-axis now uses the actual median RT within each quantile bin; Human and Model keep separate RT coordinates.

## B. S(t), DiffDecisionMultiClass, and RT distribution

The current evidence suggests that fixed-time S(t) distributions and first-passage RT distributions must be interpreted separately. R5 hard WW decision times are compressed relative to final RT, and t0 variability contributes materially to the final RT shape.

## C. Excessive incongruent errors

The excessive incongruent errors are most consistent with a mixture of upstream evidence/readout timing and t0 allocation, not a fully supported response-label mapping failure.

Strongest supported conclusion: R5 的反应时形状很大一部分来自非决策时间波动；硬性的 WW 首次越界时间本身更压缩，说明瓶颈更可能在读出/阈值/噪声与 t0 的分工，而不是“RT 应该正态”。

Most important unresolved issue: 没有在 R5 包内找到完整训练损失权重和独立 checkpoint，因此训练目标相关结论仍需进一步追踪。

Recommended next modification: 下一步优先做单因素小消融：限制 t0 波动，同时测试校准后的 accumulator noise 或 threshold/margin，不要一次加入多种新机制。
