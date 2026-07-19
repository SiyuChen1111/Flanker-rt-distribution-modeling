# Flanker 双通道表征审计

- 模式：full
- 刺激数量：9990
- 完整刺激重建通过：True
- target/flanker 来源分离通过：True
- 是否允许进入双通道 Wong–Wang 比较：True

## 各层方向解码

- target_only: conv3=0.979, conv4=0.721, conv5=0.536, pooled=0.521, final=0.700
- flanker_only: conv3=1.000, conv4=1.000, conv5=0.977, pooled=0.976, final=0.801

## 下一步

- 只有正式 full 审计通过后，才运行同步、双通道和反向时序对照的 Wong–Wang 比较。
- 若正式审计失败，停止后端拟合，并把失败定位为当前 VGG 方向证据无法可靠区分空间来源。
- 来源门槛在冒烟校准后、正式全样本运行前固定；正式结果产生后不再调整。
