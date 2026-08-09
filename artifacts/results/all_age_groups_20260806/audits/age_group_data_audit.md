# 年龄组数据审计

本审计依据 `metadata.csv` 和 `user*df.csv` 源文件建立。唯一试次键为 `subject_id + nth_play + trial`。

`raw` 是源文件行数；`valid` 要求 RT 在 0.15–10 秒且 target、flanker、response 均为 L/R/U/D；`model-input` 仅统计已有目标模型 manifest 中的实际试次。现有极端年龄结果是每组 5,000 个代表性 trial，不是每名被试 5,000 个。

基础数据审计时中间年龄组尚无 VGG cache；随后已完成 30–79 岁五组的代表性刺激缓存，并生成七组统一 trial-level predictions。当前校准后的拟合结果以 `../all_age_model_update_20260807/` 为准；该结果仍是代表性子集诊断，不是全体试次或独立留出验证。
