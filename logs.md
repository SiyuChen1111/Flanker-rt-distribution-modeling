# VAM学习日志 - 2026-03-26

## 今日更新概览

### 1. 创建的核心Notebook文件

#### `vgg_drift_rate_complete.ipynb`
- **用途**: VGG模型输出漂移率的完整实现
- **主要内容**:
  - Flanker刺激生成器（使用真实鸟图像）
  - VGG16模型输出4个漂移率
  - 四个方向数值的详细物理意义解释
  - 漂移率与认知过程的关联
  - 训练流程和可视化

#### `vgg_drift_rate_fixed.ipynb`
- **用途**: 修复SSL下载问题的版本
- **改进**:
  - 添加SSL证书绕过
  - 更新torchvision API使用方式
  - 提供多种模型加载方式

### 2. 创建的辅助脚本

#### `download_vgg16.py`
- **功能**: 手动下载VGG16预训练模型
- **解决**: SSL证书验证失败问题

#### `vgg16_download_alternatives.py`
- **功能**: 提供多种VGG16下载替代方案
- **包括**: 国内镜像、手动下载、不使用预训练等

#### `test_vgg16_loading.py`
- **功能**: 测试VGG16模型加载
- **验证**: 预训练模型是否正确安装

#### `test_vgg_dimensions.py`
- **功能**: 测试VGG16特征维度
- **发现**: 128×128输入下特征维度为25088（512×7×7）

#### `fix_vgg_dimensions.py`
- **功能**: 修复特征维度不匹配问题
- **修改**: 将8192改为25088

#### `fix_notebook_complete.py`
- **功能**: 完整修复notebook
- **包括**: 添加形状显示 + 修复特征维度

#### `add_visualization.py`
- **功能**: 添加Flanker刺激可视化
- **效果**: 显示8个样本，区分Congruent/Incongruent

#### `fix_visualization.py`
- **功能**: 修复可视化（使用matplotlib箭头）
- **解决**: 字体显示问题

#### `update_with_real_birds.py`
- **功能**: 更新使用真实鸟图像
- **改进**: 与原始代码make_model_inputs.py保持一致

#### `add_shape_display.py`
- **功能**: 添加test_images形状显示单元格

### 3. 关键发现和修复

#### 问题1: SSL下载错误
- **症状**: `URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING]>`
- **解决**: 将模型文件复制到`~/.cache/torch/hub/checkpoints/`

#### 问题2: 特征维度不匹配
- **症状**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (10x25088 and 8192x1024)`
- **原因**: VGG16在128×128输入下产生25088维特征，不是8192
- **解决**: 修改`self.feature_dim = 512 * 7 * 7`（25088）

#### 问题3: 字体显示问题
- **症状**: 箭头显示为方块乱码
- **解决**: 使用matplotlib的arrow函数代替字体

#### 问题4: 图像生成不一致
- **症状**: 使用箭头而非真实鸟图像
- **解决**: 使用bird0.png - bird3.png真实图像

### 4. 四个方向漂移率的物理意义

```
输出: [漂移率_左, 漂移率_右, 漂移率_上, 漂移率_下]
索引: [    0,      1,      2,      3]
方向: [   L,      R,      U,      D]
```

**漂移率 = 信息累积速度**

- **高值 (>2.0)**: 强烈支持该方向
- **中值 (1.0-2.0)**: 中等支持
- **低值 (<1.0)**: 弱支持或不支持

**与认知过程的关联**:
1. **反应时间**: RT ≈ threshold / drift_rate
2. **选择概率**: 通过Softmax计算
3. **一致性判断**: 通过漂移率集中度判断

### 5. 与原始VAM代码的区别

#### Notebook实现
- **框架**: PyTorch
- **训练方式**: 监督学习（简化版损失函数）
- **损失函数**: 自定义drift_rate_loss
- **目的**: 演示VGG输出漂移率

#### 原始VAM代码
- **框架**: JAX/Flax
- **训练方式**: 变分推断（ELBO优化）
- **损失函数**: 证据下界（ELBO）
- **目的**: 完整认知计算模型

**ELBO组成**:
- 重建误差（LBA对数似然）
- 雅可比行列式
- 先验项
- 熵项

### 6. 文件管理建议

#### 核心文件（保留）
- `vgg_drift_rate_complete.ipynb` - 主要学习材料
- `vgg_drift_rate_fixed.ipynb` - 备用版本
- `logs.md` - 本日志文件
- `siyu_study.md` - 学习笔记

#### 辅助脚本（可归档）
- `download_vgg16.py`
- `vgg16_download_alternatives.py`
- `test_vgg16_loading.py`
- `test_vgg_dimensions.py`
- `fix_vgg_dimensions.py`
- `fix_notebook_complete.py`
- `add_visualization.py`
- `fix_visualization.py`
- `update_with_real_birds.py`
- `add_shape_display.py`

#### 预训练模型（已安装）
- `vgg16-397923af.pth` → `~/.cache/torch/hub/checkpoints/`

### 7. 下一步建议

1. **运行Notebook**: 测试完整的VGG漂移率输出流程
2. **对比学习**: 理解监督学习vs变分推断的区别
3. **深入研究**: 阅读原始VAM论文和代码
4. **实验扩展**: 尝试不同的损失函数和训练策略

---

**记录时间**: 2026-03-26
**记录人**: AI Assistant
**状态**: 活跃开发中
