#!/usr/bin/env python3
"""
完整修复notebook：添加形状显示 + 修复特征维度
"""

import json

# 读取notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 1. 在"生成测试数据"单元格后添加形状显示
shape_display_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 显示test_images的形状\n",
        "print(\"=\"*80)\n",
        "print(\"test_images的形状和类型\")\n",
        "print(\"=\"*80)\n",
        "print()\n",
        "print(f\"test_images类型: {type(test_images)}\")\n",
        "print(f\"test_images形状: {test_images.shape}\")\n",
        "print(f\"  - 样本数: {test_images.shape[0]}\")\n",
        "print(f\"  - 通道数: {test_images.shape[1]}\")\n",
        "print(f\"  - 高度: {test_images.shape[2]}\")\n",
        "print(f\"  - 宽度: {test_images.shape[3]}\")\n",
        "print()\n",
        "print(\"数据格式: (样本数, 通道数, 高度, 宽度)\")\n",
        "print(\"         (N, C, H, W)\")\n",
        "print()\n",
        "print(\"这是PyTorch的标准数据格式:\")\n",
        "  - N: batch size (样本数量)\")\n",
        "  - C: channels (颜色通道，RGB=3)\")\n",
        "  - H: height (图像高度)\")\n",
        "  - W: width (图像宽度)\")\n",
        "print()\n",
        "print(\"注意：图像数据已经归一化到[0, 1]范围\")"
    ]
}

# 2. 修复特征维度问题
# 找到VGGDriftRateModel类定义中的feature_dim
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code' and 'self.feature_dim = 512 * 4 * 4' in str(cell.get('source', [])):
        # 修复特征维度
        old_source = ''.join(cell['source'])
        new_source = old_source.replace(
            'self.feature_dim = 512 * 4 * 4  # VGG16在128x128输入下的特征维度',
            'self.feature_dim = 512 * 7 * 7  # VGG16在128x128输入下的特征维度 (修复)'
        )
        new_source = new_source.replace(
            '# (batch, 8192)',
            '# (batch, 25088)  # 512*7*7'
        )
        cell['source'] = new_source.split('\n')
        print(f"✓ 修复了单元格 {i+1} 的特征维度")
        break

# 3. 在"生成测试数据"单元格后插入形状显示
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code' and '生成测试数据' in str(cell.get('source', [])):
        # 在这个单元格后插入新单元格
        notebook['cells'].insert(i + 1, shape_display_cell)
        print(f"✓ 在单元格 {i+1} 后添加了显示test_images形状的单元格")
        break

# 保存修改后的notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print()
print("="*80)
print("修复完成！")
print("="*80)
print()
print("修复内容：")
print("  1. ✓ 添加了显示test_images形状的单元格")
print("  2. ✓ 修复了VGG16特征维度问题")
print("     - 从 8192 (512*4*4) 改为 25088 (512*7*7)")
print()
print("现在可以运行notebook了！")
print()
print("预期输出：")
print("  test_images形状: (100, 3, 128, 128)")
print("  - 100个样本")
print("  - 3个颜色通道 (RGB)")
print("  - 128x128 像素")
