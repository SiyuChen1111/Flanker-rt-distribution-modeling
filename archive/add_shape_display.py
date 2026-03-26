#!/usr/bin/env python3
"""
添加显示test_images形状的单元格
"""

import json

# 读取notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 在"生成测试数据"单元格后添加一个新单元格
new_cell = {
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
        "# 显示第一个样本的信息\n",
        "print(f\"第一个样本:\")\n",
        "print(f\"  形状: {test_images[0].shape}\")\n",
        "print(f\"  数据类型: {test_images[0].dtype}\")\n",
        "print(f\"  值范围: [{test_images[0].min():.3f}, {test_images[0].max():.3f}]\")\n",
        "print()\n",
        "print(\"注意：图像数据已经归一化到[0, 1]范围\")"
    ]
}

# 找到"生成测试数据"单元格的索引
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code' and '生成测试数据' in str(cell.get('source', [])):
        # 在这个单元格后插入新单元格
        notebook['cells'].insert(i + 1, new_cell)
        print(f"✓ 在单元格 {i+1} 后添加了显示test_images形状的单元格")
        break

# 保存修改后的notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✓ Notebook已更新")
print()
print("现在您可以运行notebook，在'生成测试数据'单元格后会看到test_images的形状信息")
