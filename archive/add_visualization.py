#!/usr/bin/env python3
"""
添加图像可视化单元格到notebook
"""

import json

# 读取notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 添加可视化单元格
visualization_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 可视化生成的Flanker刺激图像\n",
        "print(\"=\"*80)\n",
        "print(\"可视化生成的Flanker刺激\")\n",
        "print(\"=\"*80)\n",
        "print()\n",
        "# 选择一些样本进行可视化\n",
        "n_samples_to_show = 8\n",
        "indices_to_show = list(range(n_samples_to_show))\n",
        "\n",
        "# 创建图形\n",
        "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
        "fig.suptitle('生成的Flanker刺激示例', fontsize=16)\n",
        "\n",
        "for i, idx in enumerate(indices_to_show):\n",
        "    row = i // 4\n",
        "    col = i % 4\n",
        "    \n",
        "    # 转换回HWC格式用于显示\n",
        "    img = test_images[idx].transpose(1, 2, 0)\n",
        "    \n",
        "    # 显示图像\n",
        "    axes[row, col].imshow(img)\n",
        "    \n",
        "    # 获取元数据\n",
        "    target_dir = test_metadata['target_dirs'][idx]\n",
        "    flanker_dir = test_metadata['flanker_dirs'][idx]\n",
        "    layout = test_metadata['layouts'][idx]\n",
        "    is_congruent = test_labels[idx] == 0\n",
        "    \n",
        "    # 设置标题\n",
        "    title = f'目标:{target_dir} 干扰:{flanker_dir}\\n'\n",
        "    title += f'{\"Congruent\" if is_congruent else \"Incongruent\"}\\n'\n",
        "    title += f'布局:{layout}'\n",
        "    axes[row, col].set_title(title, fontsize=10)\n",
        "    axes[row, col].axis('off')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"图例说明：\")\n",
        "print(\"  - 红色箭头：目标鸟（需要关注的）\")\n",
        "print(\"  - 蓝色箭头：干扰鸟（需要抑制的）\")\n",
        "print(\"  - Congruent: 目标和干扰项方向相同\")\n",
        "print(\"  - Incongruent: 目标和干扰项方向不同\")\n",
        "print()\n",
        "print(\"注意：在原始Flanker任务中，目标鸟和干扰鸟颜色相同（黑色）\")\n",
        "print(\"      这里为了演示方便，用不同颜色区分\")"
    ]
}

# 找到"生成测试数据"单元格
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code' and 'test_tensor = torch.from_numpy(test_images)' in str(cell.get('source', [])):
        # 在这个单元格后插入可视化单元格
        notebook['cells'].insert(i + 1, visualization_cell)
        print(f"✓ 在单元格 {i+1} 后添加了图像可视化单元格")
        break

# 保存修改后的notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✓ Notebook已更新")
print()
print("添加的可视化功能：")
print("  1. 显示8个Flanker刺激样本")
print("  2. 区分Congruent和Incongruent条件")
print("  3. 显示目标方向、干扰方向和布局")
print("  4. 使用不同颜色区分目标鸟和干扰鸟")
