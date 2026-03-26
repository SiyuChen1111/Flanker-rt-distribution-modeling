#!/usr/bin/env python3
"""
修复Flanker刺激可视化 - 使用箭头图像代替字体
"""

import json

# 读取notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 修复后的可视化单元格 - 使用matplotlib的arrow函数
fixed_visualization_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 修复后的Flanker刺激可视化\n",
        "import matplotlib.patches as patches\n",
        "\n",
        "def draw_arrow(ax, x, y, direction, color, size=0.15):\n",
        "    \"\"\"\n",
        "    使用matplotlib绘制箭头\n",
        "    \"\"\"\n",
        "    if direction == 'L':  # 左\n",
        "        dx, dy = -size, 0\n",
        "    elif direction == 'R':  # 右\n",
        "        dx, dy = size, 0\n",
        "    elif direction == 'U':  # 上\n",
        "        dx, dy = 0, size\n",
        "    elif direction == 'D':  # 下\n",
        "        dx, dy = 0, -size\n",
        "    \n",
        "    ax.arrow(x, y, dx, dy, head_width=0.08, head_length=0.08, \n",
        "             fc=color, ec=color, linewidth=2)\n",
        "\n",
        "def draw_flanker_stimulus(ax, target_dir, flanker_dir, layout, target_color='red', flanker_color='blue'):\n",
        "    \"\"\"\n",
        "    绘制单个Flanker刺激\n",
        "    \"\"\"\n",
        "    ax.set_xlim(-1.5, 1.5)\n",
        "    ax.set_ylim(-1.5, 1.5)\n",
        "    ax.set_aspect('equal')\n",
        "    ax.axis('off')\n",
        "    \n",
        "    # 目标位置（中心）\n",
        "    draw_arrow(ax, 0, 0, target_dir, target_color)\n",
        "    \n",
        "    # 干扰项位置\n",
        "    if layout == 'horizontal':\n",
        "        positions = [(-0.8, 0), (-0.4, 0), (0.4, 0), (0.8, 0)]\n",
        "    elif layout == 'vertical':\n",
        "        positions = [(0, -0.8), (0, -0.4), (0, 0.4), (0, 0.8)]\n",
        "    elif layout == 'cross':\n",
        "        positions = [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]\n",
        "    elif layout == 'diagonal':\n",
        "        positions = [(-0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5)]\n",
        "    \n",
        "    for pos in positions:\n",
        "        draw_arrow(ax, pos[0], pos[1], flanker_dir, flanker_color)\n",
        "\n",
        "# 可视化\n",
        "print(\"=\"*80)\n",
        "print(\"可视化生成的Flanker刺激（修复版）\")\n",
        "print(\"=\"*80)\n",
        "print()\n",
        "\n",
        "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
        "fig.suptitle('生成的Flanker刺激示例', fontsize=16)\n",
        "\n",
        "for i in range(8):\n",
        "    row = i // 4\n",
        "    col = i % 4\n",
        "    \n",
        "    target_dir = test_metadata['target_dirs'][i]\n",
        "    flanker_dir = test_metadata['flanker_dirs'][i]\n",
        "    layout = test_metadata['layouts'][i]\n",
        "    is_congruent = test_labels[i] == 0\n",
        "    \n",
        "    # 绘制刺激\n",
        "    draw_flanker_stimulus(axes[row, col], target_dir, flanker_dir, layout)\n",
        "    \n",
        "    # 设置标题\n",
        "    title = f'目标:{target_dir} 干扰:{flanker_dir}\\n'\n",
        "    title += f'{\"Congruent\" if is_congruent else \"Incongruent\"}\\n'\n",
        "    title += f'布局:{layout}'\n",
        "    axes[row, col].set_title(title, fontsize=10)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"图例说明：\")\n",
        "print(\"  - 红色箭头：目标鸟（需要关注的）\")\n",
        "print(\"  - 蓝色箭头：干扰鸟（需要抑制的）\")\n",
        "print(\"  - Congruent: 目标和干扰项方向相同\")\n",
        "print(\"  - Incongruent: 目标和干扰项方向不同\")"
    ]
}

# 找到并替换可视化单元格
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code' and '可视化生成的Flanker刺激' in str(cell.get('source', [])):
        notebook['cells'][i] = fixed_visualization_cell
        print(f"✓ 修复了单元格 {i+1} 的可视化代码")
        break

# 保存修改后的notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✓ Notebook已更新")
print()
print("修复内容：")
print("  - 使用matplotlib的arrow函数绘制箭头")
print("  - 红色箭头表示目标鸟")
print("  - 蓝色箭头表示干扰鸟")
print("  - 避免了字体显示问题")
