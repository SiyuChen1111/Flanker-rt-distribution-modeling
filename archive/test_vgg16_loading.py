#!/usr/bin/env python3
"""
测试VGG16预训练模型加载
"""

import torch
from torchvision import models

print("="*80)
print("测试VGG16预训练模型加载")
print("="*80)
print()

try:
    print("正在加载VGG16预训练模型...")
    # 使用新的API加载预训练模型
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    print("✓ 成功加载预训练VGG16模型！")
    print()
    
    # 打印模型信息
    print("模型信息:")
    print(f"  - 类型: {type(model).__name__}")
    print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 特征提取器层数: {len(model.features)}")
    print(f"  - 分类器层数: {len(model.classifier)}")
    print()
    
    # 测试前向传播
    print("测试前向传播...")
    model.eval()
    with torch.no_grad():
        # 创建随机输入
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"  - 输入形状: {dummy_input.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print()
    print("✓ 所有测试通过！模型可以正常使用。")
    print()
    print("现在可以运行notebook了：")
    print("  - vgg_drift_rate_fixed.ipynb (推荐)")
    print("  - vgg_drift_rate_complete.ipynb")
    
except Exception as e:
    print(f"✗ 加载失败: {e}")
    print()
    print("请检查文件是否正确放置在:")
    print("  ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
