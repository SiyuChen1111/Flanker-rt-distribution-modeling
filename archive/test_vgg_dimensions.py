#!/usr/bin/env python3
"""
测试VGG16在128x128输入下的实际特征维度
"""

import torch
from torchvision import models

print("="*80)
print("测试VGG16特征维度")
print("="*80)
print()

# 加载VGG16模型
model = models.vgg16(weights=None)

# 创建128x128的输入
dummy_input = torch.randn(1, 3, 128, 128)

# 测试特征提取
model.eval()
with torch.no_grad():
    # 特征提取
    features = model.features(dummy_input)
    print(f"卷积特征输出形状: {features.shape}")
    
    # 自适应池化
    pooled = model.avgpool(features)
    print(f"池化后特征形状: {pooled.shape}")
    
    # 展平
    flattened = torch.flatten(pooled, 1)
    print(f"展平后特征维度: {flattened.shape[1]}")
    
    print()
    print(f"✓ 正确的特征维度: {flattened.shape[1]}")
    print(f"  计算: 512 * 7 * 7 = {512 * 7 * 7}")
