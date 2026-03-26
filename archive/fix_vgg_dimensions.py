#!/usr/bin/env python3
"""
修复VGG16特征维度问题
"""

import torch
import torch.nn as nn
from torchvision import models

print("="*80)
print("修复VGG16特征维度问题")
print("="*80)
print()

# 测试原始VGG16
model = models.vgg16(weights=None)
dummy_input = torch.randn(1, 3, 128, 128)

with torch.no_grad():
    features = model.features(dummy_input)
    print(f"卷积后特征: {features.shape}")
    
    pooled = model.avgpool(features)
    print(f"池化后特征: {pooled.shape}")
    
    flattened = torch.flatten(pooled, 1)
    print(f"展平后维度: {flattened.shape[1]}")
    
    print()
    print(f"✓ 正确的特征维度: {flattened.shape[1]}")
    print(f"  计算: 512 * 7 * 7 = {512 * 7 * 7}")
    print()
    print("修复方法：")
    print("  1. 将 feature_dim 改为 25088")
    print("  2. 或添加自适应池化层确保输出为 4x4")
