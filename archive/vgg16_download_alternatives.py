#!/usr/bin/env python3
"""
VGG16模型下载替代方案
"""

import os
import urllib.request
import ssl

print("="*80)
print("VGG16预训练模型下载替代方案")
print("="*80)
print()

# 方案1: 使用国内镜像源
print("【方案1】使用国内镜像源")
print("-"*80)
print("清华镜像:")
print("  wget https://mirrors.tuna.tsinghua.edu.cn/pytorch/models/vgg16-397923af.pth -O ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
print()
print("阿里镜像:")
print("  wget https://mirrors.aliyun.com/pytorch/models/vgg16-397923af.pth -O ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
print()

# 方案2: 手动下载
print("【方案2】手动下载")
print("-"*80)
print("1. 访问以下网站之一:")
print("   - https://github.com/pytorch/vision/tree/main/torchvision/models")
print("   - https://download.pytorch.org/models/")
print()
print("2. 下载文件: vgg16-397923af.pth (约528MB)")
print()
print("3. 放到指定路径:")
cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
print(f"   {cache_dir}/vgg16-397923af.pth")
print()

# 方案3: 使用不预训练的模型
print("【方案3】使用不预训练的模型（从头训练）")
print("-"*80)
print("在notebook中将 pretrained=True 改为 pretrained=False")
print("缺点: 需要更多数据和训练时间")
print()

# 方案4: 使用其他预训练模型
print("【方案4】使用其他预训练模型")
print("-"*80)
print("如果VGG16下载失败，可以尝试:")
print("  - ResNet: models.resnet18(pretrained=True)")
print("  - AlexNet: models.alexnet(pretrained=True)")
print("  - SqueezeNet: models.squeezenet1_0(pretrained=True)")
print()

# 创建缓存目录
os.makedirs(cache_dir, exist_ok=True)
print(f"缓存目录已创建: {cache_dir}")
print()

# 检查是否已有模型
model_path = os.path.join(cache_dir, 'vgg16-397923af.pth')
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"✓ 发现已下载的模型: {model_path}")
    print(f"  文件大小: {file_size:.2f} MB")
else:
    print("✗ 未找到VGG16模型文件")
    print()
    print("建议: 先尝试方案1（国内镜像），如果失败则使用方案3（不预训练）")
