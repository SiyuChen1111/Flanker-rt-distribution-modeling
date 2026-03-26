#!/usr/bin/env python3
"""
手动下载VGG16预训练模型
解决SSL证书问题
"""

import os
import urllib.request
import ssl

# 解决SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# 模型保存路径
cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
os.makedirs(cache_dir, exist_ok=True)

# VGG16模型URL
url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
model_path = os.path.join(cache_dir, 'vgg16-397923af.pth')

print(f"正在下载VGG16预训练模型...")
print(f"URL: {url}")
print(f"保存路径: {model_path}")

try:
    # 下载模型
    urllib.request.urlretrieve(url, model_path)
    print(f"\n✓ 下载成功！")
    print(f"模型已保存到: {model_path}")
    
    # 验证文件
    file_size = os.path.getsize(model_path)
    print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
    
except Exception as e:
    print(f"\n✗ 下载失败: {e}")
    print("\n请尝试以下替代方案：")
    print("1. 使用命令行下载:")
    print(f"   wget {url} -O {model_path}")
    print(f"   或")
    print(f"   curl {url} -o {model_path}")
    print("\n2. 使用镜像源:")
    print("   从以下镜像源手动下载后放到指定路径:")
    print(f"   - 清华镜像: https://mirrors.tuna.tsinghua.edu.cn/")
    print(f"   - 阿里镜像: https://mirrors.aliyun.com/")
