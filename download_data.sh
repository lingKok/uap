#!/bin/bash
# ================================================================
# download_data.sh — 在登录节点下载数据集
# 必须在登录节点执行（计算节点不能联网）
# ================================================================

module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

# [协议修复 P0-1.1] 数据根目录统一：优先位置参数 $1，其次 DATA_ROOT 环境变量，最后默认值
# 与 config.py / utils/data.py 的默认路径保持一致
ROOT="${1:-${DATA_ROOT:-/data/public/cifar}}"
echo "===== 下载 CIFAR-10 数据集到: $ROOT ====="
python -c "
from torchvision import datasets
datasets.CIFAR10(root='$ROOT', train=True, download=True)
datasets.CIFAR10(root='$ROOT', train=False, download=True)
print('CIFAR-10 下载完成!')
"

echo "===== 数据集位置: $ROOT/cifar-10-batches-py/ ====="
