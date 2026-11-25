#!/bin/bash
# ================================================================
# setup_env.sh — 宁夏超算云 N30 区环境初始化脚本
# 在登录节点 (ln01) 上执行一次即可
# ================================================================

set -e

echo "===== 宁夏超算云 N30 区 — Meta-UAP 环境初始化 ====="

# 1. 加载 miniforge 模块（替代已取消的 anaconda）
echo "[1/4] 加载 miniforge 模块..."
module load miniforge/24.9.2
module list

# 2. 创建 Python 虚拟环境
echo "[2/4] 创建 meta_uap 虚拟环境..."
conda create -n meta_uap python=3.10 -y
conda activate meta_uap

# 3. 安装 PyTorch（匹配 CUDA 12.4，RTX 3090 支持）
echo "[3/4] 安装 PyTorch + torchvision..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. 安装 higher 和其他依赖
echo "[4/4] 安装 higher 和其他依赖..."
pip install higher numpy

# 验证
echo ""
echo "===== 验证安装 ====="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "===== 环境初始化完成 ====="
echo "每次登录后执行:"
echo "  module load miniforge/24.9.2"
echo "  conda activate meta_uap"
