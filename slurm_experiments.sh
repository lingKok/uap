#!/bin/bash
#SBATCH --job-name=sota_compare
#SBATCH --gpus=1

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

echo "============================================"
echo "SOTA 对比 + 效率分析 (CIFAR-10)"
echo "============================================"
python experiments.py --dataset cifar

echo ""
echo "============================================"
echo "SOTA 对比 + 效率分析 (ImageNet)"
echo "============================================"
python experiments.py --dataset imagenet
