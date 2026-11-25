#!/bin/bash
#SBATCH --job-name=sota_redo
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

echo "============================================"
echo "SOTA 对比 + Gen-UAP + ε敏感性 (CIFAR-10, 固定种子)"
echo "============================================"
python experiments.py --dataset cifar \
    --meta_checkpoint checkpoints/meta_uap_epoch50.pt

echo ""
echo "============================================"
echo "SOTA 对比 + Gen-UAP + ε敏感性 (ImageNet, 固定种子, 3模型)"
echo "============================================"
python experiments.py --dataset imagenet \
    --meta_checkpoint checkpoints_imagenet_3m/meta_uap_imagenet_3m_ensemble3.pt
