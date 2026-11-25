#!/bin/bash
#SBATCH --job-name=sota_imagenet_4m
#SBATCH --gpus=1

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

echo "============================================"
echo "SOTA 对比 (ImageNet, 4-model Meta-UAP)"
echo "============================================"
python experiments.py --dataset imagenet \
    --meta_checkpoint checkpoints_imagenet_4m/meta_uap_imagenet_4m_ensemble3.pt
