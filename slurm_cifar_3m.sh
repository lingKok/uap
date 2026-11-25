#!/bin/bash
#SBATCH --job-name=meta_cifar_3m
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

# CIFAR-10 3模型训练（resnet18/34/50），50 epochs
# 复制预训练权重到新目录
mkdir -p ./checkpoints_cifar_3m
cp ./checkpoints/resnet18.pt ./checkpoints/resnet34.pt ./checkpoints/resnet50.pt \
   ./checkpoints/resnet101.pt ./checkpoints/mobilenet_v2.pt ./checkpoints/shufflenet_v2_x1_0.pt \
   ./checkpoints_cifar_3m/ 2>/dev/null || true

echo "============================================"
echo "CIFAR-10 训练 (3模型配置)"
echo "============================================"
python train.py \
    --epochs 50 --inner_steps 5 --meta_lr 0.01 --epsilon 0.05 \
    --support_size 256 --query_size 64 --seed 42 \
    --train_models resnet18,resnet34,resnet50 \
    --save_dir ./checkpoints_cifar_3m

echo ""
echo "============================================"
echo "SOTA 对比 (CIFAR-10, 3-model Meta-UAP)"
echo "============================================"
python experiments.py --dataset cifar \
    --meta_checkpoint checkpoints_cifar_3m/meta_uap_epoch50.pt
