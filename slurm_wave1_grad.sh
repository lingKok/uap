#!/bin/bash
# ================================================================
# Wave 1 — 内循环梯度余弦冲突矩阵（3m / 4m / 4md, 方向三机制证据）
#
# 用途:
#   每个 surrogate 在 train split 同一批数据上计算 uap_init 的一步
#   inner-loop 梯度（与 train_imagenet.py 内循环第一步一致, 穿过 clamp）,
#   输出 N×N 余弦相似度矩阵（含元测试模型列, 仅作对照）:
#     results_wave0/grad_conflict_{3m,4m,4md}.json
#
# 预估: 10-15 GPU·分钟; --time=01:00:00 留足余量
#
# Go/No-Go（裁决文件 §4 方向三 GO③）:
#   矩阵与干预证据（Wave 1 zoo 消融）方向一致时方可声称 "consistent with";
#   不得单独支撑因果。
#
# 依赖:
#   · slurm_wave0c_4m4md.sh 产出的修复协议 ensemble checkpoints:
#       checkpoints_imagenet_3m_fixed/meta_uap_imagenet_3m_fixed_ensemble3.pt
#       checkpoints_imagenet_4m/meta_uap_imagenet_4m_ensemble3.pt
#       checkpoints_imagenet_4md/meta_uap_imagenet_4md_ensemble3.pt
#     （3m 用修复后重训版, 保证 3m/4m/4md 协议一致; 缺失时 .py fail-fast）
#   · imagenet-mini train/ 子目录; torchvision 权重缓存（登录节点预热）
# ================================================================
#SBATCH --job-name=wave1_grad
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-wave1-grad-%j.out

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

echo "============================================"
echo "Wave 1 grad: 内循环梯度余弦冲突矩阵 (3m / 4m / 4md)"
echo "============================================"

for subset in 3m 4m 4md; do
    echo "--- gradient_conflict: subset=${subset} ---"
    python gradient_conflict.py \
        --model_subset "${subset}" \
        --data_split train \
        --data_seed 42 \
        --grad_samples 32 \
        --epsilon 0.05 \
        --out_dir results_wave0 \
        > "slurm-wave1-grad-${subset}.out" 2>&1 || \
        echo "[错误] subset=${subset} 失败, 详见 slurm-wave1-grad-${subset}.out"
done

echo "Wave 1 grad 完成, 结果见 results_wave0/grad_conflict_*.json"
