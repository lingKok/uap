#!/bin/bash
# ================================================================
# Wave 1 (c) — 梯度冲突矩阵多 data_seed 方差（GO③ 稳健性）
#
# 对 3m / 4m / 4md 三个 subset，各跑 data_seed ∈ {42, 1042, 2342}，
# 输出 grad_conflict_{tag}_s{seed}.json（不互相覆盖）。
# 4m 含 VGG，是关键机制证据，优先看其跨 seed 稳定性。
#
# 预估: 9 runs 串行 ≈ 15-25 GPU·分钟（每 run 仅 32 张图 × 7 模型，轻量）
# 依赖: checkpoints_imagenet_3m_fixed / 4m / 4md；torchvision 权重已预热
# ================================================================
#SBATCH --job-name=wave1_gradseed
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-wave1-gradseed-%j.out

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

echo "============================================"
echo "Wave 1 (c): 梯度矩阵多 seed 方差"
echo "============================================"

for subset in 3m 4m 4md; do
  for seed in 42 1042 2342; do
    echo "--- gradient_conflict: subset=${subset}, data_seed=${seed} ---"
    python gradient_conflict.py --model_subset ${subset} \
        --data_seed ${seed} --grad_samples 32 --epsilon 0.05 \
        --data_root /data/public/imagenet-mini --out_dir results_wave0 \
      || echo "[错误] subset=${subset} seed=${seed} 失败, 详见日志"
  done
done

echo "所有梯度矩阵完成! 结果见 results_wave0/grad_conflict_{3m,4m,4md}_s{42,1042,2342}.json"
