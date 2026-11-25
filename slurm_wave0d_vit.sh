#!/bin/bash
# ================================================================
# Wave 0d — ViT-B/16 0-step 迁移探针（方向二前提确认 + Limitations）
#
# 用途:
#   现有 3m ensemble UAP 在 torchvision ViT-B/16（IMAGENET1K_V1,
#   NormalizedModel 包装）上 0-step 评估:
#     ε ∈ {0.05, 0.1, 0.2} × {raw, scaled} 双形态,
#     clean_acc / fr_all / asr 双口径（results_wave0/vit_probe.json）
#   评估集与 0a 的 experiments.py 完全同口径（val / seed 42 / 2000 张）——
#   ViT 探针数字可与 Table 5 的 CNN 行并列比较。
#   对照文献预期: TSC-UAP 实测 CNN→ViT 迁移 ASR 仅 14-24%。
#
# 预估: 15-25 GPU·分钟（2000 张前向）; --time=01:00:00 留足余量
#
# Go/No-Go（裁决文件 §4 方向二 GO①, Addendum 修订）:
#   探针 asr 与文献（14-24%）一致即确认前提成立（不再承担证伪功能）;
#   无论结果如何都写入 Limitations。
#
# 依赖:
#   · vit_probe.py 内置 fail-fast 预检（checkpoint / imagenet-mini val 目录 /
#     vit_b_16 权重缓存——权重需登录节点预热, 见 RUNBOOK_wave01.md §3）
#   · 默认用现有 3m ensemble（旧协议训练; 探针对 2.4% 评估集重叠不敏感,
#     如需严格口径可用 VIT_CKPT 指向修复后重训的 3m）
# ================================================================
#SBATCH --job-name=wave0d_vit
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-wave0d-%j.out

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

VIT_CKPT="${VIT_CKPT:-checkpoints_imagenet_3m/meta_uap_imagenet_3m_ensemble3.pt}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"

echo "============================================"
echo "Wave 0d: ViT-B/16 0-step 探针"
echo "  checkpoint: ${VIT_CKPT}"
echo "  eval_samples: ${EVAL_SAMPLES}"
echo "============================================"

python vit_probe.py \
    --checkpoint "${VIT_CKPT}" \
    --eval_split val \
    --eval_samples "${EVAL_SAMPLES}" \
    --eval_seed 42 \
    --epsilons "0.05,0.1,0.2" \
    --train_epsilon 0.05 \
    --out_dir results_wave0 \
    > slurm-wave0d-vit.out 2>&1

echo "Wave 0d 完成, 结果见 results_wave0/vit_probe.json（python 日志: slurm-wave0d-vit.out）"
