#!/bin/bash
# ================================================================
# Wave 1 (b) — zoo 消融逐样本 fooled 掩码 dump（供 McNemar 配对检验）
#
# 与 slurm_wave1_zoo.sh 同口径重跑，仅额外加 --dump_masks：
#   · meta 版: subsets 3,3vgg,3dn,5, inner_steps=5, ablations=3, 3 seeds
#   · ensemble 版: subsets 3,5, inner_steps=0, ablations=3, 3 seeds
# 结果写入 results_wave0/results_ablation_zoo_masks.json（不覆盖 *_meta/_ensemble）
#
# 预估: 与 zoo 同量级 ≈ 60-90 GPU·分钟（meta 为主；ensemble 快）
# 依赖: CIFAR-10 数据、meta_uap 环境、各 surrogate/test 模型 .pt 权重
# ================================================================
#SBATCH --job-name=wave1_zoomask
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-wave1-zoomask-%j.out

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

echo "============================================"
echo "Wave 1 (b): zoo 逐样本掩码 dump (McNemar)"
echo "============================================"

# ---- meta 版（inner_steps=5, 子集 3/3vgg/3dn/5）----
echo ">>> meta 版掩码 dump ..."
python ablation.py --seeds "42,1342,2342" \
    --model_subsets "3,3vgg,3dn,5" --ablations "3" --inner_steps 5 \
    --train_split train --eval_split test --out_dir results_wave0 \
    --dump_masks

# ---- ensemble 版（inner_steps=0, 子集 3/5，方向三 (a) 对照）----
echo ">>> ensemble 版掩码 dump ..."
python ablation.py --seeds "42,1342,2342" \
    --model_subsets "3,5" --ablations "3" --inner_steps 0 \
    --train_split train --eval_split test --out_dir results_wave0 \
    --dump_masks

echo "掩码 dump 完成! 结果: results_wave0/results_ablation_zoo_masks.json"
