#!/bin/bash
# ================================================================
# Wave 0a — 协议修复验证 + ε-sweep 重算 + 方向一 veto 电池
#
# 用途:
#   修复后的 experiments.py --dataset imagenet 全量跑:
#     · SOTA 对比（UAP-SGD / PD-UA / SSA / Gen-UAP / Meta-0/5/10）
#     · ε 敏感性三行（R50 / R101 / VGG）——方向一所有数字的口径基线
#     · Scaled-UAP-SGD 对照（ε_train=0.05 训练, ×4 放大到 ε_test=0.2）
#     · UAP-SGD@0.2 的 100/200 步对照
#
# 预估: 60-90 GPU·分钟（单卡）; --time=04:00:00 留足余量
#
# Go/No-Go（裁决文件 §4 方向一, 预承诺）:
#   GO  = Scaled-UAP-SGD 在 R50 与 R101 均低于 Meta-scaled ≥8pp（且 McNemar 显著）
#         + UAP-SGD@0.2 100/200 步未追平 + VGG 反例如实入表入文
#   任一不满足 → 删 "globally superior trajectory"，改写为观察 + 基线调优诚实讨论
#
# 依赖:
#   · 参数名按【接口约定】, 待洗澄明修复 experiments.py 后生效
#   · meta checkpoint: 默认现有 3m ensemble（旧协议训练, provisional 口径）,
#     以便与 0c/0d 同时提交——SGD 基线电池（最耗时部分）与 checkpoint 无关;
#     0c 产出修复后 3m 后, 用
#       META_CKPT=checkpoints_imagenet_3m_fixed/meta_uap_imagenet_3m_fixed_ensemble3.pt \
#       OUT_DIR=results_wave0_fixed3m sbatch slurm_wave0a_epsweep.sh
#     重跑本脚本得到干净口径的 Meta 行（SGD 行不变, 仅 Meta 行更新）
# ================================================================
#SBATCH --job-name=wave0a_epsweep
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-wave0a-%j.out

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

META_CKPT="${META_CKPT:-checkpoints_imagenet_3m/meta_uap_imagenet_3m_ensemble3.pt}"
OUT_DIR="${OUT_DIR:-results_wave0}"

echo "============================================"
echo "Wave 0a: ε-sweep 重算 + 方向一 veto 电池 (ImageNet, 修复协议)"
echo "  meta checkpoint: ${META_CKPT}"
echo "  out_dir:         ${OUT_DIR}"
echo "============================================"

if [ ! -f "${META_CKPT}" ]; then
    echo "[FATAL] 找不到 meta checkpoint: ${META_CKPT}"
    echo "  现有 3m ensemble 由 slurm_train_imagenet_3m.sh 在集群生成;"
    echo "  若仅有 checkpoints_imagenet_3m_run{0,1,2}/, 可按该脚本末尾方式"
    echo "  对 3 个 run 的 uap_init 取平均生成 meta_uap_imagenet_3m_ensemble3.pt。"
    echo "  或先提交 slurm_wave0c_4m4md.sh 重训 3m 后, 用"
    echo "    META_CKPT=checkpoints_imagenet_3m_fixed/meta_uap_imagenet_3m_fixed_ensemble3.pt 重跑本脚本。"
    exit 1
fi

python experiments.py --dataset imagenet \
    --epsilon 0.05 \
    --seed 42 \
    --train_split train \
    --eval_split val \
    --eval_samples 2000 \
    --scaled_sgd \
    --sgd_extra_steps "100,200" \
    --meta_checkpoint "${META_CKPT}" \
    --out_dir "${OUT_DIR}" \
    > slurm-wave0a-epsweep.out 2>&1

echo "Wave 0a 完成, 结果见 ${OUT_DIR}/ （python 日志: slurm-wave0a-epsweep.out）"
