#!/bin/bash
# ================================================================
# Wave 0c — ImageNet 3m/4m/4md 修复协议补训（各 3 seeds + ensemble 平均）
#
# 用途:
#   · 4m = [resnet18,resnet34,resnet50,vgg16_bn]      （方向三 VGG 分离）
#   · 4md = [resnet18,resnet34,resnet50,densenet121]  （方向三 DenseNet 分离）
#   · 3m_fixed = [resnet18,resnet34,resnet50] 修复协议重训
#     —— 使 4m<3m / 4md≈3m 的比较与梯度矩阵都在同一干净协议下;
#        同时给 Wave 0a 重跑提供干净口径的 3m checkpoint
#   （仓库中 4m/4md 从未跑过; 3m 旧 run 为旧协议, 不覆盖, 写入 _fixed 目录）
#
# 预估: 9 runs 串行 ≈ 25-40 GPU·分钟; --time=03:00:00 留足余量
#   （RETRAIN_3M=0 可跳过 3m 重训, 退回 6 runs ≈ 15-25 分钟）
#
# Go/No-Go（裁决文件 §4 方向三 GO②）:
#   ImageNet 4m < 3m 且 4md ≈ 3m（各 run 最终评估的 0-step FR, 3 个元测试
#   模型, 跨 3 seeds 取 mean±std 比较; 判读表见 RUNBOOK_wave01.md §4）
#
# 依赖:
#   · 参数名按【接口约定】, 待洗澄明修复 train_imagenet.py 后生效
#     （--train_models / --train_split / --eval_split / --data_seed 等;
#       当前代码硬编码 4md 模型集, 故 --train_models 必须显式传入）
#   · imagenet-mini 需有 train/ 子目录（预检见 RUNBOOK §3）
#   · [数据独立性裁决] --data_seed 随 run 变（1042/2042/3042）, --seed 控制
#     训练随机性（42/1042/2042）; 评估集种子固定（协议一致性, train_imagenet.py 内部）
# ================================================================
#SBATCH --job-name=wave0c_4m4md
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-wave0c-%j.out

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

RETRAIN_3M="${RETRAIN_3M:-1}"

# 训练一个 subset 的 3 个 seeds（串行）
# $1=tag（日志与 ensemble 文件名用）  $2=逗号分隔模型列表  $3=run 目录前缀
# [数据独立性裁决] --seed 只控制训练随机性（task 顺序等）; --data_seed 控制元训练
# 数据采样, 随 run 变化（依据 analysis_robustness.md §1.2: 旧 3m 三 run 数据完全
# 相同导致 std≈5pp 被低估）; 评估集种子固定由 train_imagenet.py 内部保证
run_seeds () {
    local tag="$1" models="$2" outbase="$3"
    for i in 0 1 2; do
        seed=$((42 + i * 1000))
        data_seed=$((1042 + i * 1000))
        echo "启动 ${tag} 实验 $i (seed=$seed, data_seed=$data_seed)..."
        python train_imagenet.py \
            --train_models "${models}" \
            --train_split train \
            --eval_split val \
            --data_seed "${data_seed}" \
            --epochs 50 \
            --inner_steps 3 \
            --meta_lr 0.01 \
            --epsilon 0.05 \
            --support_size 8 \
            --query_size 4 \
            --seed "${seed}" \
            --save_dir "./${outbase}_run${i}" \
            > "slurm-${tag}-run${i}.out" 2>&1
        echo "${tag} 实验 $i 完成（日志: slurm-${tag}-run${i}.out）"
    done
}

# 对 3 个 run 的 uap_init 取平均, 生成 ensemble checkpoint
# $1=tag  $2=run 目录前缀（同时是 ensemble 输出目录）
merge_ensemble () {
    local tag="$1" outbase="$2"
    mkdir -p "./${outbase}"
    python - "./${outbase}" "${tag}" <<'PYEOF'
import sys
import torch
import numpy as np

outbase, tag = sys.argv[1], sys.argv[2]
uaps = []
for i in range(3):
    path = f"{outbase}_run{i}/meta_uap_imagenet_epoch50.pt"
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        uap = ckpt["uap_init"].float().numpy()
        uaps.append(uap)
        print(f"  加载 {tag} run{i}: uap_norm={np.abs(uap).max():.4f}")
    except Exception as e:
        print(f"  跳过 {tag} run{i}: {e}")

if not uaps:
    print(f"[FATAL] {tag} 没有成功运行的实验, 无法生成 ensemble")
    sys.exit(1)

avg = np.mean(uaps, axis=0)
out_path = f"{outbase}/meta_uap_{tag}_ensemble3.pt"
torch.save({"uap_init": torch.from_numpy(avg), "num_runs": len(uaps),
            "protocol": "fixed-train-split"}, out_path)
print(f"  集成 UAP 已保存: {out_path}（uap_norm={np.abs(avg).max():.4f}）")
PYEOF
}

echo "============================================"
echo "Wave 0c: ImageNet 3m_fixed / 4m / 4md 修复协议补训"
echo "  RETRAIN_3M=${RETRAIN_3M}"
echo "============================================"

if [ "${RETRAIN_3M}" = "1" ]; then
    run_seeds "imagenet_3m_fixed" "resnet18,resnet34,resnet50" "checkpoints_imagenet_3m_fixed"
    merge_ensemble "imagenet_3m_fixed" "checkpoints_imagenet_3m_fixed"
else
    echo "[跳过] RETRAIN_3M=0, 不重训 3m（梯度矩阵与 0a 重跑将缺少修复协议 3m checkpoint）"
fi

run_seeds "imagenet_4m" "resnet18,resnet34,resnet50,vgg16_bn" "checkpoints_imagenet_4m"
merge_ensemble "imagenet_4m" "checkpoints_imagenet_4m"

run_seeds "imagenet_4md" "resnet18,resnet34,resnet50,densenet121" "checkpoints_imagenet_4md"
merge_ensemble "imagenet_4md" "checkpoints_imagenet_4md"

echo "所有实验完成!"
echo "ensemble checkpoints:"
echo "  checkpoints_imagenet_3m_fixed/meta_uap_imagenet_3m_fixed_ensemble3.pt（RETRAIN_3M=1 时）"
echo "  checkpoints_imagenet_4m/meta_uap_imagenet_4m_ensemble3.pt"
echo "  checkpoints_imagenet_4md/meta_uap_imagenet_4md_ensemble3.pt"
