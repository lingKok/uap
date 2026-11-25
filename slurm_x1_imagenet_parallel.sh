#!/bin/bash
#SBATCH --job-name=x1_imagenet_par
#SBATCH --output=slurm-x1-imagenet-par-%j.out
#SBATCH --error=slurm-x1-imagenet-par-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00

cd $SLURM_SUBMIT_DIR
source /HOME/scz0bqb/.bashrc 2>/dev/null || true
conda activate meta_uap

echo "=== X1 ImageNet 跨预算反转 — 每模型1卡并行（4 卡）==="
echo "提交时间: $(date)"
echo "节点: $(hostname)"

# [预检] 计算节点不能联网，需登录节点预热 torchvision 权重（仅 4 个实际模型）
prefix=("resnet50" "vgg16_bn" "resnet101" "mobilenet_v2")
python - <<'PY'
import torch, sys
from pathlib import Path
prefix = {"resnet50":"resnet50","vgg16_bn":"vgg16_bn","resnet101":"resnet101","mobilenet_v2":"mobilenet_v2"}
ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
missing = [k for k in prefix if not (ckpt_dir.is_dir() and list(ckpt_dir.glob(prefix[k]+"*.pth")))]
if missing:
    print(f"[FATAL] torchvision 权重缓存缺失: {missing}")
    print(f"缓存目录: {ckpt_dir}")
    print("计算节点不能联网，请在登录节点执行预热后重新提交:")
    print("  python -c \"import torchvision; [getattr(torchvision.models, n)(weights='IMAGENET1K_V1') for n in ['resnet50','vgg16_bn','resnet101','mobilenet_v2']]\"")
    sys.exit(1)
print(f"[预检通过] torchvision 权重缓存齐全: {ckpt_dir}")
PY

# 4 个模型，各占 1 张卡并发（CUDA_VISIBLE_DEVICES 取 SLURM 分配的设备列表）
MODELS=("resnet50" "vgg16_bn" "resnet101" "mobilenet_v2")
COMMON="--dataset imagenet --eps_train 0.05 --eps_targets 0.1,0.2,0.3 --opt_few 200 --opt_many 2000 --steps_train 200 --steps_few 200 --steps_many 500 --lr 0.01 --seeds 42,1342,2342 --out_dir results_x1"

mkdir -p results_x1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    DEVS=("0" "1" "2" "3")
else
    IFS=',' read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
fi

for i in "${!MODELS[@]}"; do
    m="${MODELS[$i]}"
    echo "启动 $m on GPU ${DEVS[$i]}"
    CUDA_VISIBLE_DEVICES=${DEVS[$i]} python x1_cross_budget.py $COMMON \
        --models "$m" --out_tag "$m" > results_x1/log_${m}.txt 2>&1 &
done

echo "等待 4 个模型作业完成（后台并发）..."
wait
echo "全部模型作业完成: $(date)"

# 跨模型合并聚合
python merge_x1.py --dataset imagenet --out_dir results_x1

echo "=== X1 ImageNet 并行完成: $(date) ==="
