#!/bin/bash
#SBATCH --job-name=x1_imagenet
#SBATCH --output=slurm-x1-imagenet-%j.out
#SBATCH --error=slurm-x1-imagenet-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00

cd $SLURM_SUBMIT_DIR
source /HOME/scz0bqb/.bashrc 2>/dev/null || true
conda activate meta_uap

echo "=== X1 ImageNet 跨预算反转可行性实验 ==="
echo "提交时间: $(date)"
echo "节点: $(hostname)"

# [预检] 计算节点不能联网，需登录节点预热 torchvision 权重（RUNBOOK 预检项）
# X1 实际只用 4 个模型，预检仅查这 4 个，避免无关模型缓存缺失误杀
# 期望权重前缀：resnet50/vgg16_bn/resnet101/mobilenet_v2
python - <<'PY'
import torch, torchvision, os, sys
from pathlib import Path
prefix = {
    "resnet50":"resnet50",
    "vgg16_bn":"vgg16_bn",
    "resnet101":"resnet101",
    "mobilenet_v2":"mobilenet_v2",
}
ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
missing = []
for k in prefix:
    if not (ckpt_dir.is_dir() and list(ckpt_dir.glob(prefix[k] + "*.pth"))):
        missing.append(k)
if missing:
    print(f"[FATAL] torchvision 权重缓存缺失: {missing}")
    print(f"缓存目录: {ckpt_dir}")
    print("计算节点不能联网，请在登录节点执行预热后重新提交:")
    print("  python -c \"import torchvision; [getattr(torchvision.models, n)(weights='IMAGENET1K_V1') for n in ['resnet50','vgg16_bn','resnet101','mobilenet_v2']]\"")
    sys.exit(1)
print(f"[预检通过] torchvision 权重缓存齐全: {ckpt_dir}")
PY

python x1_cross_budget.py \
  --dataset imagenet \
  --eps_train 0.05 \
  --eps_targets 0.1,0.2,0.3 \
  --opt_few 200 \
  --opt_many 2000 \
  --steps_train 200 \
  --steps_few 200 \
  --steps_many 500 \
  --lr 0.01 \
  --seeds 42,1342,2342 \
  --models "resnet50,vgg16_bn,resnet101,mobilenet_v2" \
  --out_dir results_x1

echo "=== X1 ImageNet 完成: $(date) ==="
