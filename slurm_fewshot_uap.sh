#!/bin/bash
#SBATCH --job-name=fewshot_uap_par
#SBATCH --output=slurm-fewshot-uap-par-%j.out
#SBATCH --error=slurm-fewshot-uap-par-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

cd $SLURM_SUBMIT_DIR
source /HOME/scz0bqb/.bashrc 2>/dev/null || true
conda activate meta_uap

echo "=== Few-shot UAP 方法化对比 — 每 surrogate 1 卡并行（4 卡）==="
echo "提交时间: $(date)"
echo "节点: $(hostname)"

# [预检] 计算节点不能联网，需登录节点预热 torchvision 权重（仅 4 个实际模型）
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

# 4 个 surrogate，各占 1 张卡并发（CUDA_VISIBLE_DEVICES 取 SLURM 分配的设备列表）
SURROGATES=("resnet50" "vgg16_bn" "resnet101" "mobilenet_v2")
COMMON="--dataset imagenet --models resnet50,vgg16_bn,resnet101,mobilenet_v2 --eps_targets 0.1,0.2,0.3 --eps_w 0.05 --few_k 200 --many 2000 --steps 200 --steps_ceiling 500 --refine 50 --lr 0.01 --seeds 42,1342,2342 --out_dir results_fewshot"

mkdir -p results_fewshot
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    DEVS=("0" "1" "2" "3")
else
    IFS=',' read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
fi

for i in "${!SURROGATES[@]}"; do
    s="${SURROGATES[$i]}"
    echo "启动 surrogate=$s on GPU ${DEVS[$i]}"
    CUDA_VISIBLE_DEVICES=${DEVS[$i]} python run_fewshot_uap.py $COMMON \
        --surrogate "$s" --out_tag "$s" > results_fewshot/log_${s}.txt 2>&1 &
done

echo "等待 4 个 surrogate 作业完成（后台并发）..."
wait
echo "全部 surrogate 作业完成: $(date)"

# 跨 surrogate 合并聚合
python merge_fewshot.py --dataset imagenet --out_dir results_fewshot

echo "=== Few-shot UAP 并行完成: $(date) ==="
