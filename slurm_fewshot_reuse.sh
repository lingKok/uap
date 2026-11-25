#!/bin/bash
#SBATCH --job-name=fewshot_reuse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm-fewshot-reuse-%A.out
#SBATCH --error=slurm-fewshot-reuse-%A.err

# 与能跑通的 X1 并行脚本一致的环境初始化（关键：先 source .bashrc 加载 conda）
cd $SLURM_SUBMIT_DIR
source /HOME/scz0bqb/.bashrc 2>/dev/null || true
conda activate meta_uap

echo "=== 实验 (A) LWS 复用部署 — 4 模型各占 1 卡并行 ==="
echo "提交时间: $(date)"
echo "节点: $(hostname)"

# 权重已在 fewshot 实验中确认缓存齐全，跳过预检

mkdir -p results_fewshot_reuse
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    DEVS=("0" "1" "2" "3")
else
    IFS=',' read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
fi

MODELS=("resnet50" "vgg16_bn" "resnet101" "mobilenet_v2")
for i in "${!MODELS[@]}"; do
    m="${MODELS[$i]}"
    echo "启动 $m on GPU ${DEVS[$i]}"
    CUDA_VISIBLE_DEVICES=${DEVS[$i]} python run_fewshot_reuse.py \
        --surrogate "$m" --out_tag "$m" --seeds 42,1342,2342 \
        > log_reuse_${m}.txt 2>&1 &
done

echo "等待 4 个模型作业完成（后台并发）..."
wait
echo "=== 全部 surrogate 完成: $(date) ==="
