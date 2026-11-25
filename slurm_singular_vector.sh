#!/bin/bash
#SBATCH --job-name=svduap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=log_svduap_%x.out

# P1: singular-vector baseline（Khrulkov-Babu SVD-UAP，零优化 budget-agnostic 竞争者）
# 单卡串行；无训练循环，成本以评估为主（4 配置 x 3 eps x 3 seeds）。
source /HOME/scz0bqb/.bashrc
conda activate meta_uap
cd $SLURM_SUBMIT_DIR

set -e
SURROGATE="${SURROGATE:-resnet50}"
SEEDS="${SEEDS:-42,1342,2342}"
EVAL_N="${EVAL_N:-2000}"
EVAL_BATCH="${EVAL_BATCH:-64}"
SVK_KS="${SVK_KS:-64,200}"
SVK_LAYERS="${SVK_LAYERS:-layer2,layer4}"
OUT_DIR="${OUT_DIR:-results_svduap}"

echo "SURROGATE  = $SURROGATE | SEEDS = $SEEDS"
echo "EVAL_N     = $EVAL_N | EVAL_BATCH = $EVAL_BATCH | OUT_DIR = $OUT_DIR"
echo "svd_ks     = $SVK_KS | svd_layers = $SVK_LAYERS"

# 预检：依赖文件 + GPU
python -c "import torch, torchvision; assert torch.cuda.is_available()" \
  || { echo "FATAL: cuda/依赖不可用"; exit 1; }

python run_singular_vector.py \
  --surrogate "$SURROGATE" \
  --seeds "$SEEDS" \
  --eval_n "$EVAL_N" \
  --eval_batch "$EVAL_BATCH" \
  --svd_ks "$SVK_KS" \
  --svd_layers "$SVK_LAYERS" \
  --out_dir "$OUT_DIR"

echo "ALL DONE $(date +%H:%M:%S)"
