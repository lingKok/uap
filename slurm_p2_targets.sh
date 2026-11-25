#!/bin/bash
#SBATCH --job-name=p2tgt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=log_p2_%x.out

# P2: 现代架构评估目标扩容（ConvNeXt-T / Swin-T / EffNet-B0）
# resnet50 surrogate x 3 seeds x 4 方法（reuse_lws/reuse_tsc/direct/tsc）。
source /HOME/scz0bqb/.bashrc
conda activate meta_uap
cd $SLURM_SUBMIT_DIR

set -e
SURROGATE="${SURROGATE:-resnet50}"
SEEDS="${SEEDS:-42,1342,2342}"
EVAL_N="${EVAL_N:-2000}"
EVAL_BATCH="${EVAL_BATCH:-64}"
OUT_DIR="${OUT_DIR:-results_p2}"

echo "SURROGATE  = $SURROGATE | SEEDS = $SEEDS"
echo "EVAL_N     = $EVAL_N | EVAL_BATCH = $EVAL_BATCH | OUT_DIR = $OUT_DIR"

python -c "import torch, torchvision; assert torch.cuda.is_available()" \
  || { echo "FATAL: cuda/依赖不可用"; exit 1; }

python run_p2_targets.py \
  --surrogate "$SURROGATE" \
  --seeds "$SEEDS" \
  --eval_n "$EVAL_N" \
  --eval_batch "$EVAL_BATCH" \
  --out_dir "$OUT_DIR"

echo "ALL DONE $(date +%H:%M:%S)"
