#!/bin/bash
#SBATCH --job-name=suppl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=log_suppl_%x.out

# 补充实验四合一（串行，单卡）。旋钮均可用环境变量前缀覆盖：
#   EXPS="epsw bsweep" sbatch slurm_supplementary.sh
source /HOME/scz0bqb/.bashrc
conda activate meta_uap
cd $SLURM_SUBMIT_DIR

set -e
EXPS="${EXPS:-epsw bsweep ksweep l2}"
SURROGATE="${SURROGATE:-resnet50}"
SEEDS="${SEEDS:-42,1342,2342}"
EVAL_N="${EVAL_N:-2000}"
EVAL_BATCH="${EVAL_BATCH:-64}"
OUT_DIR="${OUT_DIR:-results_suppl}"

echo "EXPS       = $EXPS"
echo "SURROGATE  = $SURROGATE | SEEDS = $SEEDS"
echo "EVAL_N     = $EVAL_N | EVAL_BATCH = $EVAL_BATCH | OUT_DIR = $OUT_DIR"

# 预检：依赖文件 + GPU
python -c "import torch, torchvision; assert torch.cuda.is_available()" \
  || { echo "FATAL: cuda/依赖不可用"; exit 1; }

for EXP in $EXPS; do
  echo "===== EXP $EXP 开始 $(date +%H:%M:%S) ====="
  python run_supplementary.py \
    --exp "$EXP" \
    --surrogate "$SURROGATE" \
    --seeds "$SEEDS" \
    --eval_n "$EVAL_N" \
    --eval_batch "$EVAL_BATCH" \
    --out_dir "$OUT_DIR" || { echo "FATAL: exp=$EXP 失败"; exit 1; }
done
echo "ALL DONE $(date +%H:%M:%S)"
