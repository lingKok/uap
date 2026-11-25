#!/bin/bash
#SBATCH --job-name=x1_cifar
#SBATCH --output=slurm-x1-cifar-%j.out
#SBATCH --error=slurm-x1-cifar-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR
source /HOME/scz0bqb/.bashrc 2>/dev/null || true
conda activate meta_uap

echo "=== X1 CIFAR 跨预算反转可行性实验 ==="
echo "提交时间: $(date)"
echo "节点: $(hostname)"

python x1_cross_budget.py \
  --dataset cifar \
  --eps_train 0.05 \
  --eps_targets 0.1,0.2,0.3 \
  --opt_few 200 \
  --opt_many 2000 \
  --steps_train 200 \
  --steps_few 200 \
  --steps_many 500 \
  --lr 0.01 \
  --seeds 42,1342,2342 \
  --models auto \
  --out_dir results_x1

echo "=== X1 CIFAR 完成: $(date) ==="
