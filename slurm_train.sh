#!/bin/bash
#SBATCH --job-name=meta_train
#SBATCH --gpus=1

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

python train.py --epochs 50 --inner_steps 5 --meta_lr 0.01 --epsilon 0.05 --support_size 256 --query_size 64
