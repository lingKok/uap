#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --gpus=1

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

python ablation.py
