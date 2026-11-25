#!/bin/bash
#SBATCH --job-name=meta_imagenet
#SBATCH --gpus=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

# 8 卡并行，每卡一个实验（不同随机种子）
for i in 0 1 2 3 4 5 6 7; do
    seed=$((42 + i * 1000))
    echo "启动实验 $i (seed=$seed)..."
    CUDA_VISIBLE_DEVICES=$i python train_imagenet.py \
        --epochs 50 --inner_steps 3 --meta_lr 0.01 --epsilon 0.05 \
        --support_size 8 --query_size 4 --seed $seed \
        --save_dir ./checkpoints_imagenet_run${i} \
        > slurm-imagenet-run${i}.out 2>&1 &
done

echo "已启动 8 个并行实验，等待完成..."
wait
echo "所有实验完成!"

# 合并结果：取 8 次运行的 UAP 平均
python -c "
import torch
import numpy as np

uaps = []
for i in range(8):
    path = f'./checkpoints_imagenet_run{i}/meta_uap_imagenet_epoch50.pt'
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        uap = ckpt['uap_init'].numpy()
        uaps.append(uap)
        print(f'  加载 run{i}: uap_norm={np.abs(uap).max():.4f}')
    except Exception as e:
        print(f'  跳过 run{i}: {e}')

if uaps:
    avg_uap = np.mean(uaps, axis=0)
    avg_tensor = torch.from_numpy(avg_uap)
    torch.save({'uap_init': avg_tensor, 'num_runs': len(uaps)}, './checkpoints_imagenet/meta_uap_imagenet_ensemble8.pt')
    print(f'\n集成 UAP 已保存: ./checkpoints_imagenet/meta_uap_imagenet_ensemble8.pt')
    print(f'  uap_norm={np.abs(avg_uap).max():.4f}')
else:
    print('没有成功运行的实验')
"
