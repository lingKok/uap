#!/bin/bash
#SBATCH --job-name=meta_imagenet_4md
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

# 单卡串行，3个独立实验（不同随机种子）
# 使用4个训练模型（resnet18/34/50/densenet121），去掉vgg16_bn
for i in 0 1 2; do
    seed=$((42 + i * 1000))
    echo "启动实验 $i (seed=$seed)..."
    python train_imagenet.py \
        --epochs 50 --inner_steps 3 --meta_lr 0.01 --epsilon 0.05 \
        --support_size 8 --query_size 4 --seed $seed \
        --save_dir ./checkpoints_imagenet_4md_run${i} \
        > slurm-imagenet-4md-run${i}.out 2>&1
    echo "实验 $i 完成"
done

echo "所有实验完成!"

# 合并结果：取 3 次运行的 UAP 平均
mkdir -p ./checkpoints_imagenet_4md
python -c "
import torch
import numpy as np

uaps = []
for i in range(3):
    path = f'./checkpoints_imagenet_4md_run{i}/meta_uap_imagenet_epoch50.pt'
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
    torch.save({'uap_init': avg_tensor, 'num_runs': len(uaps)}, './checkpoints_imagenet_4md/meta_uap_imagenet_4md_ensemble3.pt')
    print(f'\n集成 UAP 已保存: ./checkpoints_imagenet_4md/meta_uap_imagenet_4md_ensemble3.pt')
    print(f'  uap_norm={np.abs(avg_uap).max():.4f}')
else:
    print('没有成功运行的实验')
"
