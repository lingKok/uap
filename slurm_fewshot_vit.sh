#!/bin/bash
#SBATCH --job-name=fewshot_vit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-fewshot-vit-%A.out
#SBATCH --error=slurm-fewshot-vit-%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6

# ===== 环境初始化（对齐 X1 并行脚本，关键：先 source .bashrc 才能用 conda）=====
cd $SLURM_SUBMIT_DIR
source /HOME/scz0bqb/.bashrc 2>/dev/null || true
conda activate meta_uap

# ===== 预检：5 模型权重缓存是否齐全（计算节点离线，缺失则秒退并给预热命令）=====
python - <<'PY'
import sys, torchvision.models as M
need = {
    "resnet50": M.resnet50, "vgg16_bn": M.vgg16_bn, "resnet101": M.resnet101,
    "mobilenet_v2": M.mobilenet_v2, "vit_b_16": M.vit_b_16,
}
missing = []
for n, fn in need.items():
    try:
        fn(weights="IMAGENET1K_V1")
    except Exception as e:
        missing.append(n)
if missing:
    print("FATAL: 以下模型权重缓存缺失（计算节点离线，无法下载）:", missing)
    print('请在【登录节点】执行预热（有网）：')
    print("  python -c \"import torchvision; [getattr(torchvision.models, n)(weights='IMAGENET1K_V1') "
          "for n in ['resnet50','vgg16_bn','resnet101','mobilenet_v2','vit_b_16']]\"")
    sys.exit(1)
print("PREFLIGHT OK: 5 模型权重缓存齐全")
PY
if [ $? -ne 0 ]; then exit 1; fi

# ===== 单卡串行：5 surrogate 顺序跑（只占 1 张卡，排队更快；总时长≈5×单模型）=====
SURROGATES="resnet50 vgg16_bn resnet101 mobilenet_v2 vit_b_16"
MODELS_ARG="resnet50,vgg16_bn,resnet101,mobilenet_v2,vit_b_16"

for s in $SURROGATES; do
  echo ">>> 开始 surrogate=$s ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES=0 python -u run_fewshot_reuse.py \
    --dataset imagenet --surrogate $s --out_tag $s \
    --models "$MODELS_ARG" --out_dir results_fewshot_vit \
    > log_vit_$s.txt 2>&1
  echo "<<< 完成 surrogate=$s ($(date +%H:%M:%S)) 退出码=$?"
done

echo "ALL DONE: results_fewshot_vit/ 已生成 15 个 json（5 模型 × 3 seed）"
