#!/bin/bash
#SBATCH --job-name=fewshot_sota
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-fewshot-sota-%A.out
#SBATCH --error=slurm-fewshot-sota-%A.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=6

# ===== 环境初始化（对齐 X1：先 source .bashrc 才能用 conda）=====
cd $SLURM_SUBMIT_DIR
source /HOME/scz0bqb/.bashrc 2>/dev/null || true
conda activate meta_uap

# ===== 预检：5 个评估模型权重缓存是否齐全（计算节点离线，缺失则秒退）=====
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
    except Exception:
        missing.append(n)
if missing:
    print("FATAL: 权重缓存缺失（计算节点离线）:", missing)
    print("请在登录节点预热：python -c \"import torchvision; "
          "[getattr(torchvision.models, n)(weights='IMAGENET1K_V1') for n in "
          "['resnet50','vgg16_bn','resnet101','mobilenet_v2','vit_b_16']]\"")
    sys.exit(1)
print("PREFLIGHT OK: 5 模型权重缓存齐全")
PY
if [ $? -ne 0 ]; then exit 1; fi

# ===== 预检：DM-UAP 官方代码是否已上传（缺文件会到运行时才报 ImportError）=====
for f in sota_modern.py third_party/__init__.py \
         third_party/dmuap_vendor/attack_min_x_theta_adam.py \
         third_party/dmuap_vendor/PGDL2.py third_party/dmuap_vendor/utils_dm.py \
         third_party/dmuap_vendor/device.py; do
  if [ ! -f "$f" ]; then
    echo "FATAL: 缺文件 $f"
    echo "请上传 sota_modern.py 与整个 third_party/ 目录（见 upload_checklist_fewshot_sota.md）"
    exit 1
  fi
done
echo "PREFLIGHT OK: DM-UAP vendor 文件齐全"

# ===== 单卡串行 =====
# 默认只跑 3 个有 headroom 的 surrogate（resnet50/resnet101/vit_b_16）——
# vgg16_bn 与 mobilenet_v2 在 A-ViT 已证实被 FR 天花板屏蔽（无提升空间），
# 作为 surrogate 加入只会白烧算力；它们仍作为【评估 target】出现在 5 列矩阵里。
# 如需全 5 个 surrogate：SURROGATES="resnet50 vgg16_bn resnet101 mobilenet_v2 vit_b_16"
SURROGATES=${SURROGATES:-"resnet50 resnet101 vit_b_16"}
MODELS_ARG=${MODELS_ARG:-"resnet50,vgg16_bn,resnet101,mobilenet_v2,vit_b_16"}
OUT_DIR=${OUT_DIR:-results_fewshot_sota}
SEEDS=${SEEDS:-42,1342,2342}
STEPS=${STEPS:-200}
CEIL_STEPS=${CEIL_STEPS:-500}
# 评估是主要耗时来源（评测集 × 5 模型 × 30 次调用）。冒烟时压小 EVAL_N；
# 正式实验保持 2000。EVAL_BATCH 调大只提速、不改 FR（模型 eval() 模式，BN 用 running stats）。
EVAL_N=${EVAL_N:-2000}
EVAL_BATCH=${EVAL_BATCH:-0}

# DM-UAP（AAAI'25）算力旋钮：默认 60 次外层更新（≈1260 次图像回传，偏袒基线）。
# 严格外层步数对齐：DM_STEPS=200 sbatch ...；先看别的方法：SKIP_DM=1 sbatch ...
DM_STEPS=${DM_STEPS:-60}
DM_THETA=${DM_THETA:-10}
DM_CC=${DM_CC:-10}
DM_BATCH=${DM_BATCH:-50}

EXTRA="--steps $STEPS --steps_ceiling $CEIL_STEPS --seeds $SEEDS --dm_theta_steps $DM_THETA --dm_cc $DM_CC --dm_batch $DM_BATCH --eval_n $EVAL_N --eval_batch $EVAL_BATCH"
if [ "${SKIP_DM:-0}" = "1" ]; then EXTRA="$EXTRA --skip_modern"; fi

echo "SURROGATES = $SURROGATES"
echo "OUT_DIR    = $OUT_DIR | SEEDS = $SEEDS | STEPS = $STEPS | CEIL_STEPS = $CEIL_STEPS"
echo "EVAL_N     = $EVAL_N | EVAL_BATCH = $EVAL_BATCH（0=自动）"
echo "DM_ARGS    = --dm_steps $DM_STEPS --dm_theta_steps $DM_THETA --dm_cc $DM_CC (skip=${SKIP_DM:-0})"

FAILED=""
for s in $SURROGATES; do
  echo ">>> 开始 surrogate=$s ($(date +%H:%M:%S))"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python -u run_fewshot_sota.py \
    --dataset imagenet --surrogate $s --out_tag $s \
    --models "$MODELS_ARG" --out_dir "$OUT_DIR" --dm_steps $DM_STEPS $EXTRA \
    > log_sota_$s.txt 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ]; then
    echo "!! surrogate=$s 失败（退出码=$rc，耗时 ${dt}s）。log_sota_$s.txt 末尾 40 行："
    tail -40 log_sota_$s.txt
    FAILED="$FAILED $s"
  else
    echo "<<< 完成 surrogate=$s（耗时 ${dt}s）"
  fi
done

if [ -n "$FAILED" ]; then
  echo "======================================================"
  echo "!! PARTIAL FAILURE —— 失败的 surrogate:$FAILED"
  echo "!! $OUT_DIR 里只有成功的部分，切勿当作完整结果分析"
  exit 1
fi
echo "ALL DONE: $OUT_DIR 已生成全部（$SURROGATES）的结果"
