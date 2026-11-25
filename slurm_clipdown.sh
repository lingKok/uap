#!/bin/bash
#SBATCH --job-name=clipdown
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-clipdown-%A.out
#SBATCH --error=slurm-clipdown-%A.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=6

# ===== 环境初始化（对齐 X1/sota：先 source .bashrc 才能用 conda）=====
cd $SLURM_SUBMIT_DIR
source /HOME/scz0bqb/.bashrc 2>/dev/null || true
conda activate meta_uap

# ===== 预检 1：5 个评估模型权重缓存 =====
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
    sys.exit(1)
print("PREFLIGHT OK: 5 模型权重缓存齐全")
PY
if [ $? -ne 0 ]; then exit 1; fi

# ===== 预检 2：依赖文件齐备 =====
for f in run_clipdown.py sota_modern.py sota_uap_baselines.py fewshot_uap_method.py \
         third_party/__init__.py \
         third_party/dmuap_vendor/attack_min_x_theta_adam.py \
         third_party/dmuap_vendor/PGDL2.py third_party/dmuap_vendor/utils_dm.py \
         third_party/dmuap_vendor/device.py; do
  if [ ! -f "$f" ]; then
    echo "FATAL: 缺文件 $f"
    echo "请按 upload_checklist_clipdown.md 补齐后重提"
    exit 1
  fi
done
echo "PREFLIGHT OK: 依赖文件齐全"

# ===== 旋钮（全部可被环境变量覆盖：VAR=val sbatch slurm_clipdown.sh）=====
SURROGATES=${SURROGATES:-"resnet50 resnet101 vit_b_16"}
MODELS_ARG=${MODELS_ARG:-"resnet50,vgg16_bn,resnet101,mobilenet_v2,vit_b_16"}
SEEDS=${SEEDS:-42,1342,2342}
OUT_DIR=${OUT_DIR:-results_clipdown}

# clip-down 的起点：必须是最大预算，否则「裁剪到更小」就无意义
EPS_HI=${EPS_HI:-0.3}
EPS_TARGETS=${EPS_TARGETS:-0.1,0.2,0.3}

METHODS=${METHODS:-"direct,tsc,uap2017,dmuap"}
STEPS=${STEPS:-200}
EVAL_N=${EVAL_N:-2000}
# EVAL_BATCH 调大只提速、不改 FR（模型 eval() 模式，BN 用 running stats）。
# 共 4 方法 × 3 预算 = 12 次评估/seed，评估是主要耗时 → 建议 64。
EVAL_BATCH=${EVAL_BATCH:-64}

DM_STEPS=${DM_STEPS:-60}
DM_THETA=${DM_THETA:-10}
DM_CC=${DM_CC:-10}
DM_BATCH=${DM_BATCH:-50}

EXTRA="--eps_hi $EPS_HI --eps_targets $EPS_TARGETS --methods $METHODS \
--steps $STEPS --seeds $SEEDS --eval_n $EVAL_N --eval_batch $EVAL_BATCH \
--dm_steps $DM_STEPS --dm_theta_steps $DM_THETA --dm_cc $DM_CC --dm_batch $DM_BATCH"
if [ "${SKIP_DM:-0}" = "1" ]; then EXTRA="$EXTRA --skip_modern"; fi

echo "================================================================"
echo "SURROGATES  = $SURROGATES"
echo "METHODS     = $METHODS"
echo "EPS_HI      = $EPS_HI  (训练预算) ; EPS_TARGETS = $EPS_TARGETS (评估预算)"
echo "OUT_DIR     = $OUT_DIR | SEEDS = $SEEDS | STEPS = $STEPS"
echo "EVAL_N      = $EVAL_N | EVAL_BATCH = $EVAL_BATCH"
echo "DM_ARGS     = --dm_steps $DM_STEPS --dm_theta_steps $DM_THETA --dm_cc $DM_CC (skip=${SKIP_DM:-0})"
echo "================================================================"

FAIL=0
for S in $SURROGATES; do
  echo ""
  echo ">>> [$(date +%H:%M:%S)] surrogate = $S"
  python -u run_clipdown.py \
      --dataset imagenet \
      --surrogate "$S" \
      --models "$MODELS_ARG" \
      --out_dir "$OUT_DIR" \
      --out_tag "$S" \
      $EXTRA
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "FAILED: surrogate=$S 退出码=$RC"
    FAIL=1
  fi
done

if [ $FAIL -ne 0 ]; then
  echo ""
  echo "!!!! 有 surrogate 失败，请检查上面的 traceback !!!!"
  exit 1
fi

echo ""
echo "ALL DONE [$(date +%H:%M:%S)]"
ls -l "$OUT_DIR" | tail -20
