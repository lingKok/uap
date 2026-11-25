#!/bin/bash
#SBATCH --job-name=cifarctl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-cifarctl-%A.out
#SBATCH --error=slurm-cifarctl-%A.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=6

# CIFAR-10 侧对照：三策略（pereps/clipdown/scaleup）× 4 方法 × 3 surrogate × 3 seed
# 预注册预测：CIFAR 全饱和 → scaleup 优势消失、三策略贴近（与 ImageNet 相反）

# ===== 环境初始化（对齐既有 launcher）=====
cd $SLURM_SUBMIT_DIR
source /HOME/scz0bqb/.bashrc 2>/dev/null || true
conda activate meta_uap

# ===== 预检 1：依赖文件齐备 =====
for f in run_cifar_control.py sota_modern.py sota_uap_baselines.py fewshot_uap_method.py \
         third_party/__init__.py \
         third_party/dmuap_vendor/attack_min_x_theta_adam.py \
         third_party/dmuap_vendor/PGDL2.py third_party/dmuap_vendor/utils_dm.py \
         third_party/dmuap_vendor/device.py; do
  if [ ! -f "$f" ]; then
    echo "FATAL: 缺文件 $f"
    exit 1
  fi
done
echo "PREFLIGHT OK: 依赖文件齐全"

# ===== 预检 2：CIFAR checkpoint（checkpoints/{model}.pt，train.py 约定）=====
# PRETRAIN=1 sbatch ... 可自动补训缺失权重（每次全量训练 ~1-3 min/epoch × 100 epochs）
python - <<'PY'
import sys, os
from config import Config
need = Config().meta_train_models + Config().meta_test_models
missing = [n for n in need if not os.path.exists(f"checkpoints/{n}.pt")]
if missing:
    print("FATAL: CIFAR checkpoint 缺失:", missing)
    print("修复: PRETRAIN=1 sbatch slurm_cifar_control.sh  （自动补训）")
    print("   或: python pretrain_models.py --skip_existing")
    sys.exit(1)
print("PREFLIGHT OK: 8 个 CIFAR checkpoint 齐全")
PY
if [ $? -ne 0 ]; then
  if [ "${PRETRAIN:-0}" = "1" ]; then
    echo ">>> PRETRAIN=1：自动补训缺失权重"
    python -u pretrain_models.py --skip_existing || exit 1
  else
    exit 1
  fi
fi

# ===== 预检 3：CIFAR-10 数据集 =====
python - <<'PY'
import sys
from config import Config
from utils.data import get_cifar10_samples
try:
    imgs, lbls = get_cifar10_samples(Config().data_root, num_samples=8, split="test", seed=42)
    print(f"PREFLIGHT OK: CIFAR-10 数据可读 shape={tuple(imgs.shape)}")
except Exception as e:
    print("FATAL: CIFAR-10 不可读:", e, "| DATA_ROOT =", Config().data_root)
    sys.exit(1)
PY
if [ $? -ne 0 ]; then exit 1; fi

# ===== 旋钮（VAR=val sbatch slurm_cifar_control.sh 覆盖）=====
SURROGATES=${SURROGATES:-"resnet50 vgg16_bn mobilenet_v2"}
MODELS_ARG=${MODELS_ARG:-"auto"}
SEEDS=${SEEDS:-42,1342,2342}
OUT_DIR=${OUT_DIR:-results_cifar_control}
METHODS=${METHODS:-"direct,tsc,uap2017,dmuap"}
EPS_W=${EPS_W:-0.05}
EPS_HI=${EPS_HI:-0.3}
EPS_TARGETS=${EPS_TARGETS:-0.1,0.2,0.3}
STEPS=${STEPS:-200}
EVAL_N=${EVAL_N:-2000}
EVAL_BATCH=${EVAL_BATCH:-512}
DM_STEPS=${DM_STEPS:-60}
DM_THETA=${DM_THETA:-10}
DM_CC=${DM_CC:-10}

EXTRA="--eps_w $EPS_W --eps_hi $EPS_HI --eps_targets $EPS_TARGETS --methods $METHODS \
--steps $STEPS --seeds $SEEDS --eval_n $EVAL_N --eval_batch $EVAL_BATCH \
--dm_steps $DM_STEPS --dm_theta_steps $DM_THETA --dm_cc $DM_CC"
if [ "${SKIP_DM:-0}" = "1" ]; then EXTRA="$EXTRA --skip_modern"; fi

echo "================================================================"
echo "SURROGATES  = $SURROGATES"
echo "METHODS     = $METHODS"
echo "STRATEGIES  = pereps | clipdown(train@$EPS_HI→clip) | scaleup(train@$EPS_W→scale)"
echo "EPS_TARGETS = $EPS_TARGETS"
echo "OUT_DIR     = $OUT_DIR | SEEDS = $SEEDS | STEPS = $STEPS"
echo "EVAL_N      = $EVAL_N | EVAL_BATCH = $EVAL_BATCH"
echo "================================================================"

FAIL=0
for S in $SURROGATES; do
  echo ""
  echo ">>> [$(date +%H:%M:%S)] surrogate = $S"
  python -u run_cifar_control.py \
      --surrogate "$S" \
      --models "$MODELS_ARG" \
      --out_dir "$OUT_DIR" \
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
