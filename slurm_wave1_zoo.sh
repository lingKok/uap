#!/bin/bash
# ================================================================
# Wave 1 — 代理集消融重做 + ensemble-UAP 3v5 定义性对照（CIFAR-10）
#
# 用途:
#   1) 元学习版消融: {3, 3+VGG, 3+DenseNet, 5} × seeds {42,1342,2342}
#      × 20 epochs, 输出 mean±std（修复协议: train split 训练 / test split 评估）
#   2) ensemble 版对照（无元学习, inner_steps=0 退化）: {3, 5} × 同 seeds,
#      与元学习版同批同种子——方向三的**定义性实验**:
#      若 ensemble 也呈 3>5, 叙事从"元学习特有"降为"transfer 一般性质"
#      （裁决文件 §4 方向三前置判定）
#
# 预估: 60-90 GPU·分钟; --time=04:00:00 留足余量
#
# Go/No-Go（裁决文件 §4）:
#   方向三 GO①: 3 seeds 下 3+VGG 显著低于 3（≥3pp + 配对检验）且
#               3+DenseNet 不低于 3
#   前置判定: ensemble 3v5 若也 3>5 → 表述改 "transfer generally", ceiling 降一级
#
# 依赖:
#   · 不依赖 Wave 0, 可随时提交
#   · 参数名按【接口约定】, 待洗澄明修复 ablation.py 后生效
#     （--seeds / --model_subsets / --train_split / --eval_split / --out_dir）
#   · ensemble 块的调用形式按主理人裁定: `--inner_steps 0 --ablations "3"
#     --model_subsets "3,5"`（只跑消融3 模型子集表, 与元学习版同 seeds 同子集）;
#     脚本运行时静态 grep 检测 `"--inner_steps"` 与 `"--ablations"` 两个 flag,
#     未交付则警告跳过（不能用 `--help` 探测: 未修复版无 argparse, --help 会直接
#     开跑旧消融浪费 GPU）
# ================================================================
#SBATCH --job-name=wave1_zoo
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-wave1-zoo-%j.out

source /etc/profile.d/modules.sh
module load cuda/12.4
module load miniforge/24.9.2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate meta_uap

# CIFAR 数据路径（config.py 读环境变量, 默认 /data/public/cifar）
export DATA_ROOT="${DATA_ROOT:-/data/public/cifar}"

echo "============================================"
echo "Wave 1 zoo: 代理集消融 + ensemble 3v5 (CIFAR-10, 修复协议)"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "============================================"

echo "--- [1/2] 元学习版消融: {3, 3+VGG, 3+DN, 5} × 3 seeds ---"
python ablation.py \
    --seeds "42,1342,2342" \
    --model_subsets "3,3vgg,3dn,5" \
    --train_split train \
    --eval_split test \
    --out_dir results_wave0 \
    > slurm-wave1-zoo-meta.out 2>&1
# 元学习版（全量消融, inner_steps 默认 5）输出 results_ablation_zoo.json;
# mv 归档为 _meta.json 与 ensemble 版的 _ensemble.json 命名统一
# （互相覆盖风险已由 ablation.py 第三轮的按模式命名在代码层解决, mv 仅为命名统一）
if [ -f results_wave0/results_ablation_zoo.json ]; then
    mv results_wave0/results_ablation_zoo.json results_wave0/results_ablation_zoo_meta.json
    echo "[1/2] 完成（归档: results_wave0/results_ablation_zoo_meta.json, 日志: slurm-wave1-zoo-meta.out）"
else
    echo "[1/2][错误] 未生成 results_wave0/results_ablation_zoo.json, 详见 slurm-wave1-zoo-meta.out"
fi

echo "--- [2/2] ensemble 版对照（inner_steps=0, 无元学习）: {3, 5} × 3 seeds ---"
# 静态检测 argparse flag（不能用 `python ablation.py --help` 探测: 未修复版无
# argparse, --help 会直接开跑整个旧消融, 浪费 GPU）。
# 调用形式按主理人裁定: --inner_steps 0 + --ablations "3"（只跑消融3 模型子集表,
# 不重跑消融1/2）+ --model_subsets "3,5"（与元学习版同 seeds 同子集）
if grep -q -- '"--inner_steps"' ablation.py && grep -q -- '"--ablations"' ablation.py; then
    python ablation.py \
        --seeds "42,1342,2342" \
        --model_subsets "3,5" \
        --inner_steps 0 \
        --ablations "3" \
        --train_split train \
        --eval_split test \
        --out_dir results_wave0 \
        > slurm-wave1-zoo-ensemble.out 2>&1 || \
        echo "[错误] ensemble 块运行失败, 详见 slurm-wave1-zoo-ensemble.out"
    # ablation.py 第三轮起 inner_steps=0 直接输出 results_ablation_zoo_ensemble.json
    # （文件名带 _ensemble 后缀, 与全量版天然不覆盖）——只需确认存在, 无需 mv
    if [ -f results_wave0/results_ablation_zoo_ensemble.json ]; then
        echo "[2/2] 完成（产物: results_wave0/results_ablation_zoo_ensemble.json, 日志: slurm-wave1-zoo-ensemble.out）"
    else
        echo "[2/2][错误] 未生成 results_wave0/results_ablation_zoo_ensemble.json, 详见 slurm-wave1-zoo-ensemble.out"
    fi
else
    echo "[2/2][警告] 当前 ablation.py 不支持 --inner_steps/--ablations（待洗澄明添加）。"
    echo "            ensemble-UAP 3v5 定义性对照未运行——方向三前置判定缺失。"
    echo "            交付后请手动执行:"
    echo "            python ablation.py --seeds \"42,1342,2342\" --model_subsets \"3,5\" \\"
    echo "                --inner_steps 0 --ablations \"3\" --train_split train --eval_split test --out_dir results_wave0"
fi

echo "Wave 1 zoo 完成, 结果见 results_wave0/"
