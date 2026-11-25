#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_fewshot_vit.py — 聚合「ViT + 跨架构黑盒迁移」实验的 15 个 seed 文件，
构建 5x5 迁移矩阵，拆分白盒 / 同架构黑盒(CNN×CNN) / CNN→ViT / ViT→CNN 四块，
对比 reuse_lws 相对 per-ε direct 与 direct_lr0.05 在每块的表现，
量化跨架构边界上「预算解耦部署原语」是否依然成立。纯 json + numpy，无需 torch。
"""
import json, glob
import numpy as np
from pathlib import Path

R = Path("results_fewshot_vit")
FILES = sorted(glob.glob(str(R / "results_fewshot_reuse_imagenet_*_seed*.json")))
assert FILES, "没找到 ViT 复用实验结果（results_fewshot_vit/ 为空或路径错）"

CNN = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2"]
VIT = "vit_b_16"
MODELS = CNN + [VIT]
METHODS = ["reuse_lws", "direct", "direct_lr0.05"]
EPSS = ["0.1", "0.2", "0.3"]

runs = {s: {} for s in MODELS}
for f in FILES:
    d = json.load(open(f)); runs[d["surrogate"]][d["seed"]] = d

def mean_cell(method, eps, src, dst):
    if method == "ceiling":
        v = [runs[src][sd]["ceiling"][eps][dst]["fr_all"] for sd in runs[src]]
    else:
        v = [runs[src][sd]["methods"][method][eps][dst]["fr_all"] for sd in runs[src]]
    return float(np.mean(v))

def agg_blocks(method, eps):
    """返回 (wb, within_cnn_bb, cnn_to_vit, vit_to_cnn) 的均值。"""
    wb = np.mean([mean_cell(method, eps, s, s) for s in MODELS])
    within = [mean_cell(method, eps, s, d) for s in CNN for d in CNN if s != d]
    cv = [mean_cell(method, eps, s, VIT) for s in CNN]          # CNN→ViT（黑盒）
    vc = [mean_cell(method, eps, VIT, d) for d in CNN]          # ViT→CNN（黑盒）
    return (float(wb), float(np.mean(within)), float(np.mean(cv)), float(np.mean(vc)))

def delta_blocks(a, b, eps):
    aw, ac, acv, avc = agg_blocks(a, eps)
    bw, bc, bcv, bvc = agg_blocks(b, eps)
    return (aw - bw, ac - bc, acv - bcv, avc - bvc)

print("=" * 110)
print("表1：各方法 白盒(WB) / 同架构黑盒(CNN×CNN) / CNN→ViT / ViT→CNN 的 FR 均值（跨 5 模型 × 3 seed）")
print("=" * 110)
hdr = f"{'method':<16}" + "".join(f"{l:>24}" for l in
      ["WB", "CNN×CNN-BB", "CNN→ViT", "ViT→CNN"])
print(hdr)
for m in METHODS + ["ceiling"]:
    vals = []
    for eps in EPSS:
        wb, ac, acv, avc = agg_blocks(m, eps)
        vals += [wb, ac, acv, avc]
    # 为可读，按 ε 分行
for eps in EPSS:
    print(f"\n--- ε={eps} ---")
    print(f"{'method':<16}{'WB':>12}{'CNN×CNN':>12}{'CNN→ViT':>12}{'ViT→CNN':>12}")
    for m in METHODS + ["ceiling"]:
        wb, ac, acv, avc = agg_blocks(m, eps)
        print(f"{m:<16}{wb:>12.3f}{ac:>12.3f}{acv:>12.3f}{avc:>12.3f}")

print("\n" + "=" * 110)
print("表2：reuse_lws 相对 per-ε 基线的 Δ（正=reuse 赢）。重点看跨架构块是否仍成立")
print("=" * 110)
for b in ["direct", "direct_lr0.05"]:
    print(f"\n--- reuse_lws − {b} ---")
    print(f"{'ε':>5}{'WB Δ':>12}{'CNN×CNN Δ':>14}{'CNN→ViT Δ':>14}{'ViT→CNN Δ':>14}")
    for eps in EPSS:
        dwb, dac, dacv, dvc = delta_blocks("reuse_lws", b, eps)
        print(f"{eps:>5}{dwb:>+12.3f}{dac:>+14.3f}{dacv:>+14.3f}{dvc:>+14.3f}")

print("\n" + "=" * 110)
print("表3：reuse_lws 在全 5x5 黑盒矩阵上的「跨架构转移率」")
print("=" * 110)
for eps in EPSS:
    # 跨架构黑盒单元（CNN→ViT 与 ViT→CNN，共 4×2=8 个单元）
    cross_src_dst = [(s, VIT) for s in CNN] + [(VIT, d) for d in CNN]
    reuse_vals = [mean_cell("reuse_lws", eps, s, d) for s, d in cross_src_dst]
    direct_vals = [mean_cell("direct", eps, s, d) for s, d in cross_src_dst]
    n_win = sum(1 for r, t in zip(reuse_vals, direct_vals) if r >= t)
    # 相对 direct_lr0.05
    lr_vals = [mean_cell("direct_lr0.05", eps, s, d) for s, d in cross_src_dst]
    n_win_lr = sum(1 for r, t in zip(reuse_vals, lr_vals) if r >= t)
    print(f"  ε={eps}: 跨架构黑盒单元 {len(cross_src_dst)} 个 | "
          f"reuse≥per-ε-direct 占 {n_win}/{len(cross_src_dst)} "
          f"({n_win/len(cross_src_dst)*100:.0f}%) | "
          f"reuse≥direct_lr0.05 占 {n_win_lr}/{len(cross_src_dst)} "
          f"({n_win_lr/len(cross_src_dst)*100:.0f}%)")

print("\n" + "=" * 110)
print("表4：训练次数 / 算力节省（各 surrogate 的 meta 计时，跨 seed 均值）")
print("=" * 110)
for s in MODELS:
    seeds = runs[s]
    n_reuse = np.mean([seeds[sd]["meta"]["reuse_n_train_runs"] for sd in seeds])
    n_direct = np.mean([seeds[sd]["meta"]["direct_n_train_runs"] for sd in seeds])
    tw = np.mean([seeds[sd]["meta"]["reuse_train_time_w"] for sd in seeds])
    td = np.mean([seeds[sd]["meta"]["direct_train_time_total"] for sd in seeds])
    print(f"  {s:<14} reuse:{int(n_reuse)}次({tw:.1f}s) | direct:{int(n_direct)}次({td:.1f}s) "
          f"→ reuse 省 {(1-tw/td)*100:.0f}% 训练时长")

print("\n" + "=" * 110)
print("跨架构 (A) 裁决诊断")
print("=" * 110)
# 关键：reuse_lws 相对 per-ε direct 在「CNN×CNN」「CNN→ViT」「ViT→CNN」三块是否全正
def block_means(baseline):
    out = {}
    for eps in EPSS:
        dwb, dac, dacv, dvc = delta_blocks("reuse_lws", baseline, eps)
        out[eps] = (dac, dacv, dvc)
    return out

rd = block_means("direct")
rl = block_means("direct_lr0.05")
print("reuse_lws − per-ε direct（每 ε: CNN×CNN, CNN→ViT, ViT→CNN 三块 Δ）:")
all_pos = True
for eps in EPSS:
    dac, dacv, dvc = rd[eps]
    pos = (dac > 0) and (dacv > 0) and (dvc > 0)
    all_pos = all_pos and pos
    print(f"  ε={eps}: CNN×CNN={dac:+.3f}  CNN→ViT={dacv:+.3f}  ViT→CNN={dvc:+.3f}  "
          f"三块全正={pos}")
print(f"\n  → 跨架构边界上 reuse 相对 per-ε direct 三块全正: {all_pos}")
print("reuse_lws − direct_lr0.05（预期为负/持平，贡献是部署效率非质量碾压）:")
for eps in EPSS:
    dac, dacv, dvc = rl[eps]
    print(f"  ε={eps}: CNN×CNN={dac:+.3f}  CNN→ViT={dacv:+.3f}  ViT→CNN={dvc:+.3f}")

print("\n判读：")
print("  • 若『跨架构三块全正』成立 → 预算解耦部署原语在架构家族间泛化，广度证据强；")
print("  • 若 CNN→ViT / ViT→CNN 一块塌陷为负 → 该性质偏 CNN 特有，须缩小声明并诚实陈述；")
print("  • 无论哪种，reuse 训练次数=1（vs direct 3 次）的部署效率优势不变。")
