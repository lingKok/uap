#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_fewshot_reuse.py — 聚合实验 (A) 的 12 个 seed 文件，
构建 4x4 迁移矩阵，按白盒/黑盒对比 reuse_lws 与 per-ε direct / direct_lr0.05 / ceiling，
并量化「训练次数 / 算力」节省。纯 json + numpy，无需 torch。
"""
import json, glob
import numpy as np
from pathlib import Path

R = Path("results_fewshot_reuse")
FILES = sorted(glob.glob(str(R / "results_fewshot_reuse_imagenet_*_seed*.json")))
assert FILES, "没找到复用实验结果"

MODELS = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2"]
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

def agg(method, eps):
    wb = np.mean([mean_cell(method, eps, s, s) for s in MODELS])
    bb = np.mean([mean_cell(method, eps, s, d) for s in MODELS for d in MODELS if d != s])
    return wb, bb

def delta(a, b, eps):
    awb, abb = agg(a, eps); bwb, bbb = agg(b, eps)
    return awb - bwb, abb - bbb

print("=" * 96)
print("表1：各方法 白盒(WB)/黑盒(BB) FR 均值（跨 4 surrogate × 3 seed）")
print("=" * 96)
print(f"{'method':<16}{'ε=0.1 WB':>11}{'BB':>11}{'ε=0.2 WB':>11}{'BB':>11}{'ε=0.3 WB':>11}{'BB':>11}")
for m in METHODS + ["ceiling"]:
    row = []
    for eps in EPSS:
        wb, bb = agg(m, eps); row += [f"{wb:.3f}", f"{bb:.3f}"]
    print(f"{m:<16}" + "".join(f"{x:>11}" for x in row))

print("\n" + "=" * 96)
print("表2：reuse_lws 相对 per-ε 基线的 Δ（正=reuse 赢）。重点看 reuse 是否≈per-ε direct 且少训练")
print("=" * 96)
for b in ["direct", "direct_lr0.05"]:
    print(f"\n--- reuse_lws − {b} ---")
    print(f"{'ε':>5}{'WB Δ':>11}{'BB Δ':>11}")
    for eps in EPSS:
        dwb, dbb = delta("reuse_lws", b, eps)
        print(f"{eps:>5}{dwb:>+11.3f}{dbb:>+11.3f}")

print("\n" + "=" * 96)
print("表3：训练次数 / 算力节省（取自各 surrogate 的 meta 计时，跨 seed 均值）")
print("=" * 96)
for s in MODELS:
    seeds = runs[s]
    n_reuse = np.mean([seeds[sd]["meta"]["reuse_n_train_runs"] for sd in seeds])
    n_direct = np.mean([seeds[sd]["meta"]["direct_n_train_runs"] for sd in seeds])
    n_lr = np.mean([seeds[sd]["meta"]["direct_lr0.05_n_train_runs"] for sd in seeds])
    tw = np.mean([seeds[sd]["meta"]["reuse_train_time_w"] for sd in seeds])
    td = np.mean([seeds[sd]["meta"]["direct_train_time_total"] for sd in seeds])
    tl = np.mean([seeds[sd]["meta"]["direct_lr0.05_train_time_total"] for sd in seeds])
    print(f"  {s:<14} reuse:{int(n_reuse)}次({tw:.1f}s) | direct:{int(n_direct)}次({td:.1f}s) "
          f"| direct_lr0.05:{int(n_lr)}次({tl:.1f}s)  → reuse 省 {(1-tw/td)*100:.0f}% 训练时长")

print("\n" + "=" * 96)
print("方法化 (A) 裁决诊断")
print("=" * 96)
# reuse vs direct(default): 应当全正（救援欠饱和）
rd_wb = [delta("reuse_lws","direct",e)[0] for e in EPSS]
rd_bb = [delta("reuse_lws","direct",e)[1] for e in EPSS]
# reuse vs direct_lr0.05: 应当持平或略负（诚实），但训练次数 1 vs 3
rl_wb = [delta("reuse_lws","direct_lr0.05",e)[0] for e in EPSS]
rl_bb = [delta("reuse_lws","direct_lr0.05",e)[1] for e in EPSS]
print(f"reuse−direct(默认)  白盒均值={np.mean(rd_wb):+.3f} 全正={all(x>0 for x in rd_wb)} | 黑盒均值={np.mean(rd_bb):+.3f} 全正={all(x>0 for x in rd_bb)}")
print(f"reuse−direct_lr0.05 白盒均值={np.mean(rl_wb):+.3f} 全正={all(x>0 for x in rl_wb)} | 黑盒均值={np.mean(rl_bb):+.3f} 全正={all(x>0 for x in rl_bb)}")
print("\n判读：若 reuse−direct 全正 且 reuse 训练次数=1（vs direct 的 3），则『预算解耦单扰动部署』成立；")
print("      reuse−direct_lr0.05 为负/持平属预期——贡献是部署效率而非质量碾压。")
