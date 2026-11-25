#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_fewshot.py — 聚合 12 个 seed 文件，构建 4x4 迁移矩阵，
按白盒/黑盒对比 LWS 与各基线，产出方法化裁决。
纯 json + numpy，无需 torch。
"""
import json
import glob
import numpy as np
from pathlib import Path

R = Path("results_fewshot")
FILES = sorted(glob.glob(str(R / "results_fewshot_imagenet_*_seed*.json")))
assert FILES, "没找到结果文件"

MODELS = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2"]
METHODS = ["direct", "direct_lr0.002", "direct_lr0.05", "pdua", "lws", "lws_r"]
EPSS = ["0.1", "0.2", "0.3"]

# 聚合：runs[surrogate][seed][method][eps][eval_model] = fr_all
runs = {s: {} for s in MODELS}
for f in FILES:
    d = json.load(open(f))
    s = d["surrogate"]; seed = d["seed"]
    runs[s][seed] = d

def mean_fr(surrogate, method, eps, eval_model):
    vals = [runs[surrogate][sd]["methods"][method][eps][eval_model]["fr_all"]
            for sd in runs[surrogate]]
    return float(np.mean(vals)), float(np.std(vals))

# 构建迁移矩阵 mean over seeds: mat[method][eps][src][dst] = fr
mat = {}
for m in METHODS + ["ceiling"]:
    mat[m] = {}
    for eps in EPSS:
        mat[m][eps] = {}
        for src in MODELS:
            mat[m][eps][src] = {}
            for dst in MODELS:
                if m == "ceiling":
                    v = np.mean([runs[src][sd]["ceiling"][eps][dst]["fr_all"] for sd in runs[src]])
                else:
                    v = np.mean([runs[src][sd]["methods"][m][eps][dst]["fr_all"] for sd in runs[src]])
                mat[m][eps][src][dst] = float(v)

def agg(method, eps):
    """白盒=对角线均值；黑盒=行内非对角(迁移到其余3)均值。"""
    wb = np.mean([mat[method][eps][s][s] for s in MODELS])
    bb = np.mean([mat[method][eps][s][d]
                  for s in MODELS for d in MODELS if d != s])
    return wb, bb

print("=" * 100)
print("表1：各方法白盒(WB)/黑盒(BB) FR 均值（跨 4 surrogate × 3 seed），FR 越高=攻击越强")
print("=" * 100)
print(f"{'method':<16}{'ε=0.1 WB':>10}{'BB':>10}{'ε=0.2 WB':>10}{'BB':>10}{'ε=0.3 WB':>10}{'BB':>10}")
for m in METHODS + ["ceiling"]:
    row = []
    for eps in EPSS:
        wb, bb = agg(m, eps)
        row += [f"{wb:.3f}", f"{bb:.3f}"]
    print(f"{m:<16}" + "".join(f"{x:>10}" for x in row))

def delta(a, b, eps):
    awb, abb = agg(a, eps); bwb, bbb = agg(b, eps)
    return awb - bwb, abb - bbb

print("\n" + "=" * 100)
print("表2：LWS 相对基线的白盒/黑盒 Δ（正=LWS 赢）。关键诊断：direct_lr0.05 是否吃掉 LWS 优势")
print("=" * 100)
for b in ["direct", "direct_lr0.05", "pdua"]:
    print(f"\n--- LWS − {b} ---")
    print(f"{'ε':>5}{'WB Δ':>10}{'BB Δ':>10}")
    for eps in EPSS:
        dwb, dbb = delta("lws", b, eps)
        print(f"{eps:>5}{dwb:>+10.3f}{dbb:>+10.3f}")

print("\n" + "=" * 100)
print("表3：lws_r（带精炼）相对 direct_lr0.05，看精炼能否翻盘")
print("=" * 100)
print(f"{'ε':>5}{'WB Δ':>10}{'BB Δ':>10}")
for eps in EPSS:
    dwb, dbb = delta("lws_r", "direct_lr0.05", eps)
    print(f"{eps:>5}{dwb:>+10.3f}{dbb:>+10.3f}")

print("\n" + "=" * 100)
print("表4：饱和率 sat_frac（白盒 surrogate，跨 seed 均值）——解释机制")
print("=" * 100)
print(f"{'method':<16}{'ε=0.1':>10}{'ε=0.2':>10}{'ε=0.3':>10}")
for m in METHODS:
    row = []
    for eps in EPSS:
        sat = np.mean([runs[s][sd]["methods"][m][eps][s]["sat_frac"]
                       for s in MODELS for sd in runs[s]])
        row.append(f"{sat:.3f}")
    print(f"{m:<16}" + "".join(f"{x:>10}" for x in row))

print("\n" + "=" * 100)
print("表5：LWS 相对 direct(lr=0.01) 的逐 surrogate 白盒 Δ（看是否全正 / 是否均匀）")
print("=" * 100)
for eps in EPSS:
    print(f"  ε={eps}:")
    for s in MODELS:
        awb = mat["lws"][eps][s][s]; bwb = mat["direct"][eps][s][s]
        print(f"    {s:<14} lws={awb:.3f} direct={bwb:.3f} Δ={awb-bwb:+.3f}")

# 汇总诊断
print("\n" + "=" * 100)
print("方法化裁决诊断")
print("=" * 100)
lws_vs_direct_bb = [delta("lws","direct",e)[1] for e in EPSS]
lws_vs_lr_bb     = [delta("lws","direct_lr0.05",e)[1] for e in EPSS]
lws_vs_direct_wb = [delta("lws","direct",e)[0] for e in EPSS]
lws_vs_lr_wb     = [delta("lws","direct_lr0.05",e)[0] for e in EPSS]
print(f"LWS−direct  白盒 均值={np.mean(lws_vs_direct_wb):+.3f}  全正={all(x>0 for x in lws_vs_direct_wb)}")
print(f"LWS−direct  黑盒 均值={np.mean(lws_vs_direct_bb):+.3f}  全正={all(x>0 for x in lws_vs_direct_bb)}")
print(f"LWS−lr0.05  白盒 均值={np.mean(lws_vs_lr_wb):+.3f}  全正={all(x>0 for x in lws_vs_lr_wb)}")
print(f"LWS−lr0.05  黑盒 均值={np.mean(lws_vs_lr_bb):+.3f}  全正={all(x>0 for x in lws_vs_lr_bb)}")
