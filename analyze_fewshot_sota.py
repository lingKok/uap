#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_fewshot_sota.py — 与已发表 UAP 方法同预算对比的裁决分析。

核心问题（决定论文能否堵死「只是没用对方法」）：
  Q1 reuse_lws 是否 ≥ 三个已发表基线（uap2017 / ffa / ssa）？
  Q2 若只与「调大 lr 的 direct」持平，那么相对**已发表方法**是否仍有优势？
  Q3 data-free 方法在 few-shot 预算下的表现如何（它们甚至不用数据）？
"""
import json
import glob
import numpy as np

D = "results_fewshot_sota"
MODELS = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "vit_b_16"]
EPS = ["0.1", "0.2", "0.3"]
METHODS = ["reuse_lws", "reuse_tsc", "direct", "direct_lr0.05", "uap2017", "ffa", "ssa_energy",
           "tsc", "dmuap"]
# 已发表基线：T1 奠基层 + T2 现代层
PUBLISHED_T1 = ["uap2017", "ffa", "ssa_energy"]
PUBLISHED_T2 = ["tsc", "dmuap"]
PUBLISHED = PUBLISHED_T1 + PUBLISHED_T2
# 关键对照：与本文"少样本"主张最直接竞争的方法
KEY_RIVAL = "dmuap"

fr, sat = {}, {}
for f in glob.glob(f"{D}/*.json"):
    d = json.load(open(f))
    s, sd = d["surrogate"], d["seed"]
    for m in METHODS:
        for e in EPS:
            for t, v in d["methods"].get(m, {}).get(e, {}).items():
                fr.setdefault((s, m, e, t), {})[sd] = v["fr_all"]
                sat.setdefault((s, m, e), {})[sd] = v.get("sat_frac", float("nan"))
    if isinstance(d.get("ceiling"), dict):
        for e in EPS:
            if e in d["ceiling"]:
                for t, v in d["ceiling"][e].items():
                    fr.setdefault((s, "ceiling", e, t), {})[sd] = v["fr_all"]


def M(s, m, e, t):
    v = fr.get((s, m, e, t), {})
    return float(np.mean(list(v.values()))) if v else float("nan")


def S(s, m, e):
    for t in MODELS:
        v = sat.get((s, m, e), {})
        if v:
            return float(np.mean(list(v.values())))
    return float("nan")


def blk(s, t):
    if s == t: return "WB"
    if s != "vit_b_16" and t == "vit_b_16": return "CNN→ViT"
    if s == "vit_b_16" and t != "vit_b_16": return "ViT→CNN"
    return "CNN×CNN"


SURROGATES = sorted({k[0] for k in fr})
SEEDS = sorted({sd for k, v in fr.items() for sd in v})
print(f"surrogates found: {SURROGATES}")
print(f"seeds: {SEEDS}\n")

# ---------- 表 1：逐块方法排名 ----------
print("=" * 104)
print("表1：各块 FR 均值与方法排名（★=该块最优；数据驱动与 data-free 同预算 steps=200）")
print("=" * 104)
for e in EPS:
    print(f"\n--- ε={e} ---")
    print(f"  {'block':<10}" + "".join(f"{m[:11]:>12}" for m in METHODS) + f"{'ceiling':>10}")
    for bname in ["WB", "CNN×CNN", "CNN→ViT", "ViT→CNN"]:
        vals = {}
        for m in METHODS + ["ceiling"]:
            acc = [M(s, m, e, t) for s in SURROGATES for t in MODELS if blk(s, t) == bname]
            acc = [x for x in acc if not np.isnan(x)]
            if acc: vals[m] = float(np.mean(acc))
        if not vals: continue
        best = max((v, k) for k, v in vals.items() if k != "ceiling")[1]
        row = f"  {bname:<10}"
        for m in METHODS:
            v = vals.get(m, float("nan"))
            mark = "★" if m == best else " "
            row += f"{v:>10.3f}{mark:>2}" if not np.isnan(v) else f"{'-':>12}"
        row += f"{vals.get('ceiling', float('nan')):>10.3f}"
        print(row)

# ---------- 表 2：reuse vs 每个已发表基线 ----------
print("\n" + "=" * 104)
print("表2：reuse_lws − 每个已发表基线（Δ>0 = reuse 更强）；括号内为黑盒 seed 级为正的比例")
print("=" * 104)
for e in EPS:
    print(f"\n--- ε={e} ---")
    print(f"  {'vs':<14}{'WB Δ':>10}{'CNN×CNN Δ':>13}{'CNN→ViT Δ':>13}{'ViT→CNN Δ':>13}{'全黑盒 Δ':>12}")
    for base in PUBLISHED:
        row = f"  {base:<14}"
        allbb = []
        for bname in ["WB", "CNN×CNN", "CNN→ViT", "ViT→CNN"]:
            ds = []
            for s in SURROGATES:
                for t in MODELS:
                    if blk(s, t) != bname: continue
                    a, b = M(s, "reuse_lws", e, t), M(s, base, e, t)
                    if not (np.isnan(a) or np.isnan(b)): ds.append(a - b)
            row += f"{np.mean(ds):>+13.3f}" if ds else f"{'-':>13}"
            if bname != "WB": allbb += ds
        row += f"{np.mean(allbb):>+12.3f}"
        print(row)

# ---------- 表 2b：与 TSC-UAP（TIFS 2024）的组合增益 ----------
print("\n" + "=" * 104)
print("表2b：TSC 组合增益（reuse_tsc − tsc = 我们的预算解耦在 2024 SOTA 之上是否还有加值）")
print("=" * 104)
for e in EPS:
    row = f"  ε={e}: "
    for bname in ["WB", "CNN×CNN", "CNN→ViT", "ViT→CNN"]:
        ds = [M(s, "reuse_tsc", e, t) - M(s, "tsc", e, t)
              for s in SURROGATES for t in MODELS if blk(s, t) == bname]
        ds = [x for x in ds if not np.isnan(x)]
        row += f"{bname}={np.mean(ds):+.3f}  " if ds else f"{bname}=NA  "
    print(row)
print("  （另附 reuse_tsc − reuse_lws，看 TSC 是否也能增强我们的方法：）")
for e in EPS:
    ds = [M(s, "reuse_tsc", e, t) - M(s, "reuse_lws", e, t)
          for s in SURROGATES for t in MODELS if s != t]
    ds = [x for x in ds if not np.isnan(x)]
    if ds:
        print(f"  ε={e}: 全黑盒 {np.mean(ds):+.3f}（{(np.array(ds)>0).mean()*100:.0f}% 为正）")

# ---------- 表 2c：与 DM-UAP（AAAI 2025）正面对决 ----------
print("\n" + "=" * 104)
print("表2c：与 DM-UAP（AAAI 2025）正面对决 —— 它用 500 样本，我们用 200")
print("=" * 104)
print(f"  {'ε':<5}{'block':<10}{'reuse_lws':>11}{'dmuap':>9}{'Δ':>9}{'ceiling':>9}   备注")
for e in EPS:
    for bname in ["WB", "CNN×CNN", "CNN→ViT", "ViT→CNN"]:
        pair = [(M(s, "reuse_lws", e, t), M(s, "dmuap", e, t), M(s, "ceiling", e, t))
                for s in SURROGATES for t in MODELS if blk(s, t) == bname]
        pair = [p for p in pair if not any(np.isnan(p))]
        if not pair: continue
        a, b, c = (float(np.mean([p[i] for p in pair])) for i in range(3))
        note = "ours 更强" if a - b > 0.01 else ("dmuap 更强" if b - a > 0.01 else "持平")
        print(f"  {e:<5}{bname:<10}{a:>11.3f}{b:>9.3f}{a-b:>+9.3f}{c:>9.3f}   {note}")
print("  注：dmuap 的 sat_frac 与训练时长见 表4/表5；算力上 DM-UAP 单次外层更新"
      "≈21× direct 单步，故本表若 ours 不输，已是在偏袒基线的设置下取得。")

# ---------- 表 3：全黑盒总体 + 天花板达成率 ----------
print("\n" + "=" * 104)
print("表3：全黑盒总体均值与天花板达成率")
print("=" * 104)
SHORT = {"reuse_lws": "reuse", "reuse_tsc": "reuseTSC", "direct": "direct",
         "direct_lr0.05": "lr.05", "uap2017": "uap17", "ffa": "ffa",
         "ssa_energy": "ssa", "tsc": "tsc", "dmuap": "dmuap"}
print(f"  {'ε':<5}" + "".join(f"{SHORT[m]:>10}" for m in METHODS) + f"{'ceil':>9}{'reuse/ceil':>12}")
for e in EPS:
    vals = {}
    for m in METHODS + ["ceiling"]:
        acc = [M(s, m, e, t) for s in SURROGATES for t in MODELS if s != t]
        acc = [x for x in acc if not np.isnan(x)]
        if acc: vals[m] = float(np.mean(acc))
    ce = vals.get("ceiling", float("nan"))
    print(f"  {e:<5}" + "".join(f"{vals.get(m, float('nan')):>10.3f}" for m in METHODS)
          + f"{ce:>9.3f}{vals.get('reuse_lws', float('nan'))/ce:>12.1%}")
    order = sorted([m for m in METHODS if m in vals], key=lambda m: -vals[m])
    print("       排名: " + " > ".join(f"{SHORT[m]}({vals[m]:.3f})" for m in order))

# ---------- 表 4：饱和率对比（机制） ----------
print("\n" + "=" * 104)
print("表4：各方法 sat_frac（UAP 坐标顶到 ε 边界的比例）——解释谁欠饱和")
print("=" * 104)
print(f"  {'surrogate':<13}{'ε':>5}" + "".join(f"{m[:10]:>12}" for m in METHODS))
for s in SURROGATES:
    for e in EPS:
        print(f"  {s:<13}{e:>5}" + "".join(f"{S(s, m, e):>12.3f}" for m in METHODS))

# ---------- 表 5：训练次数/时长 ----------
print("\n" + "=" * 104)
print("表5：训练次数与时长（跨 seed 均值，仅列出记录的基线）")
print("=" * 104)
for s in SURROGATES:
    fs = glob.glob(f"{D}/*{s}_seed*.json")
    if not fs: continue
    d0 = json.load(open(fs[0]))
    keys = [k for k in d0["meta"] if k.endswith("_train_time_total")] + ["reuse_train_time_w"]
    row = f"  {s:<13}"
    for k in keys:
        v = np.mean([json.load(open(f))["meta"].get(k, float("nan")) for f in fs])
        row += f"{k.replace('_train_time_total','').replace('_train_time_w','')}={v:.0f}s  "
    print(row)

# ---------- 裁决 ----------
print("\n" + "=" * 104)
print("裁决诊断")
print("=" * 104)
for e in EPS:
    line = f"  ε={e}: "
    for base in PUBLISHED:
        ds = [M(s, "reuse_lws", e, t) - M(s, base, e, t)
              for s in SURROGATES for t in MODELS if s != t]
        ds = [x for x in ds if not np.isnan(x)]
        line += f"vs {base}={np.mean(ds):+.3f}({(np.array(ds)>0).mean()*100:.0f}%正)  "
    print(line)
print("""
  判读：
   • 若 reuse 对已发表基线全为正 → 「只是没用对方法」被堵死，方法贡献站得住；
   • 若仅对部分为正 → 如实按「相对 X 更强、相对 Y 持平/更弱」条件化声明；
   • ffa/ssa 是 data-free（不用任何真实数据），若它们已很强，须承认
     「few-shot 数据的边际价值有限」，这本身就是有价值的结论；
   • dmuap（AAAI'25）是最强对手：它在 500 样本上比 SOTA 高 12.1%。若我们在
     200 样本上仍 ≥ 它，这是全文最强的一句；若输，就只保留"效率（1/3 算力）+
     机制（两因子）+ 跨架构"三条贡献，并如实把它列为上界对照。
""")
