"""A-ViT 深度拆解：per-surrogate 白盒、逐 cell 一致性、天花板达成率、饱和诊断。"""
import json, glob
import numpy as np

D = "results_fewshot_vit"
MODELS = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "vit_b_16"]
EPS = ["0.1", "0.2", "0.3"]
METHODS = ["reuse_lws", "direct", "direct_lr0.05"]

data = {}    # fr
sat = {}     # sat_frac
for f in glob.glob(f"{D}/*.json"):
    d = json.load(open(f))
    s, sd = d["surrogate"], d["seed"]
    data.setdefault(s, {}); sat.setdefault(s, {})
    for m in METHODS:
        data[s].setdefault(m, {}); sat[s].setdefault(m, {})
        for e in EPS:
            data[s][m].setdefault(e, {}); sat[s][m].setdefault(e, {})
            for t, v in d["methods"][m][e].items():
                data[s][m][e].setdefault(t, {})[sd] = v["fr_all"]
                sat[s][m][e].setdefault(t, {})[sd] = v.get("sat_frac", float("nan"))
    if isinstance(d.get("ceiling"), dict):
        data[s].setdefault("ceiling", {}); sat[s].setdefault("ceiling", {})
        for e in EPS:
            if e in d["ceiling"]:
                data[s]["ceiling"].setdefault(e, {}); sat[s]["ceiling"].setdefault(e, {})
                for t, v in d["ceiling"][e].items():
                    data[s]["ceiling"][e].setdefault(t, {})[sd] = v["fr_all"]
                    sat[s]["ceiling"][e].setdefault(t, {})[sd] = v.get("sat_frac", float("nan"))

SEEDS = [42, 1342, 2342]
print(f"seeds: {SEEDS}\n")

def m_(s, m, e, t):
    v = data[s].get(m, {}).get(e, {}).get(t, {})
    return float(np.mean(list(v.values()))) if v else float("nan")

def s_(s, m, e):
    """该 (surrogate, method, eps) 的 sat_frac（所有 target 共用，取首個）"""
    for t in MODELS:
        v = sat[s].get(m, {}).get(e, {}).get(t, {})
        if v: return float(np.mean(list(v.values())))
    return float("nan")

def blk(s, t):
    if s == t: return "WB"
    if s != "vit_b_16" and t == "vit_b_16": return "CNN→ViT"
    if s == "vit_b_16" and t != "vit_b_16": return "ViT→CNN"
    return "CNN×CNN"

# ---------- A) per-surrogate 白盒 ----------
print("=" * 100)
print("表A：各 surrogate 的【白盒】FR（自身模型上，跨 3 seed 均值；FR 越高=攻击越强）")
print("=" * 100)
print(f"{'surrogate':<14}" + "".join(f"{'ε='+e:>26}" for e in EPS))
print(f"{'':14}" + "".join(f"{'reuse':>8}{'direct':>8}{'lr.05':>8}" for e in EPS))
for s in MODELS:
    row = f"{s:<14}"
    for e in EPS:
        row += f"{m_(s,'reuse_lws',e,s):>8.3f}{m_(s,'direct',e,s):>8.3f}{m_(s,'direct_lr0.05',e,s):>8.3f}"
    print(row)

# ---------- B) 逐 cell seed 一致性 ----------
print("\n" + "=" * 100)
print("表B：reuse_lws − direct 逐 cell（行=surrogate，列=target；括号=3 seed 中为正的个数）")
print("=" * 100)
for e in EPS:
    print(f"\n--- ε={e} ---")
    print(f"  {'s/t':<14}" + "".join(f"{t[:11]:>16}" for t in MODELS))
    win = tot = 0
    for s in MODELS:
        row = f"  {s:<14}"
        for t in MODELS:
            ds = [data[s]["reuse_lws"][e][t][sd] - data[s]["direct"][e][t][sd] for sd in SEEDS]
            row += f"{np.mean(ds):+.3f}({sum(1 for x in ds if x>0)}/3)".rjust(16)
            if s != t: win += sum(1 for x in ds if x > 0); tot += 3
        print(row)
    print(f"  → 黑盒 seed 级 reuse>direct: {win}/{tot} ({win/tot*100:.0f}%)")

# ---------- C) 天花板达成率 ----------
print("\n" + "=" * 100)
print("表C：天花板达成率 = 方法 FR / many-shot ceiling FR（越高=越接近充分训练上界）")
print("=" * 100)
for e in EPS:
    print(f"\n--- ε={e} ---")
    print(f"  {'block':<10}{'reuse':>9}{'direct':>9}{'lr.05':>9}{'ceil':>8}   {'reuse/ceil':>11}{'direct/ceil':>12}")
    for bname in ["CNN×CNN", "CNN→ViT", "ViT→CNN"]:
        acc = {m: [] for m in METHODS + ["ceiling"]}
        for s in MODELS:
            for t in MODELS:
                if blk(s, t) != bname: continue
                for m in METHODS + ["ceiling"]:
                    acc[m].append(m_(s, m, e, t))
        r, dd, lr, ce = [float(np.mean(acc[m])) for m in ["reuse_lws", "direct", "direct_lr0.05", "ceiling"]]
        print(f"  {bname:<10}{r:>9.3f}{dd:>9.3f}{lr:>9.3f}{ce:>8.3f}   {r/ce:>11.1%}{dd/ce:>12.1%}")

# ---------- D) 全黑盒汇总 ----------
print("\n" + "=" * 100)
print("表D：全 20 个黑盒单元总体均值（5 surrogate × 4 非自身 target × 3 seed）")
print("=" * 100)
print(f"  {'ε':<5}{'reuse':>8}{'direct':>8}{'lr.05':>8}{'ceil':>8}{'Δ reuse-direct':>17}{'Δ vs lr.05':>13}")
for e in EPS:
    acc = {m: [] for m in METHODS + ["ceiling"]}
    for s in MODELS:
        for t in MODELS:
            if s == t: continue
            for m in METHODS + ["ceiling"]:
                acc[m].append(m_(s, m, e, t))
    r, dd, lr, ce = [float(np.mean(acc[m])) for m in ["reuse_lws", "direct", "direct_lr0.05", "ceiling"]]
    print(f"  {e:<5}{r:>8.3f}{dd:>8.3f}{lr:>8.3f}{ce:>8.3f}{r-dd:>+17.3f}{r-lr:>+13.3f}")

# ---------- E) 饱和诊断 ----------
print("\n" + "=" * 100)
print("表E：sat_frac 饱和率诊断（UAP 坐标被 clamp 到 ε 边界的比例）")
print("=" * 100)
print(f"  {'surrogate':<14}" + "".join(f"{'ε='+e:>24}" for e in EPS))
print(f"  {'':14}" + "".join(f"{'reuse':>8}{'direct':>8}{'lr.05':>8}" for e in EPS))
for s in MODELS:
    row = f"  {s:<14}"
    for e in EPS:
        row += f"{s_(s,'reuse_lws',e):>8.3f}{s_(s,'direct',e):>8.3f}{s_(s,'direct_lr0.05',e):>8.3f}"
    print(row)
print("  （注：reuse 的 sat 恒定=缩放保形；direct 的 sat 随 ε 骤降=欠饱和）")

# ---------- F) 算力 ----------
print("\n" + "=" * 100)
print("表F：训练次数与时长（跨 3 seed 均值）")
print("=" * 100)
tot_r = tot_d = 0
for s in MODELS:
    fs = glob.glob(f"{D}/*{s}_seed*.json")
    rt = np.mean([json.load(open(f))["meta"]["reuse_train_time_w"] for f in fs])
    dt = np.mean([json.load(open(f))["meta"]["direct_train_time_total"] for f in fs])
    tot_r += rt; tot_d += dt
    print(f"  {s:<14} reuse 1次={rt:>7.1f}s | direct 3次={dt:>7.1f}s | 省 {(1-rt/dt)*100:>3.0f}%")
print(f"  {'合计':<14} reuse={tot_r:>6.0f}s | direct={tot_d:>6.0f}s | 省 {(1-tot_r/tot_d)*100:.0f}%")
