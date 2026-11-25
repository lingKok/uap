#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_supplementary.py — 补充实验四表分析。

  表1 epsw  : warm-start 敏感性（FR vs eps_w；机制预测：eps_w 越低 sat 越高）
  表2 bsweep: 预算覆盖曲线（scale-up 1 次训练 vs per-eps 6 次；算力账随 B 线性）
  表3 ksweep: k 扫描（机制预测：k 增大 -> direct 追上 -> Δ 收窄）
  表4 l2    : L2 spot-check（定性 parity 即成功）

用法：python analyze_supplementary.py [--dir results_suppl]
"""
import argparse
import json
import glob
import numpy as np

EPS = [0.1, 0.2, 0.3]
MODELS = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "vit_b_16"]
SEEDS = [42, 1342, 2342]


def load(d, exp):
    out = {}
    for f in glob.glob(f"{d}/suppl_{exp}_*_seed*.json"):
        j = json.load(open(f))
        out.setdefault((j["surrogate"], j["seed"]), []).append(j)
    return out


def bb_blackbox(run_key_fr, surrogate):
    """黑盒均值：剔除白盒 surrogate 自身。"""
    vals = [run_key_fr[m] for m in MODELS if m != surrogate and m in run_key_fr]
    return float(np.mean(vals)) if vals else np.nan


def wb(run_key_fr, surrogate):
    return run_key_fr.get(surrogate, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_suppl")
    args = ap.parse_args()
    d = args.dir

    # ---------------- 表 1: epsw ----------------
    E = load(d, "epsw")
    if E:
        print("=" * 100)
        print("表1  warm-start 敏感性（resnet50 surrogate，黑盒均值 / 白盒，3 seeds 均值）")
        print(f"{'method':<8}{'eps_w':>7} | " + "".join(f"{e:>16}" for e in EPS)
              + "  |  sat_w(训练时)")
        agg = {}
        for (s, seed), js in E.items():
            for j in js:
                for key, r in j["runs"].items():
                    name, rest = key.split("@")
                    eps_w = float(rest.split("->")[0].split("=")[1])
                    eps = float(rest.split("eps=")[1])
                    agg.setdefault((name, eps_w, eps), {})[seed] = r
        for name in ["direct", "tsc"]:
            for eps_w in [0.02, 0.05, 0.1]:
                cells, sat_ws = [], []
                for eps in EPS:
                    seeds_d = agg.get((name, eps_w, eps), {})
                    bb = [np.mean([seeds_d[sd]["fr"][m] for m in MODELS
                                   if m != "resnet50" and m in seeds_d[sd]["fr"]])
                          for sd in seeds_d]
                    wbv = [seeds_d[sd]["fr"]["resnet50"] for sd in seeds_d]
                    cells.append(f"{np.mean(bb):.3f}/{np.mean(wbv):.3f}" if bb else "  -  ")
                sat_ws = [next(iter(v.values()))["sat_w"]
                          for (n, w, e), v in agg.items()
                          if n == name and w == eps_w and e == EPS[0]]
                sat_str = f"{np.mean(sat_ws):.3f}" if sat_ws else "  -  "
                print(f"{name:<8}{eps_w:>7} | " + "".join(f"{c:>16}" for c in cells)
                      + f"  |  {sat_str}")
        print("  (cell = blackbox/whitebox)")

    # ---------------- 表 2: bsweep ----------------
    B = load(d, "bsweep")
    if B:
        print("\n" + "=" * 100)
        print("表2  预算覆盖扫描（resnet50，3 seeds 黑盒均值）")
        eps_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        su, pe, ts = {}, {}, []
        for (s, seed), js in B.items():
            for j in js:
                for key, r in j["runs"].items():
                    eps = float(key.split("eps=")[1])
                    bbv = np.mean([r["fr"][m] for m in MODELS
                                   if m != "resnet50" and m in r["fr"]])
                    if key.startswith("scaleup"):
                        su.setdefault(eps, []).append(bbv)
                        ts.append(r["train_s"])
                    else:
                        pe.setdefault(eps, []).append(bbv)
                        ts.append(r["train_s"])
        print(f"{'eps':>6}{'scale-up':>10}{'per-eps':>10}{'delta':>9}")
        for e in eps_list:
            a, b = np.mean(su.get(e, [np.nan])), np.mean(pe.get(e, [np.nan]))
            print(f"{e:>6}{a:>10.3f}{b:>10.3f}{a-b:>+9.3f}")
        # 算力账：B 个预算需要训练次数
        print(f"  算力：scale-up 训 1 次（均 {np.mean([t for t in ts[:3]]):.0f}s/次）"
              f" vs per-eps 训 {len(eps_list)} 次 → B={len(eps_list)} 时省 "
              f"{1 - 1/len(eps_list):.0%}")

    # ---------------- 表 3: ksweep ----------------
    K = load(d, "ksweep")
    if K:
        print("\n" + "=" * 100)
        print("表3  数据量扫描（机制预测：k 增大 -> direct 追上 -> Δ 收窄）")
        print(f"{'k':>6}{'eps':>6}{'direct(bb)':>12}{'direct sat':>12}"
              f"{'scaleup(bb)':>13}{'delta':>9}")
        agg = {}
        for (s, seed), js in K.items():
            for j in js:
                for key, r in j["runs"].items():
                    agg.setdefault(key, {})[seed] = r
        for k in [200, 500, 1000]:
            for e in EPS:
                dk = agg.get(f"direct@k={k},eps={e}", {})
                sk = agg.get(f"scaleup@k={k},eps={e}", {})
                def m(vals):
                    return np.mean([np.mean([v["fr"][m2] for m2 in MODELS
                                             if m2 != "resnet50" and m2 in v["fr"]])
                                    for v in vals.values()]) if vals else np.nan
                dsat = np.mean([v["sat"] for v in dk.values()]) if dk else np.nan
                dd = m(sk) - m(dk)
                print(f"{k:>6}{e:>6}{m(dk):>12.3f}{dsat:>12.3f}{m(sk):>13.3f}{dd:>+9.3f}")

    # ---------------- 表 4: l2 ----------------
    L = load(d, "l2")
    if L:
        print("\n" + "=" * 100)
        print("表4  L2 spot-check（定性 parity 即支持「结论不限于 L∞」）")
        agg = {}
        for (s, seed), js in L.items():
            for j in js:
                for key, r in j["runs"].items():
                    agg.setdefault(key, {})[seed] = r
        print(f"{'eps_l2':>7}{'direct_l2(bb)':>14}{'scaleup_l2(bb)':>15}"
              f"{'delta':>8}{'|delta|_2 实测':>14}")
        for e in [5.0, 10.0, 20.0]:
            dk = agg.get(f"direct_l2@eps={e}", {})
            sk = agg.get(f"scaleup_l2@eps={e}", {})
            def m2(vals):
                return np.mean([np.mean([v["fr"][mm] for mm in MODELS
                                         if mm != "resnet50" and mm in v["fr"]])
                                for v in vals.values()]) if vals else np.nan
            l2n = np.mean([v.get("l2_norm", np.nan) for v in sk.values()])
            print(f"{e:>7}{m2(dk):>14.3f}{m2(sk):>15.3f}{m2(sk)-m2(dk):>+8.3f}"
                  f"{l2n:>14.2f}")


if __name__ == "__main__":
    main()
