#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_cifar_control.py — CIFAR 侧对照分析。

判定逻辑（对应 ImageNet 裁决的跨数据集检验）：
  表2a scaleup − pereps : CIFAR 预测 ≈0（全饱和、direct 不欠饱和）；
                          若显著为正 → 「饱和调节」叙事被证伪。
  表2b clipdown − pereps: CIFAR 预测 ≈0 或温和为负（与 ImageNet direct 的
                          −13.8~−25.7pp 相对照，量级应大幅缩水）。
  表2c scaleup − clipdown: ImageNet direct 上 +14~+26pp；CIFAR 预测 ≈0。
用法: python analyze_cifar_control.py [--dir results_cifar_control] [--out_md paper/analysis_cifar_control_raw.txt]
"""
import argparse
import glob
import json

import numpy as np

MODELS = None  # 从 json 动态读
STRATS = ["pereps", "clipdown", "scaleup"]


def load(d):
    runs = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        j = json.load(open(f))
        runs[(j["surrogate"], j["seed"])] = j
    return runs


def fr(j, m, e, strat, t):
    cell = j["methods"].get(m, {}).get(str(e), {}).get(strat, {}).get(t, {})
    return cell.get("fr_all", np.nan)


def bb_mean(j, m, e, strat):
    """全黑盒均值：排除白盒 surrogate。"""
    ts = [t for t in j["methods"][m][str(e)][strat] if t != j["surrogate"]]
    v = [fr(j, m, e, strat, t) for t in ts]
    return float(np.nanmean(v))


def wb(j, m, e, strat):
    return fr(j, m, e, strat, j["surrogate"])


def paired(runs, m, e, sa, sb, fn=bb_mean):
    """配对 delta：每个 (surrogate, seed) 一对，n = len(runs)。"""
    ds = []
    for key in sorted(runs):
        j = runs[key]
        try:
            ds.append(fn(j, m, e, sa) - fn(j, m, e, sb))
        except (KeyError, TypeError):
            pass
    return np.array(ds)


def tstat(d):
    if len(d) < 2 or np.std(d, ddof=1) == 0:
        return np.nan
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_cifar_control")
    ap.add_argument("--out_md", default="")
    args = ap.parse_args()

    lines = []
    p = print if not args.out_md else (
        lambda *a, **k: (lines.append(" ".join(str(x) for x in a)), print(*a, **k)))

    runs = load(args.dir)
    if not runs:
        raise SystemExit(f"no json under {args.dir}")
    p(f"loaded {len(runs)} runs from {args.dir}\n")

    surr = sorted({k[0] for k in runs})
    seeds = sorted({k[1] for k in runs})
    methods = sorted({m for j in runs.values() for m in j["methods"]})
    budgets = sorted({float(e) for j in runs.values() for e in j["methods"]["direct"]})
    p(f"surrogates={surr} seeds={seeds} methods={methods} budgets={budgets}")

    # ---------- 表 0：同进程协议自洽 ----------
    p("\n" + "=" * 100)
    p("表0：协议自洽 —— clipdown@eps_hi 与 pereps@eps_hi 复用同一次训练（应恒等）")
    bad = 0
    for key in sorted(runs):
        j = runs[key]
        e_hi = str(j["eps_hi"])
        for m in methods:
            if m not in j["methods"] or e_hi not in j["methods"][m]:
                continue
            x, y = bb_mean(j, m, e_hi, "clipdown"), bb_mean(j, m, e_hi, "pereps")
            if abs(x - y) > 1e-9:
                bad += 1
                p(f"  {key} {m}: clipdown={x:.4f} pereps={y:.4f}  <-- 异常")
    p(f"  → {'全部恒等（同进程同 δ）' if bad == 0 else f'{bad} 个异常'}")

    # ---------- 表 1：全黑盒均值 ----------
    p("\n" + "=" * 100)
    p("表1：全黑盒均值 FR（3 surrogate × 3 seed × 异族 target）")
    header = f"  {'method':<10}{'eps':>5}" + "".join(f"{s:>11}" for s in STRATS)
    for m in methods:
        p(header)
        for e in budgets:
            row = f"  {m:<10}{e:>5.2f}"
            for s in STRATS:
                acc = [bb_mean(j, m, e, s) for j in runs.values()
                       if m in j["methods"] and str(e) in j["methods"][m]]
                row += f"{np.nanmean(acc):>11.3f}"
            p(row)
        p("")

    # ---------- 表 2：决定性配对检验 ----------
    for tag, sa, sb in [("2a", "scaleup", "pereps"), ("2b", "clipdown", "pereps"),
                        ("2c", "scaleup", "clipdown")]:
        p("=" * 100)
        p(f"表{tag}：{sa} − {sb}（黑盒，seed 级配对，n={len(runs)}）")
        p(f"  {'method':<10}{'eps':>5}{'mean':>10}{'sd':>9}{'t':>8}{'正比例':>8}"
          f"{'min':>9}{'max':>9}")
        for m in methods:
            for e in budgets:
                d = paired(runs, m, e, sa, sb)
                if len(d) == 0:
                    continue
                pos = (d > 0).mean()
                p(f"  {m:<10}{e:>5.2f}{d.mean():>+10.4f}{d.std(ddof=1):>9.4f}"
                  f"{tstat(d):>+8.2f}{pos:>7.0%}{d.min():>+9.4f}{d.max():>+9.4f}")
            p("")

    # ---------- 表 3：机制量 ----------
    p("=" * 100)
    p("表3：机制量均值（sat_pereps / sat_scaleup / clip_frac）")
    p(f"  {'method':<10}{'eps':>5}{'sat_pereps':>12}{'sat_scaleup':>13}"
      f"{'clip_frac':>11}{'sat@eps_w':>11}")
    for m in methods:
        for e in budgets:
            acc = {k: [] for k in
                   ["sat_pereps", "sat_scaleup", "clip_frac", "sat_train@eps_w"]}
            for j in runs.values():
                st = j.get("delta_stats", {}).get(f"{m}@{e}")
                if not st:
                    continue
                for k in acc:
                    if k in st:
                        acc[k].append(st[k])
            if not acc["sat_pereps"]:
                continue
            p(f"  {m:<10}{e:>5.2f}" + "".join(
                f"{np.nanmean(acc[k]):>{w}.3f}" for k, w in
                [("sat_pereps", 12), ("sat_scaleup", 13),
                 ("clip_frac", 11), ("sat_train@eps_w", 11)]))

    # ---------- 表 4：白盒对照 ----------
    p("\n" + "=" * 100)
    p("表4：白盒 FR（策略间对照，供附录）")
    for m in methods:
        p(f"  {m:<10}{'eps':>5}" + "".join(f"{s:>11}" for s in STRATS))
        for e in budgets:
            row = f"  {'':<10}{e:>5.2f}"
            for s in STRATS:
                acc = [wb(j, m, e, s) for j in runs.values()
                       if m in j["methods"] and str(e) in j["methods"][m]]
                row += f"{np.nanmean(acc):>11.3f}"
            p(row)
        p("")

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[已写] {args.out_md}")


if __name__ == "__main__":
    main()
