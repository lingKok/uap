#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_fewshot.py — 合并 run_fewshot_uap.py 的分散 json（跨 surrogate × seed 聚合）。

输入：results_fewshot/results_fewshot_{dataset}_{surrogate}_seed{seed}.json
输出：results_fewshot/results_fewshot_{dataset}_aggregate.json
      + 控制台打印 白盒(同模型) 与 黑盒(跨模型迁移) 摘要表。
"""
import argparse
import glob
import json
from pathlib import Path
import numpy as np


def agg_cell(vals):
    arr = np.array(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenet")
    p.add_argument("--out_dir", default="results_fewshot")
    args = p.parse_args()
    d = Path(args.out_dir)

    files = sorted(glob.glob(str(d / f"results_fewshot_{args.dataset}_*_seed*.json")))
    files = [f for f in files if "_aggregate" not in f]
    assert files, f"未找到 results_fewshot_{args.dataset}_*_seed*.json"

    surrogates, seeds, methods, targets, evals = set(), set(), set(), set(), set()
    # raw[surrogate][method][eps][eval][metric] = [values across seeds]
    raw = {}
    for f in files:
        data = json.load(open(f))
        sg, sd = data["surrogate"], data["seed"]
        surrogates.add(sg); seeds.add(sd)
        for mkey in ("methods", "ceiling"):
            block = data[mkey]
            for method, epsd in block.items():
                methods.add(method if mkey == "methods" else "ceiling")
                for eps, evdict in epsd.items():
                    targets.add(eps); evals.update(evdict.keys())
                    for ev, met in evdict.items():
                        for metric in ("fr_all", "asr_correct", "sat_frac"):
                            key = (sg, method, eps, ev, metric)
                            raw.setdefault(key, []).append(met[metric])

    surrogates, methods, targets, evals = (sorted(surrogates), sorted(methods),
                                          sorted(targets, key=float), sorted(evals))

    aggregate = {}
    for sg in surrogates:
        aggregate[sg] = {}
        for method in methods:
            aggregate[sg][method] = {}
            for eps in targets:
                aggregate[sg][method][eps] = {}
                for ev in evals:
                    cell = {}
                    for metric in ("fr_all", "asr_correct", "sat_frac"):
                        cell[metric] = agg_cell(raw.get((sg, method, eps, ev, metric), [float("nan")]))
                    aggregate[sg][method][eps][ev] = cell

    out = {"surrogates": surrogates, "methods": methods, "targets": targets,
           "eval_models": evals, "aggregate": aggregate}
    out_path = d / f"results_fewshot_{args.dataset}_aggregate.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[聚合] {out_path}")
    print(f"surrogates={surrogates}\nmethods={methods}\ntargets={targets}\neval_models={evals}")

    # ---- 白盒摘要（surrogate 训练、同模型评估）----
    print("\n=== 白盒 FR（surrogate==eval，越高越好）===")
    hdr = "surrogate ε    " + "".join(f"{m:>14}" for m in methods)
    print(hdr)
    for sg in surrogates:
        for eps in targets:
            row = f"{sg:<9} {eps:>4} " + "".join(
                f"{aggregate[sg][m][eps][sg]['fr_all']['mean']:>14.3f}" for m in methods)
            print(row)

    # ---- 黑盒迁移摘要（surrogate 训，迁移到其它模型，平均）----
    print("\n=== 黑盒迁移 FR（surrogate 训 → 其它模型均值，越高越好）===")
    print("surrogate ε    " + "".join(f"{m:>14}" for m in methods))
    for sg in surrogates:
        for eps in targets:
            bb = [ev for ev in evals if ev != sg]
            vals = []
            for m in methods:
                v = np.mean([aggregate[sg][m][eps][ev]['fr_all']['mean'] for ev in bb])
                vals.append(v)
            row = f"{sg:<9} {eps:>4} " + "".join(f"{v:>14.3f}" for v in vals)
            print(row)


if __name__ == "__main__":
    main()
