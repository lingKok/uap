"""
合并 X1 多卡 / 多作业并行产物。

各 (模型, seed) 作业独立落盘：
    results_x1_{dataset}_{tag}_seed{seed}.json   （tag = 模型名）
本脚本把它们合并为跨模型 × 跨 seed 的总聚合，并复用 x1_cross_budget 的
aggregate + print_judgment，输出 results_x1_{dataset}_aggregate.json。

用法：
    python merge_x1.py --dataset imagenet --out_dir results_x1
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from x1_cross_budget import aggregate, print_judgment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="imagenet")
    p.add_argument("--out_dir", type=str, default="results_x1")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    pat = str(out_dir / f"results_x1_{args.dataset}_*_seed*.json")
    files = sorted(glob.glob(pat))

    all_runs = []
    names = set()
    targets = set()
    skipped = 0
    for f in files:
        if "_aggregate" in f:        # 跳过历史聚合文件
            continue
        data = json.load(open(f))
        if not isinstance(data, dict) or not data:
            skipped += 1
            continue
        # 仅接受带 tag 的并行产物（模型名非空）；旧式无 tag 文件会被跳过以免混算
        kept = {m: mres for m, mres in data.items() if m}
        if not kept:
            skipped += 1
            continue
        for m, mres in kept.items():
            names.add(m)
            for t in mres:
                targets.add(float(t))
        all_runs.append(kept)

    if not all_runs:
        print(f"[merge] 未找到任何并行产物（已跳过 {skipped} 个不匹配文件），退出")
        return

    names = sorted(names)
    targets = sorted(targets)
    print(f"[merge] 读入 {len(all_runs)} 个 (模型,seed) 文件，"
          f"模型={names}，目标ε={targets}")

    agg = aggregate(all_runs, names, targets)
    with open(out_dir / f"results_x1_{args.dataset}_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    print_judgment(agg, names, targets)
    print(f"\n[保存] {out_dir}/results_x1_{args.dataset}_aggregate.json")


if __name__ == "__main__":
    main()
