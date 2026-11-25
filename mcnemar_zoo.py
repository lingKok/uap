"""
mcnemar_zoo.py — 对 zoo 消融做逐样本配对 McNemar 检验（方向三 GO① 显著性认证）

输入: results_ablation_zoo_masks.json（ablation.py --dump_masks 产出）
      {eval_split, eval_seed, num_eval, test_models, subsets, seeds,
       masks: [ {subset: {model: [0/1,...]}}, ... ] }   # 每 seed 一项

方法: 对每个 test model，比较子集 A 与 B 在同批 eval 图像上的 fooled 标签。
  逐 seed 用精确（条件）McNemar 双侧 p：
      discordant n = n10(A独有) + n01(B独有)
      在 H0 下 n10 ~ Bin(n, 0.5)；p = 2 * P(X <= min(n10,n01))
  合并：跨 seed 累加 discordant 计数后做一次合并精确 McNemar（独立 seed 求和仍服从
      Bin(sum n, 0.5)，无需 scipy）。

用法:
  python mcnemar_zoo.py --masks results_wave0/results_ablation_zoo_masks.json \
      --a models_3 --b models_3vgg
  python mcnemar_zoo.py --masks results_wave0/results_ablation_zoo_masks.json \
      --a models_3 --b models_3dn
  # 不指定 --model 则对全部 test model 分别输出；可加 --out_dir 落盘 JSON
"""
import argparse
import json
import math
from pathlib import Path


def exact_mcnemar(n10, n01):
    """精确（条件）McNemar 双侧 p；n10=A 独有 fooled, n01=B 独有 fooled"""
    n = n10 + n01
    if n == 0:
        return 1.0, n, 0
    k = min(n10, n01)
    # P(X <= k) for X ~ Bin(n, 0.5)
    p_le = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(2.0 * p_le, 1.0)
    return p, n, k


def main():
    ap = argparse.ArgumentParser(description="zoo 消融逐样本配对 McNemar 检验")
    ap.add_argument("--masks", required=True, help="results_ablation_zoo_masks.json")
    ap.add_argument("--a", required=True, help="子集 A，如 models_3")
    ap.add_argument("--b", required=True, help="子集 B，如 models_3vgg")
    ap.add_argument("--model", default=None, help="指定 test model；省略则全部")
    ap.add_argument("--out_dir", default=None, help="结果落盘目录")
    args = ap.parse_args()

    data = json.load(open(args.masks))
    masks = data["masks"]
    test_models = data["test_models"]
    targets = [args.model] if args.model else test_models

    print(f"McNemar 配对检验: {args.a}  vs  {args.b}")
    print(f"  eval: split={data['eval_split']} seed={data['eval_seed']} "
          f"n={data['num_eval']}  seeds={data['seeds']}\n")

    out_rows = []
    for m in targets:
        pool_10 = pool_01 = 0
        print(f"=== test_model = {m} ===")
        for si, seed_masks in enumerate(masks):
            ma = (seed_masks.get(args.a) or {}).get(m)
            mb = (seed_masks.get(args.b) or {}).get(m)
            if ma is None or mb is None:
                print(f"  seed[{si}] 缺掩码，跳过")
                continue
            n10 = sum(1 for x, y in zip(ma, mb) if x == 1 and y == 0)  # A 独有
            n01 = sum(1 for x, y in zip(ma, mb) if x == 0 and y == 1)  # B 独有
            frA = sum(ma) / len(ma)
            frB = sum(mb) / len(mb)
            p, n, k = exact_mcnemar(n10, n01)
            print(f"  seed[{si}] discordant={n} A独有={n10} B独有={n01} "
                  f"FR_A={frA:.3f} FR_B={frB:.3f} p={p:.2e}")
            pool_10 += n10
            pool_01 += n01
            out_rows.append({"seed": si, "n_discordant": n, "n10": n10,
                             "n01": n01, "p": p, "frA": frA, "frB": frB})
        p_pool, n_pool, k_pool = exact_mcnemar(pool_10, pool_01)
        print(f"  >>> 合并精确 McNemar: discordant={n_pool} "
              f"A独有={pool_10} B独有={pool_01} p={p_pool:.2e}")
        out_rows.append({"pooled": True, "n_discordant": n_pool,
                         "n10": pool_10, "n01": pool_01, "p": p_pool})
        print()

    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        outp = Path(args.out_dir) / f"mcnemar_{args.a}_vs_{args.b}.json"
        json.dump({"a": args.a, "b": args.b, "rows": out_rows}, open(outp, "w"), indent=2)
        print(f"已保存: {outp}")


if __name__ == "__main__":
    main()
