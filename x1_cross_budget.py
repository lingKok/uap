"""
X1 可行性实验：跨预算通用攻击反转（低-ε 训 + 放大部署 > 目标-ε 训）

核心目的：验证 Wave 0a 观察到的「Scaled-UAP-SGD（ε_train 训好后放大到目标 ε）
显著强于在目标 ε 从头训的 UAP-SGD」究竟是真现象，还是高-ε SGD 在小优化集上
过拟合的 artifact。

四组对照（均在同一模型上白盒评估，与 Wave 0a 协议一致以保证可比性）：
  (a) scaled_from_train_few : ε_train 上用 few 张图训 → 放大到目标 ε
  (b) scaled_from_train_many: ε_train 上用 many 张图训 → 放大到目标 ε   [机制分离]
  (c) sgd_target_few        : 目标 ε 上用 few 张图从头训               [原基线]
  (d) sgd_target_many       : 目标 ε 上用 many 张图从头训 + 更多步      [artifact 控制]

判读（FR=攻击成功率，越高越好；越高=迁移/攻击越强）：
  - 反转成立 vs 基线： (a) > (c) 在多数模型×目标 ε 上成立。
  - 抗 artifact：     (a) >= (d) → 即便目标 ε 给足数据/步数，放大仍赢或持平 → 现象稳健。
                     若 (a) ≈ (d) 且 (a) > (c) → 低-ε 优势只是「更抗小样本过拟合」的数据效率效应。
  - 机制分离：       (b) vs (d)：若 (b) > (d) → 低-ε 预算本身有益（超数据量）；若 (b) ≈ (d) → 低-ε 无额外增益。

协议修复：优化集(few) ⊂ 训练集(many) 同一 seed 切分；评估集用不同 split+seed，零交集。
输出：results_x1/results_x1_{dataset}_seed{seed}.json（逐 seed 落盘，互不覆盖）+ 末尾聚合打印。
"""
import argparse
import json
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from config import Config
from models import create_all_models, create_all_imagenet_models
from utils.data import get_cifar10_samples, get_imagenet_mini_samples
from utils.losses import clamp_uap
from experiments import uap_sgd, compute_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(is_imagenet, names, device):
    if is_imagenet:
        models = create_all_imagenet_models(names)
    else:
        cfg = Config()
        models = create_all_models(names, cfg.num_classes, cfg.image_size)
        for n, m in models.items():
            ckpt = cfg.save_dir / f"{n}.pt"
            if ckpt.exists():
                m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
            else:
                print(f"  [警告] {n}: checkpoint 缺失 ({ckpt})，使用随机初始化权重！")
    for n in names:
        models[n] = models[n].to(device).eval()
    return models


def get_data(is_imagenet, many, device, fallback_val_holdout=False):
    if is_imagenet:
        many_imgs, many_lbls = get_imagenet_mini_samples(
            num_samples=many, split="train", image_size=224, seed=100,
            fallback_val_holdout=fallback_val_holdout)
        eval_imgs, eval_lbls = get_imagenet_mini_samples(
            num_samples=2000, split="val", image_size=224, seed=42,
            fallback_val_holdout=fallback_val_holdout)
        eb = 16
    else:
        cfg = Config()
        many_imgs, many_lbls = get_cifar10_samples(
            cfg.data_root, num_samples=many, split="train", seed=100)
        eval_imgs, eval_lbls = get_cifar10_samples(
            cfg.data_root, num_samples=2000, split="test", seed=42)
        eb = 128
    return (many_imgs.to(device), many_lbls.to(device),
            eval_imgs.to(device), eval_lbls.to(device), eb)


def eval_uap(model, uap, eps, eval_imgs, eval_lbls, eb):
    m = compute_metrics(model, uap, eval_imgs, eval_lbls, eb)
    sat = float((uap.abs() >= eps - 1e-4).float().mean().item())  # 饱和坐标占比
    m["uap_linf"] = float(uap.abs().max().item())
    m["sat_frac"] = sat
    return m


def run_seed(args, seed, device):
    set_seed(seed)
    is_img = args.dataset == "imagenet"
    names = _resolve_models(args, is_img)
    models = load_models(is_img, names, device)
    many_imgs, many_lbls, eval_imgs, eval_lbls, eb = get_data(
        is_img, args.opt_many, device, args.fallback_val_holdout)
    few_n = args.opt_few
    few_imgs, few_lbls = many_imgs[:few_n], many_lbls[:few_n]   # few ⊂ many
    eps_tr = args.eps_train
    targets = [float(e) for e in args.eps_targets.split(",") if e.strip()]
    lr = args.lr
    out = {}

    print(f"\n{'='*72}\nSEED {seed}  dataset={args.dataset}  ε_train={eps_tr}  "
          f"few={few_n} many={args.opt_many}  steps_train={args.steps_train} "
          f"steps_few={args.steps_few} steps_many={args.steps_many}\n{'='*72}")

    for name in names:
        model = models[name]
        t0 = time.time()
        # ε_train 上训两个 UAP（few / many），整个 seed 复用，不随目标 ε 重训
        uap_tr_few = uap_sgd(model, few_imgs, few_lbls, eps_tr, steps=args.steps_train, lr=lr)
        uap_tr_many = uap_sgd(model, many_imgs, many_lbls, eps_tr, steps=args.steps_train, lr=lr)
        model_res = {}
        for tgt in targets:
            sf = tgt / eps_tr
            uap_sf = clamp_uap(uap_tr_few * sf, tgt)
            uap_sm = clamp_uap(uap_tr_many * sf, tgt)
            uap_tf = uap_sgd(model, few_imgs, few_lbls, tgt, steps=args.steps_few, lr=lr)
            uap_tm = uap_sgd(model, many_imgs, many_lbls, tgt, steps=args.steps_many, lr=lr)
            model_res[str(tgt)] = {
                "scaled_from_train_few":  eval_uap(model, uap_sf, tgt, eval_imgs, eval_lbls, eb),
                "scaled_from_train_many": eval_uap(model, uap_sm, tgt, eval_imgs, eval_lbls, eb),
                "sgd_target_few":         eval_uap(model, uap_tf, tgt, eval_imgs, eval_lbls, eb),
                "sgd_target_many":        eval_uap(model, uap_tm, tgt, eval_imgs, eval_lbls, eb),
            }
            r = model_res[str(tgt)]
            print(f"  [{name}] ε={tgt:.2f}  "
                  f"scaled_few={r['scaled_from_train_few']['fr_all']:.3f}  "
                  f"scaled_many={r['scaled_from_train_many']['fr_all']:.3f}  "
                  f"tgt_few={r['sgd_target_few']['fr_all']:.3f}  "
                  f"tgt_many={r['sgd_target_many']['fr_all']:.3f}  "
                  f"| sat%={r['scaled_from_train_few']['sat_frac']*100:.0f}")
        out[name] = model_res
        print(f"  -> {name} 完成 ({time.time()-t0:.1f}s)")
        model.cpu(); torch.cuda.empty_cache(); model.to(device)

    return out


def _resolve_models(args, is_img):
    if args.models and args.models != "auto":
        return [m.strip() for m in args.models.split(",") if m.strip()]
    if is_img:
        # 仅用 models.py 已支持的 ImageNet 模型（不含 ViT，ViT 由 wave0d 单独探针处理）
        return ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2"]
    cfg = Config()
    return cfg.meta_train_models + cfg.meta_test_models


def aggregate(all_runs, names, targets):
    """跨 seed 聚合每个 (模型, 目标ε, 条件) 的 FR 均值/std。"""
    agg = {}
    for name in names:
        agg[name] = {}
        for tgt in targets:
            agg[name][str(tgt)] = {}
            for cond in ["scaled_from_train_few", "scaled_from_train_many",
                         "sgd_target_few", "sgd_target_many"]:
                vals = [run[name][str(tgt)][cond]["fr_all"]
                        for run in all_runs if name in run]
                if vals:
                    agg[name][str(tgt)][cond] = {
                        "mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def print_judgment(agg, names, targets):
    """跨 seed 聚合判读（FR 均值；越高=攻击越强）。供 main 与 merge_x1 复用。"""
    print("\n" + "=" * 72)
    print("聚合判读（FR 均值；越高=攻击越强）：反转 vs 基线 / 抗 artifact / 机制分离")
    print("=" * 72)
    rev_vs_base, rev_vs_artifact, mech_low_eps = [], [], []
    for name in names:
        for tgt in targets:
            a = agg[name][str(tgt)]
            sf = a["scaled_from_train_few"]["mean"]
            sm = a["scaled_from_train_many"]["mean"]
            tf = a["sgd_target_few"]["mean"]
            tm = a["sgd_target_many"]["mean"]
            rev_vs_base.append(sf - tf)
            rev_vs_artifact.append(sf - tm)
            mech_low_eps.append(sm - tm)
            print(f"  {name:<14} ε={tgt:.2f}  scaled_few={sf:.3f} scaled_many={sm:.3f} "
                  f"tgt_few={tf:.3f} tgt_many={tm:.3f}  "
                  f"| vs_base={sf-tf:+.3f} vs_artifact={sf-tm:+.3f} mech={sm-tm:+.3f}")
    print("-" * 72)
    print(f"  平均 反转vs基线(>0=成立):     {np.mean(rev_vs_base):+.3f}  "
          f"({(np.array(rev_vs_base)>0).mean()*100:.0f}% 单元为正)")
    print(f"  平均 抗artifact(>=0=稳健):    {np.mean(rev_vs_artifact):+.3f}  "
          f"({(np.array(rev_vs_artifact)>=0).mean()*100:.0f}% 单元>=0)")
    print(f"  平均 机制(低ε超数据,>0):      {np.mean(mech_low_eps):+.3f}  "
          f"({(np.array(mech_low_eps)>0).mean()*100:.0f}% 单元为正)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "imagenet"])
    p.add_argument("--eps_train", type=float, default=0.05)
    p.add_argument("--eps_targets", type=str, default="0.1,0.2,0.3")
    p.add_argument("--opt_few", type=int, default=200)
    p.add_argument("--opt_many", type=int, default=2000)
    p.add_argument("--steps_train", type=int, default=200)
    p.add_argument("--steps_few", type=int, default=200)
    p.add_argument("--steps_many", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seeds", type=str, default="42,1342,2342")
    p.add_argument("--models", type=str, default="auto",
                  help="逗号分隔模型名，或 auto（按数据集取默认集）")
    p.add_argument("--fallback_val_holdout", action="store_true",
                  help="imagenet-mini 无 train/ 时用 val 内不重合切分产生优化/评估数据")
    p.add_argument("--out_dir", type=str, default="results_x1")
    p.add_argument("--out_tag", type=str, default="",
                  help="输出文件名标签（并行时传模型名，避免各作业覆盖彼此的 seed 文件）")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    is_img = args.dataset == "imagenet"
    names = _resolve_models(args, is_img)
    targets = [float(e) for e in args.eps_targets.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag
    seed_pat = (f"results_x1_{args.dataset}_{tag}_seed{{seed}}.json"
                if tag else f"results_x1_{args.dataset}_seed{{seed}}.json")
    agg_pat = (f"results_x1_{args.dataset}_{tag}_aggregate.json"
               if tag else f"results_x1_{args.dataset}_aggregate.json")
    all_runs = []
    for seed in seeds:
        res = run_seed(args, seed, device)
        all_runs.append(res)
        # 逐 seed 落盘，互不覆盖（并行时按 tag 区分各模型作业，避免互相覆盖）
        with open(out_dir / seed_pat.format(seed=seed), "w") as f:
            json.dump(res, f, indent=2)
        print(f"[保存] {out_dir}/{seed_pat.format(seed=seed)}")

    # 跨 seed 聚合
    agg = aggregate(all_runs, names, targets)
    with open(out_dir / agg_pat, "w") as f:
        json.dump(agg, f, indent=2)

    # 聚合判读（FR 越高越好）
    print_judgment(agg, names, targets)
    print(f"\n[保存] {out_dir}/{agg_pat}")


if __name__ == "__main__":
    main()
