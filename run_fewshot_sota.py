#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_fewshot_sota.py — 与已发表 UAP 方法同 few-shot 预算对比（堵「只是没用对方法」）。

方法（同一 few-shot 预算 K=200、同 ε 球、同步数 200，除 ceiling 外）：
  【T1 奠基层 2017–2018】
  direct       : uap_sgd(Adam, lr=0.01)，每 ε_t 单独训           ← 默认基线
  direct_lr0.05: uap_sgd(Adam, lr=0.05)，每 ε_t 单独训           ← 「只是没调 lr」对照
  uap2017      : Moosavi-Dezfooli CVPR'17 迭代聚合，每 ε_t 单独训
  ffa          : Mopuri BMVC'17 Fast Feature Fool，data-free
  ssa_energy   : Mopuri TPAMI'18 激活能量变体，data-free
  【T2 现代层 2024+】
  tsc          : TSC-UAP（TIFS 2024），每 ε_t 单独训纹理块
  dmuap        : DM-UAP（AAAI 2025，官方代码接入），动态 maximin，每 ε_t 单独训
                 ← 与本文"少样本"主张最直接竞争的方法（其论文用 500 样本）
  【OURS】
  reuse_lws    : ε_w=0.05 训 1 次 → 缩放到 {ε_t}（不重训）        ← 1 次训练
  reuse_tsc    : 本方法 × TSC（TIFS'24 组合），仍只训 1 次         ← 1 次训练
  ceiling      : many(2000) 图充分训（上界，非同预算）

注：早先计划的 "PG-UAP" 文献查无此方法，已替换为上述可核验基线（见
sota_uap_baselines.py 顶部说明）。

落盘：results_fewshot_sota/results_fewshot_sota_{dataset}_{surrogate}_seed{seed}.json
"""
import argparse
import json
import time
import random
import torch
import numpy as np
from pathlib import Path

from config import Config
from models import create_all_models, create_all_imagenet_models
from utils.data import get_cifar10_samples, get_imagenet_mini_samples
from experiments import uap_sgd
from utils.losses import clamp_uap
from fewshot_uap_method import eval_on_models
from sota_uap_baselines import uap2017_md, ffa, ssa, tsc_uap, tsc_patch, tsc_assemble
from sota_modern import dmuap


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(is_imagenet, names, device):
    models = (create_all_imagenet_models(names) if is_imagenet
              else create_all_models(names, Config().num_classes, Config().image_size))
    for n in names:
        models[n] = models[n].to(device).eval()
    return models


def get_data(is_imagenet, many, few_k, device, eval_n=2000, eval_batch=0):
    """eval_n / eval_batch 只影响【评估成本】，不影响训练；缩小它们仅用于冒烟测试。

    注意：FR 由模型预测与标签决定，与 eval batch 大小无关（模型处于 eval() 模式，
    BN 用 running stats），所以 eval_batch 可以放心调大提速。
    """
    if is_imagenet:
        many_imgs, many_lbls = get_imagenet_mini_samples(num_samples=many, split="train", image_size=224, seed=100)
        eval_imgs, eval_lbls = get_imagenet_mini_samples(num_samples=eval_n, split="val", image_size=224, seed=42)
        eb = eval_batch if eval_batch > 0 else 16
    else:
        many_imgs, many_lbls = get_cifar10_samples(Config().data_root, num_samples=many, split="train", seed=100)
        eval_imgs, eval_lbls = get_cifar10_samples(Config().data_root, num_samples=eval_n, split="test", seed=42)
        eb = eval_batch if eval_batch > 0 else 128
    few_imgs, few_lbls = many_imgs[:few_k], many_lbls[:few_k]
    return (many_imgs.to(device), many_lbls.to(device),
            few_imgs.to(device), few_lbls.to(device),
            eval_imgs.to(device), eval_lbls.to(device), eb)


def run_seed(a, surrogate, seed, device, all_names):
    set_seed(seed)
    models = load_models(a.dataset == "imagenet", all_names, device)
    many_imgs, many_lbls, few_imgs, few_lbls, eval_imgs, eval_lbls, eb = get_data(
        a.dataset == "imagenet", a.many, a.few_k, device,
        eval_n=a.eval_n, eval_batch=a.eval_batch)
    model = models[surrogate]
    targets = [float(e) for e in a.eps_targets.split(",") if e.strip()]
    shape = tuple(few_imgs[0].shape)

    result = {"surrogate": surrogate, "seed": seed, "config": vars(a),
              "methods": {}, "ceiling": {}, "meta": {}}

    # ---- 1) reuse_lws：训 1 次 ε_w，缩放到全部目标 ε ----
    t0 = time.time()
    uap_w = uap_sgd(model, few_imgs, few_lbls, a.eps_w, steps=a.steps, lr=a.lr)
    result["meta"]["reuse_train_time_w"] = time.time() - t0
    result["meta"]["reuse_n_train_runs"] = 1
    result["methods"]["reuse_lws"] = {}
    for tgt in targets:
        uap_t = clamp_uap(uap_w * (tgt / a.eps_w), tgt)
        print(f"  [{surrogate}] seed{seed} reuse_lws ε={tgt:.2f} 评估中 ...", flush=True)
        result["methods"]["reuse_lws"][str(tgt)] = eval_on_models(
            uap_t, tgt, models, eval_imgs, eval_lbls, eb)
        print(f"  [{surrogate}] seed{seed} reuse_lws ε={tgt:.2f} "
              f"白盒FR={result['methods']['reuse_lws'][str(tgt)][surrogate]['fr_all']:.3f}", flush=True)

    # ---- 1b) reuse_tsc：本方法 × TSC（TIFS'24）组合，仍只训 1 次 ----
    t0 = time.time()
    v_w = tsc_patch(model, few_imgs, few_lbls, a.eps_w, alpha=a.tsc_alpha,
                    steps=a.steps, lr=a.lr)
    result["meta"]["reuse_tsc_train_time_w"] = time.time() - t0
    result["meta"]["reuse_tsc_n_train_runs"] = 1
    result["methods"]["reuse_tsc"] = {}
    for tgt in targets:
        uap_t = tsc_assemble(v_w, tgt, a.tsc_alpha, eps_in=a.eps_w)
        print(f"  [{surrogate}] seed{seed} reuse_tsc ε={tgt:.2f} 评估中 ...", flush=True)
        result["methods"]["reuse_tsc"][str(tgt)] = eval_on_models(
            uap_t, tgt, models, eval_imgs, eval_lbls, eb)
        print(f"  [{surrogate}] seed{seed} reuse_tsc ε={tgt:.2f} "
              f"白盒FR={result['methods']['reuse_tsc'][str(tgt)][surrogate]['fr_all']:.3f}", flush=True)

    # ---- 2) 需要按 ε 重训的基线 ----
    per_eps = {
        "direct":        lambda tgt: uap_sgd(model, few_imgs, few_lbls, tgt, steps=a.steps, lr=a.lr),
        "direct_lr0.05": lambda tgt: uap_sgd(model, few_imgs, few_lbls, tgt, steps=a.steps, lr=0.05),
        "uap2017":       lambda tgt: uap2017_md(model, few_imgs, few_lbls, tgt, steps=a.steps,
                                                lr=a.lr_uap2017, batch_size=a.batch_size),
        "ffa":           lambda tgt: ffa(model, shape, tgt, steps=a.steps, lr=a.lr_df,
                                         n_layers=a.df_layers, batch=a.batch_df, device=device),
        "ssa_energy":    lambda tgt: ssa(model, shape, tgt, steps=a.steps, lr=a.lr_df,
                                         n_layers=a.df_layers, batch=a.batch_df, device=device),
        "tsc":           lambda tgt: tsc_uap(model, few_imgs, few_lbls, tgt, alpha=a.tsc_alpha,
                                             steps=a.steps, lr=a.lr, batch_size=a.batch_size),
    }
    if not a.skip_modern:
        per_eps["dmuap"] = lambda tgt: dmuap(
            model, few_imgs, tgt, steps=a.dm_steps, batch_size=a.dm_batch,
            beta=a.dm_beta, step_decay=a.dm_step_decay, rho=a.dm_rho, aa=a.dm_aa,
            cc=a.dm_cc, theta_steps=a.dm_theta_steps, smooth_rate=a.dm_smooth,
            momentum=a.dm_momentum, cross_loss=a.dm_cross_loss)
    for m_name, fn in per_eps.items():
        result["methods"][m_name] = {}
        t_total = 0.0
        for tgt in targets:
            t0 = time.time()
            uap = fn(tgt)
            t_total += time.time() - t0
            print(f"  [{surrogate}] seed{seed} {m_name} ε={tgt:.2f} 训练完成({t_total:.0f}s)，"
                  f"评估 5 模型 × {len(eval_imgs)} 图 ...", flush=True)
            result["methods"][m_name][str(tgt)] = eval_on_models(
                uap, tgt, models, eval_imgs, eval_lbls, eb)
            print(f"  [{surrogate}] seed{seed} {m_name} ε={tgt:.2f} "
                  f"白盒FR={result['methods'][m_name][str(tgt)][surrogate]['fr_all']:.3f}", flush=True)
        result["meta"][f"{m_name}_train_time_total"] = t_total
        result["meta"][f"{m_name}_n_train_runs"] = len(targets)

    # ---- 3) ceiling ----
    for tgt in targets:
        uap_c = uap_sgd(model, many_imgs, many_lbls, tgt, steps=a.steps_ceiling, lr=a.lr)
        result["ceiling"][str(tgt)] = eval_on_models(uap_c, tgt, models, eval_imgs, eval_lbls, eb)
        print(f"  [{surrogate}] seed{seed} ceiling ε={tgt:.2f} "
              f"白盒FR={result['ceiling'][str(tgt)][surrogate]['fr_all']:.3f}", flush=True)

    model.cpu(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenet")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--models", default="auto")
    p.add_argument("--eps_targets", default="0.1,0.2,0.3")
    p.add_argument("--eps_w", type=float, default=0.05)
    p.add_argument("--few_k", type=int, default=200)
    p.add_argument("--many", type=int, default=2000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--steps_ceiling", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.01, help="direct/reuse 的 Adam lr")
    p.add_argument("--lr_uap2017", type=float, default=0.02, help="CVPR'17 累积步长")
    p.add_argument("--lr_df", type=float, default=0.02, help="data-free(ffa/ssa) Adam lr")
    p.add_argument("--df_layers", type=int, default=5, help="data-free 激活探针层数")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--batch_df", type=int, default=16, help="data-free 随机输入 batch")
    p.add_argument("--tsc_alpha", type=int, default=4,
                   help="TSC-UAP 纹理分割比 α（须整除 224；α=4 → 56×56 纹理块）")
    # ---- DM-UAP (AAAI 2025) 官方超参，见官方 run.sh ----
    p.add_argument("--skip_modern", action="store_true", help="跳过 DM-UAP（只跑 T1/T2/ours）")
    p.add_argument("--dm_steps", type=int, default=60,
                   help="DM-UAP 外层更新次数（默认 60≈1260 次图像回传，算力上偏向基线）")
    p.add_argument("--dm_batch", type=int, default=50, help="DM-UAP batch（官方 125）")
    p.add_argument("--dm_beta", type=float, default=9.0, help="clamped-CE 上界 β（官方 9）")
    p.add_argument("--dm_step_decay", type=float, default=0.1, help="官方 step_decay=0.1")
    p.add_argument("--dm_rho", type=float, default=4.0, help="权重扰动半径 ρ（官方 4）")
    p.add_argument("--dm_aa", type=float, default=25.0, help="x 对抗课程上界 aa（官方 25）")
    p.add_argument("--dm_cc", type=int, default=10, help="x 步数 cc（官方 10）")
    p.add_argument("--dm_theta_steps", type=int, default=10, help="权重扰动步数（官方 10）")
    p.add_argument("--dm_smooth", type=float, default=0.2, help="label smoothing（官方 0.2）")
    p.add_argument("--dm_momentum", type=int, default=0, help="官方 Momentum=0")
    p.add_argument("--dm_cross_loss", type=int, default=1, help="官方 cross_loss=1")
    p.add_argument("--seeds", default="42,1342,2342")
    p.add_argument("--eval_n", type=int, default=2000,
                   help="评估集大小（评估是主要耗时来源；冒烟可设 64，正式实验保持 2000）")
    p.add_argument("--eval_batch", type=int, default=0,
                   help="评估 batch；0=自动(imagenet 16 / cifar 128)。调大只提速不改结果")
    p.add_argument("--out_dir", default="results_fewshot_sota")
    p.add_argument("--out_tag", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_img = args.dataset == "imagenet"
    if args.models == "auto":
        names = (["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "vit_b_16"]
                 if is_img else Config().meta_train_models + Config().meta_test_models)
    else:
        names = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or args.surrogate

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        res = run_seed(args, args.surrogate, seed, device, names)
        with open(out_dir / f"results_fewshot_sota_{args.dataset}_{tag}_seed{seed}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"[保存] {out_dir}/results_fewshot_sota_{args.dataset}_{tag}_seed{seed}.json", flush=True)


if __name__ == "__main__":
    main()
