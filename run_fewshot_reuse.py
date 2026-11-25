#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_fewshot_reuse.py — 实验 (A)：LWS 复用部署验证。

核心问题：能否「训一个低 ε_w UAP → 部署期缩放到任意大 ε_t，不重训」，
且质量匹配「每个 ε_t 单独训的 direct」？若是，则 LWS 的贡献 =
「预算解耦的单扰动部署（1 次训练 vs N 次）」，而非「只是调大 lr」。

方法（同 few-shot 数据预算 K=200，除 Ceiling 外）：
  reuse_lws       : 在 ε_w=0.05 训 1 个 UAP（uap_sgd, lr=0.01, 200 步），
                    对 {0.1,0.2,0.3} 各线性缩放（保形），不重训。  ← OURS / 1 次训练
  direct          : 每个 ε_t 单独用 uap_sgd(lr=0.01) 训。          ← 默认基线 / 3 次训练
  direct_lr0.05   : 每个 ε_t 单独用 uap_sgd(lr=0.05) 训。          ← 强基线 / 3 次训练
  ceiling         : 每个 ε_t 在 many(2000) 图上充分训（上界）。     ← 上界 / 非同预算

评估：每个 UAP 在 surrogate（白盒）+ 其余 3 黑盒上测 FR/ASR/饱和率。
落盘：results_fewshot_reuse/results_fewshot_reuse_{dataset}_{surrogate}_seed{seed}.json
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
from fewshot_uap_method import eval_on_models, sat_frac


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


def get_data(is_imagenet, many, few_k, device):
    if is_imagenet:
        many_imgs, many_lbls = get_imagenet_mini_samples(num_samples=many, split="train", image_size=224, seed=100)
        eval_imgs, eval_lbls = get_imagenet_mini_samples(num_samples=2000, split="val", image_size=224, seed=42)
        eb = 16
    else:
        many_imgs, many_lbls = get_cifar10_samples(Config().data_root, num_samples=many, split="train", seed=100)
        eval_imgs, eval_lbls = get_cifar10_samples(Config().data_root, num_samples=2000, split="test", seed=42)
        eb = 128
    few_imgs, few_lbls = many_imgs[:few_k], many_lbls[:few_k]
    return (many_imgs.to(device), many_lbls.to(device),
            few_imgs.to(device), few_lbls.to(device),
            eval_imgs.to(device), eval_lbls.to(device), eb)


def run_seed(a, surrogate, seed, device, all_names):
    set_seed(seed)
    models = load_models(a.dataset == "imagenet", all_names, device)
    many_imgs, many_lbls, few_imgs, few_lbls, eval_imgs, eval_lbls, eb = get_data(
        a.dataset == "imagenet", a.many, a.few_k, device)
    model = models[surrogate]
    targets = [float(e) for e in a.eps_targets.split(",") if e.strip()]

    result = {"surrogate": surrogate, "seed": seed, "config": vars(a),
              "methods": {}, "ceiling": {}, "meta": {}}

    # ---- reuse_lws：训 1 个 ε_w UAP，缩放到所有目标 ε（不重训）----
    t0 = time.time()
    uap_w = uap_sgd(model, few_imgs, few_lbls, a.eps_w, steps=a.steps, lr=a.lr)
    train_time_w = time.time() - t0
    result["meta"]["reuse_train_time_w"] = train_time_w
    result["meta"]["reuse_n_train_runs"] = 1
    result["methods"]["reuse_lws"] = {}
    for tgt in targets:
        uap_t = clamp_uap(uap_w * (tgt / a.eps_w), tgt)   # 保形缩放，零重训
        result["methods"]["reuse_lws"][str(tgt)] = eval_on_models(
            uap_t, tgt, models, eval_imgs, eval_lbls, eb)
        print(f"  [{surrogate}] seed{seed} reuse_lws ε={tgt:.2f} "
              f"白盒FR={result['methods']['reuse_lws'][str(tgt)][surrogate]['fr_all']:.3f}")

    # ---- direct / direct_lr0.05：每个 ε_t 单独训 ----
    for m_name, lr in [("direct", a.lr), ("direct_lr0.05", 0.05)]:
        result["methods"][m_name] = {}
        t_total = 0.0
        for tgt in targets:
            t0 = time.time()
            uap = uap_sgd(model, few_imgs, few_lbls, tgt, steps=a.steps, lr=lr)
            t_total += time.time() - t0
            result["methods"][m_name][str(tgt)] = eval_on_models(
                uap, tgt, models, eval_imgs, eval_lbls, eb)
            print(f"  [{surrogate}] seed{seed} {m_name} ε={tgt:.2f} "
                  f"白盒FR={result['methods'][m_name][str(tgt)][surrogate]['fr_all']:.3f}")
        result["meta"][f"{m_name}_train_time_total"] = t_total
        result["meta"][f"{m_name}_n_train_runs"] = len(targets)

    # ---- ceiling：many 数据充分训（上界）----
    for tgt in targets:
        uap_c = uap_sgd(model, many_imgs, many_lbls, tgt, steps=a.steps_ceiling, lr=a.lr)
        result["ceiling"][str(tgt)] = eval_on_models(uap_c, tgt, models, eval_imgs, eval_lbls, eb)
        print(f"  [{surrogate}] seed{seed} ceiling ε={tgt:.2f} "
              f"白盒FR={result['ceiling'][str(tgt)][surrogate]['fr_all']:.3f}")

    model.cpu(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenet")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--models", default="auto")
    p.add_argument("--eps_targets", default="0.1,0.2,0.3")
    p.add_argument("--eps_w", type=float, default=0.05, help="复用训练 ε（热身预算）")
    p.add_argument("--few_k", type=int, default=200)
    p.add_argument("--many", type=int, default=2000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--steps_ceiling", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seeds", default="42,1342,2342")
    p.add_argument("--out_dir", default="results_fewshot_reuse")
    p.add_argument("--out_tag", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_img = args.dataset == "imagenet"
    if args.models == "auto":
        names = (["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2"]
                 if is_img else Config().meta_train_models + Config().meta_test_models)
    else:
        names = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or args.surrogate

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        res = run_seed(args, args.surrogate, seed, device, names)
        with open(out_dir / f"results_fewshot_reuse_{args.dataset}_{tag}_seed{seed}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"[保存] {out_dir}/results_fewshot_reuse_{args.dataset}_{tag}_seed{seed}.json")


if __name__ == "__main__":
    main()
