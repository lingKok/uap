#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_clipdown.py — clip-down 反驳基线：判定「预算解耦」是否只是「任何扰动都能裁」。

背景
----
LWS（ours）主张：低 ε_w 训 1 次 → 线性缩放到任意 ε，省掉重复训练。
但存在一个同样只训 1 次、更朴素的竞争策略：

    clip-down：在最大预算 ε_hi 训 1 次 → 把同一个 δ 裁剪到各小预算。

若 clip-down 也能达到相近 FR，则「预算解耦」不需要 LWS（任何扰动都能被裁），
我们的主张当场失效；若 clip-down 显著掉点，则「缩放保形 > 裁剪截断」成立，
预算解耦从工程 trick 升格为有机制解释的结论。

对照组（训练次数严格对齐，均为 1 次）
----
  clipdown_{m} : train@{ε_hi} → clip(δ, ε)          ← 本脚本（1 次训练）
  reuse_{m}    : train@{ε_w}  → scale(δ, ε/ε_w)     ← results_fewshot_sota/（1 次训练）
  {m}          : 每个 ε 单独训                        ← results_fewshot_sota/（3 次训练，非对齐）

方法覆盖
----
  direct   : uap_sgd(Adam, lr=0.01)，LWS 的基优化器
  tsc      : TSC-UAP（TIFS 2024）
  uap2017  : Moovasi-Dezfooli（CVPR 2017）
  dmuap    : DM-UAP（AAAI 2025，官方代码）

落盘
----
  results_clipdown/results_clipdown_{dataset}_{surrogate}_seed{seed}.json

自包含说明：本脚本不 import run_fewshot_sota，避免集群上残留旧版驱动导致
函数签名不匹配（本项目已多次踩到这个坑）。
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
from sota_uap_baselines import uap2017_md, tsc_patch, tsc_assemble
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
    """eval_n / eval_batch 只影响评估成本，不改 FR（模型 eval() 模式）。"""
    if is_imagenet:
        many_imgs, many_lbls = get_imagenet_mini_samples(
            num_samples=many, split="train", image_size=224, seed=100)
        eval_imgs, eval_lbls = get_imagenet_mini_samples(
            num_samples=eval_n, split="val", image_size=224, seed=42)
        eb = eval_batch if eval_batch > 0 else 16
    else:
        many_imgs, many_lbls = get_cifar10_samples(
            Config().data_root, num_samples=many, split="train", seed=100)
        eval_imgs, eval_lbls = get_cifar10_samples(
            Config().data_root, num_samples=eval_n, split="test", seed=42)
        eb = eval_batch if eval_batch > 0 else 128
    few_imgs, few_lbls = many_imgs[:few_k], many_lbls[:few_k]
    return (many_imgs.to(device), many_lbls.to(device),
            few_imgs.to(device), few_lbls.to(device),
            eval_imgs.to(device), eval_lbls.to(device), eb)


def train_at_eps_hi(a, name, model, few_imgs, few_lbls, shape, device, eps_hi):
    """在 ε_hi 上训 1 次，返回全尺寸扰动 δ（|δ|∞ ≤ ε_hi）。"""
    if name == "direct":
        return uap_sgd(model, few_imgs, few_lbls, eps_hi, steps=a.steps, lr=a.lr)
    if name == "direct_lr0.05":
        return uap_sgd(model, few_imgs, few_lbls, eps_hi, steps=a.steps, lr=0.05)
    if name == "uap2017":
        return uap2017_md(model, few_imgs, few_lbls, eps_hi, steps=a.steps,
                          lr=a.lr_uap2017, batch_size=a.batch_size)
    if name == "tsc":
        v = tsc_patch(model, few_imgs, few_lbls, eps_hi, alpha=a.tsc_alpha,
                      steps=a.steps, lr=a.lr, batch_size=a.batch_size)
        return tsc_assemble(v, eps_hi, a.tsc_alpha)
    if name == "dmuap":
        return dmuap(
            model, few_imgs, eps_hi, steps=a.dm_steps, batch_size=a.dm_batch,
            beta=a.dm_beta, step_decay=a.dm_step_decay, rho=a.dm_rho, aa=a.dm_aa,
            cc=a.dm_cc, theta_steps=a.dm_theta_steps, smooth_rate=a.dm_smooth,
            momentum=a.dm_momentum, cross_loss=a.dm_cross_loss)
    raise ValueError(f"unknown method: {name}")


def run_seed(a, surrogate, seed, device, all_names):
    set_seed(seed)
    models = load_models(a.dataset == "imagenet", all_names, device)
    many_imgs, many_lbls, few_imgs, few_lbls, eval_imgs, eval_lbls, eb = get_data(
        a.dataset == "imagenet", a.many, a.few_k, device,
        eval_n=a.eval_n, eval_batch=a.eval_batch)
    model = models[surrogate]
    shape = tuple(few_imgs[0].shape)
    eps_hi = float(a.eps_hi)
    budgets = sorted({float(e) for e in a.eps_targets.split(",") if e.strip()} | {eps_hi})

    result = {
        "surrogate": surrogate, "seed": seed, "config": vars(a),
        "eps_hi": eps_hi, "budgets": budgets,
        "methods": {}, "delta_stats": {}, "meta": {},
    }

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    if a.skip_modern:
        methods = [m for m in methods if m != "dmuap"]

    for name in methods:
        t0 = time.time()
        delta = train_at_eps_hi(a, name, model, few_imgs, few_lbls, shape, device, eps_hi)
        result["meta"][f"{name}_train_time_hi"] = time.time() - t0
        result["meta"][f"{name}_n_train_runs"] = 1
        linf = float(delta.abs().max().item())
        l2 = float(delta.flatten().norm(2).item())
        print(f"  [{surrogate}] seed{seed} {name} train@{eps_hi:.2f} 完成 "
              f"({result['meta'][f'{name}_train_time_hi']:.0f}s) "
              f"Linf={linf:.4f} L2={l2:.2f}", flush=True)

        result["methods"][name] = {}
        for eps in budgets:
            if eps < eps_hi - 1e-9:
                d = clamp_uap(delta, eps)     # clip-down：同一个 δ，硬截断到 ε 球
                mode = "clipdown"
            else:
                d = delta                      # 原样（γ 不改）
                mode = "asis"
            clip_frac = float((delta.abs() > eps + 1e-4).float().mean().item())
            result["delta_stats"][f"{name}@{eps}"] = {
                "mode": mode, "linf": linf, "l2": l2, "clip_frac": clip_frac,
            }
            print(f"  [{surrogate}] seed{seed} {name} [{mode}] ε={eps:.2f} "
                  f"(clip_frac={clip_frac:.3f}) 评估中 ...", flush=True)
            result["methods"][name][str(eps)] = eval_on_models(
                d, eps, models, eval_imgs, eval_lbls, eb)
            print(f"  [{surrogate}] seed{seed} {name} [{mode}] ε={eps:.2f} "
                  f"白盒FR={result['methods'][name][str(eps)][surrogate]['fr_all']:.3f} "
                  f"sat={result['methods'][name][str(eps)][surrogate]['sat_frac']:.3f}", flush=True)

    model.cpu(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenet")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--models", default="auto")
    p.add_argument("--methods", default="direct,tsc,uap2017,dmuap",
                   help="逗号分隔；可选 direct,direct_lr0.05,uap2017,tsc,dmuap")
    p.add_argument("--eps_hi", type=float, default=0.3,
                   help="训练预算（clip-down 的起点，通常取最大预算）")
    p.add_argument("--eps_targets", default="0.1,0.2,0.3",
                   help="评估预算；会自动并入 eps_hi")
    p.add_argument("--few_k", type=int, default=200)
    p.add_argument("--many", type=int, default=2000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr_uap2017", type=float, default=0.02)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--tsc_alpha", type=int, default=4)
    p.add_argument("--skip_modern", action="store_true", help="跳过 DM-UAP（省时）")
    p.add_argument("--dm_steps", type=int, default=60)
    p.add_argument("--dm_batch", type=int, default=50)
    p.add_argument("--dm_beta", type=float, default=9.0)
    p.add_argument("--dm_step_decay", type=float, default=0.1)
    p.add_argument("--dm_rho", type=float, default=4.0)
    p.add_argument("--dm_aa", type=float, default=25.0)
    p.add_argument("--dm_cc", type=int, default=10)
    p.add_argument("--dm_theta_steps", type=int, default=10)
    p.add_argument("--dm_smooth", type=float, default=0.2)
    p.add_argument("--dm_momentum", type=int, default=0)
    p.add_argument("--dm_cross_loss", type=int, default=1)
    p.add_argument("--seeds", default="42,1342,2342")
    p.add_argument("--eval_n", type=int, default=2000)
    p.add_argument("--eval_batch", type=int, default=0)
    p.add_argument("--out_dir", default="results_clipdown")
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
        path = out_dir / f"results_clipdown_{args.dataset}_{tag}_seed{seed}.json"
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[保存] {path}", flush=True)


if __name__ == "__main__":
    main()
