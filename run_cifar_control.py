#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_cifar_control.py — CIFAR-10 侧对照实验：验证「饱和区结论」跨数据集成立。

目的
----
ImageNet clip-down 裁决（analysis_clipdown_verdict.md）发现：
  · scale-up（低ε训1次→缩放）相对 clip-down（高ε训1次→裁剪）在欠饱和优化器
    （direct）上优势 14–26pp，在饱和优化器（tsc）上无差异；
  · 裁剪破坏力取决于「方向质量」而非「截断量」（tsc 裁 90% 坐标只掉 1.5pp，
    direct 裁 36% 崩 13.8pp）；
  · 机制量：sat_frac 解释 per-eps 直接训练为何崩，clip_frac 解释 clip-down
    的代价为何因方法而异。

CIFAR-10 的先验事实（X1 轮）：所有模型在 ε∈{0.1,0.2,0.3} 上 sat≈99% 全饱和。
两因子机制的预测（预注册）：
  · direct 在 CIFAR 上不欠饱和 → scale-up 无优势（Δ≈0，与 ImageNet 相反）；
  · per-eps direct 本身就强 → clipdown 与 scaleup 都应贴近 per-eps；
  · 若 CIFAR 上 scale-up 仍有优势，则「饱和调节」叙事被证伪。

三策略（同进程、同 seed、同数据 → 协议严格对齐）
----
  pereps   : 每个 ε 单独训（4 次训练：0.05/0.1/0.2/0.3）
  clipdown : train@ε_hi=0.3 → clamp_uap(δ, ε)          （复用 pereps@0.3 的 δ）
  scaleup  : train@ε_w=0.05 → δ × (ε/0.05)             （复用同一次低ε训练）

方法：direct / tsc / uap2017 / dmuap（与 ImageNet 主实验一致）
每个 (surrogate, seed, method) 只需 4 次训练同时支撑三策略。

落盘
----
  results_cifar_control/results_cifar_control_{surrogate}_seed{seed}.json

自包含：不 import run_clipdown / run_fewshot_sota（避免集群旧版驱动踩坑）。
"""
import argparse
import json
import time
import random
import torch
import numpy as np
from pathlib import Path

from config import Config
from models import create_all_models
from utils.data import get_cifar10_samples
from experiments import uap_sgd
from utils.losses import clamp_uap
from fewshot_uap_method import eval_on_models, sat_frac
from sota_uap_baselines import uap2017_md, tsc_patch, tsc_assemble
from sota_modern import dmuap


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_method(a, name, model, few_imgs, few_lbls, eps):
    """在预算 eps 上训练一次，返回 δ（|δ|∞ ≤ eps）。四种方法与 ImageNet 主实验同实现。"""
    if name == "direct":
        return uap_sgd(model, few_imgs, few_lbls, eps, steps=a.steps, lr=a.lr)
    if name == "uap2017":
        return uap2017_md(model, few_imgs, few_lbls, eps, steps=a.steps,
                          lr=a.lr_uap2017, batch_size=a.batch_size)
    if name == "tsc":
        v = tsc_patch(model, few_imgs, few_lbls, eps, alpha=a.tsc_alpha,
                      steps=a.steps, lr=a.lr, batch_size=a.batch_size)
        return tsc_assemble(v, eps, a.tsc_alpha)
    if name == "dmuap":
        return dmuap(
            model, few_imgs, eps, steps=a.dm_steps, batch_size=a.dm_batch,
            beta=a.dm_beta, step_decay=a.dm_step_decay, rho=a.dm_rho, aa=a.dm_aa,
            cc=a.dm_cc, theta_steps=a.dm_theta_steps, smooth_rate=a.dm_smooth,
            momentum=a.dm_momentum, cross_loss=a.dm_cross_loss)
    raise ValueError(f"unknown method: {name}")


def run_seed(a, surrogate, seed, device, all_names):
    set_seed(seed)
    models = create_all_models(all_names, Config().num_classes, Config().image_size)
    for n in all_names:
        models[n] = models[n].to(device).eval()

    many_imgs, many_lbls = get_cifar10_samples(
        Config().data_root, num_samples=a.many, split="train", seed=100)
    eval_imgs, eval_lbls = get_cifar10_samples(
        Config().data_root, num_samples=a.eval_n, split="test", seed=42)
    few_imgs, few_lbls = many_imgs[:a.few_k], many_lbls[:a.few_k]
    few_imgs, few_lbls = few_imgs.to(device), few_lbls.to(device)
    eval_imgs, eval_lbls = eval_imgs.to(device), eval_lbls.to(device)

    model = models[surrogate]
    eps_w, eps_hi = float(a.eps_w), float(a.eps_hi)
    budgets = sorted({float(e) for e in a.eps_targets.split(",") if e.strip()} | {eps_hi})
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    if a.skip_modern:
        methods = [m for m in methods if m != "dmuap"]

    result = {
        "dataset": "cifar10", "surrogate": surrogate, "seed": seed,
        "config": vars(a), "eps_w": eps_w, "eps_hi": eps_hi, "budgets": budgets,
        "methods": {}, "delta_stats": {}, "meta": {},
    }

    for name in methods:
        result["methods"][name] = {}

        # ---- 训练：每个 eps 点各训 1 次（pereps 直接用；clipdown 复用 eps_hi；scaleup 复用 eps_w）----
        deltas = {}
        for eps in [eps_w] + budgets:
            t0 = time.time()
            d = train_method(a, name, model, few_imgs, few_lbls, eps)
            deltas[eps] = d
            result["meta"][f"{name}_train_time@{eps}"] = time.time() - t0
            print(f"  [{surrogate}] seed{seed} {name} train@{eps:.2f} 完成 "
                  f"({time.time()-t0:.0f}s) Linf={d.abs().max().item():.4f} "
                  f"sat={sat_frac(d, eps):.3f}", flush=True)

        # ---- 三策略构造 + 评估 ----
        for eps in budgets:
            strategies = {
                "pereps": deltas[eps],                      # 直接训出来的
                "clipdown": (clamp_uap(deltas[eps_hi], eps)   # 高ε训1次→裁剪
                             if eps < eps_hi - 1e-9 else deltas[eps_hi]),
                "scaleup": deltas[eps_w] * (eps / eps_w),   # 低ε训1次→线性放大
            }
            result["methods"][name][str(eps)] = {}
            for sname, d in strategies.items():
                tag = "asis" if sname == "pereps" else sname
                result["methods"][name][str(eps)][sname] = eval_on_models(
                    d, eps, models, eval_imgs, eval_lbls, a.eval_batch)
                fr = result["methods"][name][str(eps)][sname][surrogate]["fr_all"]
                print(f"  [{surrogate}] seed{seed} {name} ε={eps:.2f} [{sname}] "
                      f"白盒FR={fr:.3f}", flush=True)
            # 机制量：clipdown 的截断比例、scaleup 后的饱和比例
            d_hi, d_lo = deltas[eps_hi], deltas[eps_w]
            result["delta_stats"][f"{name}@{eps}"] = {
                "clip_frac": float((d_hi.abs() > eps + 1e-4).float().mean().item()),
                "sat_clipdown": sat_frac(strategies["clipdown"], eps),
                "sat_scaleup": sat_frac(strategies["scaleup"], eps),
                "sat_pereps": sat_frac(strategies["pereps"], eps),
                "sat_train@eps_w": sat_frac(d_lo, eps_w),
                "sat_train@eps_hi": sat_frac(d_hi, eps_hi),
            }

    for n in all_names:
        models[n].cpu()
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate", required=True)
    p.add_argument("--models", default="auto",
                   help="auto = Config.meta_train + meta_test 共 8 个 CIFAR 模型")
    p.add_argument("--methods", default="direct,tsc,uap2017,dmuap")
    p.add_argument("--eps_w", type=float, default=0.05, help="scaleup 的热身预算")
    p.add_argument("--eps_hi", type=float, default=0.3, help="clipdown 的训练预算")
    p.add_argument("--eps_targets", default="0.1,0.2,0.3")
    p.add_argument("--few_k", type=int, default=200)
    p.add_argument("--many", type=int, default=2000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr_uap2017", type=float, default=0.02)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--tsc_alpha", type=int, default=4)
    p.add_argument("--skip_modern", action="store_true")
    p.add_argument("--dm_steps", type=int, default=60)
    p.add_argument("--dm_batch", type=int, default=128)
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
    p.add_argument("--eval_batch", type=int, default=512)
    p.add_argument("--out_dir", default="results_cifar_control")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.models == "auto":
        names = Config().meta_train_models + Config().meta_test_models
    else:
        names = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        res = run_seed(args, args.surrogate, seed, device, names)
        path = out_dir / f"results_cifar_control_{args.surrogate}_seed{seed}.json"
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[保存] {path}", flush=True)


if __name__ == "__main__":
    main()
