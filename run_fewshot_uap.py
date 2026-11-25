#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_fewshot_uap.py — LWS 方法化对比驱动（白盒 + 黑盒迁移）。

实验矩阵（均在同一 few-shot 数据预算 K 下公平对比，除 Ceiling 外）：
  方法            | 训练 ε     | lr          | 步数        | 说明
  ----------------+-----------+-------------+-------------+------------------------
  direct          | ε_t       | lr          | steps       | 主竞争者（目标ε直接训）
  direct_lr0.002  | ε_t       | 0.002       | steps       | lr 扫描，反驳“只是没调lr”
  direct_lr0.05   | ε_t       | 0.05        | steps       | lr 扫描
  pdua            | ε_t       | lr          | steps       | Moosavi-Dezfooli 已发表优化器
  lws             | ε_w→ε_t   | lr          | steps(+0)   | OURS：低ε热身+缩放
  lws_r           | ε_w→ε_t   | lr          | steps(+ref) | OURS：LWS + 精炼
  Ceiling         | ε_t       | lr          | steps_ceiling | 上界（2000图/多步，非同预算）

评估：每个 UAP 在 surrogate（白盒）+ 其余 3 个黑盒模型上测 FR/ASR/饱和率。
输出：results_fewshot/results_fewshot_{dataset}_{surrogate}_seed{seed}.json（逐 seed 落盘）。
"""
import argparse
import json
import random
import time
import torch
import numpy as np
from pathlib import Path

from config import Config
from models import create_all_models, create_all_imagenet_models
from utils.data import get_cifar10_samples, get_imagenet_mini_samples
from experiments import uap_sgd, pd_ua
from fewshot_uap_method import lws_uap, eval_on_models


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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
        many_imgs, many_lbls = get_imagenet_mini_samples(
            num_samples=many, split="train", image_size=224, seed=100)
        eval_imgs, eval_lbls = get_imagenet_mini_samples(
            num_samples=2000, split="val", image_size=224, seed=42)
        eb = 16
    else:
        many_imgs, many_lbls = get_cifar10_samples(Config().data_root,
                                                   num_samples=many, split="train", seed=100)
        eval_imgs, eval_lbls = get_cifar10_samples(Config().data_root,
                                                   num_samples=2000, split="test", seed=42)
        eb = 128
    few_imgs, few_lbls = many_imgs[:few_k], many_lbls[:few_k]
    return (many_imgs.to(device), many_lbls.to(device),
            few_imgs.to(device), few_lbls.to(device),
            eval_imgs.to(device), eval_lbls.to(device), eb)


def train_uap(method, model, few_imgs, few_lbls, eps_t, a):
    """按方法名训 UAP（均在 few 数据预算下，除 ceiling 外）。"""
    if method == "direct":
        return uap_sgd(model, few_imgs, few_lbls, eps_t, steps=a.steps, lr=a.lr)
    if method.startswith("direct_lr"):
        lr = float(method.split("direct_lr")[1])
        return uap_sgd(model, few_imgs, few_lbls, eps_t, steps=a.steps, lr=lr)
    if method == "pdua":
        return pd_ua(model, few_imgs, few_lbls, eps_t, steps=a.steps, lr=a.lr)
    if method == "lws":
        return lws_uap(model, few_imgs, few_lbls, a.eps_w, eps_t,
                       a.steps, a.lr, refine_steps=0)
    if method == "lws_r":
        return lws_uap(model, few_imgs, few_lbls, a.eps_w, eps_t,
                       a.steps, a.lr, refine_steps=a.refine)
    raise ValueError(f"未知方法: {method}")


def run_seed(a, surrogate, seed, device, all_names):
    set_seed(seed)
    models = load_models(a.dataset == "imagenet", all_names, device)
    many_imgs, many_lbls, few_imgs, few_lbls, eval_imgs, eval_lbls, eb = get_data(
        a.dataset == "imagenet", a.many, a.few_k, device)
    model = models[surrogate]

    methods = ["direct", "direct_lr0.002", "direct_lr0.05", "pdua", "lws", "lws_r"]
    targets = [float(e) for e in a.eps_targets.split(",") if e.strip()]

    result = {"surrogate": surrogate, "seed": seed,
              "config": vars(a), "methods": {}, "ceiling": {}}

    for m in methods:
        result["methods"][m] = {}
        for tgt in targets:
            t0 = time.time()
            uap = train_uap(m, model, few_imgs, few_lbls, tgt, a)
            result["methods"][m][str(tgt)] = eval_on_models(
                uap, tgt, models, eval_imgs, eval_lbls, eb)
            print(f"  [{surrogate}] seed{seed} {m} ε={tgt:.2f} "
                  f"白盒FR={result['methods'][m][str(tgt)][surrogate]['fr_all']:.3f} "
                  f"({time.time()-t0:.1f}s)")

    # Ceiling：目标 ε 在 many(2000) 图上充分训（上界参考，非同数据预算）
    for tgt in targets:
        uap_c = uap_sgd(model, many_imgs, many_lbls, tgt,
                        steps=a.steps_ceiling, lr=a.lr)
        result["ceiling"][str(tgt)] = eval_on_models(
            uap_c, tgt, models, eval_imgs, eval_lbls, eb)
        print(f"  [{surrogate}] seed{seed} ceiling ε={tgt:.2f} "
              f"白盒FR={result['ceiling'][str(tgt)][surrogate]['fr_all']:.3f}")

    model.cpu(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenet")
    p.add_argument("--surrogate", required=True,
                   help="训练 UAP 的模型（白盒）；其余作为黑盒迁移目标")
    p.add_argument("--models", default="auto",
                   help="加载的全部模型（含 surrogate + 黑盒）。auto=imagenet 4 模型")
    p.add_argument("--eps_targets", default="0.1,0.2,0.3")
    p.add_argument("--eps_w", type=float, default=0.05, help="LWS 热身 ε")
    p.add_argument("--few_k", type=int, default=200, help="few-shot 数据预算")
    p.add_argument("--many", type=int, default=2000)
    p.add_argument("--steps", type=int, default=200, help="方法训练步数")
    p.add_argument("--steps_ceiling", type=int, default=500)
    p.add_argument("--refine", type=int, default=50, help="lws_r 精炼步数")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seeds", default="42,1342,2342")
    p.add_argument("--out_dir", default="results_fewshot")
    p.add_argument("--out_tag", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_img = args.dataset == "imagenet"
    if args.models == "auto":
        names = (["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2"]
                 if is_img else Config().meta_train_models + Config().meta_test_models)
    else:
        names = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or args.surrogate

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        res = run_seed(args, args.surrogate, seed, device, names)
        with open(out_dir / f"results_fewshot_{args.dataset}_{tag}_seed{seed}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"[保存] {out_dir}/results_fewshot_{args.dataset}_{tag}_seed{seed}.json")


if __name__ == "__main__":
    main()
