#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_supplementary.py — 论文补充实验四合一（审稿人预防性弹药）。

四种模式（--exp）：
  epsw   : warm-start 敏感性。eps_w in {0.02, 0.05, 0.1}，train@eps_w -> scale-up
           评估 {0.1,0.2,0.3}；回答「为什么 eps_w=0.05？对选择多敏感？」
  bsweep : 预算覆盖扫描。预算 {0.05,0.1,0.15,0.2,0.25,0.3}：
           scale-up(eps_w=0.05) 训 1 次评 6 点 vs direct per-eps 训 6 次；
           支撑 §6.5「savings grow linearly in B」。
  ksweep : 数据量扫描。k in {200,500,1000}，direct@{0.1,0.2,0.3} + scale-up；
           机制预测：k 增大 -> direct 欠饱和缓解 -> Δ 收窄（§5 机制的第三根支柱）。
           训练池嵌套（k=200 ⊂ 500 ⊂ 1000，同一 split seed），受控比较。
  l2     : L2 spot-check。eps_l2 in {5,10,20}，per-eps direct-L2 vs scale-up-L2；
           软化 Limitation 5「仅 L∞」。

协议与 run_fewshot_sota 完全一致：k=200 默认、split seed 100/42、steps=200、
lr=0.01（Adam 实现，与既有 direct 基线同一实现）、seeds {42,1342,2342}。

落盘：results_suppl/suppl_{exp}_{surrogate}_seed{seed}.json
自包含说明：复用 run_clipdown.py（自包含模块）的 load_models/get_data/set_seed，
不 import run_fewshot_sota（集群旧版签名坑）。
"""
import argparse
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from config import Config
from models import create_all_models, create_all_imagenet_models
from experiments import uap_sgd
from sota_uap_baselines import tsc_patch, tsc_assemble
from fewshot_uap_method import eval_on_models, sat_frac
from run_clipdown import set_seed, load_models, get_data


# ---------------------------------------------------------------- L2 基元
def project_l2(delta, eps):
    """投影到全局 L2 球：delta * min(1, eps/||delta||_2)（全张量标量缩放）。"""
    with torch.no_grad():
        nrm = float(delta.norm().item())
        scale = min(1.0, eps / max(nrm, 1e-12))
    return delta * scale


def uap_sgd_l2(model, images, labels, eps_l2, steps=200, lr=0.01, batch_size=128):
    """L2 版 UAP-SGD：与 uap_sgd 同结构（Adam, 同采样），仅换投影。"""
    uap = torch.zeros_like(images[0], requires_grad=True)
    optimizer = torch.optim.Adam([uap], lr=lr)
    n = min(len(images), batch_size)
    for step in range(steps):
        optimizer.zero_grad()
        idx = np.random.choice(len(images), n, replace=False)
        adv = (images[idx] + project_l2(uap, eps_l2)).clamp(0, 1)
        loss = -F.cross_entropy(model(adv), labels[idx])
        loss.backward()
        optimizer.step()
    return project_l2(uap.detach(), eps_l2)


# ---------------------------------------------------------------- 训练分派
def train_uap(name, model, imgs, lbls, eps, a, norm="linf"):
    """norm='l2' 时 eps 语义为 L2 半径。"""
    if norm == "l2":
        return uap_sgd_l2(model, imgs, lbls, eps, steps=a.steps, lr=a.lr,
                          batch_size=a.batch_size)
    if name == "direct":
        return uap_sgd(model, imgs, lbls, eps, steps=a.steps, lr=a.lr)
    if name == "tsc":
        v = tsc_patch(model, imgs, lbls, eps, alpha=a.tsc_alpha,
                      steps=a.steps, lr=a.lr, batch_size=a.batch_size)
        return tsc_assemble(v, eps, a.tsc_alpha)
    raise ValueError(name)


def run_seed(a, exp, surrogate, seed, device, all_names):
    set_seed(seed)
    models = load_models(a.dataset == "imagenet", all_names, device)
    model = models[surrogate]
    result = {"exp": exp, "surrogate": surrogate, "seed": seed,
              "config": vars(a), "runs": {}, "meta": {}}
    print(f"  [{surrogate}] seed{seed} exp={exp} 开始", flush=True)

    if exp == "epsw":
        few_imgs, few_lbls, eval_imgs, eval_lbls, eb = _data(a, device, a.few_k)
        for eps_w in a.eps_ws:
            for name in a.methods.split(","):
                t0 = time.time()
                dw = train_uap(name, model, few_imgs, few_lbls, eps_w, a)
                tr_t = time.time() - t0
                sw = float(dw.abs().max().item())
                sat_w = float(sat_frac(dw, eps_w))
                for eps in a.eps_targets:
                    d = dw * (eps / eps_w)          # scale-up
                    r = eval_on_models(d, eps, models, eval_imgs, eval_lbls, eb)
                    result["runs"][f"{name}@eps_w={eps_w}->eps={eps}"] = {
                        "fr": {m: r[m]["fr_all"] for m in r},
                        "sat": {m: r[m]["sat_frac"] for m in r},
                        "sat_w": sat_w, "train_s": tr_t,
                    }
                print(f"  [{surrogate}] s{seed} {name} eps_w={eps_w} done "
                      f"(sat_w={sat_w:.3f}, {tr_t:.0f}s)", flush=True)

    elif exp == "bsweep":
        few_imgs, few_lbls, eval_imgs, eval_lbls, eb = _data(a, device, a.few_k)
        eps_list = a.bsweep_budgets
        # ① scale-up：训 1 次评 |eps_list| 点
        t0 = time.time()
        dw = train_uap("tsc", model, few_imgs, few_lbls, a.eps_w, a)
        tr_t = time.time() - t0
        for eps in eps_list:
            d = dw * (eps / a.eps_w)
            r = eval_on_models(d, eps, models, eval_imgs, eval_lbls, eb)
            result["runs"][f"scaleup@eps={eps}"] = {
                "fr": {m: r[m]["fr_all"] for m in r}, "train_s": tr_t}
        # ② per-eps direct：每个预算重训
        for eps in eps_list:
            t0 = time.time()
            d = train_uap("direct", model, few_imgs, few_lbls, eps, a)
            tr_t = time.time() - t0
            sf = float(sat_frac(d, eps))
            r = eval_on_models(d, eps, models, eval_imgs, eval_lbls, eb)
            result["runs"][f"pereps@eps={eps}"] = {
                "fr": {m: r[m]["fr_all"] for m in r},
                "train_s": tr_t, "sat": sf}
        print(f"  [{surrogate}] s{seed} bsweep done "
              f"({len(eps_list)} budgets)", flush=True)

    elif exp == "ksweep":
        for k in a.ks:
            few_imgs, few_lbls, eval_imgs, eval_lbls, eb = _data(a, device, k)
            # direct per-eps
            for eps in a.eps_targets:
                t0 = time.time()
                d = train_uap("direct", model, few_imgs, few_lbls, eps, a)
                tr_t = time.time() - t0
                sf = float(sat_frac(d, eps))
                r = eval_on_models(d, eps, models, eval_imgs, eval_lbls, eb)
                result["runs"][f"direct@k={k},eps={eps}"] = {
                    "fr": {m: r[m]["fr_all"] for m in r},
                    "train_s": tr_t, "sat": sf}
            # scale-up
            t0 = time.time()
            dw = train_uap("tsc", model, few_imgs, few_lbls, a.eps_w, a)
            tr_t = time.time() - t0
            for eps in a.eps_targets:
                d = dw * (eps / a.eps_w)
                r = eval_on_models(d, eps, models, eval_imgs, eval_lbls, eb)
                result["runs"][f"scaleup@k={k},eps={eps}"] = {
                    "fr": {m: r[m]["fr_all"] for m in r}, "train_s": tr_t}
            print(f"  [{surrogate}] s{seed} k={k} done", flush=True)

    elif exp == "l2":
        few_imgs, few_lbls, eval_imgs, eval_lbls, eb = _data(a, device, a.few_k)
        for eps_l2 in a.l2_budgets:
            # per-eps L2 direct
            t0 = time.time()
            d = uap_sgd_l2(model, few_imgs, few_lbls, eps_l2,
                           steps=a.steps, lr=a.lr, batch_size=a.batch_size)
            tr_t = time.time() - t0
            l2n = float(d.flatten().norm(2).item())
            r = eval_on_models(d, eps_l2, models, eval_imgs, eval_lbls, eb)
            result["runs"][f"direct_l2@eps={eps_l2}"] = {
                "fr": {m: r[m]["fr_all"] for m in r},
                "train_s": tr_t, "l2_norm": l2n}
        # scale-up L2：训 1 次 eps_w_l2 -> 缩放到全部预算
        t0 = time.time()
        dw = uap_sgd_l2(model, few_imgs, few_lbls, a.eps_w_l2,
                        steps=a.steps, lr=a.lr, batch_size=a.batch_size)
        tr_t = time.time() - t0
        for eps_l2 in a.l2_budgets:
            d = project_l2(dw * (eps_l2 / a.eps_w_l2), eps_l2)
            r = eval_on_models(d, eps_l2, models, eval_imgs, eval_lbls, eb)
            result["runs"][f"scaleup_l2@eps={eps_l2}"] = {
                "fr": {m: r[m]["fr_all"] for m in r},
                "train_s": tr_t, "l2_norm": float(d.flatten().norm(2).item())}
        print(f"  [{surrogate}] s{seed} l2 done", flush=True)

    model.cpu(); torch.cuda.empty_cache()
    return result


def _data(a, device, few_k):
    """只取训练池前 few_k 与评估集（嵌套子集，受控比较）。"""
    (_, _, few_imgs, few_lbls, eval_imgs, eval_lbls, eb) = get_data(
        a.dataset == "imagenet", max(a.many, few_k), few_k, device,
        eval_n=a.eval_n, eval_batch=a.eval_batch)
    return few_imgs, few_lbls, eval_imgs, eval_lbls, eb


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True,
                   choices=["epsw", "bsweep", "ksweep", "l2"])
    p.add_argument("--dataset", default="imagenet")
    p.add_argument("--surrogate", default="resnet50")
    p.add_argument("--models", default="auto")
    p.add_argument("--methods", default="direct,tsc", help="epsw 模式用")
    p.add_argument("--eps_ws", type=_floats, default="0.02,0.05,0.1")
    p.add_argument("--eps_targets", type=_floats, default="0.1,0.2,0.3")
    p.add_argument("--bsweep_budgets", type=_floats,
                   default="0.05,0.1,0.15,0.2,0.25,0.3")
    p.add_argument("--ks", type=_ints, default="200,500,1000")
    p.add_argument("--l2_budgets", type=_floats, default="5,10,20")
    p.add_argument("--eps_w", type=float, default=0.05)
    p.add_argument("--eps_w_l2", type=float, default=2.0)
    p.add_argument("--few_k", type=int, default=200)
    p.add_argument("--many", type=int, default=2000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--tsc_alpha", type=int, default=4)
    p.add_argument("--seeds", type=_ints, default="42,1342,2342")
    p.add_argument("--eval_n", type=int, default=2000)
    p.add_argument("--eval_batch", type=int, default=0)
    p.add_argument("--out_dir", default="results_suppl")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_img = args.dataset == "imagenet"
    if args.models == "auto":
        names = (["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "vit_b_16"]
                 if is_img else Config().meta_train_models + Config().meta_test_models)
    else:
        names = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        res = run_seed(args, args.exp, args.surrogate, seed, device, names)
        path = out_dir / f"suppl_{args.exp}_{args.surrogate}_seed{seed}.json"
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[保存] {path}", flush=True)


if __name__ == "__main__":
    main()
