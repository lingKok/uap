#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_p2_targets.py -- P2: 现代架构评估目标扩容（ConvNeXt-T / Swin-T / EffNet-B0）。

动机
----
现有 5 个评估目标 = 4 CNN + 1 ViT（vit_b_16），最年轻的 CNN 是 2019 年的
EfficientNet 没进池、ConvNeXt(2022)/Swin(2021) 都不在。审稿人会问：
「换现代架构做目标，结论还成立吗？」

设计
----
- surrogate 固定 resnet50（论文主 surrogate），3 seeds；
- 方法取论文四条主线：reuse_lws / reuse_tsc（各训 1 次 eps_w=0.05），
  direct / tsc（每 ε 重训）——实现与超参逐字对齐 run_fewshot_sota.py；
- 评估池扩为 8 目标：原 5 个 + convnext_tiny + swin_t + efficientnet_b0，
  每个 (方法, ε) 都在全部 8 个上评估（原 5 列同时充当重训一致性参照）；
- 配对比较在同一批新训 delta 内部进行（scale-up vs direct 同 seed 配对），
  与旧结果 json 的逐位一致性不构成主张（独立进程 RNG 顺序噪声 ~<=2pp，
  见表 0 判据）。

预注册预测
----
- P-a：scale-up 在 3 个新目标上仍 >= direct（与主表方向一致）；
- P-b：新目标绝对 FR 落在原池范围内（无"现代架构免疫"效应）；
- 若 P-a 反转：条件化边界须加"目标架构代际"一条。

落盘
----
  results_p2/results_p2_imagenet_<surrogate>_seed<seed>.json
  结构同 results_fewshot_sota（methods -> {eps -> {model -> metrics}}），
  make_tables.py 复用 fr_sota 访问器。

权重预下载（登录节点，一次性）：
  python -c "from torchvision.models import convnext_tiny, swin_t, efficientnet_b0;\
  convnext_tiny(weights='IMAGENET1K_V1'); swin_t(weights='IMAGENET1K_V1');\
  efficientnet_b0(weights='IMAGENET1K_V1')"
若登录节点无外网：本地下载 3 个权重放入 ~/.cache/torch/hub/checkpoints/ 再上传，
文件名 convnext_tiny-983f1562.pth / swin_t-704fadba.pth /
efficientnet_b0_rwightman-23ab8bcd.pth。
"""
import argparse
import json
import time
import random
import traceback
import torch
import numpy as np
from pathlib import Path

from models import create_all_imagenet_models
from utils.data import get_imagenet_mini_samples
from experiments import uap_sgd
from utils.losses import clamp_uap
from fewshot_uap_method import eval_on_models
from sota_uap_baselines import tsc_uap, tsc_patch, tsc_assemble

BASE_TARGETS = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "vit_b_16"]
NEW_TARGETS = ["convnext_tiny", "swin_t", "efficientnet_b0"]


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_seed(a, surrogate, seed, device):
    set_seed(seed)
    all_names = BASE_TARGETS + NEW_TARGETS
    models = create_all_imagenet_models(all_names)
    for n in all_names:
        models[n] = models[n].to(device).eval()

    many_imgs, many_lbls = get_imagenet_mini_samples(
        num_samples=a.many, split="train", image_size=224, seed=100)
    eval_imgs, eval_lbls = get_imagenet_mini_samples(
        num_samples=a.eval_n, split="val", image_size=224, seed=42)
    eb = a.eval_batch if a.eval_batch > 0 else 64
    few_imgs, few_lbls = many_imgs[:a.few_k], many_lbls[:a.few_k]
    model = models[surrogate]
    targets = sorted({float(e) for e in a.eps_targets.split(",") if e.strip()})

    result = {"surrogate": surrogate, "seed": seed, "config": vars(a),
              "targets": all_names, "new_targets": NEW_TARGETS,
              "methods": {}, "meta": {}}

    def ev(m_name, uap, tgt, t_train):
        print(f"  [{surrogate}] seed{seed} {m_name} eps={tgt:.2f} "
              f"评估 8 模型 x {len(eval_imgs)} 图 ...", flush=True)
        res = eval_on_models(uap, tgt, models, eval_imgs, eval_lbls, eb)
        result["methods"][m_name][str(tgt)] = res
        print(f"  [{surrogate}] seed{seed} {m_name} eps={tgt:.2f} "
              f"白盒FR={res[surrogate]['fr_all']:.3f} "
              f"新目标均值={np.mean([res[t]['fr_all'] for t in NEW_TARGETS]):.3f} "
              f"(训练 {t_train:.0f}s)", flush=True)

    # ---- 1) reuse_lws：训 1 次 eps_w，缩放到全部目标 eps ----
    result["methods"]["reuse_lws"] = {}
    t0 = time.time()
    uap_w = uap_sgd(model, few_imgs, few_lbls, a.eps_w, steps=a.steps, lr=a.lr)
    result["meta"]["reuse_train_time_w"] = time.time() - t0
    for tgt in targets:
        ev("reuse_lws", clamp_uap(uap_w * (tgt / a.eps_w), tgt), tgt,
           result["meta"]["reuse_train_time_w"])

    # ---- 2) reuse_tsc：本方法 x TSC，仍只训 1 次 ----
    result["methods"]["reuse_tsc"] = {}
    t0 = time.time()
    v_w = tsc_patch(model, few_imgs, few_lbls, a.eps_w, alpha=a.tsc_alpha,
                    steps=a.steps, lr=a.lr)
    result["meta"]["reuse_tsc_train_time_w"] = time.time() - t0
    for tgt in targets:
        ev("reuse_tsc", tsc_assemble(v_w, tgt, a.tsc_alpha, eps_in=a.eps_w),
           tgt, result["meta"]["reuse_tsc_train_time_w"])

    # ---- 3) per-eps 基线 ----
    per_eps = {
        "direct": lambda tgt: uap_sgd(model, few_imgs, few_lbls, tgt,
                                      steps=a.steps, lr=a.lr),
        "tsc":    lambda tgt: tsc_uap(model, few_imgs, few_lbls, tgt,
                                      alpha=a.tsc_alpha, steps=a.steps,
                                      lr=a.lr, batch_size=a.batch_size),
    }
    for m_name, fn in per_eps.items():
        result["methods"][m_name] = {}
        t_total = 0.0
        for tgt in targets:
            t0 = time.time()
            uap = fn(tgt)
            t_total += time.time() - t0
            ev(m_name, uap, tgt, t_total)
        result["meta"][f"{m_name}_train_time_total"] = t_total

    model.cpu(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenet")
    p.add_argument("--surrogate", default="resnet50")
    p.add_argument("--eps_targets", default="0.1,0.2,0.3")
    p.add_argument("--eps_w", type=float, default=0.05)
    p.add_argument("--few_k", type=int, default=200)
    p.add_argument("--many", type=int, default=2000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--tsc_alpha", type=int, default=4)
    p.add_argument("--seeds", default="42,1342,2342")
    p.add_argument("--eval_n", type=int, default=2000)
    p.add_argument("--eval_batch", type=int, default=0)
    p.add_argument("--out_dir", default="results_p2")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset != "imagenet":
        raise ValueError("P2 只在 ImageNet 上跑（对照主表协议）")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[P2] surrogate={args.surrogate} seeds={args.seeds} "
          f"new_targets={NEW_TARGETS} n_iter-free (eval 扩容)", flush=True)
    # 权重预检：新目标模型能否成功构建（触发缓存下载，失败早退）
    try:
        for n in NEW_TARGETS:
            m = create_all_imagenet_models([n])[n]
            del m
        print("[P2] 新目标权重预检通过", flush=True)
    except Exception as e:
        print(f"FATAL: 新目标权重不可用（{type(e).__name__}: {e}）\n"
              f"请在登录节点预下载（见脚本头部说明）后重试", flush=True)
        raise SystemExit(1)

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        res = run_seed(args, args.surrogate, seed, device)
        path = out_dir / f"results_p2_imagenet_{args.surrogate}_seed{seed}.json"
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[保存] {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
