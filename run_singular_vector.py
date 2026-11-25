#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_singular_vector.py -- P1: singular-vector baseline (Khrulkov & Babu, CVPR 2018).

动机
----
审稿人最尖锐的问题是：「budget-agnostic 部署有什么稀奇的？拿数据集的顶层
奇异向量（SVD-UAP，64 张图、零优化）按 epsilon 缩放，天然预算无关。」

本脚本实现这个零优化竞争者并评估：
  - 对每张图，取所选层特征 F_l(x) 对输入 x 的 Jacobian 的顶层右奇异向量
    （幂迭代 + forward-mode JVP / backward VJP，从不显式构造 J）；
  - 跨图像对齐符号后取均值，归一化到 ||v||_inf = 1；
  - 部署 delta(eps) = eps * v_hat：对任意 eps 精确成立、零截断、零训练
    ——按构造就是 budget-agnostic 的。

实现要点（v2，2026-10-31 修正）
----
v1 直接 getattr(wrapped_model, layer) 崩在 NormalizedModel 上。模型是
NormalizedModel(backbone)，且归一化 (x-mean)/std 必须留在 Jacobian 里
（扰动在 [0,1] 空间，dF/dx = J_backbone/std != J_backbone）。因此：
  - 在 backbone 的**子模块**上挂 forward hook 抓中间层输出；
  - 前向仍走完整 wrapped model（含归一化），JVP/VJP 因此都是对输入图像求导；
  - JVP: forward_ad.make_dual 输入 -> 完整前向 -> hook 里的 dual 张量 unpack；
    VJP: hook 保存 out_l，autograd.grad(<u, out_l>, x)。
  - forward_ad 失败时回退 functional.jvp（闭包内做同样的 hook 抓取）。

判定
----
  - 若 SVD 方向逼近 scale-up SGD：部署原语的主张被"免费方向"拆解，须重写；
  - 若 SVD 显著更弱：budget-agnostic 是必要非充分条件，优化得到的方向质量
    才是优势来源（预注册预期）。

覆盖配置
----
  k x layer：k in {64, 200}（64 = Khrulkov-Babu 原设定；200 = 与 few-shot 对齐），
  layer in {layer2, layer4}（浅/深特征）。

落盘
----
  results_svduap/results_svduap_imagenet_<surrogate>_seed<seed>.json
  json 结构与 run_clipdown 对齐（"methods" -> {eps -> {model -> metrics}}），
  make_tables.py 可直接复用 fr_cd 访问器。

自包含说明：不 import run_fewshot_sota / run_clipdown，避免集群残留旧版
驱动的签名漂移（本项目已多次踩坑）。
"""
import argparse
import json
import time
import random
import traceback
import torch
import numpy as np
from pathlib import Path

from config import Config
from models import create_all_imagenet_models
from utils.data import get_imagenet_mini_samples
from fewshot_uap_method import eval_on_models


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    """剥掉 NormalizedModel 之类的外层包装，拿到真正的 backbone（拿子模块用）。"""
    seen = set()
    while isinstance(getattr(model, "model", None), torch.nn.Module):
        if id(model) in seen:
            break
        seen.add(id(model))
        model = model.model
    return model


def _feat_with_hook(wrapped, sub, x):
    """跑一次完整前向（含归一化），借 hook 抓 sub 的输出。"""
    holder = {}

    def hook(_m, _i, o):
        holder["o"] = o

    h = sub.register_forward_hook(hook)
    try:
        wrapped(x)
    finally:
        h.remove()
    if "o" not in holder:
        raise RuntimeError("hook 未捕获到层输出：层名是否存在于该 backbone？")
    return holder["o"]


_FWD_AD_OK = True   # 一次探测失败后全局关闭，避免每轮迭代重复抛异常/刷日志


def jvp_at_layer(wrapped, sub, x, v):
    """u = J v，J = dF_l/dx（对 [0,1] 输入空间，含归一化）。"""
    global _FWD_AD_OK
    if _FWD_AD_OK:
        try:
            with torch.autograd.forward_ad.dual_level():
                d_x = torch.autograd.forward_ad.make_dual(x, v)
                out = _feat_with_hook(wrapped, sub, d_x)
                _, tangent = torch.autograd.forward_ad.unpack_dual(out)
            if tangent is None:
                raise RuntimeError("tangent is None")
            return tangent.detach()
        except Exception as e:
            _FWD_AD_OK = False
            print(f"  [svd] forward_ad 不可用（{type(e).__name__}: {e}），"
                  f"整体回退 functional.jvp", flush=True)

    def feat(inp):
        return _feat_with_hook(wrapped, sub, inp)

    u, _ = torch.autograd.functional.jvp(feat, (x,), (v,))
    return u.detach()


def vjp_at_layer(wrapped, sub, x, u):
    """J^T u = grad_x <u, F_l(x)>。"""
    x_req = x.detach().requires_grad_(True)
    out = _feat_with_hook(wrapped, sub, x_req)
    g = torch.autograd.grad((out * u).sum(), x_req)[0]
    return g


def top_sv_directions(wrapped, sub, imgs, n_iter, chunk, seed):
    """逐图顶层右奇异向量（幂迭代，批内并行），返回 (N,C,H,W) 单位向量堆叠。"""
    device = imgs.device
    N = imgs.shape[0]
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    outs = []
    resid_last = float("nan")
    for s in range(0, N, chunk):
        x = imgs[s:s + chunk]
        v = torch.randn(x.shape, generator=g, device=device)
        v = v / v.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        for it in range(n_iter):
            u = jvp_at_layer(wrapped, sub, x, v)
            v2 = vjp_at_layer(wrapped, sub, x, u)
            v2 = v2 / v2.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
            if it == n_iter - 1:
                cos = (v2 * v).flatten(1).sum(dim=1).abs()
                resid_last = float((1.0 - cos).mean().item())
            if s == 0 and (it % 10 == 0 or it == n_iter - 1):
                print(f"    [svd] iter {it+1}/{n_iter}", flush=True)
            v = v2.detach()
        outs.append(v)
    return torch.cat(outs, dim=0), resid_last


def svd_uap(wrapped, sub, imgs, n_iter, chunk, seed):
    """Khrulkov-Babu：逐图顶奇异向量 -> 符号对齐 -> 均值 -> ||.||_inf = 1。"""
    t0 = time.time()
    V, resid = top_sv_directions(wrapped, sub, imgs, n_iter, chunk, seed)
    ref = V[0]
    sign = torch.sign((V * ref.unsqueeze(0)).flatten(1).sum(dim=1))
    sign[sign == 0] = 1.0
    V = V * sign.view(-1, 1, 1, 1)
    v_mean = V.mean(dim=0)
    v_hat = v_mean / v_mean.abs().max().clamp_min(1e-12)
    dt = time.time() - t0
    return v_hat, dt, resid


def load_models(names, device):
    models = create_all_imagenet_models(names)
    for n in names:
        models[n] = models[n].to(device).eval()
    return models


def run_seed(a, surrogate, seed, device, all_names):
    set_seed(seed)
    models = load_models(all_names, device)
    many_imgs, many_lbls = get_imagenet_mini_samples(
        num_samples=a.many, split="train", image_size=224, seed=100)
    eval_imgs, eval_lbls = get_imagenet_mini_samples(
        num_samples=a.eval_n, split="val", image_size=224, seed=42)
    eb = a.eval_batch if a.eval_batch > 0 else 64
    model = models[surrogate]
    backbone = unwrap(model)

    if not surrogate.startswith("resnet"):
        raise ValueError(f"SVD 基线暂只支持 resnet 系 surrogate，得到 {surrogate}")
    layer_names = [s.strip() for s in a.svd_layers.split(",") if s.strip()]
    subs = {}
    for ln in layer_names:
        if not hasattr(backbone, ln):
            avail = [n for n, _ in backbone.named_children()]
            raise AttributeError(f"{type(backbone).__name__} 无子模块 {ln}；"
                                 f"可用: {avail}")
        subs[ln] = getattr(backbone, ln)

    budgets = sorted({float(e) for e in a.eps_targets.split(",") if e.strip()})
    ks = [int(k) for k in a.svd_ks.split(",") if k.strip()]

    result = {
        "surrogate": surrogate, "seed": seed, "config": vars(a),
        "eps_targets": budgets, "methods": {}, "delta_stats": {}, "meta": {},
    }

    for k in ks:
        imgs = many_imgs[:k]
        for ln, sub in subs.items():
            name = f"svd_k{k}_{ln}"
            v_hat, dt, resid = svd_uap(model, sub, imgs, a.n_power_iter,
                                       a.svd_chunk, seed * 1000 + k)
            result["meta"][f"{name}_time"] = dt
            result["meta"][f"{name}_power_resid"] = resid   # 收敛残差（越小越好）
            result["meta"][f"{name}_n_train_runs"] = 0      # 零优化
            result["delta_stats"][name] = {
                "linf": float(v_hat.abs().max().item()),
                "l2": float(v_hat.flatten().norm(2).item()),
                "n_power_iter": a.n_power_iter,
                "power_resid": resid,
            }
            print(f"  [{surrogate}] seed{seed} {name} 方向就绪 ({dt:.0f}s, "
                  f"resid={resid:.4f}) Linf={result['delta_stats'][name]['linf']:.4f}",
                  flush=True)

            result["methods"][name] = {}
            for eps in budgets:
                delta = eps * v_hat
                print(f"  [{surrogate}] seed{seed} {name} eps={eps:.2f} 评估中 ...",
                      flush=True)
                res = eval_on_models(delta, eps, models, eval_imgs, eval_lbls, eb)
                result["methods"][name][str(eps)] = res
                print(f"  [{surrogate}] seed{seed} {name} eps={eps:.2f} "
                      f"白盒FR={res[surrogate]['fr_all']:.3f} "
                      f"sat={res[surrogate]['sat_frac']:.3f}", flush=True)

    model.cpu(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenet")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--models", default="auto")
    p.add_argument("--eps_targets", default="0.1,0.2,0.3")
    p.add_argument("--svd_ks", default="64,200",
                   help="逗号分隔的样本数（64 = Khrulkov-Babu 原设定）")
    p.add_argument("--svd_layers", default="layer2,layer4",
                   help="逗号分隔的 backbone 子模块名（resnet: layer1..layer4）")
    p.add_argument("--n_power_iter", type=int, default=50)
    p.add_argument("--svd_chunk", type=int, default=25,
                   help="幂迭代批大小（显存旋钮；forward_ad 路径显存约 2x）")
    p.add_argument("--many", type=int, default=2000)
    p.add_argument("--seeds", default="42,1342,2342")
    p.add_argument("--eval_n", type=int, default=2000)
    p.add_argument("--eval_batch", type=int, default=0)
    p.add_argument("--out_dir", default="results_svduap")
    p.add_argument("--out_tag", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset != "imagenet":
        raise ValueError("P1 基线只在 ImageNet 上跑（对照主表）")
    if args.models == "auto":
        names = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "vit_b_16"]
    else:
        names = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or args.surrogate

    print(f"[P1] surrogate={args.surrogate} ks={args.svd_ks} "
          f"layers={args.svd_layers} seeds={args.seeds} "
          f"n_iter={args.n_power_iter} chunk={args.svd_chunk}", flush=True)
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        res = run_seed(args, args.surrogate, seed, device, names)
        path = out_dir / f"results_svduap_{args.dataset}_{tag}_seed{seed}.json"
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[保存] {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
