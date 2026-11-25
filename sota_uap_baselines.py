#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sota_uap_baselines.py — 与已发表 UAP 方法做同预算对比的基线实现。

基线的两层设计（回应「基线太旧」的审稿质疑）：
  【T1 奠基层 2017–2018】定义了我们所修改的标准优化协议，是可比性的锚点：
      uap2017_md — Moosavi-Dezfooli et al., CVPR 2017（原始 UAP，数据驱动）
      ffa        — Mopuri et al., BMVC 2017（Fast Feature Fool，data-free）
      ssa_energy — Mopuri et al., TPAMI 2018（data-free，激活能量变体）
  【T2 现代层 2024+】证明我们不是只挑老弱基线：
      tsc        — Huang et al., TIFS 2024（TSC-UAP，可套任意方法的纹理尺度增强）
      reuse_tsc  — 本方法 × TSC 组合（检验增益是否可与 2024 SOTA 叠加）

  待接入官方代码的现代 SOTA（未在此实现，避免不忠实复现）：
      DM-UAP（AAAI 2025, arXiv:2503.12793）   — 动态 maximin，500 样本 +12.1% FR
      PSP-UAP（CVPR 2025, arXiv:2502.21048）  — data-free 伪语义先验
      IntriUAP（AAAI 2025, arXiv:2503.22205） — data-free 右奇异向量对齐

⚠️ 关于命名的诚实说明
  早先计划里的 "PG-UAP" 在文献中查无对应方法（两轮检索均无命中），已弃用。
  另注意 `ssa` 在本文件指 Mopuri 的 data-free SSA（TPAMI 2018），
  **不是** SSA: Spectrum Simulation Attack（Long et al., ECCV 2022）。
  为免混淆，run/analyze 中记为 `ssa_energy`。

三者共享同样的步数预算（steps=200）、同一 ε 球、同一 clamp 语义。
另有（experiments.py / fewshot_uap_method.py 已有，不重复实现）：
  - direct        : uap_sgd(Adam, lr=0.01) 目标 ε 直接训（默认基线）
  - direct_lr0.05 : uap_sgd(Adam, lr=0.05)（「只是没调 lr」的对照）
  - ceiling       : many(2000) 数据充分训（上界，非 few-shot 预算）
"""
import numpy as np
import torch
import torch.nn.functional as F

from utils.losses import clamp_uap


# ==================== 1) Moosavi-Dezfooli et al., CVPR 2017 ====================

def uap2017_md(model, images, labels, epsilon, steps=200, lr=0.02, batch_size=128,
               device=None):
    """原始 UAP（CVPR 2017）的迭代聚合实现。

    与 pd_ua 的区别（保持可区分、可辩护）：
      - 只让「当前 UAP 还骗不过」的样本贡献更新（原论文的 skip 逻辑）；
      - 逐样本取 sign 后再对 batch 求平均（= 累积逐样本最小扰动）；
      - 无动量、无 Adam（原论文是纯投影累积）。
    """
    device = device or images.device
    uap = torch.zeros_like(images[0], device=device)
    n = min(len(images), batch_size)

    for _ in range(steps):
        idx = np.random.choice(len(images), n, replace=False)
        x, y = images[idx], labels[idx]

        # 只保留尚未被愚弄的样本
        with torch.no_grad():
            pred = model((x + uap).clamp(0, 1)).argmax(1)
        mask = (pred == y)
        if mask.sum() == 0:
            continue
        xf, yf = x[mask], y[mask]

        # 逐样本最小扰动方向（单步 sign 近似 DeepFool 的最小扰动）
        delta = torch.zeros_like(xf, requires_grad=True)
        adv = (xf + clamp_uap(uap + delta, epsilon)).clamp(0, 1)
        loss = -F.cross_entropy(model(adv), yf)
        g = torch.autograd.grad(loss, delta)[0]

        # 累积逐样本 sign 摄动后投影回 ε 球
        uap = clamp_uap(uap - lr * g.sign().mean(dim=0), epsilon).detach()

    return clamp_uap(uap, epsilon)


# ==================== 2) Fast Feature Fool（data-free）====================

def _pick_layers(model, k=5):
    """按等间隔挑选 k 个卷积层作为激活探针。"""
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        return []
    idx = sorted(set(np.linspace(0, len(convs) - 1, min(k, len(convs))).astype(int).tolist()))
    return [convs[i] for i in idx]


def _activate_objective(model, layer_list, x, uap, epsilon, mode="mean"):
    """前向并读出探针层激活；mode=mean（FFA）或 energy（SSA 能量变体）。"""
    acts = []
    hooks = [l.register_forward_hook(lambda m, i, o, acts=acts: acts.append(o))
             for l in layer_list]
    try:
        model((x + clamp_uap(uap, epsilon)).clamp(0, 1))
    finally:
        for h in hooks:
            h.remove()
    if mode == "mean":
        return sum(a.float().mean() for a in acts) / max(len(acts), 1)
    else:  # energy: 逐样本 L2 范数均值
        return sum(a.float().flatten(1).norm(dim=1).mean() for a in acts) / max(len(acts), 1)


def _data_free_uap(model, shape, epsilon, steps, lr, n_layers, batch, mode, device):
    layer_list = _pick_layers(model, n_layers)
    if not layer_list:
        raise RuntimeError("未找到 Conv2d 层，无法注册 FFA/SSA 探针")
    uap = torch.zeros(shape, device=device, requires_grad=True)
    opt = torch.optim.Adam([uap], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x = torch.rand(batch, *shape, device=device)   # data-free：随机输入
        loss = -_activate_objective(model, layer_list, x, uap, epsilon, mode=mode)
        loss.backward()
        opt.step()
    return clamp_uap(uap.detach(), epsilon)


def ffa(model, shape, epsilon, steps=200, lr=0.02, n_layers=5, batch=16, device=None):
    """FFA（Mopuri et al., BMVC 2017）：最大化多层激活均值，完全不需要真实数据。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _data_free_uap(model, shape, epsilon, steps, lr, n_layers, batch, "mean", device)


def ssa(model, shape, epsilon, steps=200, lr=0.02, n_layers=5, batch=16, device=None):
    """SSA 激活能量变体（Mopuri et al., TPAMI 2018 的 data-free 目标，L2 形式）。

    ⚠️ 命名注意：此处的 `ssa` 是 **Mopuri 的 data-free SSA（激活能量变体）**，
    与「SSA: Spectrum Simulation Attack（Long et al., ECCV 2022）」**不是同一个方法**。
    为避免混淆，run/analyze 中统一记为 `ssa_energy`。
    原方法用逐层激活的主导奇异向量构造目标；此处用各层激活的逐样本 L2 能量近似
    （省去每步 SVD）。论文中须如实标注为「能量变体」。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _data_free_uap(model, shape, epsilon, steps, lr, n_layers, batch, "energy", device)


# ==================== 3) TSC-UAP（TIFS 2024，现代基线）====================
# Huang et al., "Texture Re-scalable Universal Adversarial Perturbation", TIFS 2024.
# 论文 Algorithm 1 完整给出，可忠实复现：
#   把 UAP 约束成一个 (H/α, W/α) 的小纹理块 v，训练时 tile 平铺成整幅 δ=T(v,α)，
#   梯度只更新 v，再 clip v 到 [-ε, ε]。α 越大 → 纹理块越小。
# TSC 是「可套在任意 UAP 方法上的增强」，因此既能当独立基线，也能与本方法组合。

def _tile_patch(v, alpha):
    """T(v, α)：把 (C, H/α, W/α) 的小块在纵横方向各平铺 α 次 → (C, H, W)。"""
    return v.repeat(1, alpha, alpha)


def tsc_patch(model, images, labels, epsilon, alpha=4, steps=200, lr=0.01,
              batch_size=128, device=None):
    """训练 TSC 小纹理块 v（只更新 v，返回裁到 ε 球内的 v）。"""
    device = device or images.device
    C, H, W = images[0].shape
    if H % alpha or W % alpha:
        raise ValueError(f"alpha={alpha} 不能整除 {H}x{W}")
    v = torch.zeros(C, H // alpha, W // alpha, device=device, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr)
    n = min(len(images), batch_size)
    for _ in range(steps):
        opt.zero_grad()
        delta = _tile_patch(clamp_uap(v, epsilon), alpha)
        idx = np.random.choice(len(images), n, replace=False)
        adv = (images[idx] + delta).clamp(0, 1)
        loss = -F.cross_entropy(model(adv), labels[idx])
        loss.backward()
        opt.step()
    return clamp_uap(v.detach(), epsilon)


def tsc_assemble(v, eps_out, alpha, eps_in=None):
    """把纹理块缩放（可选，用于预算解耦）后平铺成整幅 UAP。"""
    if eps_in is not None and eps_in != eps_out:
        v = clamp_uap(v * (eps_out / eps_in), eps_out)
    return _tile_patch(clamp_uap(v, eps_out), alpha)


def tsc_uap(model, images, labels, epsilon, alpha=4, steps=200, lr=0.01,
            batch_size=128, device=None):
    """TSC-UAP 独立基线：在目标 ε 上训练纹理块再平铺（每个 ε 单独训）。"""
    v = tsc_patch(model, images, labels, epsilon, alpha=alpha, steps=steps,
                  lr=lr, batch_size=batch_size, device=device)
    return tsc_assemble(v, epsilon, alpha)

