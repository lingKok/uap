"""
fewshot_uap_method.py — LWS 方法与有限数据 UAP 训练协议的基元。

方法（ours）：
  - lws_uap(model, imgs, lbls, eps_w, eps_t, steps, lr, refine_steps=0)
      在极小 ε_w 上用 uap_sgd 训 UAP，再线性缩放到目标 ε_t 部署；
      可选 refine_steps 步在 ε_t 上热启动精炼（仍远少于直接训）。

基线基元（均来自 experiments.py，保证与 X1 同协议）：
  - uap_sgd  : Adam，主竞争者（目标 ε 直接训）
  - pd_ua    : Moosavi-Dezfooli 投影梯度+动量（真实已发表 UAP 优化器）
  - uap_sgd_warm : 从 uap_init 热启动在指定 ε 继续 Adam（供 LWS 精炼）

评估：
  - eval_on_models : 同一 UAP 在 surrogate（白盒）+ 各黑盒模型上评估，返回 FR/ASR/饱和率。
"""
import torch
import torch.nn.functional as F
import numpy as np
from experiments import uap_sgd, pd_ua, compute_metrics
from utils.losses import clamp_uap


def uap_sgd_warm(model, images, labels, epsilon, uap_init, steps=50, lr=0.01, batch_size=128):
    """从 uap_init 热启动，在 epsilon 球内继续 Adam 优化。"""
    uap = uap_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([uap], lr=lr)
    n = min(len(images), batch_size)
    for step in range(steps):
        optimizer.zero_grad()
        idx = np.random.choice(len(images), n, replace=False)
        adv = (images[idx] + clamp_uap(uap, epsilon)).clamp(0, 1)
        loss = -F.cross_entropy(model(adv), labels[idx])
        loss.backward()
        optimizer.step()
    return clamp_uap(uap.detach(), epsilon)


def lws_uap(model, imgs, lbls, eps_w, eps_t, steps, lr, refine_steps=0):
    """Low-budget Warm-start Scaling (LWS)。

    核心：在低 ε_w（小预算）上训练能稳定填满预算、找到良形扰动；
    缩放到 ε_t 是保形、无损失的（饱和比例不变）。高 ε 直接训在 few-shot
    + 固定 lr 下会欠饱和（见 analysis_x1_imagenet_verdict.md 机制）。
    """
    uap_w = uap_sgd(model, imgs, lbls, eps_w, steps=steps, lr=lr)
    uap_t = clamp_uap(uap_w * (eps_t / eps_w), eps_t)
    if refine_steps and refine_steps > 0:
        uap_t = uap_sgd_warm(model, imgs, lbls, eps_t, uap_t, steps=refine_steps, lr=lr)
    return uap_t


def sat_frac(uap, eps):
    """饱和坐标占比：|uap| 顶到 ε 边界的比例。"""
    return float((uap.abs() >= eps - 1e-4).float().mean().item())


def eval_on_models(uap, eps_t, models, eval_imgs, eval_lbls, eb=16):
    """同一 UAP 在多个模型（含白盒 surrogate 与各黑盒）上评估。

    返回 {model_name: {fr_all, asr_correct, sat_frac}}。
    """
    out = {}
    for name, mdl in models.items():
        m = compute_metrics(mdl, uap, eval_imgs, eval_lbls, eb)
        out[name] = {
            "fr_all": m["fr_all"],
            "asr_correct": m["asr_correct"],
            "sat_frac": sat_frac(uap, eps_t),
        }
    return out
