#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sota_modern.py — 现代已发表 UAP 方法（2024–2025）的接入层。

已接入
------
dmuap : DM-UAP (Zhang et al., **AAAI 2025**, arXiv:2503.12793)
        官方实现：https://github.com/yechao-zhang/DM-UAP (MIT)
        本地副本：third_party/DM-UAP-main/（**原样保留，未修改**）
        可导入副本：third_party/dmuap_vendor/（仅两处机械改动：
            ① `from utils import ...` → 相对导入，避免与本项目 utils/ 包撞名；
            ② 设备硬编码 "cuda:0" → 读 UAP_DEV 环境变量，便于本地 CPU 冒烟测试。
            算法、超参、更新规则一行未改。）

未接入（附原因，见 unavailable()）
----------------------------------
psp_uap : PSP-UAP (Lee et al., CVPR 2025) — 代码已下载至 third_party/PSP-UAP-main/，
          但依赖 cv2 / skimage / imageio，且其 prepare_for_model 返回的模型**未做
          ImageNet 归一化**（transform 只有 Resize/CenterCrop/ToTensor），与我们协议
          的 [0,1]+内部归一化语义不一致；强行接入会重演 pdua 那类"基线失效"。
intriuap: IntriUAP (Yan & Tian, AAAI 2025) — 代码已下载，但仓库**不含**所需资源：
          各层右奇异向量需从 Google Drive 下载；且核心文件硬编码作者本机路径
          (/data/yyt/project1/data)。无法在不引入不确定性的前提下忠实复现。

关于预算的诚实约定
------------------
DM-UAP 的"一次外层更新"内部含 theta_steps 步权重扰动 + cc 步输入扰动，
单次更新成本约为普通 UAP-SGD 的 (theta_steps + cc + 1) 倍。因此本文默认给它
`dm_steps=60` 次外层更新（≈1260 次图像梯度回传，约为 direct(200 步) 的 6 倍），
即**在算力上偏向基线**；若我们仍胜出，结论更稳。需要严格外层步数对齐时用
`--dm_steps 200`。两种设置都会记录 wall-clock（meta.*_train_time_total）。
"""
import math
import shutil
import tempfile

import torch
from torch.utils.data import Dataset, DataLoader

from third_party.dmuap_vendor.attack_min_x_theta_adam import uap_dm


class _ImageOnly(Dataset):
    """DM-UAP 的 loader 只迭代图像（内部用模型自身预测当伪标签）。"""

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i]


def dmuap(model, images, epsilon, steps=60, batch_size=50, beta=9.0, step_decay=0.1,
          rho=4, aa=25, cc=10, theta_steps=10, smooth_rate=0.2, momentum=0,
          cross_loss=1, keep_ckpt=False):
    """DM-UAP：动态 maximin 优化生成 UAP。

    参数对应官方 run.sh：
        --alpha {epsilon*255} --rho {rho} --aa {aa} --cc {cc}
        --steps {theta_steps} --smooth_rate {smooth_rate} --Momentum {momentum}
        --cross_loss {cross_loss} --batch_size {batch_size}

    Args:
        model   : NormalizedModel（内部做 ImageNet 归一化，UAP 在 [0,1] 空间）
        images  : (N,3,H,W) float [0,1]，few-shot 训练集
        epsilon : L∞ 预算（[0,1] 尺度，等价官方 alpha/255）
        steps   : 目标**外层更新次数**（官方为 epoch 数；这里由 batch 数换算）

    Returns:
        (3,H,W) 的 UAP 张量，已 clamp 到 ±epsilon。
    """
    n = len(images)
    bs = max(1, min(int(batch_size), n))
    per_epoch = max(1, math.ceil(n / bs))
    nb_epoch = max(1, math.ceil(int(steps) / per_epoch))

    loader = DataLoader(_ImageOnly(images), batch_size=bs, shuffle=True,
                        num_workers=0, drop_last=False)

    scratch = tempfile.mkdtemp(prefix="dmuap_")
    try:
        uap, _losses, _lmin = uap_dm(
            model=model,
            loader=loader,
            nb_epoch=nb_epoch,
            eps=float(epsilon),
            beta=float(beta),
            step_decay=float(step_decay),
            loss_function=int(cross_loss),
            uap_init=None,
            batch_size=bs,
            loader_eval=loader,
            dir_uap=scratch.rstrip("/") + "/",
            center_crop=int(images.shape[-1]),
            Momentum=int(momentum),
            img_num=n,
            rho=float(rho),
            aa=float(aa),
            cc=int(cc),
            steps=int(theta_steps),
            smooth_rate=float(smooth_rate),
        )
    finally:
        if not keep_ckpt:
            shutil.rmtree(scratch, ignore_errors=True)

    uap = uap.detach()
    if uap.dim() == 4:
        uap = uap[0]
    return torch.clamp(uap, -float(epsilon), float(epsilon)).detach()


def unavailable():
    """未接入的现代基线及其确切原因（供报告/审稿回复直接引用）。"""
    return {
        "psp_uap": "代码已下载(third_party/PSP-UAP-main)，但依赖 cv2/skimage/imageio，"
                   "且官方 prepare_for_model 未做 ImageNet 归一化，与本协议语义不一致。",
        "intriuap": "代码已下载(third_party/Intri_Attack-main)，但仓库不含各层右奇异向量"
                    "（需 Google Drive 下载），且硬编码作者本机路径。",
    }


def dmuap_cost_ratio(theta_steps=10, cc=10):
    """DM-UAP 单次外层更新 ≈ direct 单步的多少倍（用于论文算力表脚注）。"""
    return theta_steps + cc + 1
