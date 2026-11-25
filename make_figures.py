#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_figures.py — 论文图（英文标签，PDF 输出，投稿可用）。

fig1: clip-frac vs clip-down 劣化散点（机制二的视觉证据）。
  x = clip_frac（被截断坐标比例，clipdown json delta_stats）
  y = FR 劣化 = per-eps FR − clipdown FR（黑盒均值，正=劣化）
  每点 = 一个 (surrogate, seed, method, eps=0.1) 运行单元。
  预期形态：direct 高劣化/低 clip_frac，tsc 低劣化/高 clip_frac ——
  说明破坏力在方向质量而非截断量。
"""
import glob
import json
import os

import numpy as np

EPS = [0.1, 0.2, 0.3]
IMAGENET_MODELS = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "vit_b_16"]
METHOD_STYLE = {
    "direct": dict(label="Direct (SGD)", color="#d62728", marker="o"),
    "tsc": dict(label="TSC-UAP", color="#1f77b4", marker="s"),
    "uap2017": dict(label="UAP 2017", color="#2ca02c", marker="^"),
    "dmuap": dict(label="DM-UAP", color="#9467bd", marker="D"),
}


def load(d):
    out = {}
    for f in glob.glob(f"{d}/*.json"):
        j = json.load(open(f))
        out[(j["surrogate"], j["seed"])] = j
    return out


def bb_sota(j, m, e):
    src = j["methods"]
    v = [src[m][str(e)][t]["fr_all"] for t in IMAGENET_MODELS if t != j["surrogate"]]
    return float(np.nanmean(v))


def bb_cd(j, m, e):
    v = [j["methods"][m][str(e)][t]["fr_all"] for t in IMAGENET_MODELS if t != j["surrogate"]]
    return float(np.nanmean(v))


def fig_clipfrac(runs_sota, runs_cd, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m, st in METHOD_STYLE.items():
        xs, ys = [], []
        for k in sorted(set(runs_sota) & set(runs_cd)):
            js, jc = runs_sota[k], runs_cd[k]
            if m not in js["methods"] or m not in jc["methods"]:
                continue
            for e in EPS:
                cf = jc.get("delta_stats", {}).get(f"{m}@{e}", {}).get("clip_frac")
                if cf is None:
                    continue
                xs.append(100 * cf)
                ys.append(100 * (bb_sota(js, m, e) - bb_cd(jc, m, e)))
        ax.scatter(xs, ys, s=28, alpha=0.75, color=st["color"],
                   marker=st["marker"], label=st["label"], edgecolors="none")
    ax.set_xlabel("Clipped coordinate fraction (\\%)", fontsize=10)
    ax.set_ylabel("FR degradation of clip-down (pp)", fontsize=10)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=200)
    print(f"[写入] {outpath}")


def fig_deploy_vs_retrain(runs_sota, runs_cd, outpath):
    """逐预算比较两种 budget-agnostic 部署策略相对 per-eps 重训的 FR 变化。

    Δ = method − 重训（与 Table 4 同约定）：正 = FR 更高 = 比重训强，负 = 更差。
    clip-down 在 0.3 训练后裁剪；scale-up 在 0.05 训练后线性放大。两者都只
    训练 1 次，与 3 次重训对比，训练次数严格对齐。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [("direct", "reuse_lws", "SGD (Direct)"), ("tsc", "reuse_tsc", "TSC-UAP")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    width = 0.36
    xpos = np.arange(len(EPS))
    for ax, (base, sc, title) in zip(axes, pairs):
        d_cd, d_su, e_cd, e_su = [], [], [], []
        for e in EPS:
            cd_v, su_v = [], []
            for k in sorted(set(runs_sota) & set(runs_cd)):
                js, jc = runs_sota[k], runs_cd[k]
                if base not in js["methods"] or base not in jc["methods"]:
                    continue
                ref = bb_sota(js, base, e)
                cd_v.append(100 * (bb_cd(jc, base, e) - ref))
                if sc in js["methods"]:
                    su_v.append(100 * (bb_sota(js, sc, e) - ref))
            d_cd.append(np.mean(cd_v) if cd_v else np.nan)
            e_cd.append(np.std(cd_v, ddof=1) if len(cd_v) > 1 else 0.0)
            d_su.append(np.mean(su_v) if su_v else np.nan)
            e_su.append(np.std(su_v, ddof=1) if len(su_v) > 1 else 0.0)
        ax.bar(xpos - width / 2, d_cd, width, yerr=e_cd, capsize=2, color="#9e9e9e",
               edgecolor="black", lw=0.4, label="Clip-down")
        ax.bar(xpos + width / 2, d_su, width, yerr=e_su, capsize=2, color="#1f77b4",
               edgecolor="black", lw=0.4, label="Scale-up (ours)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{e:.1f}" for e in EPS], fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Deployment budget $\\epsilon$", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("$\\Delta$FR vs. per-$\\epsilon$ retraining (pp; $>0$ better)",
                       fontsize=9)
    axes[0].text(0.03, 0.93, "better $\\uparrow$", transform=axes[0].transAxes,
                 fontsize=7.5, color="#1f77b4", ha="left", va="top")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left",
                   bbox_to_anchor=(0.02, 0.02))
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=200)
    print(f"[写入] {outpath}")


def main():
    runs_sota = load("results_fewshot_sota")
    runs_cd = load("results_clipdown")
    print(f"sota={len(runs_sota)} clipdown={len(runs_cd)}")
    if runs_sota and runs_cd:
        fig_clipfrac(runs_sota, runs_cd, "paper/figs/clipfrac_scatter.pdf")
        fig_deploy_vs_retrain(runs_sota, runs_cd, "paper/figs/deploy_vs_retrain.pdf")


if __name__ == "__main__":
    main()
