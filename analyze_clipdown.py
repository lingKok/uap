#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_clipdown.py — clip-down vs LWS 的决定性检验。

核心问题
--------
「低 ε 训 1 次 → 缩放到全预算」（LWS/ours）与「高 ε 训 1 次 → 裁剪到小预算」
（clip-down）都是 1 次训练。哪个更好？

    Δ = clipdown − lws
        Δ > 0  → clip-down 更强，LWS 的预算解耦主张失效
        Δ < 0  → 缩放保形优于裁剪截断，LWS 主张成立
        Δ ≈ 0  → 两者等价，LWS 只是众多 1-次训练方案之一（无独特性）

配对关系（同为 1 次训练，严格对齐；且按 (surrogate, seed) 逐对匹配）
    clipdown_direct  ↔  reuse_lws     （同一基优化器 uap_sgd lr=0.01）
    clipdown_tsc     ↔  reuse_tsc     （同一 TSC 纹理优化器）

安全网
------
表0 检查 clipdown@ε_hi 是否与该 seed 的 per-ε 重训@ε_hi 一致（应为同一结果）。
不一致说明两次运行协议漂移，此时先别解释 Δ。

用法
----
    python analyze_clipdown.py
    python analyze_clipdown.py --out_md paper/analysis_clipdown_raw.txt
"""
import argparse
import glob
import json
import numpy as np

EPS = [0.1, 0.2, 0.3]
EPS_HI = 0.3
MODELS = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "vit_b_16"]
SURR = ["resnet50", "resnet101", "vit_b_16"]
PAIRS = [("direct", "reuse_lws"), ("tsc", "reuse_tsc")]


def _fr_node(node, model):
    return node.get(model, {}).get("fr_all", np.nan)


def cd_fr(j, method, eps, model):
    return _fr_node(j.get("methods", {}).get(method, {}).get(str(eps), {}), model)


def so_fr(j, method, eps, model):
    if method == "ceiling":
        node = j.get("ceiling", {}).get(str(eps), {})
    else:
        node = j.get("methods", {}).get(method, {}).get(str(eps), {})
    return _fr_node(node, model)


def bb(j, s, method, eps, getter):
    """黑盒均值：剔除与 surrogate 同名的 target。"""
    v = [getter(j, method, eps, t) for t in MODELS if t != s]
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


def load(path, keyed_by_seed):
    out = {}
    for f in glob.glob(f"{path}/*.json"):
        j = json.load(open(f))
        k = (j["surrogate"], j["seed"])
        if keyed_by_seed:
            out[k] = j
        else:
            out.setdefault(j["surrogate"], []).append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cd_dir", default="results_clipdown")
    ap.add_argument("--sota_dir", default="results_fewshot_sota")
    ap.add_argument("--out_md", default="")
    a = ap.parse_args()

    CD = load(a.cd_dir, keyed_by_seed=True)          # (surr, seed) -> json
    SO_all = load(a.sota_dir, keyed_by_seed=False)   # surr -> [json, ...]
    SO = {(j["surrogate"], j["seed"]): j for v in SO_all.values() for j in v}
    if not CD:
        print(f"[FATAL] {a.cd_dir} 里没有 json")
        return

    pairs = sorted(k for k in CD if k in SO)
    missing = sorted(k for k in CD if k not in SO)
    print(f"clipdown runs = {len(CD)} | sota runs = {len(SO)} | 可配对 = {len(pairs)}")
    if missing:
        print(f"[警告] 这些 clipdown 运行在 sota 目录找不到同 (surrogate, seed) 对照: {missing}")
    if not pairs:
        print("[FATAL] 无可配对单元")
        return
    surr = sorted({s for s, _ in pairs}, key=SURR.index)

    L = []
    def p(s=""):
        print(s); L.append(s)

    # ---------- 表 0：协议自洽性 ----------
    # 判据：|diff| <= max(0.01, 2×同单元 seed 间 sd)。
    # 两次运行是独立进程，np.random.choice 的 batch 采样顺序不同（RNG 状态不同），
    # 逐单元不可能逐位相等；只有结构性参数（steps/lr/数据/评估集/ε）才会造成大偏差。
    p("=" * 108)
    p("表0：协议自洽性 —— clipdown@ε_hi vs 同 seed 的 per-ε 重训@ε_hi（结构参数应一致）")
    p("     判据 |diff| ≤ max(0.01, 2×seed间sd)；超出者列出（RNG 顺序噪声 vs 协议漂移）")
    p("=" * 108)
    p(f"  {'surrogate':<12}{'seed':>7}{'method':>10}{'clipdown':>12}{'per-eps':>11}{'diff':>10}  状态")
    bad = 0
    for (s, seed) in pairs:
        for m in ["direct", "tsc", "uap2017", "dmuap"]:
            if m not in CD[(s, seed)].get("methods", {}):
                continue
            x = bb(CD[(s, seed)], s, m, EPS_HI, cd_fr)
            y = bb(SO[(s, seed)], s, m, EPS_HI, so_fr)
            if np.isnan(x) or np.isnan(y):
                continue
            # 同单元 seed 间 sd（用 sota 运行内的 3 个 seed 估计噪声）
            same_s = [k for k in SO if k[0] == s]
            ys = [bb(SO[k], k[0], m, EPS_HI, so_fr) for k in same_s]
            sd = float(np.std(ys, ddof=1)) if len(ys) > 1 else 0.0
            tol = max(0.01, 2 * sd)
            d = x - y
            ok = abs(d) <= tol
            if not ok:
                bad += 1
            p(f"  {s:<12}{seed:>7}{m:>10}{x:>12.4f}{y:>11.4f}{d:>+10.4f}  "
              f"{'OK' if ok else f'超出 (tol={tol:.3f}) <-- 查协议'}")
    p(f"  → {'全部自洽' if bad == 0 else f'{bad} 个单元超容差；若 |diff| 仍 <0.02 属 RNG 顺序噪声，>0.03 才需查协议'}")

    def mean_cd(m, e):
        v = [bb(CD[k], k[0], m, e, cd_fr) for k in pairs]
        return float(np.nanmean(v))

    def mean_so(m, e):
        v = [bb(SO[k], k[0], m, e, so_fr) for k in pairs]
        return float(np.nanmean(v))

    # ---------- 表 1：三策略 ----------
    p("")
    p("=" * 108)
    p("表1：三策略 · 全黑盒均值（训练次数 3 / 1 / 1）")
    p("     per-eps = 每个 ε 单独训（3 次） | clipdown = train@0.3 再裁（1 次） | lws = train@0.05 再缩放（1 次）")
    p("=" * 108)
    for cd_m, lws_m in PAIRS:
        p(f"\n  [基优化器 = {cd_m}]")
        p(f"    {'ε':<6}{'per-eps(3x)':>14}{'clipdown(1x)':>15}{'lws(1x)':>12}"
          f"{'ceiling':>10}{'clipdown−lws':>15}")
        for e in EPS:
            p(f"    {e:<6}{mean_so(cd_m, e):>14.4f}{mean_cd(cd_m, e):>15.4f}"
              f"{mean_so(lws_m, e):>12.4f}{mean_so('ceiling', e):>10.4f}"
              f"{mean_cd(cd_m, e) - mean_so(lws_m, e):>+15.4f}")

    # ---------- 表 2：决定性配对检验 ----------
    p("")
    p("=" * 108)
    p(f"表2：决定性检验 Δ = clipdown − lws（按 (surrogate, seed) 配对，n={len(pairs)}）")
    p("     Δ<0 → 缩放保形优于裁剪截断（LWS 主张成立）")
    p("     Δ>0 → clip-down 更强（LWS 的预算解耦主张失效）")
    p("=" * 108)
    verdict = {}
    for cd_m, lws_m in PAIRS:
        p(f"\n  [{cd_m} vs {lws_m}]")
        for e in EPS:
            ds = np.array([bb(CD[k], k[0], cd_m, e, cd_fr) - bb(SO[k], k[0], lws_m, e, so_fr)
                           for k in pairs])
            ds = ds[~np.isnan(ds)]
            sd = ds.std(ddof=1)
            t = ds.mean() / (sd / np.sqrt(len(ds))) if sd > 0 else 0.0
            verdict[(cd_m, e)] = (float(ds.mean()), float(t), float((ds > 0).mean()))
            p(f"    ε={e:<5} mean={ds.mean():+.4f}  sd={sd:.4f}  min={ds.min():+.4f} "
              f"max={ds.max():+.4f}  正比例={(ds > 0).mean():.0%}  t={t:+.2f}")

    # ---------- 表 3：裁剪强度 ----------
    p("")
    p("=" * 108)
    p("表3：裁剪强度（clip_frac = |δ| 超出目标 ε 球的坐标比例）——解释谁被截断得多")
    p("=" * 108)
    p(f"  {'surrogate':<12}{'method':>10}{'ε':>6}{'Linf':>9}{'L2':>9}{'clip_frac':>11}{'模式':>10}")
    for s in surr:
        for m in ["direct", "tsc", "uap2017", "dmuap"]:
            for e in EPS:
                vals = []
                for k in pairs:
                    if k[0] != s:
                        continue
                    st = CD[k].get("delta_stats", {}).get(f"{m}@{e}")
                    if st:
                        vals.append(st)
                if not vals:
                    continue
                lin = np.nanmean([v["linf"] for v in vals])
                l2 = np.nanmean([v["l2"] for v in vals])
                cf = np.nanmean([v["clip_frac"] for v in vals])
                p(f"  {s:<12}{m:>10}{e:>6}{lin:>9.4f}{l2:>9.2f}{cf:>11.3f}{vals[0]['mode']:>10}")

    # ---------- 表 4：裁决 ----------
    p("")
    p("=" * 108)
    p("表4：裁决")
    p("=" * 108)
    for cd_m, lws_m in PAIRS:
        rows = [(e, *verdict[(cd_m, e)]) for e in EPS]
        n_win = sum(1 for e, m, t, pos in rows if m > 0)
        n_sig_pos = sum(1 for e, m, t, pos in rows if m > 0 and t > 2.0)
        n_sig_neg = sum(1 for e, m, t, pos in rows if m < 0 and t < -2.0)
        p(f"  [{cd_m} vs {lws_m}] clip-down 更强的 ε 数 = {n_win}/3"
          f"（显著为正 {n_sig_pos}，显著为负 {n_sig_neg}）")
        for e, m, t, pos in rows:
            lab = ("clip-down 更强" if (m > 0 and t > 2) else
                   "lws 更强" if (m < 0 and t < -2) else "无显著差异")
            p(f"      ε={e:<5} Δ={m:+.4f} (t={t:+.2f}, 正比例={pos:.0%}) → {lab}")

    p("")
    p("  读法与后续动作：")
    p("    · 若 Δ ≥ 0 或全不显著 → clip-down 不劣于 LWS，「预算解耦」不是 LWS 的贡献，")
    p("      只能作为通用工程做法陈述，方法线关闭。")
    p("    · 若 Δ 显著为负（尤其 ε=0.1/0.2）→ 缩放保形优于裁剪截断，预算解耦成为")
    p("      有机制解释的结论（配 clip_frac），可写进论文。")
    p("    · 若 direct 系显著为负但 tsc 系不显著 → 结论只在「欠饱和基优化器」上成立，")
    p("      即 LWS 仍是「修复欠饱和」，不是通用技巧（与 SOTA 对比结论一致）。")

    if a.out_md:
        open(a.out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
        print(f"\n[saved] {a.out_md}")


if __name__ == "__main__":
    main()
