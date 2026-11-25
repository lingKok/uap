"""
消融实验：inner_steps 和 support_size 对 Meta-UAP 的影响

实验设计（由 --ablations 选择，默认 "1,2,3" 全跑）：
  1. inner_steps 消融：固定 support_size=256，变化 inner_steps ∈ {1, 3, 5, 10}
  2. support_size 消融：固定 inner_steps=5，变化 support_size ∈ {64, 128, 256}
  3. 模型子集消融：固定其他，变化训练模型集合 ∈ {3, 3vgg, 3dn, 5}（可附 1/2 前缀子集）；
     inner_steps 由 --inner_steps 控制，0 即 ensemble-UAP 退化（方向三 (a) 定义性对照）

[协议修复 P0-1.2] 外层按 --seeds 逐 seed 完整重跑所选消融，最终输出 mean±std
（训练数据种子随 seed 变、评估种子固定 1234）。
每组训练 20 epoch（快速验证），在测试模型上评估 0-step FR。
"""
import argparse
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from config import Config
from models import create_all_models
from utils.data import get_cifar10_samples, sample_meta_batch
from utils.losses import get_attack_loss_fn, clamp_uap, attack_success_rate, differentiable_attack_loss


def meta_train_step(uap_init, target_model, support_images, support_labels,
                    query_images, query_labels, inner_steps, inner_lr, epsilon, attack_loss_fn):
    """手写可微分内循环"""
    uap = uap_init
    for step in range(inner_steps):
        clamped_uap = clamp_uap(uap, epsilon)
        adv_images = (support_images + clamped_uap).clamp(0, 1)
        logits = target_model(adv_images)
        loss = attack_loss_fn(logits, support_labels)
        grad = torch.autograd.grad(outputs=loss, inputs=uap, create_graph=True, retain_graph=True)[0]
        uap = uap - inner_lr * grad
        # 释放中间变量的显存
        del loss, logits, adv_images, grad

    final_uap = clamp_uap(uap, epsilon)
    adv_query = (query_images + final_uap).clamp(0, 1)
    query_logits = target_model(adv_query)
    meta_loss = differentiable_attack_loss(query_logits, query_labels)
    return meta_loss


def train_uap(train_models, images, labels, config, device, epochs=20):
    """训练一个 UAP_init

    [协议修复 P0-3(a)] config.inner_steps=0 时内循环 range(0) 不迭代：meta_loss 退化为
    query 集上的普通联合目标（clamp 后的 uap_init 被直接优化），即 ensemble-UAP——
    surrogate 池联合训练单一 δ，用于方向三 (a) "元学习初始化 vs 集成训练初始化" 的定义性对照。
    """
    uap_init = torch.zeros((3, config.image_size, config.image_size), device=device, requires_grad=True)
    meta_optimizer = torch.optim.Adam([uap_init], lr=config.meta_lr)
    attack_loss_fn = get_attack_loss_fn(config.attack_loss_type, num_classes=config.num_classes)

    for epoch in range(epochs):
        meta_optimizer.zero_grad()
        model_names = list(train_models.keys())
        np.random.shuffle(model_names)
        epoch_meta_loss = 0.0

        for model_name in model_names:
            target_model = train_models[model_name].to(device)
            sup_img, sup_lbl, qry_img, qry_lbl = sample_meta_batch(
                images, labels,
                support_size=config.support_size,
                query_size=config.query_size,
            )
            meta_loss = meta_train_step(
                uap_init, target_model, sup_img, sup_lbl, qry_img, qry_lbl,
                config.inner_steps, config.inner_lr, config.uap_epsilon, attack_loss_fn,
            )
            meta_loss.backward()
            epoch_meta_loss += meta_loss.item()
            del meta_loss
            target_model.cpu()
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_([uap_init], max_norm=1.0)
        meta_optimizer.step()

    return uap_init.detach()


def evaluate_uap(uap_init, model, images, labels, epsilon, device):
    """评估 0-step FR"""
    model = model.to(device)
    uap = clamp_uap(uap_init, epsilon)
    with torch.no_grad():
        adv = (images + uap).clamp(0, 1)
        logits = model(adv)
        fr = attack_success_rate(logits, labels).item()
    model.cpu()
    torch.cuda.empty_cache()
    return fr


def evaluate_uap_masks(uap_init, model, images, labels, epsilon, device):
    """评估 0-step FR 并返回逐样本 fooled 布尔（供 McNemar 配对检验）

    fooled 口径与 attack_success_rate 完全一致：预测 != 标签（攻击下误分类）。
    eval 集（seed=1234 固定）跨子集/跨 seed 一致，故掩码可直接配对比较。
    """
    model = model.to(device)
    uap = clamp_uap(uap_init, epsilon)
    with torch.no_grad():
        adv = (images + uap).clamp(0, 1)
        logits = model(adv)
        pred = logits.argmax(dim=1)
        fooled = (pred != labels).cpu().numpy().astype(int).tolist()
        fr = float(np.mean(fooled))
    model.cpu()
    torch.cuda.empty_cache()
    return fr, fooled


def set_seed(seed):
    """[协议修复 P0-1.2] 每个 seed 完整重跑前设置全局种子
    （原 ablation.py 无任何种子设置，同参数两次运行结果不可比、方差无法估计）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# [协议修复 P0-1.5] 命名模型子集：消融 3 改用子集字典驱动，可分离 VGG / DenseNet 效应；
# "1"/"2" 保留原 config.meta_train_models[:count] 前缀逻辑的语义，作为附录消融可选
MODEL_SUBSETS = {
    "3":    ["resnet18", "resnet34", "resnet50"],
    "3vgg": ["resnet18", "resnet34", "resnet50", "vgg16_bn"],
    "3dn":  ["resnet18", "resnet34", "resnet50", "densenet121"],
    "5":    ["resnet18", "resnet34", "resnet50", "vgg16_bn", "densenet121"],
    "1":    ["resnet18"],
    "2":    ["resnet18", "resnet34"],
}


def run_single_seed(args, ablation_ids, all_models, test_model_names, images, labels,
                    eval_images, eval_labels, device, epsilon, subset_names):
    """单个 seed 下的消融运行（由 ablation_ids 决定跑哪些类别）

    返回 (results, inner_steps_list, support_list)：后两者记录实际跑了哪些配置
    （未跑的类别为空列表），供汇总表按是否非空条件打印（防 KeyError）。

    [协议修复 P0-3(a)] --ablations 控制跑哪些消融：默认 "1,2,3" 全跑。
    ensemble-UAP 对照（方向三 (a) 定义性实验）的调用形式：
      python ablation.py --seeds "42,1342,2342" --model_subsets "3,5" --inner_steps 0 \
          --ablations "3" --train_split train --eval_split test --out_dir results_wave0
    --inner_steps 只作用于消融 3 的 cfg（ensemble 即 0）；消融 1 的扫描列表仅在其为
    非默认值时退化为单点（避免完整扫描浪费机时），消融 2 完全不受影响。
    """
    config = Config()
    results = {}
    inner_steps_list = []
    support_list = []
    masks = {}  # [协议修复 P0-4] --dump_masks 时记录逐样本 fooled（仅消融 3）

    # ==================== 消融 1: inner_steps ====================
    if "1" in ablation_ids:
        print("\n" + "=" * 70)
        print("消融实验 1: inner_steps 的影响 (固定 support_size=256)")
        print("=" * 70)

        # 扫描逻辑：默认（inner_steps=5）扫描 [1,3,5,10]；传非默认值则只跑该单点。
        # [协议修复 P0-3(a)] --inner_steps 的主战场是消融 3（ensemble 对照），此处单点退化
        # 只是避免非默认调用时完整扫描浪费机时；不跑时返回空列表（汇总表跳过）。
        inner_steps_list = [1, 3, 5, 10] if args.inner_steps == 5 else [args.inner_steps]
        for inner_steps in inner_steps_list:
            print(f"\n--- inner_steps={inner_steps} ---")
            # inner_steps=10 时减小 support 防止 OOM（原设计，论文重跑消融表时须统一或如实标注）
            support = 128 if inner_steps >= 10 else 256
            cfg = Config(inner_steps=inner_steps, support_size=support, query_size=64, meta_epochs=20)

            train_subset = {k: all_models[k] for k in config.meta_train_models}
            uap = train_uap(train_subset, images, labels, cfg, device, epochs=20)

            results[f"inner_steps_{inner_steps}"] = {}
            for name in test_model_names:
                fr = evaluate_uap(uap, all_models[name], eval_images, eval_labels, epsilon, device)
                results[f"inner_steps_{inner_steps}"][name] = fr
                print(f"  {name}: 0-step FR = {fr:.4f}")

    # ==================== 消融 2: support_size ====================
    if "2" in ablation_ids:
        print("\n" + "=" * 70)
        print("消融实验 2: support_size 的影响 (固定 inner_steps=5)")
        print("=" * 70)

        support_list = [64, 128, 256]
        for support_size in support_list:
            print(f"\n--- support_size={support_size} ---")
            cfg = Config(inner_steps=5, support_size=support_size, query_size=64, meta_epochs=20)

            train_subset = {k: all_models[k] for k in config.meta_train_models}
            uap = train_uap(train_subset, images, labels, cfg, device, epochs=20)

            results[f"support_{support_size}"] = {}
            for name in test_model_names:
                fr = evaluate_uap(uap, all_models[name], eval_images, eval_labels, epsilon, device)
                results[f"support_{support_size}"][name] = fr
                print(f"  {name}: 0-step FR = {fr:.4f}")

    # ==================== 消融 3: 训练模型子集 ====================
    if "3" in ablation_ids:
        print("\n" + "=" * 70)
        print(f"消融实验 3: 训练模型子集的影响 (inner_steps={args.inner_steps}"
              f"{'，ensemble-UAP 模式' if args.inner_steps == 0 else ''}, support=256)")
        print("=" * 70)

        # [协议修复 P0-1.5] 消融 3 由 --model_subsets 命名子集驱动（原为固定 [1,2,3,5] 计数 +
        # config.meta_train_models[:count] 前缀逻辑，无法分离 VGG / DenseNet 各自的效应）
        for subset_name in subset_names:
            if subset_name not in MODEL_SUBSETS:
                raise ValueError(f"未知模型子集: {subset_name}，可选: {list(MODEL_SUBSETS.keys())}")
            selected = MODEL_SUBSETS[subset_name]
            print(f"\n--- 训练模型子集={subset_name} ({selected}) ---")
            # [协议修复 P0-3(a)] inner_steps 读 --inner_steps：0 时 train_uap 内循环 range(0)
            # 不迭代，uap_init 直接被 query 联合目标优化 = ensemble-UAP（方向三 (a) 对照）
            cfg = Config(inner_steps=args.inner_steps, support_size=256, query_size=64, meta_epochs=20)

            train_subset = {k: all_models[k] for k in selected}
            uap = train_uap(train_subset, images, labels, cfg, device, epochs=20)

            results[f"models_{subset_name}"] = {}
            for name in test_model_names:
                if args.dump_masks:
                    fr, fooled = evaluate_uap_masks(
                        uap, all_models[name], eval_images, eval_labels, epsilon, device)
                    masks.setdefault(f"models_{subset_name}", {})[name] = fooled
                else:
                    fr = evaluate_uap(uap, all_models[name], eval_images, eval_labels, epsilon, device)
                results[f"models_{subset_name}"][name] = fr
                print(f"  {name}: 0-step FR = {fr:.4f}")

    return results, inner_steps_list, support_list, masks


def run_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    config = Config()
    epsilon = config.uap_epsilon

    # 加载评估数据（一次性；种子固定 1234，跨实验协议一致）
    print("\n加载 CIFAR-10 数据...")
    # [协议修复 P0-1.1] 训练/评估 split 解耦（原同为 test split 且默认同种子 seed=42，完全重叠）
    eval_images, eval_labels = get_cifar10_samples(config.data_root, num_samples=1000,
                                                   split=args.eval_split, seed=1234)
    if args.train_split == args.eval_split:
        print(f"[警告 P0-1.1] train_split == eval_split == '{args.train_split}'，"
              f"训练与评估数据存在重叠风险，请确认是否为预期设定")
    print(f"  评估数据: {len(eval_images)} 张 (split={args.eval_split}, eval_seed=1234 固定)")

    # 加载所有模型
    print("加载预训练模型...")
    all_model_names = config.meta_train_models + config.meta_test_models
    all_models = create_all_models(all_model_names, config.num_classes, config.image_size)
    for name, model in all_models.items():
        ckpt_path = config.save_dir / f"{name}.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    test_model_names = config.meta_test_models

    # [协议修复 P0-1.2] 外层 seed 循环：每个 seed 完整重跑所选消融，最终输出 mean±std
    # [协议修复 P0-3(a)] 消融类别解析：--ablations "1,2,3"（默认）全跑；ensemble-UAP
    # 对照（方向三 (a)）用 "3" 只跑模型子集消融，避免白白多跑 7 组元学习训练
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    subset_names = [s.strip() for s in args.model_subsets.split(",") if s.strip()]
    ablation_ids = [a.strip() for a in args.ablations.split(",") if a.strip()]
    unknown_abl = [a for a in ablation_ids if a not in {"1", "2", "3"}]
    if unknown_abl:
        raise ValueError(f"未知消融类别: {unknown_abl}，可选: ['1', '2', '3']")
    if "3" in ablation_ids and not subset_names:
        raise ValueError("--ablations 含 '3' 时 --model_subsets 不能为空")
    mode_desc = ("ensemble-UAP（inner_steps=0，无元学习，方向三 (a) 对照）"
                 if args.inner_steps == 0 else f"meta-UAP (inner_steps={args.inner_steps})")
    print(f"\n种子列表: {seeds}")
    print(f"模型子集: {subset_names}")
    print(f"消融类别: {ablation_ids}｜模式: {mode_desc}")

    all_runs = []
    all_masks = []
    inner_steps_summary = []
    support_summary = []
    for seed in seeds:
        print(f"\n{'=' * 70}\n===== Seed {seed} ({len(all_runs) + 1}/{len(seeds)}) =====\n{'=' * 70}")
        set_seed(seed)
        # [协议修复 P0-1.2 裁决] 数据种子独立性：训练数据采样种子随 run 变（data_seed = seed，
        # 如 seeds "42,1342,2342" → data_seed 42,1342,2342），方差诚实化、std 不再低估；
        # 评估集种子永远固定（1234），保证跨实验协议一致。get_cifar10_samples(seed=...) 已支持。
        images, labels = get_cifar10_samples(config.data_root, num_samples=600,
                                             split=args.train_split, seed=seed)
        print(f"  训练数据: {len(images)} 张 (split={args.train_split}, data_seed={seed})")
        run_results, inner_steps_summary, support_summary, run_masks = run_single_seed(
            args, ablation_ids, all_models, test_model_names, images, labels,
            eval_images, eval_labels, device, epsilon, subset_names)
        all_runs.append(run_results)
        all_masks.append(run_masks)

    # ==================== 聚合 mean±std ====================
    # [协议修复 P0-1.2] JSON 结构: {subset: {model: {"mean":x, "std":y, "runs":[...]}}}
    final = {}
    for subset in all_runs[0]:
        final[subset] = {}
        for model_name in all_runs[0][subset]:
            vals = [r[subset][model_name] for r in all_runs]
            final[subset][model_name] = {
                "mean": float(np.mean(vals)),
                # 单 seed 时 std 记 0（样本 std ddof=1 不可估）；多 seed 用样本 std
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "runs": [float(v) for v in vals],
            }

    # ==================== 汇总表格 ====================
    print("\n" + "=" * 70)
    print(f"消融实验汇总 (mean±std over {len(seeds)} seeds)")
    print("=" * 70)

    def _print_table(title, keys):
        # [协议修复 P0-3(a)] 按 final 实际存在的键打印：--ablations 可能只跑部分消融，
        # 缺失类别整表跳过（防 KeyError）
        existing = [k for k in keys if k in final]
        if not existing:
            return
        print(f"\n{title}")
        print(f"{'config':<15}", end="")
        for name in test_model_names:
            print(f" {name:>17}", end="")
        print()
        for subset in existing:
            print(f"{subset:<15}", end="")
            for name in test_model_names:
                d = final[subset][name]
                print(f" {d['mean']:.4f}±{d['std']:.4f}", end="")
            print()

    # [协议修复 P0-3(a)] 按列表是否非空条件打印：--ablations 未选的类别（列表为空）
    # 整表跳过，防 KeyError；表内再按 final 实际存在的键过滤（双保险）
    if inner_steps_summary:
        _print_table("1. inner_steps 的影响:", [f"inner_steps_{s}" for s in inner_steps_summary])
    if support_summary:
        _print_table("2. support_size 的影响:", [f"support_{s}" for s in support_summary])
    if "3" in ablation_ids:
        _print_table("3. 训练模型子集的影响:", [f"models_{s}" for s in subset_names])

    # 保存
    # [协议修复 P0-1.2] 输出至 --out_dir；文件名随模式区分，避免不同模式互相覆盖
    # （默认全量 results_ablation_zoo.json；ensemble 模式 inner_steps=0 加 _ensemble 后缀；
    #   非全量消融加 _abl 后缀）——原靠脚本 mv 归档，非根本解法
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.inner_steps == 0:
        suffix = "_ensemble"
    elif ablation_ids != ["1", "2", "3"]:
        suffix = "_abl" + "".join(ablation_ids)
    save_path = out_dir / f"results_ablation_zoo{suffix}.json"
    with open(save_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n结果已保存: {save_path}")

    # [协议修复 P0-4] 逐样本 fooled 掩码（McNemar 配对检验用）
    # 修复: 掩码文件名随模式加 suffix（与结果 JSON 一致），否则 ensemble(inner_steps=0)
    # 会覆盖 meta 掩码，导致 3vgg/3dn 隔离掩码丢失（实测 2026-10-24 发现）。
    if args.dump_masks:
        mask_path = out_dir / f"results_ablation_zoo_masks{suffix}.json"
        json.dump({
            "eval_split": args.eval_split,
            "eval_seed": 1234,
            "num_eval": len(eval_images),
            "test_models": test_model_names,
            "subsets": subset_names,
            "seeds": seeds,
            "masks": all_masks,
        }, open(mask_path, "w"), indent=2)
        print(f"逐样本 fooled 掩码已保存: {mask_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [协议修复 P0-1.2] 多 seed 完整重跑，输出 mean±std（逗号分隔，如 "42,1234,2024"）
    parser.add_argument("--seeds", type=str, default="42")
    # [协议修复 P0-3(a)] 消融类别选择（默认 "1,2,3" 全跑；ensemble-UAP 对照只跑 "3"）。
    # 1=inner_steps 扫描, 2=support_size 扫描, 3=模型子集消融
    parser.add_argument("--ablations", type=str, default="1,2,3")
    # [协议修复 P0-3(a)] inner_steps（默认 5；0 = ensemble-UAP 退化，方向三 (a) 定义性对照）；
    # 只作用于消融 3 的 cfg（消融 1/2 本身即 inner_steps/support 消融，语义独立）。
    # ensemble 调用形式：
    # python ablation.py --seeds "42,1342,2342" --model_subsets "3,5" --inner_steps 0 \
    #     --ablations "3" --train_split train --eval_split test --out_dir results_wave0
    # （输出文件名自动变为 results_ablation_zoo_ensemble.json，与全量版互不覆盖）
    parser.add_argument("--inner_steps", type=int, default=5)
    # [协议修复 P0-1.5] 命名模型子集（默认 3,3vgg,3dn,5；附录前缀子集可选 1,2）
    parser.add_argument("--model_subsets", type=str, default="3,3vgg,3dn,5")
    # [协议修复 P0-1.1] 训练/评估 split 解耦（默认 train/test）
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    # [协议修复 P0-1.2] 输出目录（默认 results_wave0），不覆盖旧结果
    parser.add_argument("--out_dir", type=str, default="results_wave0")
    # [协议修复 P0-4] 保存逐样本 fooled 掩码（供 McNemar 配对检验；仅对消融 3 生效）
    parser.add_argument("--dump_masks", action="store_true",
                        help="保存逐样本 fooled 掩码（McNemar 用；仅消融 3 生效）")
    args = parser.parse_args()
    run_ablation(args)
