"""
实验脚本：SOTA 对比 + 效率分析

SOTA 方法：
  - UAP-SGD: 经典 SGD 优化 UAP (Haystack et al.)
  - PD-UA:   梯度投影 + 动量 (Moosavi-Dezfooli et al.)
  - SSA:     单步攻击 (快速基线)
  - Meta-UAP (ours): 0-step / 5-step

效率分析：
  - 记录每种方法达到目标 FR 所需的前向传播次数
  - 记录 wall-clock 时间

支持 CIFAR-10 和 ImageNet 两种数据集
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
from models import create_all_models, create_all_imagenet_models, NormalizedModel
from utils.data import get_cifar10_samples, get_imagenet_mini_samples
from utils.losses import clamp_uap, attack_success_rate


# ==================== SOTA 方法实现 ====================

def uap_sgd(model, images, labels, epsilon, steps=50, lr=0.01, batch_size=128):
    """UAP-SGD: 标准 SGD 优化 UAP (Haystack et al. 2018)"""
    uap = torch.zeros_like(images[0], requires_grad=True)
    optimizer = torch.optim.Adam([uap], lr=lr)

    n = min(len(images), batch_size)
    for step in range(steps):
        optimizer.zero_grad()
        idx = np.random.choice(len(images), n, replace=False)
        batch = images[idx]
        batch_lbl = labels[idx]

        clamped = clamp_uap(uap, epsilon)
        adv = (batch + clamped).clamp(0, 1)
        logits = model(adv)
        loss = -F.cross_entropy(logits, batch_lbl)
        loss.backward()
        optimizer.step()

    return clamp_uap(uap.detach(), epsilon)


def pd_ua(model, images, labels, epsilon, steps=50, lr=0.01, momentum=0.9, batch_size=128):
    """PD-UA: 投影梯度下降 + 动量 (Moosavi-Dezfooli et al. 2017)"""
    uap = torch.zeros_like(images[0], requires_grad=True)
    optimizer = torch.optim.SGD([uap], lr=lr, momentum=momentum)

    n = min(len(images), batch_size)
    for step in range(steps):
        optimizer.zero_grad()
        idx = np.random.choice(len(images), n, replace=False)
        batch = images[idx]
        batch_lbl = labels[idx]

        adv = (batch + clamp_uap(uap, epsilon)).clamp(0, 1)
        logits = model(adv)
        loss = -F.cross_entropy(logits, batch_lbl)
        loss.backward()
        optimizer.step()

    return clamp_uap(uap.detach(), epsilon)


def ssa(model, images, labels, epsilon, steps=1, lr=0.1, batch_size=32):
    """SSA: 单步攻击 (快速基线) — 只做一次大步长梯度更新"""
    uap = torch.zeros_like(images[0], requires_grad=True)

    n = min(len(images), batch_size)
    adv = (images[:n] + clamp_uap(uap, epsilon)).clamp(0, 1)
    logits = model(adv)
    loss = -F.cross_entropy(logits, labels[:n])
    grad = torch.autograd.grad(loss, uap)[0]
    uap = clamp_uap(uap - lr * grad, epsilon).detach()

    return uap


def gen_uap(model, images, labels, epsilon, steps=10, lr=0.01):
    """Gen-UAP (simplified): fast UAP via gradient accumulation with momentum.
    Inspired by Haystack et al. but with fewer steps and momentum acceleration.
    Represents the 'fast optimization' baseline category."""
    uap = torch.zeros_like(images[0], requires_grad=True)
    optimizer = torch.optim.SGD([uap], lr=lr, momentum=0.9)
    n = min(len(images), 64)
    for step in range(steps):
        optimizer.zero_grad()
        idx = np.random.choice(len(images), n, replace=False)
        batch = images[idx]
        batch_lbl = labels[idx]
        adv = (batch + clamp_uap(uap, epsilon)).clamp(0, 1)
        logits = model(adv)
        loss = -F.cross_entropy(logits, batch_lbl)
        loss.backward()
        optimizer.step()

    return clamp_uap(uap.detach(), epsilon)


def meta_uap_eval(model, uap_init, images, labels, epsilon, steps=0, lr=0.1, eval_batch=128):
    """Meta-UAP: 从 uap_init 出发，做 steps 步优化后评估"""
    uap = uap_init.clone().detach()

    for s in range(steps):
        uap = uap.requires_grad_(True)
        n = min(len(images), eval_batch)
        adv = (images[:n] + clamp_uap(uap, epsilon)).clamp(0, 1)
        logits = model(adv)
        loss = -F.cross_entropy(logits, labels[:n])
        grad = torch.autograd.grad(loss, uap)[0]
        uap = clamp_uap(uap - lr * grad, epsilon).detach()

    return clamp_uap(uap, epsilon)


# ==================== 评估工具 ====================

def compute_fr(model, uap, images, labels, batch_size=128):
    """计算 fooling rate（分批防 OOM）"""
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            adv = (images[start:end] + uap).clamp(0, 1)
            logits = model(adv)
            correct += (logits.argmax(dim=1) != labels[start:end]).sum().item()
            total += end - start
    return correct / total


def _fmt(v):
    """指标值格式化（None → N/A）"""
    return f"{v:.4f}" if v is not None else "N/A"


def compute_metrics(model, uap, images, labels, batch_size=128):
    """[协议修复 P0-1.3] 双口径指标（替代单一 compute_fr 作为论文报告口径）：
    - clean_acc:   无攻击准确率
    - fr_all:      旧口径 fooling rate = 攻击后全测试集错误率（保留供对照）
    - asr_correct: 标准口径 ASR = 攻击致错样本数 / 攻击前正确样本数
                   （分子 = clean 正确且 adv 错误的样本数；分母为 0 时记 None）
    """
    clean_correct = 0
    adv_correct = 0
    fooled = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch_img = images[start:end]
            batch_lbl = labels[start:end]
            clean_ok = model(batch_img).argmax(dim=1) == batch_lbl
            adv_ok = model((batch_img + uap).clamp(0, 1)).argmax(dim=1) == batch_lbl
            clean_correct += clean_ok.sum().item()
            adv_correct += adv_ok.sum().item()
            fooled += (clean_ok & ~adv_ok).sum().item()
            total += end - start
    return {
        "clean_acc": clean_correct / total,
        "fr_all": (total - adv_correct) / total,
        "asr_correct": (fooled / clean_correct) if clean_correct > 0 else None,
    }


def time_fn(fn, *args, **kwargs):
    """计时执行"""
    torch.cuda.synchronize()
    start = time.time()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return result, elapsed


# ==================== 主实验 ====================

def run_experiments(args):
    # [协议修复 P0-1.2] 全局种子：修复 experiments.py 全程无种子问题
    # （同代码两次运行 UAP-SGD-50@ResNet-50 得 66.25%/59.25%，Δ7.0pp）
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"全局种子: {args.seed}")

    is_imagenet = args.dataset == "imagenet"
    epsilon = args.epsilon
    # [协议修复 P0-1.1] 数据 split 解耦（ImageNet 默认 train/val，CIFAR 默认 train/test）
    train_split = args.train_split
    eval_split = args.eval_split or ("val" if is_imagenet else "test")

    # ==================== 加载数据 ====================
    print(f"\n加载 {args.dataset} 数据...")
    if is_imagenet:
        # [协议修复 P0-1.4] 评估样本数由 --eval_samples 统一控制（原 ImageNet 500 张 SE≈2.1pp → 默认 2000）
        eval_images, eval_labels = get_imagenet_mini_samples(
            num_samples=args.eval_samples, split=eval_split, image_size=224, seed=42,
            fallback_val_holdout=args.fallback_val_holdout,
        )
        # [协议修复 P0-1.1] baseline 优化数据改用 train split（原与评估同为 val split，种子不同但仍有重叠）
        train_images, train_labels = get_imagenet_mini_samples(
            num_samples=200, split=train_split, image_size=224, seed=100,
            fallback_val_holdout=args.fallback_val_holdout,
        )
        num_classes = 1000
        eval_batch = 16
    else:
        config = Config()
        # [协议修复 P0-1.4] CIFAR 评估同样受 --eval_samples 控制（原 2000）
        eval_images, eval_labels = get_cifar10_samples(
            config.data_root, num_samples=args.eval_samples, split=eval_split, seed=42,
        )
        # [协议修复 P0-1.1] CIFAR baseline 优化数据改用 train split（原与评估同为 test split）
        train_images, train_labels = get_cifar10_samples(
            config.data_root, num_samples=500, split=train_split, seed=100,
        )
        num_classes = config.num_classes
        eval_batch = 128
    print(f"  优化数据: {len(train_images)} 张 (split={train_split}, seed=100)")
    print(f"  评估数据: {len(eval_images)} 张 (split={eval_split}, seed=42)")

    # ==================== 加载模型 ====================
    print("\n加载模型...")
    if is_imagenet:
        all_model_names = ["resnet18", "resnet34", "resnet50", "vgg16_bn", "densenet121",
                           "resnet101", "mobilenet_v2", "shufflenet_v2_x1_0"]
        all_models = create_all_imagenet_models(all_model_names)
    else:
        all_model_names = config.meta_train_models + config.meta_test_models
        all_models = create_all_models(all_model_names, config.num_classes, config.image_size)
        for name, model in all_models.items():
            ckpt_path = config.save_dir / f"{name}.pt"
            if ckpt_path.exists():
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            else:
                # [协议修复 P0-1.x] checkpoint 缺失时显式告警，避免静默使用随机初始化权重导致数字不可用
                print(f"  [警告] {name}: checkpoint 缺失 ({ckpt_path})，使用随机初始化权重！")

    for name in all_model_names:
        all_models[name] = all_models[name].to(device).eval()

    # ==================== 加载 Meta-UAP ====================
    print("\n加载 Meta-UAP...")
    if is_imagenet:
        # 加载 ensemble 或单个
        uap_path = args.meta_checkpoint or "checkpoints_imagenet/meta_uap_imagenet_ensemble8.pt"
    else:
        uap_path = args.meta_checkpoint or "checkpoints/meta_uap_epoch50.pt"

    ckpt = torch.load(uap_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "uap_init" in ckpt:
        uap_init = ckpt["uap_init"].to(device)
    else:
        uap_init = ckpt.to(device)
    print(f"  UAP 路径: {uap_path}")
    print(f"  UAP 范数: {uap_init.abs().max().item():.4f}")

    # ==================== SOTA 对比实验 ====================
    print("\n" + "=" * 70)
    print("实验 1: SOTA 对比 — 各方法在所有模型上的 fooling rate（双口径: FR / ASR）")
    print("=" * 70)

    # 实验配置
    pgd_steps = [10, 30, 50]
    sota_methods = ["UAP-SGD", "PD-UA", "SSA", "Gen-UAP", "Meta-0", "Meta-5", "Meta-10"]
    results = {method: {} for method in sota_methods}
    timing = {method: {} for method in sota_methods}

    # 选几个代表模型做 SOTA 对比（训练模型 + 测试模型各选）
    if is_imagenet:
        sota_models = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "shufflenet_v2_x1_0"]
    else:
        sota_models = ["resnet50", "vgg16_bn", "resnet101", "mobilenet_v2", "shufflenet_v2_x1_0"]

    for model_name in sota_models:
        if model_name not in all_models:
            continue
        model = all_models[model_name]
        is_test = model_name in (["resnet101", "mobilenet_v2", "shufflenet_v2_x1_0"])
        marker = "[测试]" if is_test else "[训练]"
        print(f"\n{marker} {model_name}:")

        # UAP-SGD (50步)
        uap, t = time_fn(uap_sgd, model, train_images, train_labels, epsilon, steps=50)
        # [协议修复 P0-1.3] 双口径指标：fr_all（旧口径）/ asr_correct（标准口径）
        m = compute_metrics(model, uap, eval_images, eval_labels, eval_batch)
        results["UAP-SGD"][model_name] = m
        timing["UAP-SGD"][model_name] = t
        model.cpu(); torch.cuda.empty_cache(); model.to(device)
        print(f"  UAP-SGD (50步):    FR={_fmt(m['fr_all'])}  ASR={_fmt(m['asr_correct'])}  time={t:.1f}s")

        # PD-UA (50步)
        uap, t = time_fn(pd_ua, model, train_images, train_labels, epsilon, steps=50)
        m = compute_metrics(model, uap, eval_images, eval_labels, eval_batch)
        results["PD-UA"][model_name] = m
        timing["PD-UA"][model_name] = t
        model.cpu(); torch.cuda.empty_cache(); model.to(device)
        print(f"  PD-UA (50步):      FR={_fmt(m['fr_all'])}  ASR={_fmt(m['asr_correct'])}  time={t:.1f}s")

        # SSA (1步)
        ssa_batch = 16 if is_imagenet else 128
        uap, t = time_fn(ssa, model, train_images, train_labels, epsilon, steps=1, batch_size=ssa_batch)
        m = compute_metrics(model, uap, eval_images, eval_labels, eval_batch)
        results["SSA"][model_name] = m
        timing["SSA"][model_name] = t
        model.cpu(); torch.cuda.empty_cache(); model.to(device)
        print(f"  SSA (1步):         FR={_fmt(m['fr_all'])}  ASR={_fmt(m['asr_correct'])}  time={t:.1f}s")

        # Gen-UAP (10步, momentum)
        uap, t = time_fn(gen_uap, model, train_images, train_labels, epsilon, steps=10)
        m = compute_metrics(model, uap, eval_images, eval_labels, eval_batch)
        results["Gen-UAP"][model_name] = m
        timing["Gen-UAP"][model_name] = t
        model.cpu(); torch.cuda.empty_cache(); model.to(device)
        print(f"  Gen-UAP (10步):    FR={_fmt(m['fr_all'])}  ASR={_fmt(m['asr_correct'])}  time={t:.1f}s")

        # Meta-UAP 0步
        # [协议修复 P0-1.1] k-step 适配梯度改在 train_images 上进行（原传 eval_images，
        # 即在评估集上做梯度适配 = train-on-test，第三重数据污染）；steps=0 不使用 images，为一致性同改
        uap = meta_uap_eval(model, uap_init, train_images, train_labels, epsilon, steps=0)
        m = compute_metrics(model, uap, eval_images, eval_labels, eval_batch)
        results["Meta-0"][model_name] = m
        print(f"  Meta-UAP (0步):    FR={_fmt(m['fr_all'])}  ASR={_fmt(m['asr_correct'])}  time=0.0s")

        # Meta-UAP 5步
        # [协议修复 P0-1.1] 适配梯度改在 train split 上进行（与评估集零交集）
        uap, t = time_fn(meta_uap_eval, model, uap_init, train_images, train_labels, epsilon, steps=5)
        m = compute_metrics(model, uap, eval_images, eval_labels, eval_batch)
        results["Meta-5"][model_name] = m
        timing["Meta-5"][model_name] = t
        print(f"  Meta-UAP (5步):    FR={_fmt(m['fr_all'])}  ASR={_fmt(m['asr_correct'])}  time={t:.1f}s")

        # Meta-UAP 10步
        # [协议修复 P0-1.1] 适配梯度改在 train split 上进行（与评估集零交集）
        uap, t = time_fn(meta_uap_eval, model, uap_init, train_images, train_labels, epsilon, steps=10)
        m = compute_metrics(model, uap, eval_images, eval_labels, eval_batch)
        results["Meta-10"][model_name] = m
        timing["Meta-10"][model_name] = t
        print(f"  Meta-UAP (10步):   FR={_fmt(m['fr_all'])}  ASR={_fmt(m['asr_correct'])}  time={t:.1f}s")

    # ==================== 汇总表格 ====================
    print("\n" + "=" * 70)
    print("SOTA 对比汇总表")
    print("=" * 70)

    # [协议修复 P0-1.3] results[method][model] 现为双口径 dict，分三张表打印
    def _print_metric_table(metric_key, title):
        print(f"\n{title}")
        header = f"{'方法':<15}"
        for m in sota_models:
            header += f" {m:>15}"
        print(header)
        print("-" * (15 + 16 * len(sota_models)))
        for method in sota_methods:
            row = f"{method:<15}"
            for m in sota_models:
                vals = results[method].get(m)
                v = vals.get(metric_key) if isinstance(vals, dict) else vals
                row += f" {v:>15.4f}" if (v is not None and v >= 0) else f" {'N/A':>15}"
            print(row)

    _print_metric_table("fr_all", "口径 1（旧口径，保留对照）: 全测试集错误率 FR")
    _print_metric_table("asr_correct", "口径 2（标准口径）: ASR = 攻击致错数 / 攻击前正确样本数")
    _print_metric_table("clean_acc", "clean accuracy（无攻击，供口径解释参考）")

    # ==================== 效率分析 ====================
    print("\n" + "=" * 70)
    print("实验 2: 效率分析 — 达到目标 FR 所需时间")
    print("=" * 70)

    print(f"\n{'方法':<15} {'模型':<15} {'FR':>8} {'ASR':>8} {'时间(s)':>10} {'前向次数':>10}")
    print("-" * 68)
    for method in sota_methods:
        for model_name in sota_models:
            if model_name in results[method]:
                vals = results[method][model_name]
                # [协议修复 P0-1.3] 兼容双口径 dict；FR 列展示旧口径 fr_all，ASR 列为标准口径
                fr = vals["fr_all"] if isinstance(vals, dict) else vals
                asr = vals.get("asr_correct") if isinstance(vals, dict) else None
                t = timing[method].get(model_name, 0)
                if method == "Meta-0":
                    fwd = 0
                elif method == "Meta-5":
                    fwd = 5
                elif method == "Meta-10":
                    fwd = 10
                elif method == "SSA":
                    fwd = 1
                elif method == "Gen-UAP":
                    fwd = 10
                else:
                    fwd = 50
                print(f"{method:<15} {model_name:<15} {fr:>8.4f} {_fmt(asr):>8} {t:>10.1f} {fwd:>10}")

    # ==================== ε 敏感性分析 ====================
    print("\n" + "=" * 70)
    print("实验 3: ε 敏感性分析")
    print("=" * 70)
    train_epsilon = args.epsilon  # Meta-UAP 训练时的 ε
    epsilons = [0.03, 0.05, 0.1, 0.2]
    # [协议修复 P0-3(e)] UAP-SGD 大步数对照：判定大 ε 下 baseline 是欠拟合还是方向劣势
    # 仅在 --scaled_sgd 同开时生效（同一批运行一次拿全数据）
    extra_steps_list = ([int(s) for s in args.sgd_extra_steps.split(",") if s.strip()]
                        if args.sgd_extra_steps else [])
    # [协议修复 P0-3(c)] ε_train 下训练的 UAP-SGD 解缓存（不随 ε_test 变化，按模型算一次）
    sgd_train_eps_cache = {}
    epsilon_results = {}
    for eps in epsilons:
        print(f"\n--- ε={eps} ---")
        epsilon_results[eps] = {}
        # ε-scaling: 按比例放大 UAP 以利用更大预算
        scale_factor = eps / train_epsilon
        uap_scaled = clamp_uap(uap_init * scale_factor, eps)
        for model_name in sota_models[:3]:  # 取前3个模型
            model = all_models[model_name].to(device)
            # UAP-SGD（按当前 ε 直接训练）
            uap_s, _ = time_fn(uap_sgd, model, train_images, train_labels, eps, steps=50)
            m_sgd = compute_metrics(model, uap_s, eval_images, eval_labels, eval_batch)
            # Meta-UAP 0-step (原始, clamp 到当前 eps)
            # [协议修复 P0-1.1] 与 SOTA 区一致：适配数据用 train split（steps=0 不使用 images）
            uap_meta = meta_uap_eval(model, uap_init, train_images, train_labels, eps, steps=0)
            m_meta = compute_metrics(model, uap_meta, eval_images, eval_labels, eval_batch)
            # Meta-UAP 0-step (ε-scaled)
            m_meta_scaled = compute_metrics(model, uap_scaled, eval_images, eval_labels, eval_batch)
            entry = {
                "UAP-SGD": m_sgd, "Meta-0": m_meta, "Meta-0-scaled": m_meta_scaled,
            }
            if args.scaled_sgd:
                # [协议修复 P0-3(c)] Scaled-UAP-SGD：对 ε_train 下训练得到的 UAP-SGD 解
                # 施加 ×(ε_test/ε_train) 缩放后 clamp，再评估
                # （uap_sgd 返回的 uap 本身已 clamp 到 ε_train，缩放公式 clamp_uap(uap*(eps/train_eps), eps)）
                if model_name not in sgd_train_eps_cache:
                    uap_sgd_te, _ = time_fn(uap_sgd, model, train_images, train_labels,
                                            train_epsilon, steps=50)
                    sgd_train_eps_cache[model_name] = uap_sgd_te
                uap_sgd_scaled = clamp_uap(sgd_train_eps_cache[model_name] * (eps / train_epsilon), eps)
                m_sgd_scaled = compute_metrics(model, uap_sgd_scaled, eval_images, eval_labels, eval_batch)
                entry["Scaled-UAP-SGD"] = m_sgd_scaled
                # [协议修复 P0-3(e)] UAP-SGD 更大步数对照（仅 --scaled_sgd 同开时执行）
                for extra_steps in extra_steps_list:
                    uap_x, _ = time_fn(uap_sgd, model, train_images, train_labels, eps,
                                       steps=extra_steps)
                    m_x = compute_metrics(model, uap_x, eval_images, eval_labels, eval_batch)
                    entry[f"UAP-SGD-{extra_steps}"] = m_x
            epsilon_results[eps][model_name] = entry
            line = (f"  {model_name}: SGD={m_sgd['fr_all']:.4f}  Meta-0={m_meta['fr_all']:.4f}"
                    f"  Meta-0-scaled={m_meta_scaled['fr_all']:.4f}")
            if args.scaled_sgd:
                line += f"  ScaledSGD={entry['Scaled-UAP-SGD']['fr_all']:.4f}"
                for extra_steps in extra_steps_list:
                    line += f"  SGD{extra_steps}={entry[f'UAP-SGD-{extra_steps}']['fr_all']:.4f}"
            print(line)
            model.cpu(); torch.cuda.empty_cache()

    # ==================== 保存结果 ====================
    # [协议修复 P0-1.3] 结果保存至 --out_dir（默认 results_wave0）；
    # 原 results_{dataset}_sota.json 为污染口径，保留不覆盖
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"results_{args.dataset}_sota.json"
    with open(save_path, "w") as f:
        json.dump({"results": results, "timing": timing, "epsilon": epsilon_results}, f, indent=2)
    print(f"\n结果已保存: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "imagenet"])
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--meta_checkpoint", type=str, default=None)
    # [协议修复 P0-1.2] 全局种子（默认 42；多 seed 复跑由外层循环传不同值）
    parser.add_argument("--seed", type=int, default=42)
    # [协议修复 P0-1.1] 数据 split 解耦：ImageNet 默认 train/val，CIFAR 默认 train/test
    # （eval_split 默认 None → 按数据集自动解析，保证两数据集默认值与规格一致）
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default=None,
                        help="默认: imagenet→val, cifar→test")
    # [协议修复 P0-1.4] 评估样本数统一参数（原 ImageNet 500 / CIFAR 2000 硬编码不一致）
    parser.add_argument("--eval_samples", type=int, default=2000)
    # [协议修复 P0-3(c)] ε-sweep 增加 Scaled-UAP-SGD 列（对 ε_train 下 UAP-SGD 解做 ε-scaling）
    parser.add_argument("--scaled_sgd", action="store_true")
    # [协议修复 P0-3(e)] 每个 ε 增加 UAP-SGD 更大步数对照，如 "100,200"；仅 --scaled_sgd 同开时生效
    parser.add_argument("--sgd_extra_steps", type=str, default="")
    # [协议修复 P0-1.3] 结果输出目录（默认 results_wave0），不覆盖污染口径旧结果
    parser.add_argument("--out_dir", type=str, default="results_wave0")
    # [协议修复 P0-1.1] 备选方案：imagenet-mini 无 train/ 目录时，从 val 内按不重合切分
    # 产生训练/评估数据（训练池 seed=100、评估 seed=42 且排除训练池索引）；见 utils/data.py
    parser.add_argument("--fallback_val_holdout", action="store_true")
    args = parser.parse_args()
    run_experiments(args)
