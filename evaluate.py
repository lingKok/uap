"""
Idea 2: 评估脚本 — 效率-效果权衡曲线 + 跨架构泛化

使用方法:
    python evaluate.py --checkpoint checkpoints/meta_uap_epoch50.pt
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config
from models import create_all_models
from utils.data import get_cifar10_samples
from utils.losses import clamp_uap, attack_success_rate


def evaluate_uap_at_steps(
    uap_init: torch.Tensor,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: list,
    lr: float = 0.1,
) -> dict:
    """在给定步数下评估 UAP 的攻击效果"""
    results = {}
    for step in steps:
        uap = uap_init.clone().detach()

        for s in range(step):
            uap.requires_grad_(True)
            adv = (images + clamp_uap(uap, epsilon)).clamp(0, 1)
            logits = model(adv)
            loss = -F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, uap)[0]
            uap = clamp_uap(uap - lr * grad, epsilon).detach()

        with torch.no_grad():
            adv = (images + clamp_uap(uap, epsilon)).clamp(0, 1)
            logits = model(adv)
            fr = attack_success_rate(logits, labels).item()
        results[step] = fr

    return results


def baseline_uap_pgd(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int = 50,
    lr: float = 0.01,
) -> torch.Tensor:
    """基线：标准 PGD 优化 UAP"""
    uap = torch.zeros_like(images[0], requires_grad=True)
    opt = torch.optim.Adam([uap], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        clamped = clamp_uap(uap, epsilon)
        adv = (images[:500] + clamped).clamp(0, 1)  # 用子集优化
        logits = model(adv)
        loss = -F.cross_entropy(logits, labels[:500])
        loss.backward()
        opt.step()

    return clamp_uap(uap.detach(), epsilon)


def main():
    parser = argparse.ArgumentParser(description="Meta-UAP Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="UAP 初始化路径")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--eval_steps", type=str, default="0,1,3,5,10,20,50",
                        help="评估步数列表，逗号分隔")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eval_steps = [int(s) for s in args.eval_steps.split(",")]

    # 加载 UAP
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        uap_init = checkpoint["uap_init"].to(device)
        print(f"加载 UAP: {args.checkpoint}")
        if "config" in checkpoint:
            saved_config = checkpoint["config"]
            print(f"  训练 epochs: {saved_config.meta_epochs}")
            print(f"  inner_steps: {saved_config.inner_steps}")
            print(f"  epsilon: {saved_config.uap_epsilon}")
    else:
        uap_init = checkpoint.to(device)

    print(f"  UAP shape: {uap_init.shape}")
    print(f"  UAP 范数: {uap_init.abs().max().item():.4f}")

    # 加载数据
    print("\n加载数据...")
    eval_images, eval_labels = get_cifar10_samples(config.data_root, num_samples=2000, split="test")

    # 加载模型
    all_model_names = config.meta_train_models + config.meta_test_models
    all_models = create_all_models(all_model_names, config.num_classes, config.image_size)

    for name, model in all_models.items():
        ckpt_path = config.save_dir / f"{name}.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        all_models[name] = model.to(device).eval()

    # ==================== 实验 1: 跨步数评估 ====================
    print("\n" + "=" * 70)
    print("实验 1: 不同优化步数下的 fooling rate")
    print("=" * 70)

    all_results = {}
    for name in all_model_names:
        marker = "[训练]" if name in config.meta_train_models else "[测试]"
        print(f"\n{marker} {name}:")
        results = evaluate_uap_at_steps(
            uap_init, all_models[name],
            eval_images, eval_labels,
            args.epsilon, eval_steps,
        )
        all_results[name] = results

        for step, fr in results.items():
            bar = "█" * int(fr * 40)
            print(f"  step={step:3d}: {fr:.4f} {bar}")

    # ==================== 实验 2: 与基线对比 ====================
    print("\n" + "=" * 70)
    print("实验 2: 元学习 UAP vs 基线 PGD UAP")
    print("=" * 70)

    for name in ["resnet50"] + config.meta_test_models:
        if name not in all_models:
            continue

        # 基线: 在该模型上直接 PGD 优化 50 步
        print(f"\n{name}:")
        baseline_uap = baseline_uap_pgd(
            all_models[name], eval_images, eval_labels,
            args.epsilon, steps=50,
        )

        with torch.no_grad():
            # 基线效果
            adv = (eval_images + baseline_uap).clamp(0, 1)
            fr_baseline = attack_success_rate(all_models[name](adv), eval_labels).item()

            # 元学习 0 步
            meta_uap_0 = clamp_uap(uap_init, args.epsilon)
            adv_meta = (eval_images + meta_uap_0).clamp(0, 1)
            fr_meta_0 = attack_success_rate(all_models[name](adv_meta), eval_labels).item()

        print(f"  基线 PGD 50步: {fr_baseline:.4f}")
        print(f"  元学习 0步:    {fr_meta_0:.4f}")

        if name in all_results:
            fr_meta_5 = all_results[name].get(5, 0)
            print(f"  元学习 5步:    {fr_meta_5:.4f}")

    # ==================== 实验 3: 效率-效果对比 ====================
    print("\n" + "=" * 70)
    print("实验 3: 效率-效果权衡曲线（摘要）")
    print("=" * 70)

    print(f"\n{'模型':<20} {'0步':>8} {'1步':>8} {'5步':>8} {'10步':>8} {'50步':>8}")
    print("-" * 70)
    for name, results in all_results.items():
        marker = "★" if name in config.meta_test_models else " "
        row = f"{marker} {name:<18}"
        for step in [0, 1, 5, 10, 50]:
            fr = results.get(step, -1)
            row += f" {fr:>8.4f}" if fr >= 0 else f" {'N/A':>8}"
        print(row)

    # 保存结果
    save_path = Path(args.checkpoint).parent / "eval_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n评估结果已保存: {save_path}")


if __name__ == "__main__":
    from pathlib import Path
    main()
