"""
Idea 2: Meta-Learning Single-Step UAP — ImageNet 版本

核心改进：
- 使用 ImageNet 预训练模型（1000 类，224x224）
- 不需要预训练，直接用 torchvision 权重
- UAP 在 [0,1] 像素空间操作，模型内部处理归一化

使用方法:
    python train_imagenet.py --epochs 50 --inner_steps 5 --support_size 32 --query_size 8
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from models import create_all_imagenet_models
from utils.data import get_imagenet_mini_samples, sample_meta_batch
from utils.losses import get_attack_loss_fn, clamp_uap, attack_success_rate, differentiable_attack_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-UAP Training on ImageNet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--meta_lr", type=float, default=0.01)
    parser.add_argument("--inner_lr", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--support_size", type=int, default=32)
    parser.add_argument("--query_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_imagenet")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--attack_loss", type=str, default="ce", choices=["ce", "cw"])
    parser.add_argument("--seed", type=int, default=42)
    # [协议修复 P0-1.1] 元训练/评估数据 split 解耦：训练默认 train split、评估默认 val split，
    # 杜绝原"训练与评估同为 val 且同种子，40 张训练图全部落入 200 张评估图内"的完全重叠
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="val")
    # [协议修复 P0-1.1] 数据采样种子与训练随机性种子（--seed）解耦：
    # --seed 只控制训练随机性；元训练数据池采样用 --data_seed
    # [协议修复 P0-1.2 裁决] --data_seed 随 run 变（多种子复跑时传 --data_seed $seed），
    # 使 run 间方差包含数据采样差异、std 不低估（数据独立性，修复"虚假独立性"）
    parser.add_argument("--data_seed", type=int, default=42)
    # [协议修复 P0-1.2 裁决] 评估数据采样种子：固定 42，跨 run/跨实验同一把尺子；
    # 与 --data_seed（元训练数据池，随 run 变）解耦——三种子体系见 protocol_fix_notes.md §4
    parser.add_argument("--eval_seed", type=int, default=42)
    # [协议修复 P0-1.5] 元训练模型集合参数化（原硬编码 4 个模型与论文 3m 口径不符）；
    # 默认 3 个 ResNet 与论文一致，逗号分隔可扩展（如 4m/4md 补训无需改代码）
    parser.add_argument("--train_models", type=str, default="resnet18,resnet34,resnet50")
    # [协议修复 P0-1.1] 备选方案：imagenet-mini 无 train/ 目录时，从 val 内按不重合切分
    # 产生训练/评估数据（训练池 seed=100、评估 seed=42 且排除训练池索引）；见 utils/data.py
    parser.add_argument("--fallback_val_holdout", action="store_true")
    return parser.parse_args()


def meta_train_step(
    uap_init, target_model,
    support_images, support_labels,
    query_images, query_labels,
    inner_steps, inner_lr, epsilon, attack_loss_fn,
):
    """手写可微分内循环（与 CIFAR 版相同）"""
    uap = uap_init

    for step in range(inner_steps):
        clamped_uap = clamp_uap(uap, epsilon)
        adv_images = (support_images + clamped_uap).clamp(0, 1)
        logits = target_model(adv_images)
        loss = attack_loss_fn(logits, support_labels)

        grad = torch.autograd.grad(
            outputs=loss, inputs=uap,
            create_graph=True, retain_graph=True,
        )[0]
        uap = uap - inner_lr * grad

    final_uap = clamp_uap(uap, epsilon)
    adv_query = (query_images + final_uap).clamp(0, 1)
    query_logits = target_model(adv_query)
    meta_loss = differentiable_attack_loss(query_logits, query_labels)
    return meta_loss


def meta_evaluate(uap_init, model, images, labels, epsilon, max_steps=10, eval_batch=8):
    """评估 uap_init 在不同步数下的攻击效果（分批处理防 OOM）"""
    results = {}
    n = len(images)

    for step in range(max_steps + 1):
        uap = uap_init.clone().detach()
        lr = 0.1

        for s in range(step):
            # 分批计算梯度，每批算完立即释放计算图
            grad_accum = torch.zeros_like(uap)
            num_batches = 0
            for start in range(0, n, eval_batch):
                end = min(start + eval_batch, n)
                batch_img = images[start:end]
                batch_lbl = labels[start:end]

                uap_grad = uap.detach().clone().requires_grad_(True)
                adv_images = (batch_img + clamp_uap(uap_grad, epsilon)).clamp(0, 1)
                logits = model(adv_images)
                loss = -F.cross_entropy(logits, batch_lbl)
                g = torch.autograd.grad(loss, uap_grad)[0]
                grad_accum += g.detach()
                num_batches += 1
                del loss, logits, adv_images, g, uap_grad

            grad = grad_accum / num_batches
            uap = clamp_uap(uap - lr * grad, epsilon).detach()
            torch.cuda.empty_cache()

        # 分批评估 fooling rate
        uap_eval = clamp_uap(uap, epsilon)
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, n, eval_batch):
                end = min(start + eval_batch, n)
                batch_img = images[start:end]
                batch_lbl = labels[start:end]
                adv_images = (batch_img + uap_eval).clamp(0, 1)
                logits = model(adv_images)
                correct += (logits.argmax(dim=1) != batch_lbl).sum().item()
                total += len(batch_lbl)
        fr = correct / total
        results[step] = fr
        print(f"    step={step:2d}: fooling_rate = {fr:.4f}")
    return results


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 加载预训练模型（留在 CPU，按需搬 GPU） ====================
    # [协议修复 P0-1.5] 元训练模型集合由 --train_models 参数化（默认 3 个，与论文口径一致）
    train_model_names = [m.strip() for m in args.train_models.split(",") if m.strip()]
    test_model_names = ["resnet101", "mobilenet_v2", "shufflenet_v2_x1_0"]

    print("\n加载 ImageNet 预训练模型（元训练集）...")
    train_models = create_all_imagenet_models(train_model_names)
    for name, m in train_models.items():
        print(f"  [加载] {name}")

    print("\n加载 ImageNet 预训练模型（元测试集）...")
    test_models = create_all_imagenet_models(test_model_names)
    for name, m in test_models.items():
        print(f"  [加载] {name}")

    # ==================== 准备数据 ====================
    print("\n准备 ImageNet-mini 数据...")
    # [协议修复 P0-1.1] 防御性检查：train_split 与 eval_split 相同时数据可能重叠，违背协议修复初衷
    if args.train_split == args.eval_split:
        print(f"[警告 P0-1.1] train_split == eval_split == '{args.train_split}'，"
              f"训练与评估数据存在重叠风险，请确认是否为预期设定")
    total_needed = args.support_size + args.query_size
    # [协议修复 P0-1.1] 元训练数据改用 train split（原为 val split 且与评估同 split 同种子）
    images, labels = get_imagenet_mini_samples(
        num_samples=total_needed, split=args.train_split, image_size=224,
        seed=args.data_seed, fallback_val_holdout=args.fallback_val_holdout,
    )
    # [协议修复 P0-1.1] 评估数据改用 eval split，与训练 split 零交集
    # [协议修复 P0-1.2 裁决] 评估数据种子用 --eval_seed（默认固定 42，跨 run/跨实验同一把尺子），
    # 与 --data_seed（元训练数据池，随 run 变）解耦：若 0c 传变化的 --data_seed 后评估采样
    # 也跟着变，各 run 的 200 张评估尺子不同，方向三 ImageNet 判据（4m<3m 且 4md≈3m，
    # 用训练日志 FR 的 mean±std）会混入数据采样噪声，违背"评估集种子永远固定"的裁决
    eval_images, eval_labels = get_imagenet_mini_samples(
        num_samples=200, split=args.eval_split, image_size=224,
        seed=args.eval_seed, fallback_val_holdout=args.fallback_val_holdout,
    )
    print(f"  元训练数据: {len(images)} 张 (split={args.train_split}, data_seed={args.data_seed})")
    print(f"  评估数据:   {len(eval_images)} 张 (split={args.eval_split}, eval_seed={args.eval_seed} 固定)")

    attack_loss_fn = get_attack_loss_fn(args.attack_loss, num_classes=1000)

    # ==================== 初始化 UAP ====================
    uap_init = torch.zeros((3, 224, 224), device=device, requires_grad=True)
    meta_optimizer = torch.optim.Adam([uap_init], lr=args.meta_lr)

    history = {"meta_loss": [], "eval_fooling_rate": {}}

    # ==================== 元训练循环 ====================
    print(f"\n开始元训练: {args.epochs} epochs, {len(train_models)} models/epoch")
    print(f"  inner_steps={args.inner_steps}, inner_lr={args.inner_lr}, epsilon={args.epsilon}")
    print(f"  meta_lr={args.meta_lr}, support_size={args.support_size}, query_size={args.query_size}")
    print("=" * 60)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        meta_optimizer.zero_grad()

        epoch_meta_loss = 0.0
        num_tasks = 0
        model_names = list(train_models.keys())
        np.random.shuffle(model_names)

        for task_idx, model_name in enumerate(model_names):
            target_model = train_models[model_name].to(device)

            sup_img, sup_lbl, qry_img, qry_lbl = sample_meta_batch(
                images, labels,
                support_size=args.support_size,
                query_size=args.query_size,
            )

            meta_loss = meta_train_step(
                uap_init=uap_init,
                target_model=target_model,
                support_images=sup_img,
                support_labels=sup_lbl,
                query_images=qry_img,
                query_labels=qry_lbl,
                inner_steps=args.inner_steps,
                inner_lr=args.inner_lr,
                epsilon=args.epsilon,
                attack_loss_fn=attack_loss_fn,
            )

            meta_loss.backward()
            epoch_meta_loss += meta_loss.item()
            num_tasks += 1

            target_model.cpu()
            torch.cuda.empty_cache()

            if (task_idx + 1) == len(model_names):
                uap_norm = uap_init.detach().abs().max().item()
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Task {task_idx+1}/{len(model_names)} | "
                    f"meta_loss={meta_loss.item():.4f} | "
                    f"uap_norm={uap_norm:.4f}"
                )

        torch.nn.utils.clip_grad_norm_([uap_init], max_norm=1.0)
        meta_optimizer.step()

        avg_loss = epoch_meta_loss / num_tasks
        history["meta_loss"].append(avg_loss)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{args.epochs} 完成 | "
            f"avg_meta_loss={avg_loss:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # 定期评估
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            print(f"\n  ====== 评估 Epoch {epoch+1} ======")

            uap_eval_init = uap_init.detach().clone()

            for name in model_names[:2]:
                print(f"  [训练模型] {name}:")
                m_gpu = train_models[name].to(device)
                meta_evaluate(uap_eval_init, m_gpu, eval_images, eval_labels,
                              epsilon=args.epsilon, max_steps=5)
                m_gpu.cpu()
                torch.cuda.empty_cache()

            for name, model in test_models.items():
                print(f"  [测试模型] {name} (未见过):")
                model_gpu = model.to(device)
                results = meta_evaluate(uap_eval_init, model_gpu, eval_images, eval_labels,
                                        epsilon=args.epsilon, max_steps=5)
                history["eval_fooling_rate"][f"{name}_step0"] = results[0]
                model.cpu()
                torch.cuda.empty_cache()

            del uap_eval_init
            torch.cuda.empty_cache()

            print()

    # ==================== 保存模型 ====================
    # [协议修复 P0-1.5] 文件名不含模型集合信息，保持不变（调用方以 --save_dir 区分 3m/4m/4md）；
    # checkpoint 内附带训练配置，便于超算批次溯源（下游仅读取 "uap_init" 键，不受影响）
    save_path = save_dir / f"meta_uap_imagenet_epoch{args.epochs}.pt"
    torch.save({"uap_init": uap_init.detach().cpu(), "history": history,
                "train_config": vars(args)}, save_path)
    print(f"\nUAP 初始化已保存: {save_path}")

    # ==================== 最终评估报告 ====================
    print("\n" + "=" * 60)
    print("最终评估报告 (ImageNet)")
    print("=" * 60)

    for name, model in test_models.items():
        print(f"\n模型: {name} (元测试，从未在训练中出现)")
        print("-" * 40)
        model_gpu = model.to(device)
        results = meta_evaluate(uap_init, model_gpu, eval_images, eval_labels,
                                epsilon=args.epsilon, max_steps=10)
        model.cpu()
        torch.cuda.empty_cache()
        print(f"  0步(直接使用uap_init): {results[0]:.4f}")
        if results[0] > 0.25:
            print(f"  ✓ 单步攻击成功率 > 25%，方案可行！")
        else:
            print(f"  ✗ 单步攻击成功率偏低，需要调参")

    return uap_init, history


if __name__ == "__main__":
    args = parse_args()
    train(args)
