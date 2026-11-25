"""
Idea 2: Meta-Learning Single-Step Universal Adversarial Perturbation
元训练主脚本

核心思想：
- 元训练阶段：在多个模型上学习 UAP 的最优初始化 (uap_init)
- 元测试阶段：面对新模型时，仅需 0-1 步即可生成有效 UAP
- 手写可微分内循环（torch.autograd.grad + create_graph）

使用方法:
    python train.py --epochs 50 --inner_steps 5 --meta_lr 0.01
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from config import Config
from models import create_all_models
from utils.data import get_cifar10_loaders, get_cifar10_samples, sample_meta_batch
from utils.losses import get_attack_loss_fn, clamp_uap, attack_success_rate, differentiable_attack_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-UAP Training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--meta_lr", type=float, default=0.01)
    parser.add_argument("--inner_lr", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--attack_loss", type=str, default="ce", choices=["ce", "cw"])
    parser.add_argument("--support_size", type=int, default=256)
    parser.add_argument("--query_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_models", type=str, default=None,
                        help="Comma-separated model names to override config.meta_train_models")
    return parser.parse_args()


def load_pretrained_models(model_names, num_classes, image_size, checkpoint_dir):
    """
    加载预训练模型权重
    如果没有预训练权重或 shape 不匹配，则使用随机初始化并打印警告
    """
    models = create_all_models(model_names, num_classes, image_size)
    for name, model in models.items():
        ckpt_path = Path(checkpoint_dir) / f"{name}.pt"
        if ckpt_path.exists():
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                print(f"  [加载] {name}: {ckpt_path}")
            except RuntimeError as e:
                print(f"  [警告] {name}: 权重 shape 不匹配，使用随机初始化")
        else:
            print(f"  [警告] {name}: 未找到预训练权重 ({ckpt_path})，使用随机初始化")
    return models


def meta_train_step(
    uap_init: torch.Tensor,
    target_model: nn.Module,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    inner_steps: int,
    inner_lr: float,
    epsilon: float,
    attack_loss_fn,
    inner_optimizer: str = "sgd",
    track_higher_grads: bool = True,
):
    """
    单个 meta-task 的训练步骤（手写可微分内循环）

    不依赖 higher 库，直接用 torch.autograd.grad(create_graph=True) 实现可微分更新。
    内循环：uap 从 uap_init 出发，多步 SGD 优化
    外循环：meta_loss.backward() 梯度回传到 uap_init
    """
    # 从 uap_init 出发（保留梯度连接）
    uap = uap_init

    # ========== Inner Loop: 在 support set 上可微分优化 UAP ==========
    for step in range(inner_steps):
        clamped_uap = clamp_uap(uap, epsilon)
        adv_images = (support_images + clamped_uap).clamp(0, 1)
        logits = target_model(adv_images)
        loss = attack_loss_fn(logits, support_labels)

        # 计算 uap 的梯度，保留计算图（create_graph=True）
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=uap,
            create_graph=track_higher_grads,
            retain_graph=True,
        )[0]

        # SGD 更新（非原地，保留梯度链）
        uap = uap - inner_lr * grad

    # ========== Query: 在 query set 上评估优化后的 UAP ==========
    final_uap = clamp_uap(uap, epsilon)
    adv_query = (query_images + final_uap).clamp(0, 1)
    query_logits = target_model(adv_query)
    meta_loss = differentiable_attack_loss(query_logits, query_labels)

    return meta_loss


def meta_evaluate(
    uap_init: torch.Tensor,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    max_steps: int = 10,
):
    """
    在给定模型上评估 uap_init 在不同步数下的攻击效果
    """
    results = {}

    for step in range(max_steps + 1):
        uap = uap_init.clone().detach()
        lr = 0.1

        # 做 step 步优化
        for s in range(step):
            uap = uap.requires_grad_(True)
            adv_images = (images + clamp_uap(uap, epsilon)).clamp(0, 1)
            logits = model(adv_images)
            loss = -F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, uap)[0]
            uap = clamp_uap(uap - lr * grad, epsilon).detach()

        # 评估当前步数下的攻击效果
        uap_eval = clamp_uap(uap, epsilon)
        adv_images = (images + uap_eval).clamp(0, 1)
        with torch.no_grad():
            logits = model(adv_images)
            fr = attack_success_rate(logits, labels).item()

        results[step] = fr
        print(f"    step={step:2d}: fooling_rate = {fr:.4f}")

    return results


def train(config: Config):
    """元训练主函数"""

    # ==================== 设置随机种子 ====================
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ==================== 加载预训练模型 ====================
    train_model_names = config.meta_train_models

    print("\n加载元训练模型集合...")
    train_models = load_pretrained_models(
        train_model_names, config.num_classes, config.image_size, config.save_dir
    )
    train_models = {name: m.eval() for name, m in train_models.items()}
    for m in train_models.values():
        for p in m.parameters():
            p.requires_grad_(False)

    # 加载测试模型（留在 CPU，评估时临时搬 GPU，节省显存）
    print("\n加载元测试模型集合...")
    test_models = load_pretrained_models(
        config.meta_test_models, config.num_classes, config.image_size, config.save_dir
    )
    for m in test_models.values():
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # ==================== 准备数据 ====================
    print("\n准备数据...")
    images, labels = get_cifar10_samples(
        config.data_root,
        num_samples=config.support_size + config.query_size,
        split="test",
    )

    # 评估数据（独立的一批）
    eval_images, eval_labels = get_cifar10_samples(
        config.data_root,
        num_samples=1000,
        split="test",
    )

    # 攻击损失函数
    attack_loss_fn = get_attack_loss_fn(
        config.attack_loss_type,
        cw_confidence=config.cw_confidence,
        num_classes=config.num_classes,
    )

    # ==================== 初始化 UAP ====================
    image_shape = (3, config.image_size, config.image_size)
    uap_init = torch.zeros(image_shape, device=device, requires_grad=True)

    # 元优化器
    meta_optimizer = torch.optim.Adam([uap_init], lr=config.meta_lr)

    # 训练记录
    history = {"meta_loss": [], "eval_fooling_rate": {}}

    # ==================== 元训练循环 ====================
    print(f"\n开始元训练: {config.meta_epochs} epochs, {len(train_models)} models/epoch")
    print(f"  inner_steps={config.inner_steps}, inner_lr={config.inner_lr}, epsilon={config.uap_epsilon}")
    print(f"  meta_lr={config.meta_lr}, support_size={config.support_size}, query_size={config.query_size}")
    print("=" * 60)

    for epoch in range(config.meta_epochs):
        epoch_start = time.time()
        meta_optimizer.zero_grad()

        epoch_meta_loss = 0.0
        num_tasks = 0

        # 遍历每个 meta-task（每个模型 = 一个任务）
        model_names = list(train_models.keys())
        np.random.shuffle(model_names)

        for task_idx, model_name in enumerate(model_names):
            target_model = train_models[model_name].to(device)

            # 采样 meta-batch
            sup_img, sup_lbl, qry_img, qry_lbl = sample_meta_batch(
                images, labels,
                support_size=config.support_size,
                query_size=config.query_size,
            )

            # 元训练步骤
            meta_loss = meta_train_step(
                uap_init=uap_init,
                target_model=target_model,
                support_images=sup_img,
                support_labels=sup_lbl,
                query_images=qry_img,
                query_labels=qry_lbl,
                inner_steps=config.inner_steps,
                inner_lr=config.inner_lr,
                epsilon=config.uap_epsilon,
                attack_loss_fn=attack_loss_fn,
                inner_optimizer=config.inner_optimizer,
                track_higher_grads=True,
            )

            # 累积梯度（跨多个 meta-task）
            meta_loss.backward()
            epoch_meta_loss += meta_loss.item()
            num_tasks += 1

            # 释放模型显存 + 清空计算图缓存
            target_model.cpu()
            torch.cuda.empty_cache()

            # 打印日志
            if (task_idx + 1) % config.log_interval == 0 or (task_idx + 1) == len(model_names):
                uap_norm = uap_init.detach().abs().max().item()
                print(
                    f"  Epoch {epoch+1}/{config.meta_epochs} | "
                    f"Task {task_idx+1}/{len(model_names)} | "
                    f"meta_loss={meta_loss.item():.4f} | "
                    f"uap_norm={uap_norm:.4f}"
                )

        # 梯度裁剪（防止二阶梯度爆炸）
        torch.nn.utils.clip_grad_norm_([uap_init], max_norm=1.0)

        # 元更新
        meta_optimizer.step()

        avg_loss = epoch_meta_loss / num_tasks
        history["meta_loss"].append(avg_loss)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{config.meta_epochs} 完成 | "
            f"avg_meta_loss={avg_loss:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # 定期评估
        if (epoch + 1) % config.eval_interval == 0 or (epoch + 1) == config.meta_epochs:
            print(f"\n  ====== 评估 Epoch {epoch+1} ======")

            # 在训练模型上评估（临时搬 GPU）
            for name in model_names[:2]:
                print(f"  [训练模型] {name}:")
                m_gpu = train_models[name].to(device)
                meta_evaluate(
                    uap_init, m_gpu,
                    eval_images, eval_labels,
                    epsilon=config.uap_epsilon,
                    max_steps=5,
                )
                m_gpu.cpu()
                torch.cuda.empty_cache()

            # 在测试模型上评估（临时搬 GPU，用完释放）
            for name, model in test_models.items():
                print(f"  [测试模型] {name} (未见过):")
                model_gpu = model.to(device)
                results = meta_evaluate(
                    uap_init, model_gpu,
                    eval_images, eval_labels,
                    epsilon=config.uap_epsilon,
                    max_steps=5,
                )
                history["eval_fooling_rate"][f"{name}_step0"] = results[0]
                model.cpu()  # 释放显存
                torch.cuda.empty_cache()

            print()

    # ==================== 保存模型 ====================
    save_path = config.save_dir / f"meta_uap_epoch{config.meta_epochs}.pt"
    torch.save({
        "uap_init": uap_init.detach().cpu(),
        "config": config,
        "history": history,
    }, save_path)
    print(f"\nUAP 初始化已保存: {save_path}")

    # ==================== 最终评估报告 ====================
    print("\n" + "=" * 60)
    print("最终评估报告")
    print("=" * 60)

    for name, model in test_models.items():
        print(f"\n模型: {name} (元测试，从未在训练中出现)")
        print("-" * 40)
        model_gpu = model.to(device)
        results = meta_evaluate(
            uap_init, model_gpu,
            eval_images, eval_labels,
            epsilon=config.uap_epsilon,
            max_steps=10,
        )
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

    config = Config(
        meta_epochs=args.epochs,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        uap_epsilon=args.epsilon,
        batch_size=args.batch_size,
        device=args.device,
        save_dir=args.save_dir,
        eval_interval=args.eval_interval,
        attack_loss_type=args.attack_loss,
        support_size=args.support_size,
        query_size=args.query_size,
        seed=args.seed,
    )

    if args.train_models:
        config.meta_train_models = [m.strip() for m in args.train_models.split(",")]
        print(f"使用自定义训练模型: {config.meta_train_models}")

    uap_init, history = train(config)
