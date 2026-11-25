"""
Idea 2: 两周试点脚本 — 快速验证可行性

试点目标：
- 在 CIFAR-10 上用 3 个 ResNet 变体进行 MAML 元训练
- 在留出的 ResNet-101 上测试 0 步和 1 步 UAP 的 fooling rate
- 判断标准：单步 fooling rate > 30% 即方案可行

使用方法:
    python pilot.py
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import higher
except ImportError:
    print("请先安装: pip install higher")
    exit(1)

from config import Config
from models import create_all_models, create_model
from utils.data import get_cifar10_samples, sample_meta_batch
from utils.losses import get_attack_loss_fn, clamp_uap, attack_success_rate, differentiable_attack_loss


def pilot_study():
    """两周试点实验"""

    print("=" * 60)
    print("Idea 2 试点实验: 元学习单步 UAP")
    print("目标: 验证 0 步 UAP fooling rate > 30%")
    print("=" * 60)

    # ==================== 简化配置 ====================
    config = Config(
        meta_epochs=20,          # 试点只用 20 个 epoch
        inner_steps=5,          # 内循环 5 步
        inner_lr=0.1,
        meta_lr=0.01,
        uap_epsilon=0.05,
        support_size=300,       # 试点用小一点的 support
        query_size=100,
        eval_interval=5,
        attack_loss_type="ce",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    # ==================== 模型：3 个训练 + 1 个测试 ====================
    print("\n创建模型...")
    train_model_names = ["resnet18", "resnet34", "resnet50"]
    test_model_name = "resnet101"

    # 加载预训练权重（如果有的话）
    train_models = {}
    for name in train_model_names:
        model = create_model(name, config.num_classes, config.image_size).to(device).eval()
        ckpt_path = config.save_dir / f"{name}.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"  ✓ {name}: 加载预训练权重")
        else:
            print(f"  ✗ {name}: 无预训练权重，使用随机初始化（结果仅供参考）")
        for p in model.parameters():
            p.requires_grad_(False)
        train_models[name] = model

    test_model = create_model(test_model_name, config.num_classes, config.image_size).to(device).eval()
    test_ckpt = config.save_dir / f"{test_model_name}.pt"
    if test_ckpt.exists():
        test_model.load_state_dict(torch.load(test_ckpt, map_location=device))
        print(f"  ✓ {test_model_name}: 加载预训练权重")
    else:
        print(f"  ✗ {test_model_name}: 无预训练权重")
    for p in test_model.parameters():
        p.requires_grad_(False)

    # ==================== 数据 ====================
    print("\n准备数据...")
    images, labels = get_cifar10_samples(config.data_root, num_samples=500, split="test")
    eval_images, eval_labels = get_cifar10_samples(config.data_root, num_samples=1000, split="test")

    attack_loss_fn = get_attack_loss_fn(config.attack_loss_type, num_classes=config.num_classes)

    # ==================== 先跑基线：普通 UAP 优化（对比用） ====================
    print("\n" + "=" * 60)
    print("基线: 在 ResNet-50 上普通 PGD 优化 UAP（50 步）")
    print("=" * 60)

    baseline_uap = torch.zeros(3, 32, 32, device=device, requires_grad=True)
    baseline_opt = torch.optim.Adam([baseline_uap], lr=0.01)

    sup_img, sup_lbl, _, _ = sample_meta_batch(images, labels, 400, 0)

    for step in range(50):
        baseline_opt.zero_grad()
        clamped = clamp_uap(baseline_uap, config.uap_epsilon)
        adv = (sup_img + clamped).clamp(0, 1)
        logits = train_models["resnet50"](adv)
        loss = attack_loss_fn(logits, sup_lbl)
        loss.backward()
        baseline_opt.step()

        if (step + 1) % 10 == 0:
            with torch.no_grad():
                eval_adv = (eval_images + clamp_uap(baseline_uap, config.uap_epsilon)).clamp(0, 1)
                eval_logits = train_models["resnet50"](eval_adv)
                fr = attack_success_rate(eval_logits, eval_labels).item()
            print(f"  step {step+1}: loss={loss.item():.4f}, fooling_rate(RN50)={fr:.4f}")

    # 在测试模型上评估基线 UAP
    with torch.no_grad():
        baseline_uap_clamped = clamp_uap(baseline_uap.detach(), config.uap_epsilon)
        # ResNet-50 上的效果
        adv50 = (eval_images + baseline_uap_clamped).clamp(0, 1)
        fr_baseline_rn50 = attack_success_rate(train_models["resnet50"](adv50), eval_labels).item()
        # ResNet-101 上的迁移效果
        adv101 = (eval_images + baseline_uap_clamped).clamp(0, 1)
        fr_baseline_rn101 = attack_success_rate(test_model(adv101), eval_labels).item()

    print(f"\n基线结果（50步PGD优化）:")
    print(f"  ResNet-50 (训练模型): {fr_baseline_rn50:.4f}")
    print(f"  ResNet-101 (未见过):  {fr_baseline_rn101:.4f}")

    # ==================== 元训练 ====================
    print("\n" + "=" * 60)
    print("元训练: 学习跨模型通用的 UAP 初始化")
    print("=" * 60)

    uap_init = torch.zeros(3, 32, 32, device=device, requires_grad=True)
    meta_optimizer = torch.optim.Adam([uap_init], lr=config.meta_lr)

    # 导入元训练步骤
    from train import meta_train_step

    for epoch in range(config.meta_epochs):
        meta_optimizer.zero_grad()
        epoch_loss = 0.0

        for model_name in train_model_names:
            sup_img, sup_lbl, qry_img, qry_lbl = sample_meta_batch(
                images, labels,
                support_size=config.support_size,
                query_size=config.query_size,
            )

            meta_loss = meta_train_step(
                uap_init=uap_init,
                target_model=train_models[model_name],
                support_images=sup_img,
                support_labels=sup_lbl,
                query_images=qry_img,
                query_labels=qry_lbl,
                inner_steps=config.inner_steps,
                inner_lr=config.inner_lr,
                epsilon=config.uap_epsilon,
                attack_loss_fn=attack_loss_fn,
            )

            meta_loss.backward()
            epoch_loss += meta_loss.item()

        torch.nn.utils.clip_grad_norm_([uap_init], max_norm=1.0)
        meta_optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            uap_norm = uap_init.detach().abs().max().item()
            print(f"  Epoch {epoch+1:2d}/{config.meta_epochs}: "
                  f"avg_meta_loss={epoch_loss/len(train_model_names):.4f}, "
                  f"uap_norm={uap_norm:.4f}")

    # ==================== 试点评估 ====================
    print("\n" + "=" * 60)
    print("试点评估: 在 ResNet-101（从未见过）上测试")
    print("=" * 60)

    # 0 步：直接用 uap_init
    with torch.no_grad():
        uap_clamped = clamp_uap(uap_init.detach(), config.uap_epsilon)
        adv_0step = (eval_images + uap_clamped).clamp(0, 1)
        logits_0step = test_model(adv_0step)
        fr_0step = attack_success_rate(logits_0step, eval_labels).item()

    # 1 步：从 uap_init 做一步优化
    uap_1step = uap_init.detach().clone().requires_grad_(True)
    adv_1step = (eval_images + clamp_uap(uap_1step, config.uap_epsilon)).clamp(0, 1)
    logits_1step = test_model(adv_1step)
    loss_1step = -F.cross_entropy(logits_1step, eval_labels)
    grad_1step = torch.autograd.grad(loss_1step, uap_1step)[0]
    uap_1step = clamp_uap(uap_1step - 0.1 * grad_1step, config.uap_epsilon).detach()

    with torch.no_grad():
        adv_1step_final = (eval_images + uap_1step).clamp(0, 1)
        fr_1step = attack_success_rate(test_model(adv_1step_final), eval_labels).item()

    # 5 步
    uap_5step = uap_init.detach().clone()
    lr = 0.1
    for s in range(5):
        uap_5step = uap_5step.requires_grad_(True)
        adv = (eval_images + clamp_uap(uap_5step, config.uap_epsilon)).clamp(0, 1)
        logits = test_model(adv)
        loss = -F.cross_entropy(logits, eval_labels)
        grad = torch.autograd.grad(loss, uap_5step)[0]
        uap_5step = clamp_uap(uap_5step - lr * grad, config.uap_epsilon).detach()

    with torch.no_grad():
        adv_5step_final = (eval_images + clamp_uap(uap_5step, config.uap_epsilon)).clamp(0, 1)
        fr_5step = attack_success_rate(test_model(adv_5step_final), eval_labels).item()

    print(f"\n元学习 UAP 在 ResNet-101 上的效果:")
    print(f"  0 步 (直接使用): {fr_0step:.4f}")
    print(f"  1 步优化后:      {fr_1step:.4f}")
    print(f"  5 步优化后:      {fr_5step:.4f}")

    print(f"\n基线 UAP 在 ResNet-101 上的效果:")
    print(f"  50步PGD (RN50):  {fr_baseline_rn101:.4f}")

    # ==================== 判断标准 ====================
    print("\n" + "=" * 60)
    print("试点结论")
    print("=" * 60)

    if fr_0step > 0.30:
        print(f"  ✓ 0步 fooling rate = {fr_0step:.4f} > 30%，方案高度可行！")
        print(f"    → 可直接使用 uap_init，无需任何优化")
    elif fr_1step > 0.30:
        print(f"  ✓ 1步 fooling rate = {fr_1step:.4f} > 30%，方案可行！")
        print(f"    → 仅需 1 步优化即可有效攻击")
    elif fr_5step > 0.30:
        print(f"  △ 5步 fooling rate = {fr_5step:.4f} > 30%，方案部分可行")
        print(f"    → 需要少量优化步数，但远少于基线的 50 步")
    else:
        print(f"  ✗ 所有步数 fooling rate 均 < 30%，需要调整超参数")
        print(f"    → 建议：增大 epsilon、增加 meta_epochs、调整 inner_lr")

    if fr_5step > fr_baseline_rn101:
        improvement = (fr_5step - fr_baseline_rn101) / max(fr_baseline_rn101, 0.01) * 100
        print(f"\n  相比基线 UAP（{fr_baseline_rn101:.4f}），")
        print(f"  元学习 UAP 5步（{fr_5step:.4f}）提升 {improvement:.1f}%")

    # 保存 uap_init
    save_path = config.save_dir / "pilot_uap_init.pt"
    torch.save(uap_init.detach().cpu(), save_path)
    print(f"\n  UAP 初始化已保存: {save_path}")


if __name__ == "__main__":
    pilot_study()
