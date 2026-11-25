"""
Idea 2: Meta-Learning Single-Step Universal Adversarial Perturbation
配置文件 — 所有超参数集中管理
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    # ==================== 数据 ====================
    dataset: str = "cifar10"          # cifar10 / imagenet-subset
    # [协议修复 P0-1.1] data_root 改读 DATA_ROOT 环境变量（默认 /data/public/cifar），
    # 与 utils/data.py 的 DATA_ROOT 及 download_data.sh 统一；显式传参仍可覆盖
    data_root: str = os.environ.get("DATA_ROOT", "/data/public/cifar")
    num_classes: int = 10
    image_size: int = 32             # 32 for CIFAR-10, 224 for ImageNet
    support_size: int = 500           # 每个 meta-task 的 support set 大小
    query_size: int = 200             # 每个 meta-task 的 query set 大小

    # ==================== UAP 参数 ====================
    uap_epsilon: float = 0.05        # UAP 的 L_inf 约束（占像素值范围的比例）
    # CIFAR-10: epsilon = 0.05 * 255 ≈ 12.75
    # ImageNet: epsilon = 0.05 * 255 ≈ 12.75

    # ==================== 元训练 ====================
    meta_epochs: int = 50
    meta_lr: float = 0.01             # 元优化器学习率
    inner_steps: int = 5              # 内循环步数（推荐 5）
    inner_lr: float = 0.1            # 内循环学习率
    inner_optimizer: str = "sgd"     # sgd / adam

    # ==================== 攻击配置 ====================
    attack_loss_type: str = "ce"     # ce (交叉熵) / cw (CW损失)
    cw_confidence: float = 0.0       # CW 损失的 kappa 参数

    # ==================== 元训练模型集合 ====================
    # CIFAR-10: 用 ResNet 变体作为元训练任务
    meta_train_models: List[str] = field(default_factory=lambda: [
        "resnet18", "resnet34", "resnet50",
        "vgg16_bn", "densenet121",
    ])
    # 元测试（验证/泛化）：不在训练集合中出现的模型
    meta_test_models: List[str] = field(default_factory=lambda: [
        "resnet101", "mobilenet_v2", "shufflenet_v2_x1_0",
    ])

    # ==================== 训练配置 ====================
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"              # cuda / cpu

    # ==================== 日志与保存 ====================
    save_dir: str = "./checkpoints"
    log_interval: int = 10            # 每多少个 meta-task 打印日志
    eval_interval: int = 5            # 每多少个 meta_epoch 做一次评估

    # ==================== 评估配置 ====================
    eval_steps: List[int] = field(default_factory=lambda: [0, 1, 3, 5, 10, 50])
    # eval_steps[0]=0 表示直接用 uap_init 不做任何更新

    def __post_init__(self):
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_epsilon_abs(self) -> float:
        """返回绝对 epsilon 值（基于 [0,1] 归一化后的范围）"""
        return self.uap_epsilon

    def get_num_meta_tasks(self) -> int:
        return len(self.meta_train_models)
