"""
数据加载：CIFAR-10 / ImageNet 子集
"""
import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple

# 默认数据路径，可通过环境变量覆盖
DATA_ROOT = os.environ.get("DATA_ROOT", "/data/public/cifar")


def get_cifar10_loaders(
    data_root: str = None,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """获取 CIFAR-10 的训练集和测试集 DataLoader"""
    if data_root is None:
        data_root = DATA_ROOT
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def get_cifar10_samples(
    data_root: str = None,
    num_samples: int = 1000,
    split: str = "test",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取固定数量的 CIFAR-10 样本（用于 UAP 生成和评估）"""
    if data_root is None:
        data_root = DATA_ROOT
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=(split == "train"), download=False, transform=transform
    )

    # 固定种子采样，保证可复现
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)

    # 加载到内存
    all_images = []
    all_labels = []
    for img, label in subset:
        all_images.append(img)
        all_labels.append(label)

    images = torch.stack(all_images).cuda()  # (N, C, H, W)
    labels = torch.tensor(all_labels, dtype=torch.long).cuda()
    return images, labels


# [协议修复 P0-1.1] fallback 模式下训练池大小（固定常数，与 num_samples 无关，
# 保证训练请求/评估请求两侧对"训练池"的定义完全一致，从而实现不重合切分）
_VAL_HOLDOUT_TRAIN_POOL = 2000


def get_imagenet_mini_samples(
    data_root: str = "/data/public/imagenet-mini",
    num_samples: int = 500,
    split: str = "val",
    image_size: int = 224,
    seed: int = 42,
    fallback_val_holdout: bool = False,
):
    """获取 ImageNet-mini 样本（用于 UAP 元训练和评估）

    [协议修复 P0-1.1] fallback_val_holdout（默认 False，新参数不影响既有调用方）：
    当 split="train" 目录不存在且该开关开启时，从 val 内按不重合切分产生训练数据——
    训练池用固定种子 100 采样（大小 _VAL_HOLDOUT_TRAIN_POOL），评估请求（split="val"）
    排除训练池索引后按调用方 seed 采样。注意：该模式下评估集与既有纯 val 口径不可比，
    仅作环境缺失 train/ 目录时的降级方案。
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # [0, 1] 范围，不归一化（归一化在模型内处理）
    ])

    # [协议修复 P0-1.1] fallback 实际生效条件：开关开启且 train 目录不存在。
    # 生效时训练请求降级到 val 内切分，评估请求（split="val"）排除训练池索引；
    # train 目录存在时本开关完全不生效（评估集不受影响），保证正常环境行为不变。
    fallback_active = fallback_val_holdout and not os.path.isdir(f"{data_root}/train")
    if fallback_active:
        print(f"[P0-1.1 fallback] {data_root}/train 不存在，从 val 内不重合切分产生数据"
              f"（训练池 seed=100，评估排除训练池索引）")
    effective_split = "val" if (fallback_active and split == "train") else split

    dataset = torchvision.datasets.ImageFolder(
        root=f"{data_root}/{effective_split}",
        transform=transform,
    )

    if fallback_active:
        # 训练池定义：固定种子 100、固定大小，训练/评估两侧调用完全一致
        pool_rng = np.random.RandomState(100)
        pool = set(pool_rng.choice(
            len(dataset), min(_VAL_HOLDOUT_TRAIN_POOL, len(dataset)), replace=False
        ).tolist())
        if split == "train":
            # 训练请求：取训练池前 num_samples 个（不同 num_samples 的训练调用天然嵌套，均为训练侧）
            indices = sorted(pool)[:num_samples]
        else:
            # 评估请求：从 val 排除训练池后的候选集中按调用方 seed 采样
            candidates = np.array(sorted(set(range(len(dataset))) - pool))
            rng = np.random.RandomState(seed)
            indices = rng.choice(candidates, min(num_samples, len(candidates)), replace=False)
    else:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)

    all_images = []
    all_labels = []
    for img, label in subset:
        all_images.append(img)
        all_labels.append(label)

    images = torch.stack(all_images).cuda()
    labels = torch.tensor(all_labels, dtype=torch.long).cuda()
    return images, labels


def sample_meta_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    support_size: int = 500,
    query_size: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从数据中采样 meta-batch
    返回: (support_images, support_labels, query_images, query_labels)
    """
    n = min(len(images), support_size + query_size)
    indices = torch.randperm(n)
    support_idx = indices[:support_size]
    query_idx = indices[support_size : support_size + query_size]

    return (
        images[support_idx],
        labels[support_idx],
        images[query_idx],
        labels[query_idx],
    )
