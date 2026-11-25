"""
模型工厂：创建不同架构的分类模型
所有模型在 CIFAR-10 或 ImageNet 子集上训练
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Type


def modify_for_cifar(model: nn.Module, num_classes: int = 10) -> nn.Module:
    """
    修改 ImageNet 模型以适应 CIFAR-10（32x32 输入）
    仅适用于 ResNet 系列：改 conv1 stride + 替换分类头
    """
    # ResNet: 7x7 stride=2 → 3x3 stride=1
    if hasattr(model, "conv1") and hasattr(model, "fc"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 替换最后的全连接层
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def modify_vgg_for_cifar(model: nn.Module, num_classes: int = 10) -> nn.Module:
    """修改 VGG 以适应 CIFAR-10：加自适应池化防止特征图缩到 0"""
    if hasattr(model, "classifier"):
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    return model


def modify_classifier_for_cifar(model: nn.Module, num_classes: int = 10) -> nn.Module:
    """
    通用适配：仅替换分类头，不修改特征提取部分
    适用于 MobileNetV2, DenseNet, ShuffleNetV2 等自带池化的模型
    """
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            # MobileNetV2: Sequential(Dropout, Linear)
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                model.classifier[-1] = nn.Linear(last.in_features, num_classes)
            else:
                # DenseNet: Sequential(ReLU, AvgPool, Dropout, Linear) — 替换最后一个 Linear
                for i, layer in enumerate(model.classifier):
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        model.classifier[i] = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def create_model(name: str, num_classes: int = 10, image_size: int = 32) -> nn.Module:
    """
    创建指定名称的模型

    Args:
        name: 模型名称（如 "resnet18", "vgg16_bn"）
        num_classes: 分类数
        image_size: 输入图像尺寸
    """
    # 模型名称映射
    model_fn_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "vgg16_bn": models.vgg16_bn,
        "densenet121": models.densenet121,
        "mobilenet_v2": models.mobilenet_v2,
        "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
    }

    if name not in model_fn_map:
        raise ValueError(f"未知模型: {name}，可选: {list(model_fn_map.keys())}")

    model = model_fn_map[name](weights=None)

    # CIFAR-10 适配（32x32）
    if image_size == 32:
        if "vgg" in name:
            model = modify_vgg_for_cifar(model, num_classes)
        elif "resnet" in name:
            model = modify_for_cifar(model, num_classes)
        else:
            # MobileNetV2, DenseNet121, ShuffleNetV2: 只替换分类头
            model = modify_classifier_for_cifar(model, num_classes)
    else:
        # ImageNet 尺寸（224x224）：只需替换分类头
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                last = model.classifier[-1]
                if isinstance(last, nn.Linear):
                    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
            else:
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model


def create_all_models(
    model_names: list,
    num_classes: int = 10,
    image_size: int = 32,
) -> Dict[str, nn.Module]:
    """创建所有模型并返回字典"""
    model_dict = {}
    for name in model_names:
        model_dict[name] = create_model(name, num_classes, image_size)
    return model_dict


# ==================== ImageNet 预训练模型 ====================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizedModel(nn.Module):
    """包装模型，内部做 ImageNet 归一化。UAP 在 [0,1] 空间操作。"""

    def __init__(self, model: nn.Module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def create_imagenet_model(name: str) -> nn.Module:
    """创建 ImageNet 预训练模型（1000 类），含归一化包装"""
    model_fn_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "vgg16_bn": models.vgg16_bn,
        "densenet121": models.densenet121,
        "mobilenet_v2": models.mobilenet_v2,
        "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
        "vit_b_16": models.vit_b_16,
        # P2 扩容的现代评估目标（2026-10-31）
        "convnext_tiny": models.convnext_tiny,
        "swin_t": models.swin_t,
        "efficientnet_b0": models.efficientnet_b0,
    }

    if name not in model_fn_map:
        raise ValueError(f"未知模型: {name}")

    # 加载 ImageNet 预训练权重（需要先在登录节点下载缓存）
    model = model_fn_map[name](weights="IMAGENET1K_V1")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 包装归一化层，UAP 在 [0,1] 空间操作
    return NormalizedModel(model)


def create_all_imagenet_models(model_names: list) -> Dict[str, nn.Module]:
    """创建所有 ImageNet 预训练模型"""
    return {name: create_imagenet_model(name) for name in model_names}
