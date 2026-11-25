"""
攻击损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_attack_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    攻击损失（最小化此损失 = 攻击）
    取负号：最小化 -CE = 最大化 CE = 让模型分错
    """
    return -F.cross_entropy(logits, labels)


def cw_attack_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confidence: float = 0.0,
    num_classes: int = 10,
) -> torch.Tensor:
    """
    Carlini-Wagner 风格攻击损失
    目标：让 logits 在非正确类别上的最大值比正确类别大至少 kappa
    """
    # one-hot 正确标签
    one_hot = F.one_hot(labels, num_classes).float()

    # 正确类别的 logit
    correct_logits = (logits * one_hot).sum(dim=1)
    # 非正确类别的最大 logit
    wrong_logits = (logits * (1 - one_hot) - 1e4 * one_hot).max(dim=1)[0]

    # 最大化 wrong - correct - kappa
    loss = F.relu(wrong_logits - correct_logits + confidence)
    return loss.mean()


def get_attack_loss_fn(loss_type: str, **kwargs):
    """获取攻击损失函数"""
    if loss_type == "ce":
        return cross_entropy_attack_loss
    elif loss_type == "cw":
        return lambda logits, labels: cw_attack_loss(
            logits, labels,
            confidence=kwargs.get("cw_confidence", 0.0),
            num_classes=kwargs.get("num_classes", 10),
        )
    else:
        raise ValueError(f"未知损失类型: {loss_type}")


def clamp_uap(uap: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    将 UAP 投影到 L_inf 约束球内
    uap ∈ [-epsilon, epsilon]
    """
    return uap.clamp(-epsilon, epsilon)


def attack_success_rate(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算攻击成功率（不可微分，仅用于评估）
    返回 [0, 1] 之间的标量
    """
    predicted = logits.argmax(dim=1)
    return (predicted != labels).float().mean()


def differentiable_attack_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    可微分的攻击损失（用于 meta_loss）
    取负号：最小化 -CE = 最大化 CE = 让模型分错
    """
    return -F.cross_entropy(logits, labels)
