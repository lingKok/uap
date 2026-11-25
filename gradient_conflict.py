"""
gradient_conflict.py — 元训练内循环梯度余弦冲突矩阵（Wave 1 / 方向三机制证据）

用途:
  对指定 surrogate 集合（3m / 4m / 4md），每个模型在同一批 train split 数据上
  计算 uap_init 的**一步 inner-loop 梯度**，输出两两余弦相似度矩阵（含元测试
  模型列，仅作对照）。

  梯度定义与 train_imagenet.py 内循环第一步完全一致:
      loss = -CE( model_m( x + clamp_uap(uap_init, ε) ), y )
      g_m  = d loss / d uap_init        （梯度穿过 clamp，饱和坐标梯度为 0）
  这与 meta_train_step 中 autograd.grad(loss, uap)（uap 即 uap_init）一致。

  证据强度限定（裁决文件 §4 方向三 GO③）: 本矩阵是方向三的**机制性描述证据**——
  仅当与干预证据（代理集消融 3+VGG / 3+DenseNet）方向一致时方可声称
  "consistent with"，不得单独支撑因果。

用法:
  python gradient_conflict.py --model_subset 3m \
      --data_split train --data_seed 42 --epsilon 0.05 --grad_samples 32 \
      --data_root /data/public/imagenet-mini --out_dir results_wave0

输出: results_wave0/grad_conflict_{3m|4m|4md}.json

依赖:
  各 subset 默认使用**修复协议重训**的 ensemble checkpoint（slurm_wave0c_4m4md.sh
  产出）。缺失时 fail-fast 并提示先提交该作业; 也可用 --checkpoint 覆盖（例如
  指向旧协议 checkpoints_imagenet_3m/...，但会破坏 3m/4m/4md 的协议一致性）。
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

from models import create_imagenet_model
from utils.data import get_imagenet_mini_samples
from utils.losses import clamp_uap


# ==================== 配置 ====================

META_TEST_MODELS = ["resnet101", "mobilenet_v2", "shufflenet_v2_x1_0"]

# subset -> (surrogate 模型列表, 规范 tag, 默认 checkpoint)
SUBSETS = {
    "3": {
        "models": ["resnet18", "resnet34", "resnet50"],
        "tag": "3m",
        "checkpoint": "checkpoints_imagenet_3m_fixed/meta_uap_imagenet_3m_fixed_ensemble3.pt",
    },
    "3m": {
        "models": ["resnet18", "resnet34", "resnet50"],
        "tag": "3m",
        "checkpoint": "checkpoints_imagenet_3m_fixed/meta_uap_imagenet_3m_fixed_ensemble3.pt",
    },
    "4m": {
        "models": ["resnet18", "resnet34", "resnet50", "vgg16_bn"],
        "tag": "4m",
        "checkpoint": "checkpoints_imagenet_4m/meta_uap_imagenet_4m_ensemble3.pt",
    },
    "4md": {
        "models": ["resnet18", "resnet34", "resnet50", "densenet121"],
        "tag": "4md",
        "checkpoint": "checkpoints_imagenet_4md/meta_uap_imagenet_4md_ensemble3.pt",
    },
}

WEIGHT_PREFIX = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "vgg16_bn": "vgg16_bn",
    "densenet121": "densenet121",
    "mobilenet_v2": "mobilenet_v2",
    "shufflenet_v2_x1_0": "shufflenetv2_x1",
    "vit_b_16": "vit_b_16",
}

WARMUP_CMD = (
    "python -c \"import torchvision; "
    "[getattr(torchvision.models, n)(weights='IMAGENET1K_V1') for n in "
    "['resnet18','resnet34','resnet50','resnet101','vgg16_bn','densenet121',"
    "'mobilenet_v2','shufflenet_v2_x1_0','vit_b_16']]\""
)


# ==================== fail-fast 预检 ====================

def fail(msg: str):
    print(f"[FATAL] {msg}")
    sys.exit(1)


def _weight_present(ckpt_dir: Path, prefix: str) -> bool:
    # torchvision 下载的是 .pth；历史上也有 .pt 副本。两种都认。
    return (ckpt_dir.is_dir()
            and (list(ckpt_dir.glob(prefix + "*.pt"))
                 or list(ckpt_dir.glob(prefix + "*.pth"))))


def check_weight_cache(model_keys):
    ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
    missing = [WEIGHT_PREFIX[k] for k in model_keys
               if not _weight_present(ckpt_dir, WEIGHT_PREFIX[k])]
    if missing:
        fail(
            f"torchvision 权重缓存缺失: {missing}\n"
            f"  缓存目录: {ckpt_dir}\n"
            f"  计算节点不能联网，请在登录节点执行预热后重新提交:\n"
            f"    {WARMUP_CMD}"
        )


def check_data_split(data_root: str, split: str) -> int:
    d = Path(data_root) / split
    if not d.is_dir():
        fail(
            f"找不到数据目录: {d}\n"
            f"  梯度计算使用 train split（--data_split train）。\n"
            f"  imagenet-mini 应含 train/ 与 val/ 子目录（ImageFolder 结构）。\n"
            f"  预检命令: ls {data_root}/ && ls {data_root}/{split} | head"
        )
    subdirs = [p for p in d.iterdir() if p.is_dir()]
    if not subdirs:
        fail(f"数据目录 {d} 下没有类别子目录（ImageFolder 需要 <split>/<class>/*.JPEG）。")
    return len(subdirs)


def check_checkpoint(path: str, subset: str):
    p = Path(path)
    if not p.is_file():
        fail(
            f"找不到 meta checkpoint: {p}\n"
            f"  本脚本默认使用修复协议重训的 ensemble checkpoint，请先提交:\n"
            f"    sbatch slurm_wave0c_4m4md.sh\n"
            f"  或用 --checkpoint 指定其他 checkpoint（注意: 混用旧协议 3m 与\n"
            f"  新协议 4m/4md 会破坏协议一致性，仅建议临时诊断时使用）。"
        )


# ==================== 主流程 ====================

def parse_args():
    p = argparse.ArgumentParser(description="内循环梯度余弦冲突矩阵（Wave 1 / 方向三机制证据）")
    p.add_argument("--model_subset", type=str, required=True,
                   choices=["3", "3m", "4m", "4md"],
                   help="surrogate 集合: 3/3m=[r18,r34,r50]; 4m=+vgg16_bn; 4md=+densenet121")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="默认按 subset 选择修复协议 ensemble checkpoint")
    p.add_argument("--data_root", type=str, default="/data/public/imagenet-mini")
    p.add_argument("--data_split", type=str, default="train", choices=["train", "val"],
                   help="计算梯度所用的数据 split（默认 train，与元训练口径一致）")
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--grad_samples", type=int, default=32,
                   help="单批次梯度计算的图像数（同一批次用于所有模型）")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="meta 训练时的 ε（clamp_uap 用）")
    p.add_argument("--out_dir", type=str, default="results_wave0")
    return p.parse_args()


def compute_inner_grad(model, images, labels, uap_init, epsilon):
    """一步 inner-loop 梯度（与 train_imagenet.py 内循环第一步一致，穿过 clamp）"""
    uap_leaf = uap_init.clone().requires_grad_(True)
    clamped = clamp_uap(uap_leaf, epsilon)
    adv = (images + clamped).clamp(0, 1)
    logits = model(adv)
    loss = -F.cross_entropy(logits, labels)
    g = torch.autograd.grad(loss, uap_leaf)[0]
    return g.detach()


def main():
    args = parse_args()
    spec = SUBSETS[args.model_subset]
    tag = spec["tag"]
    checkpoint = args.checkpoint or spec["checkpoint"]
    surrogates = list(spec["models"])
    all_models = surrogates + META_TEST_MODELS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type != "cuda":
        print("[警告] 未检测到 GPU；get_imagenet_mini_samples 内部直接 .cuda()，CPU 下会失败。"
              "请在 GPU 计算节点运行。")

    # ---- fail-fast 预检 ----
    print("\n[预检] checkpoint / 数据目录 / torchvision 权重缓存 ...")
    check_checkpoint(checkpoint, tag)
    n_class_dirs = check_data_split(args.data_root, args.data_split)
    check_weight_cache(all_models)
    print(f"  checkpoint: {checkpoint}")
    print(f"  数据: {args.data_root}/{args.data_split}（{n_class_dirs} 个类别子目录）")
    print("  预检通过。")

    # ---- 加载 UAP ----
    print("\n加载 Meta-UAP ...")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "uap_init" in ckpt:
        uap_init = ckpt["uap_init"]
    else:
        uap_init = ckpt
    uap_init = uap_init.detach().to(device).float()
    if uap_init.dim() != 3 or uap_init.shape[0] != 3:
        fail(f"UAP 形状异常: {tuple(uap_init.shape)}，期望 (3, H, W)")
    uap_linf = float(uap_init.abs().max().item())
    print(f"  UAP: shape={tuple(uap_init.shape)}, L_inf={uap_linf:.4f}, ε={args.epsilon}")

    # ---- 数据批次（train split，所有模型共用同一批次）----
    print(f"\n加载梯度数据: {args.data_split} x {args.grad_samples} (seed={args.data_seed}) ...")
    try:
        images, labels = get_imagenet_mini_samples(
            data_root=args.data_root,
            num_samples=args.grad_samples,
            split=args.data_split,
            image_size=224,
            seed=args.data_seed,
        )
    except Exception as e:
        fail(f"数据加载失败: {e}\n  目录: {args.data_root}/{args.data_split}")
    print(f"  images: {tuple(images.shape)}, labels: {tuple(labels.shape)}")

    # ---- 逐模型计算一步 inner-loop 梯度 ----
    grads = {}
    grad_norms = {}
    for name in all_models:
        role = "surrogate" if name in surrogates else "meta_test"
        print(f"  [梯度] {name} ({role}) ...")
        model = create_imagenet_model(name).to(device).eval()
        g = compute_inner_grad(model, images, labels, uap_init, args.epsilon)
        grads[name] = g.flatten().cpu()
        grad_norms[name] = float(g.norm().item())
        model.cpu()
        torch.cuda.empty_cache()
        print(f"          ||g||_2 = {grad_norms[name]:.6f}")

    # ---- 余弦相似度矩阵 ----
    names = all_models
    zero_norm = [n for n in names if grad_norms[n] == 0.0]
    if zero_norm:
        print(f"[警告] 以下模型梯度为零范数，对应行/列置 0: {zero_norm}")
    G = torch.stack([grads[n] for n in names], dim=0)  # (M, D)
    norms = G.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Gn = G / norms
    cos = (Gn @ Gn.t()).tolist()

    # ---- 汇总统计（仅 surrogate 两两）----
    idx = [i for i, n in enumerate(names) if n in surrogates]
    pair_cos = {}
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            na, nb = names[idx[a]], names[idx[b]]
            pair_cos[f"{na}|{nb}"] = cos[idx[a]][idx[b]]
    sur_vals = list(pair_cos.values())
    mean_cos = float(sum(sur_vals) / len(sur_vals)) if sur_vals else None
    min_pair = min(pair_cos, key=pair_cos.get) if pair_cos else None

    # ---- 输出 JSON ----
    out = {
        "config": {
            "model_subset": tag,
            "checkpoint": str(checkpoint),
            "surrogates": surrogates,
            "meta_test_models": META_TEST_MODELS,
            "data_root": args.data_root,
            "data_split": args.data_split,
            "data_seed": args.data_seed,
            "grad_samples": args.grad_samples,
            "epsilon": args.epsilon,
            "device": str(device),
            "uap_linf": uap_linf,
            "gradient_definition": ("one-step inner-loop grad of -CE(x + clamp_uap(uap_init, eps)) "
                                    "w.r.t. uap_init, through the clamp (as in train_imagenet.py)"),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "models": names,
        "roles": {n: ("surrogate" if n in surrogates else "meta_test") for n in names},
        "grad_norms": grad_norms,
        "cosine_matrix": cos,
        "surrogate_pair_cosine": pair_cos,
        "summary": {
            "mean_pairwise_cosine_surrogates": mean_cos,
            "min_pair": min_pair,
            "min_pair_cosine": pair_cos.get(min_pair) if min_pair else None,
            "note": ("机制性描述证据: 仅在与干预证据（代理集消融）方向一致时声称 "
                     "'consistent with'，不得单独支撑因果（裁决文件 §4 方向三 GO③）。"),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grad_conflict_{tag}_s{args.data_seed}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- 打印矩阵 ----
    print("\n" + "=" * 60)
    print(f"内循环梯度余弦相似度矩阵（subset={tag}, ε={args.epsilon}, n={args.grad_samples}）")
    print("=" * 60)
    header = f"{'':<18}" + "".join(f"{n[:10]:>12}" for n in names)
    print(header)
    for i, n in enumerate(names):
        role = "*" if n in surrogates else " "
        row = f"{n[:16] + role:<18}" + "".join(f"{cos[i][j]:>12.4f}" for j in range(len(names)))
        print(row)
    print("(* = surrogate 元训练模型; 无标记 = 元测试模型，仅作对照)")
    if mean_cos is not None:
        print(f"\nsurrogate 两两余弦: mean={mean_cos:.4f}, min={min_pair} ({pair_cos[min_pair]:.4f})")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
