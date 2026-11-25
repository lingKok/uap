"""
vit_probe.py — ViT-B/16 0-step 迁移探针（Wave 0d / 方向二前提确认 + Limitations）

用途:
  用现有 ImageNet 元训练得到的 ensemble UAP（uap_init），在 torchvision
  ViT-B/16（IMAGENET1K_V1 权重，NormalizedModel 包装，与 models.py 现有
  ImageNet 模型一致）上做 0-step 评估:
    - ε ∈ {0.05, 0.1, 0.2}
    - 双形态: raw    = clamp(uap_init, ε)
              scaled = clamp(uap_init * ε/ε_train, ε)   （与 experiments.py 的
              ε-scaling 口径一致，缩放对象为 ensemble uap_init）
    - 双口径: fr_all = 全样本攻击后错误率（含原本就错图，与 compute_fr 一致）
              asr    = 仅在 clean 正确分类样本上的攻击成功率
  对照文献预期: TSC-UAP 实测 CNN 像素 UAP → ViT-B/16 迁移 ASR 仅 14-24%
  （裁决文件 §2 方向二 / §4 方向二 GO①: 探针数字与文献一致即确认前提成立）。

协议说明:
  - 评估走 val split（--eval_split val，seed 42），与修复后 experiments.py
    的评估口径一致。
  - 注意: 现有 3m checkpoint 为旧协议训练（meta-training 曾使用 val split
    的 12 张图, seed 42）。同种子前缀性质决定这 12 张落在评估集内
    （eval_samples=2000 时约 0.6%）。此为既有 checkpoint 的固有性质; 文献对照为
    14-24% 量级，探针对此不敏感。如需严格口径，改用修复后重训的 checkpoint
    （slurm_wave0c_4m4md.sh 产出的
    checkpoints_imagenet_3m_fixed/meta_uap_imagenet_3m_fixed_ensemble3.pt）。

用法:
  python vit_probe.py \
      --checkpoint checkpoints_imagenet_3m/meta_uap_imagenet_3m_ensemble3.pt \
      --eval_split val --eval_samples 2000 --eval_seed 42 \
      --epsilons 0.05,0.1,0.2 --train_epsilon 0.05 \
      --data_root /data/public/imagenet-mini \
      --out_dir results_wave0

输出: results_wave0/vit_probe.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

from models import NormalizedModel
from utils.data import get_imagenet_mini_samples
from utils.losses import clamp_uap


# ==================== fail-fast 预检 ====================

def fail(msg: str):
    print(f"[FATAL] {msg}")
    sys.exit(1)


# torchvision 权重文件名前缀（IMAGENET1K_V1），用于缓存预检
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


def check_weight_cache(model_keys):
    """计算节点不能联网: torchvision 权重必须在登录节点预热到 hub 缓存。"""
    ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
    missing = [WEIGHT_PREFIX[k] for k in model_keys
               if not (ckpt_dir.is_dir() and list(ckpt_dir.glob(WEIGHT_PREFIX[k] + "*.pt")))]
    if missing:
        fail(
            f"torchvision 权重缓存缺失: {missing}\n"
            f"  缓存目录: {ckpt_dir}\n"
            f"  计算节点不能联网，请在登录节点执行预热后重新提交:\n"
            f"    {WARMUP_CMD}\n"
            f"  或把已有缓存同步到计算节点可见路径并设置 TORCH_HOME。"
        )


def check_data_split(data_root: str, split: str) -> int:
    """ImageFolder 结构预检: <data_root>/<split>/<class>/*.JPEG"""
    d = Path(data_root) / split
    if not d.is_dir():
        fail(
            f"找不到数据目录: {d}\n"
            f"  imagenet-mini 应含 train/ 与 val/ 子目录（ImageFolder 结构）。\n"
            f"  预检命令: ls {data_root}/ && ls {data_root}/{split} | head"
        )
    subdirs = [p for p in d.iterdir() if p.is_dir()]
    if not subdirs:
        fail(f"数据目录 {d} 下没有类别子目录（ImageFolder 需要 <split>/<class>/*.JPEG）。")
    return len(subdirs)


def check_checkpoint(path: str):
    p = Path(path)
    if not p.is_file():
        fail(
            f"找不到 meta checkpoint: {p}\n"
            f"  预期路径（集群上由 slurm_train_imagenet_3m.sh 生成）:\n"
            f"    checkpoints_imagenet_3m/meta_uap_imagenet_3m_ensemble3.pt\n"
            f"  若仅有 checkpoints_imagenet_3m_run{{0,1,2}}/meta_uap_imagenet_epoch50.pt，\n"
            f"  可按 slurm_train_imagenet_3m.sh 末尾的方式对 3 个 run 的 uap_init 取平均生成。\n"
            f"  或用 --checkpoint 指定其他 checkpoint（如修复后重训的\n"
            f"    checkpoints_imagenet_3m_fixed/meta_uap_imagenet_3m_fixed_ensemble3.pt）。"
        )


# ==================== 评估 ====================

@torch.no_grad()
def clean_accuracy(model, images, labels, batch_size):
    n = len(images)
    correct = 0
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        logits = model(images[s:e])
        correct += (logits.argmax(dim=1) == labels[s:e]).sum().item()
    return correct / n


@torch.no_grad()
def evaluate(model, images, labels, uap, batch_size):
    """双口径评估: fr_all（全样本错误率）与 asr（clean 正确样本上的攻击成功率）"""
    n = len(images)
    clean_correct = 0
    adv_wrong_all = 0
    adv_wrong_given_correct = 0
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        x, y = images[s:e], labels[s:e]
        cc = model(x).argmax(dim=1) == y
        clean_correct += cc.sum().item()
        adv_logits = model((x + uap).clamp(0, 1))
        wrong = adv_logits.argmax(dim=1) != y
        adv_wrong_all += wrong.sum().item()
        adv_wrong_given_correct += (wrong & cc).sum().item()
    return {
        "fr_all": adv_wrong_all / n,
        "asr": (adv_wrong_given_correct / clean_correct) if clean_correct > 0 else None,
        "n": n,
        "n_clean_correct": clean_correct,
    }


# ==================== 主流程 ====================

def parse_args():
    p = argparse.ArgumentParser(description="ViT-B/16 0-step 迁移探针（Wave 0d）")
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints_imagenet_3m/meta_uap_imagenet_3m_ensemble3.pt",
                   help="meta UAP checkpoint（含 uap_init 键或纯张量）")
    p.add_argument("--data_root", type=str, default="/data/public/imagenet-mini")
    p.add_argument("--eval_split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--eval_samples", type=int, default=2000)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--epsilons", type=str, default="0.05,0.1,0.2",
                   help="逗号分隔的 ε 列表（测试预算）")
    p.add_argument("--train_epsilon", type=float, default=0.05,
                   help="meta 训练时的 ε（scaled 形态的缩放分母）")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_dir", type=str, default="results_wave0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type != "cuda":
        print("[警告] 未检测到 GPU；ViT-B/16 在 CPU 上评估数百张约需数分钟，请确认这是有意的。")

    # ---- fail-fast 预检（数据 / 权重 / checkpoint）----
    print("\n[预检] checkpoint / 数据目录 / torchvision 权重缓存 ...")
    check_checkpoint(args.checkpoint)
    n_class_dirs = check_data_split(args.data_root, args.eval_split)
    check_weight_cache(["vit_b_16"])
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  数据: {args.data_root}/{args.eval_split}（{n_class_dirs} 个类别子目录）")
    print(f"  权重缓存: {Path(torch.hub.get_dir()) / 'checkpoints'}（vit_b_16 已就绪）")
    print("  预检通过。")

    # ---- 加载 UAP ----
    print("\n加载 Meta-UAP ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "uap_init" in ckpt:
        uap_init = ckpt["uap_init"]
    else:
        uap_init = ckpt
    uap_init = uap_init.detach().to(device).float()
    if uap_init.dim() != 3 or uap_init.shape[0] != 3:
        fail(f"UAP 形状异常: {tuple(uap_init.shape)}，期望 (3, H, W)")
    uap_linf = float(uap_init.abs().max().item())
    print(f"  UAP: shape={tuple(uap_init.shape)}, L_inf={uap_linf:.4f}")

    # ---- ViT-B/16（IMAGENET1K_V1 + NormalizedModel，与 models.py 一致）----
    print("\n加载 ViT-B/16 (IMAGENET1K_V1) ...")
    try:
        vit = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
    except Exception as e:  # 权重损坏/不匹配等
        fail(
            f"ViT-B/16 权重加载失败: {e}\n"
            f"  请在登录节点重新预热: {WARMUP_CMD}"
        )
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    model = NormalizedModel(vit).to(device).eval()

    # ---- 评估数据（val split，seed 固定）----
    print(f"\n加载评估数据: {args.eval_split} x {args.eval_samples} (seed={args.eval_seed}) ...")
    try:
        images, labels = get_imagenet_mini_samples(
            data_root=args.data_root,
            num_samples=args.eval_samples,
            split=args.eval_split,
            image_size=224,
            seed=args.eval_seed,
        )
    except Exception as e:
        fail(f"数据加载失败: {e}\n  目录: {args.data_root}/{args.eval_split}")
    print(f"  images: {tuple(images.shape)}, labels: {tuple(labels.shape)}")

    epsilons = [float(x) for x in args.epsilons.split(",") if x.strip()]

    # ---- clean 基线 ----
    print("\n[clean] 无扰动基线 ...")
    cacc = clean_accuracy(model, images, labels, args.batch_size)
    print(f"  clean_acc = {cacc:.4f}")

    # ---- ε × {raw, scaled} 双形态评估 ----
    results = {}
    for eps in epsilons:
        forms = {
            "raw": clamp_uap(uap_init, eps),
            "scaled": clamp_uap(uap_init * (eps / args.train_epsilon), eps),
        }
        results[str(eps)] = {}
        for form_name, uap in forms.items():
            r = evaluate(model, images, labels, uap, args.batch_size)
            results[str(eps)][form_name] = r
            asr_str = f"{r['asr']:.4f}" if r["asr"] is not None else "N/A"
            print(f"  ε={eps:<5} {form_name:<7}: fr_all={r['fr_all']:.4f}  asr={asr_str}")

    # ---- 保存 ----
    out = {
        "probe": "vit_b_16_zero_step",
        "config": {
            "checkpoint": str(args.checkpoint),
            "model": "vit_b_16(IMAGENET1K_V1)+NormalizedModel",
            "data_root": args.data_root,
            "eval_split": args.eval_split,
            "eval_samples": args.eval_samples,
            "eval_seed": args.eval_seed,
            "train_epsilon": args.train_epsilon,
            "epsilons": epsilons,
            "batch_size": args.batch_size,
            "device": str(device),
            "uap_linf": uap_linf,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "clean_acc": cacc,
        "results": results,
        "reference": {
            "note": ("文献预期: TSC-UAP (2024) Table XI 实测 CNN 像素 UAP → ViT-B/16 "
                     "ASR 仅 14-24%; FG-UAP <40%。本探针 asr 落在该区间即确认方向二前提"
                     "（裁决文件 §4 方向二 GO①）。"),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vit_probe.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("ViT-B/16 0-step 探针汇总（fr_all / asr）")
    print("=" * 60)
    print(f"clean_acc = {cacc:.4f}")
    print(f"{'ε':<8}{'raw fr_all':<14}{'raw asr':<12}{'scaled fr_all':<16}{'scaled asr':<12}")
    for eps in epsilons:
        r_raw = results[str(eps)]["raw"]
        r_scl = results[str(eps)]["scaled"]
        a_raw = f"{r_raw['asr']:.4f}" if r_raw["asr"] is not None else "N/A"
        a_scl = f"{r_scl['asr']:.4f}" if r_scl["asr"] is not None else "N/A"
        print(f"{eps:<8}{r_raw['fr_all']:<14.4f}{a_raw:<12}{r_scl['fr_all']:<16.4f}{a_scl:<12}")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
