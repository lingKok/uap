"""
Idea 2: 预训练模型权重生成脚本

在 CIFAR-10 上训练 5 个模型作为元学习的"任务分布"
这些模型只需要训练一次，之后重复使用
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from config import Config
from models import create_model


def train_single_model(
    model_name: str,
    num_classes: int = 10,
    image_size: int = 32,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.1,
    save_path: str = "./checkpoints",
    device: str = "cuda",
):
    """训练单个模型"""
    print(f"\n训练模型: {model_name}")
    print("=" * 40)

    # 创建模型
    model = create_model(model_name, num_classes, image_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # 数据
    transform_train = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    data_root = os.environ.get("DATA_ROOT", "/data/public/cifar")
    train_set = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    best_acc = 0.0

    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        scheduler.step()

        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * test_correct / test_total

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, "
                  f"train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%")

        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_file = f"{save_path}/{model_name}.pt"
            torch.save(model.state_dict(), save_file)

    print(f"  训练完成: {model_name}, 最佳测试准确率 = {best_acc:.2f}%")
    print(f"  权重保存: {save_path}/{model_name}.pt")

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预训练模型")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已存在权重的模型")
    args = parser.parse_args()

    config = Config()

    # 训练所有模型
    all_models = config.meta_train_models + config.meta_test_models
    print(f"将训练 {len(all_models)} 个模型: {all_models}")

    import os
    os.makedirs(args.save_dir, exist_ok=True)

    # 跳过已存在的模型
    if args.skip_existing:
        skip = [n for n in all_models if os.path.exists(f"{args.save_dir}/{n}.pt")]
        remain = [n for n in all_models if not os.path.exists(f"{args.save_dir}/{n}.pt")]
        print(f"跳过已训练: {skip}")
        print(f"待训练: {remain}")
        all_models = remain

    results = {}
    for name in all_models:
        acc = train_single_model(
            model_name=name,
            num_classes=config.num_classes,
            image_size=config.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.save_dir,
            device=args.device,
        )
        results[name] = acc

    print("\n" + "=" * 60)
    print("所有模型训练完成！")
    print("=" * 60)
    for name, acc in results.items():
        marker = " [训练]" if name in config.meta_train_models else " [测试]"
        print(f"  {name}{marker}: {acc:.2f}%")
