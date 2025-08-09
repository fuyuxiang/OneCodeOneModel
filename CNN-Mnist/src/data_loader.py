from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_mnist_loaders(config):
    """创建MNIST数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载完整训练集
    full_train = datasets.MNIST(
        root=config['data']['root'],
        train=True,
        download=True,
        transform=transform
    )

    # 划分训练集和验证集 (90%/10%)
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size]
    )

    # 测试集
    test_set = datasets.MNIST(
        root=config['data']['root'],
        train=False,
        download=True,
        transform=transform
    )

    # 自动设置worker数量
    num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0

    train_loader = DataLoader(
        train_set,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader