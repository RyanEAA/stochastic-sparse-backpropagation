"""Dataset loading used by the training harness.

This deliberately contains data loading only. Dataset-specific model shapes live
under models/<dataset>/config.py.
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ALIASES = {
    "mnist": "mnist",
    "fashion-mnist": "fashion_mnist", "fashion_mnist": "fashion_mnist", "fashionmnist": "fashion_mnist",
    "kmnist": "kmnist",
    "cifar10": "cifar10", "cifar-10": "cifar10",
    "cifar100": "cifar100", "cifar-100": "cifar100",
    "svhn": "svhn",
}

SUPPORTED_DATASETS = tuple(sorted(set(ALIASES.values())))

def normalize_dataset_name(name: str) -> str:
    key = name.lower()
    if key not in ALIASES:
        raise ValueError(f"Unsupported dataset {name!r}. Available: {', '.join(SUPPORTED_DATASETS)}")
    return ALIASES[key]

def get_datasets(name: str, data_dir: str = "data"):
    name = normalize_dataset_name(name)
    root = str(Path(data_dir))
    if name == "mnist":
        t = transforms.ToTensor()
        return datasets.MNIST(root, train=True, download=True, transform=t), datasets.MNIST(root, train=False, download=True, transform=t)
    if name == "fashion_mnist":
        t = transforms.ToTensor()
        return datasets.FashionMNIST(root, train=True, download=True, transform=t), datasets.FashionMNIST(root, train=False, download=True, transform=t)
    if name == "kmnist":
        t = transforms.ToTensor()
        return datasets.KMNIST(root, train=True, download=True, transform=t), datasets.KMNIST(root, train=False, download=True, transform=t)
    if name == "cifar10":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
        return datasets.CIFAR10(root, train=True, download=True, transform=t), datasets.CIFAR10(root, train=False, download=True, transform=t)
    if name == "cifar100":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
        return datasets.CIFAR100(root, train=True, download=True, transform=t), datasets.CIFAR100(root, train=False, download=True, transform=t)
    if name == "svhn":
        t = transforms.ToTensor()
        return datasets.SVHN(root, split="train", download=True, transform=t), datasets.SVHN(root, split="test", download=True, transform=t)
    raise AssertionError(name)

def build_loaders(name, batch_size=128, seed=0, subset=0, data_dir="data", num_workers=0):
    train_ds, val_ds = get_datasets(name, data_dir)
    if subset > 0:
        train_ds = Subset(train_ds, range(min(subset, len(train_ds))))
        val_ds = Subset(val_ds, range(max(1, min(subset // 4, len(val_ds)))))
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=num_workers)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train, val
