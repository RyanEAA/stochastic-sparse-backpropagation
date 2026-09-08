"""Deterministic torchvision loaders for the six benchmark datasets."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SUPPORTED_DATASETS = (
    "mnist", "fashion_mnist", "kmnist", "cifar10", "cifar100", "svhn"
)


def normalize_dataset_name(name: str) -> str:
    normalized = name.lower().strip().replace("-", "_")
    aliases = {"fashionmnist": "fashion_mnist", "fashion": "fashion_mnist"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {', '.join(SUPPORTED_DATASETS)}"
        )
    return normalized


def _datasets(name: str, root: Path):
    transform = transforms.ToTensor()
    if name == "mnist":
        return (
            datasets.MNIST(root, train=True, download=True, transform=transform),
            datasets.MNIST(root, train=False, download=True, transform=transform),
        )
    if name == "fashion_mnist":
        return (
            datasets.FashionMNIST(root, train=True, download=True, transform=transform),
            datasets.FashionMNIST(root, train=False, download=True, transform=transform),
        )
    if name == "kmnist":
        return (
            datasets.KMNIST(root, train=True, download=True, transform=transform),
            datasets.KMNIST(root, train=False, download=True, transform=transform),
        )
    if name == "cifar10":
        return (
            datasets.CIFAR10(root, train=True, download=True, transform=transform),
            datasets.CIFAR10(root, train=False, download=True, transform=transform),
        )
    if name == "cifar100":
        return (
            datasets.CIFAR100(root, train=True, download=True, transform=transform),
            datasets.CIFAR100(root, train=False, download=True, transform=transform),
        )
    return (
        datasets.SVHN(root, split="train", download=True, transform=transform),
        datasets.SVHN(root, split="test", download=True, transform=transform),
    )


def _deterministic_subset(dataset, size: int, seed: int):
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_loaders(name, batch_size, seed, subset, data_dir, num_workers):
    name = normalize_dataset_name(name)
    train_set, val_set = _datasets(name, Path(data_dir))
    train_set = _deterministic_subset(train_set, subset, seed)
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader
