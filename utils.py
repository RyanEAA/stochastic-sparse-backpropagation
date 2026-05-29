import os
import time
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


def get_dataloaders(name, batch_size_train=128, batch_size_test=256, data_dir="./data", num_workers=4):
    name_lower = name.lower()

    if name_lower in ("mnist",):
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    elif name_lower in ("fashion-mnist", "fashion_mnist", "fashionmnist"):
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    elif name_lower in ("kmnist",):
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.KMNIST(root=data_dir, train=True, download=True, transform=transform)
        test = datasets.KMNIST(root=data_dir, train=False, download=True, transform=transform)

    elif name_lower in ("cifar10",):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
        train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    elif name_lower in ("cifar100",):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
        train = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    elif name_lower in ("svhn",):
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
        test = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)

    elif name_lower in ("tiny-imagenet", "tiny_imagenet", "tinyimagenet"):
        # expects ImageNet-like layout at data_dir/tiny-imagenet-200/{train,val}
        train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
        val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
        transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
        train = ImageFolder(train_dir, transform=transform)
        test = ImageFolder(val_dir, transform=transform)

    elif name_lower in ("uci-adult", "adult"):
        try:
            from sklearn.datasets import fetch_openml
        except Exception as e:
            raise ImportError("scikit-learn is required for UCI tabular loaders (pip install scikit-learn)")

        df = fetch_openml('adult', version=2, as_frame=True).frame
        df = df.dropna()
        y = (df['class'] == '>50K').astype(int)
        X = df.drop(columns=['class'])
        X = pd.get_dummies(X)
        X = X.astype('float32')
        X = X.values
        y = y.values.astype('int64')
        # simple 80/20 split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    elif name_lower in ("covertype",):
        try:
            from sklearn.datasets import fetch_openml
        except Exception as e:
            raise ImportError("scikit-learn is required for UCI tabular loaders (pip install scikit-learn)")

        df = fetch_openml('covertype', version=3, as_frame=True).frame
        df = df.dropna()
        y = df.iloc[:, -1].astype('int64').values
        X = df.iloc[:, :-1]
        X = pd.get_dummies(X)
        X = X.astype('float32').values
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_loader = DataLoader(train, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def evaluate(model, test_loader, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def train_model(model, train_loader, test_loader, epochs=5, lr=1e-4, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, test_loader, device=device)
        accuracies.append(acc)
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f} | Accuracy: {acc:.4f}")

    total_time = time.time() - start_time
    return accuracies, total_time


__all__ = ["get_dataloaders", "train_model", "evaluate"]
