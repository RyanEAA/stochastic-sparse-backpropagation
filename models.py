import torch
import torch.nn as nn
import torch.nn.functional as F

from SparseLinear import SparseLinear


def _make_mlp(in_features, hidden_features, out_features, use_sparse=False, keep_ratio=0.5):
    layers = []
    current_features = in_features
    for hidden in hidden_features:
        if use_sparse:
            layers.append(SparseLinear(current_features, hidden, keep_ratio))
        else:
            layers.append(nn.Linear(current_features, hidden))
        layers.append(nn.ReLU())
        current_features = hidden

    if use_sparse:
        layers.append(SparseLinear(current_features, out_features, keep_ratio))
    else:
        layers.append(nn.Linear(current_features, out_features))

    return nn.Sequential(*layers)


class DenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _make_mlp(784, [256, 128], 10, use_sparse=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class SparseMLP(nn.Module):
    def __init__(self, keep_ratio=0.2):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.net = _make_mlp(784, [256, 128], 10, use_sparse=True, keep_ratio=keep_ratio)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class DenseMLP28(nn.Module):
    def __init__(self, input_features, num_classes, hidden_features=(256, 128)):
        super().__init__()
        self.net = _make_mlp(input_features, list(hidden_features), num_classes, use_sparse=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class SparseMLP28(nn.Module):
    def __init__(self, input_features, num_classes, keep_ratio=0.2, hidden_features=(256, 128)):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.net = _make_mlp(input_features, list(hidden_features), num_classes, use_sparse=True, keep_ratio=keep_ratio)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class DenseCIFARSmallCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SparseCIFARSmallCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, keep_ratio=0.5):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            SparseLinear(128 * 4 * 4, 512, keep_ratio),
            nn.ReLU(),
            SparseLinear(512, num_classes, keep_ratio),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DenseSVHNCNN(DenseCIFARSmallCNN):
    def __init__(self, num_classes=10):
        super().__init__(in_channels=3, num_classes=num_classes)


class SparseSVHNCNN(SparseCIFARSmallCNN):
    def __init__(self, num_classes=10, keep_ratio=0.5):
        super().__init__(in_channels=3, num_classes=num_classes, keep_ratio=keep_ratio)


class DenseTinyImageNetCNN(DenseCIFARSmallCNN):
    def __init__(self, num_classes=200):
        super().__init__(in_channels=3, num_classes=num_classes)


class SparseTinyImageNetCNN(SparseCIFARSmallCNN):
    def __init__(self, num_classes=200, keep_ratio=0.5):
        super().__init__(in_channels=3, num_classes=num_classes, keep_ratio=keep_ratio)


class DenseTabularMLP(nn.Module):
    def __init__(self, input_features, num_classes, hidden_features=(256, 128)):
        super().__init__()
        self.net = _make_mlp(input_features, list(hidden_features), num_classes, use_sparse=False)

    def forward(self, x):
        return self.net(x)


class SparseTabularMLP(nn.Module):
    def __init__(self, input_features, num_classes, keep_ratio=0.2, hidden_features=(256, 128)):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.net = _make_mlp(input_features, list(hidden_features), num_classes, use_sparse=True, keep_ratio=keep_ratio)

    def forward(self, x):
        return self.net(x)


class CIFARSmallCNN(DenseCIFARSmallCNN):
    def __init__(self, use_sparse=False, keep_ratio=0.5):
        if use_sparse:
            SparseCIFARSmallCNN.__init__(self, in_channels=3, num_classes=10, keep_ratio=keep_ratio)
        else:
            DenseCIFARSmallCNN.__init__(self, in_channels=3, num_classes=10)


class MNISTDenseModel(DenseMLP):
    pass


class MNISTSparseModel(SparseMLP):
    pass


class FashionMNISTDenseModel(DenseMLP):
    pass


class FashionMNISTSparseModel(SparseMLP):
    pass


class KMNISTDenseModel(DenseMLP):
    pass


class KMNISTSparseModel(SparseMLP):
    pass


class CIFAR100DenseModel(DenseCIFARSmallCNN):
    def __init__(self):
        super().__init__(in_channels=3, num_classes=100)


class CIFAR100SparseModel(SparseCIFARSmallCNN):
    def __init__(self, keep_ratio=0.5):
        super().__init__(in_channels=3, num_classes=100, keep_ratio=keep_ratio)


class CIFAR10DenseModel(DenseCIFARSmallCNN):
    def __init__(self):
        super().__init__(in_channels=3, num_classes=10)


class CIFAR10SparseModel(SparseCIFARSmallCNN):
    def __init__(self, keep_ratio=0.5):
        super().__init__(in_channels=3, num_classes=10, keep_ratio=keep_ratio)


class SVHNDenseModel(DenseSVHNCNN):
    pass


class SVHNSparseModel(SparseSVHNCNN):
    pass


class TinyImageNetDenseModel(DenseTinyImageNetCNN):
    pass


class TinyImageNetSparseModel(SparseTinyImageNetCNN):
    pass


class UCIAdultDenseModel(DenseTabularMLP):
    def __init__(self, input_features=105, num_classes=2):
        super().__init__(input_features=input_features, num_classes=num_classes)


class UCIAdultSparseModel(SparseTabularMLP):
    def __init__(self, input_features=105, num_classes=2, keep_ratio=0.5):
        super().__init__(input_features=input_features, num_classes=num_classes, keep_ratio=keep_ratio)


class CovertypeDenseModel(DenseTabularMLP):
    def __init__(self, input_features=98, num_classes=7):
        super().__init__(input_features=input_features, num_classes=num_classes)


class CovertypeSparseModel(SparseTabularMLP):
    def __init__(self, input_features=98, num_classes=7, keep_ratio=0.5):
        super().__init__(input_features=input_features, num_classes=num_classes, keep_ratio=keep_ratio)


__all__ = [
    "DenseMLP",
    "SparseMLP",
    "DenseMLP28",
    "SparseMLP28",
    "DenseCIFARSmallCNN",
    "SparseCIFARSmallCNN",
    "CIFARSmallCNN",
    "DenseSVHNCNN",
    "SparseSVHNCNN",
    "DenseTinyImageNetCNN",
    "SparseTinyImageNetCNN",
    "DenseTabularMLP",
    "SparseTabularMLP",
    "MNISTDenseModel",
    "MNISTSparseModel",
    "FashionMNISTDenseModel",
    "FashionMNISTSparseModel",
    "KMNISTDenseModel",
    "KMNISTSparseModel",
    "CIFAR10DenseModel",
    "CIFAR10SparseModel",
    "CIFAR100DenseModel",
    "CIFAR100SparseModel",
    "SVHNDenseModel",
    "SVHNSparseModel",
    "TinyImageNetDenseModel",
    "TinyImageNetSparseModel",
    "UCIAdultDenseModel",
    "UCIAdultSparseModel",
    "CovertypeDenseModel",
    "CovertypeSparseModel",
]
