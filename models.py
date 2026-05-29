import torch
import torch.nn as nn
from SparseLinear import SparseLinear


class CIFARSmallCNN(nn.Module):
    def __init__(self, use_sparse=False, keep_ratio=0.5):
        super().__init__()
        self.use_sparse = use_sparse
        self.keep_ratio = keep_ratio
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # flattened feature size: 128 * 4 * 4 = 2048
        flat = 128 * 4 * 4
        if use_sparse:
            self.classifier = nn.Sequential(
                SparseLinear(flat, 512, keep_ratio),
                nn.ReLU(),
                SparseLinear(512, 10, keep_ratio),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(flat, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


__all__ = ["CIFARSmallCNN"]
