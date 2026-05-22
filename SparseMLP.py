import torch
import torch.nn as nn
import torch.nn.functional as F
from SparseLinear import SparseLinear
class SparseMLP(nn.Module):

    def __init__(self, keep_ratio=0.2):

        super().__init__()

        self.fc1 = SparseLinear(
            784,
            256,
            keep_ratio
        )

        self.fc2 = SparseLinear(
            256,
            128,
            keep_ratio
        )

        self.fc3 = SparseLinear(
            128,
            10,
            keep_ratio
        )

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)