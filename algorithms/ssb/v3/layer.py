import torch
import torch.nn as nn

from .function import SparseLinearFunctionV3


class SparseLinearV3(nn.Module):
    """Forward+backward stochastic neuron sparsity using one shared mask."""

    def __init__(self, in_features, out_features, keep_ratio=0.2):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in the range (0, 1].")

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.keep_ratio = keep_ratio

    def forward(self, x):
        if not self.training:
            return x @ self.weight.t() + self.bias

        active_mask = torch.rand(
            self.weight.size(0), device=self.weight.device
        ) < self.keep_ratio
        return SparseLinearFunctionV3.apply(x, self.weight, self.bias, active_mask)
