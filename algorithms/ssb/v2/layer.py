import torch
import torch.nn as nn

from .function import SparseLinearFunctionV2


class SparseLinearV2(nn.Module):
    def __init__(self, in_features, out_features, keep_ratio=0.2):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in the range (0, 1].")

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.keep_ratio = keep_ratio

    def forward(self, x):
        if self.training:
            active_mask = torch.rand(
                self.weight.size(0), device=self.weight.device
            ) < self.keep_ratio
        else:
            active_mask = torch.ones(
                self.weight.size(0), device=self.weight.device, dtype=torch.bool
            )

        return SparseLinearFunctionV2.apply(
            x, self.weight, self.bias, active_mask, self.keep_ratio
        )
