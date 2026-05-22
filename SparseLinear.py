import torch
import torch.nn as nn
import torch.nn.functional as F

from SparseLinearFunction import SparseLinearFunction

class SparseLinear(nn.Module):

    def __init__(self, in_features, out_features, keep_ratio=0.2):
        super().__init__()
        # Initialize weights and biases
        self.weight = nn.Parameter(
        torch.empty(out_features, in_features)
        )

        nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_features))
        # keep_raito 1 means all neurons are active, 0.5 means half of the neurons are active on average
        self.keep_ratio = keep_ratio


    def forward(self, x):

        if self.training:
            active_mask = torch.rand(self.weight.size(0), device=self.weight.device) < self.keep_ratio
        else:
            active_mask = torch.ones(self.weight.size(0), device=self.weight.device, dtype=torch.bool)

        return SparseLinearFunction.apply(x, self.weight, self.bias, active_mask)
    