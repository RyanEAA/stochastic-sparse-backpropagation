import math
import torch
import torch.nn as nn

class DenseMLP(nn.Module):
    def __init__(self, input_features, hidden_dims, num_classes):
        super().__init__()
        dims = [input_features, *hidden_dims, num_classes]
        layers = [nn.Flatten()]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class SparseMLP(nn.Module):
    def __init__(self, input_features, hidden_dims, num_classes, sparse_linear_cls, keep_ratio):
        super().__init__()
        dims = [input_features, *hidden_dims, num_classes]
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([
            sparse_linear_cls(dims[i], dims[i+1], keep_ratio=keep_ratio)
            for i in range(len(dims)-1)
        ])
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.flatten(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1: x = self.relu(x)
        return x

class DropoutMLP(nn.Module):
    """Standard dropout learning baseline. p = 1 - keep_ratio."""
    def __init__(self, input_features, hidden_dims, num_classes, keep_ratio):
        super().__init__()
        if not 0 < keep_ratio <= 1: raise ValueError('keep_ratio must be in (0, 1].')
        dims = [input_features, *hidden_dims, num_classes]
        layers = [nn.Flatten()]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers += [nn.ReLU(), nn.Dropout(p=1.0-keep_ratio)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class StaticNeuronPrunedLinear(nn.Module):
    """Simple random static neuron-pruning baseline.

    This mask is fixed for the run. It is a learning/control baseline; because a
    dense matmul is still used, it should not be interpreted as structural speed pruning.
    """
    def __init__(self, in_features, out_features, keep_ratio):
        super().__init__()
        if not 0 < keep_ratio <= 1: raise ValueError('keep_ratio must be in (0, 1].')
        self.linear = nn.Linear(in_features, out_features)
        keep = max(1, round(out_features * keep_ratio))
        indices = torch.randperm(out_features)[:keep]
        mask = torch.zeros(out_features)
        mask[indices] = 1.0
        self.register_buffer('mask', mask)
    def forward(self, x):
        return self.linear(x) * self.mask

class PrunedMLP(nn.Module):
    def __init__(self, input_features, hidden_dims, num_classes, keep_ratio):
        super().__init__()
        dims = [input_features, *hidden_dims, num_classes]
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([
            StaticNeuronPrunedLinear(dims[i], dims[i+1], keep_ratio)
            for i in range(len(dims)-1)
        ])
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.flatten(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1: x = self.relu(x)
        return x
