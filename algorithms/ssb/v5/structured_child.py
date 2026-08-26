"""SSB V5: dense-master / structured-dense-child training.

The master network retains the full parameter tensors.  Training executes on a
physically smaller child composed of ordinary dense PyTorch Linear/Conv2d
modules.  After ``refresh_steps`` optimizer updates, the active child weights
are scattered back into the master, a new structured child is sampled from the
master, and training continues.

This intentionally uses structured neuron/channel removal rather than pruning
masks so the active child executes smaller dense kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn
from torch import optim


@dataclass
class LayerMap:
    kind: str
    master: nn.Module
    child: nn.Module
    out_idx: torch.Tensor
    in_idx: torch.Tensor


class ChildMLP(nn.Module):
    def __init__(self, layers: Sequence[nn.Linear]):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)
        return x


class ChildCNN(nn.Module):
    def __init__(self, feature_modules: Sequence[nn.Module], classifier_layers: Sequence[nn.Linear]):
        super().__init__()
        self.features = nn.Sequential(*feature_modules)
        self.flatten = nn.Flatten()
        self.classifier = nn.ModuleList(classifier_layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(self.features(x))
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i < len(self.classifier) - 1:
                x = self.relu(x)
        return x


def _choose_indices(total: int, keep_ratio: float, device: torch.device) -> torch.Tensor:
    keep = max(1, min(total, round(total * keep_ratio)))
    if keep == total:
        return torch.arange(total, device=device, dtype=torch.long)
    # Sort after sampling so child rows/channels preserve master order and memory locality.
    return torch.randperm(total)[:keep].sort().values.to(device)


def _copy_linear(master: nn.Linear, out_idx: torch.Tensor, in_idx: torch.Tensor) -> nn.Linear:
    device, dtype = master.weight.device, master.weight.dtype
    child = nn.Linear(len(in_idx), len(out_idx), bias=master.bias is not None, device=device, dtype=dtype)
    with torch.no_grad():
        child.weight.copy_(master.weight.index_select(0, out_idx).index_select(1, in_idx))
        if master.bias is not None:
            child.bias.copy_(master.bias.index_select(0, out_idx))
    return child


def _copy_conv(master: nn.Conv2d, out_idx: torch.Tensor, in_idx: torch.Tensor) -> nn.Conv2d:
    if master.groups != 1:
        raise NotImplementedError("SSB V5 currently supports Conv2d(groups=1) only.")
    device, dtype = master.weight.device, master.weight.dtype
    child = nn.Conv2d(
        len(in_idx), len(out_idx), master.kernel_size, master.stride, master.padding,
        master.dilation, 1, master.bias is not None, master.padding_mode,
        device=device, dtype=dtype,
    )
    with torch.no_grad():
        child.weight.copy_(master.weight.index_select(0, out_idx).index_select(1, in_idx))
        if master.bias is not None:
            child.bias.copy_(master.bias.index_select(0, out_idx))
    return child


def _scatter_linear(mapping: LayerMap) -> None:
    master: nn.Linear = mapping.master  # type: ignore[assignment]
    child: nn.Linear = mapping.child  # type: ignore[assignment]
    with torch.no_grad():
        rows = master.weight.index_select(0, mapping.out_idx).clone()
        rows.index_copy_(1, mapping.in_idx, child.weight)
        master.weight.index_copy_(0, mapping.out_idx, rows)
        if master.bias is not None:
            master.bias.index_copy_(0, mapping.out_idx, child.bias)


def _scatter_conv(mapping: LayerMap) -> None:
    master: nn.Conv2d = mapping.master  # type: ignore[assignment]
    child: nn.Conv2d = mapping.child  # type: ignore[assignment]
    with torch.no_grad():
        rows = master.weight.index_select(0, mapping.out_idx).clone()
        rows.index_copy_(1, mapping.in_idx, child.weight)
        master.weight.index_copy_(0, mapping.out_idx, rows)
        if master.bias is not None:
            master.bias.index_copy_(0, mapping.out_idx, child.bias)


class StructuredChildModelV5(nn.Module):
    """Dense master with a periodically resampled structured dense child."""

    is_structured_child = True
    is_v5_structured_child = True

    def __init__(self, master: nn.Module, keep_ratio: float, refresh_steps: int = 100):
        super().__init__()
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1].")
        if refresh_steps < 1:
            raise ValueError("refresh_steps must be >= 1.")
        self.master = master
        self.keep_ratio = float(keep_ratio)
        self.refresh_steps = int(refresh_steps)
        self.child: nn.Module | None = None
        self._maps: List[LayerMap] = []
        self._steps_since_refresh = 0
        self.refresh_count = 0
        # Persistent Adam state lives in dense-master tensor shapes.
        # Keys are master Parameter objects; inactive slices retain their moments.
        self._master_adam_state = {}

    def training_parameters(self):
        if self.child is None:
            raise RuntimeError("Structured child has not been initialized.")
        return self.child.parameters()

    def master_parameter_count(self) -> int:
        return sum(p.numel() for p in self.master.parameters())

    def child_parameter_count(self) -> int:
        if self.child is None:
            return 0
        return sum(p.numel() for p in self.child.parameters())

    def refresh_child(self) -> None:
        if self.child is not None:
            self.sync_child_to_master()
        self._maps = []
        if hasattr(self.master, "net") and isinstance(self.master.net, nn.Sequential):
            self.child = self._build_mlp_child()
        elif hasattr(self.master, "features") and hasattr(self.master, "classifier"):
            self.child = self._build_cnn_child()
        else:
            raise TypeError(f"Unsupported dense master type for SSB V5: {type(self.master).__name__}")
        self._steps_since_refresh = 0
        self.refresh_count += 1

    def sync_child_to_master(self) -> None:
        if self.child is None:
            return
        for mapping in self._maps:
            if mapping.kind == "linear":
                _scatter_linear(mapping)
            elif mapping.kind == "conv":
                _scatter_conv(mapping)
            else:
                raise RuntimeError(f"Unknown V5 mapping kind {mapping.kind!r}")

    def _master_state_for(self, parameter: nn.Parameter):
        state = self._master_adam_state.get(parameter)
        if state is None:
            state = {
                "step": 0.0,
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
            self._master_adam_state[parameter] = state
        return state

    @staticmethod
    def _gather_state_tensor(full: torch.Tensor, mapping: LayerMap, is_bias: bool):
        if is_bias:
            return full.index_select(0, mapping.out_idx).clone()
        return full.index_select(0, mapping.out_idx).index_select(1, mapping.in_idx).clone()

    @staticmethod
    def _scatter_state_tensor(full: torch.Tensor, child_value: torch.Tensor, mapping: LayerMap, is_bias: bool):
        if is_bias:
            full.index_copy_(0, mapping.out_idx, child_value)
            return
        rows = full.index_select(0, mapping.out_idx).clone()
        rows.index_copy_(1, mapping.in_idx, child_value)
        full.index_copy_(0, mapping.out_idx, rows)

    def make_optimizer(self, learning_rate: float):
        """Create Adam for the current child and hydrate it from master-owned moments."""
        if self.child is None:
            raise RuntimeError("Structured child has not been initialized.")
        optimizer = optim.Adam(self.child.parameters(), lr=learning_rate)
        for mapping in self._maps:
            for attr in ("weight", "bias"):
                master_parameter = getattr(mapping.master, attr, None)
                child_parameter = getattr(mapping.child, attr, None)
                if master_parameter is None or child_parameter is None:
                    continue
                master_state = self._master_state_for(master_parameter)
                optimizer.state[child_parameter] = {
                    "step": torch.tensor(master_state["step"], device=child_parameter.device),
                    "exp_avg": self._gather_state_tensor(master_state["exp_avg"], mapping, attr == "bias"),
                    "exp_avg_sq": self._gather_state_tensor(master_state["exp_avg_sq"], mapping, attr == "bias"),
                }
        return optimizer

    def sync_optimizer_state_to_master(self, optimizer) -> None:
        """Scatter active child's Adam moments back into persistent master-shaped state."""
        with torch.no_grad():
            for mapping in self._maps:
                for attr in ("weight", "bias"):
                    master_parameter = getattr(mapping.master, attr, None)
                    child_parameter = getattr(mapping.child, attr, None)
                    if master_parameter is None or child_parameter is None:
                        continue
                    child_state = optimizer.state.get(child_parameter)
                    if not child_state or "exp_avg" not in child_state:
                        continue
                    master_state = self._master_state_for(master_parameter)
                    step = child_state.get("step", 0.0)
                    master_state["step"] = float(step.item() if torch.is_tensor(step) else step)
                    self._scatter_state_tensor(master_state["exp_avg"], child_state["exp_avg"], mapping, attr == "bias")
                    self._scatter_state_tensor(master_state["exp_avg_sq"], child_state["exp_avg_sq"], mapping, attr == "bias")

    def after_optimizer_step(self, optimizer, learning_rate: float):
        # Persist moments after every update so refresh=1 is a first-class configuration.
        self.sync_optimizer_state_to_master(optimizer)
        self._steps_since_refresh += 1
        if self._steps_since_refresh < self.refresh_steps:
            return optimizer
        self.refresh_child()
        return self.make_optimizer(learning_rate)

    def master_optimizer_state_bytes(self) -> int:
        total = 0
        for state in self._master_adam_state.values():
            total += state["exp_avg"].numel() * state["exp_avg"].element_size()
            total += state["exp_avg_sq"].numel() * state["exp_avg_sq"].element_size()
        return total

    def forward(self, x):
        if self.training:
            if self.child is None:
                raise RuntimeError("Structured child has not been initialized.")
            return self.child(x)
        return self.master(x)

    def _build_mlp_child(self) -> nn.Module:
        master_layers = [m for m in self.master.net if isinstance(m, nn.Linear)]
        device = master_layers[0].weight.device
        in_idx = torch.arange(master_layers[0].in_features, device=device)
        child_layers = []
        for i, master_layer in enumerate(master_layers):
            is_output = i == len(master_layers) - 1
            out_idx = (
                torch.arange(master_layer.out_features, device=device)
                if is_output
                else _choose_indices(master_layer.out_features, self.keep_ratio, device)
            )
            child_layer = _copy_linear(master_layer, out_idx, in_idx)
            child_layers.append(child_layer)
            self._maps.append(LayerMap("linear", master_layer, child_layer, out_idx, in_idx))
            in_idx = out_idx
        return ChildMLP(child_layers)

    def _build_cnn_child(self) -> nn.Module:
        master_feature_modules = list(self.master.features.net)
        convs = [m for m in master_feature_modules if isinstance(m, nn.Conv2d)]
        if not convs:
            raise TypeError("CNN master has no Conv2d layers.")
        device = convs[0].weight.device
        current_in = torch.arange(convs[0].in_channels, device=device)
        selected_by_conv = []
        for conv in convs:
            out_idx = _choose_indices(conv.out_channels, self.keep_ratio, device)
            selected_by_conv.append((conv, current_in, out_idx))
            current_in = out_idx

        # Rebuild feature stack, replacing each conv by a physically smaller dense conv.
        feature_modules = []
        conv_cursor = 0
        for module in master_feature_modules:
            if isinstance(module, nn.Conv2d):
                master_conv, in_idx, out_idx = selected_by_conv[conv_cursor]
                child_conv = _copy_conv(master_conv, out_idx, in_idx)
                feature_modules.append(child_conv)
                self._maps.append(LayerMap("conv", master_conv, child_conv, out_idx, in_idx))
                conv_cursor += 1
            else:
                # ReLU/pooling/adaptive pooling modules are parameter-free and safe to reuse by type.
                if isinstance(module, nn.ReLU):
                    feature_modules.append(nn.ReLU(inplace=module.inplace))
                elif isinstance(module, nn.MaxPool2d):
                    feature_modules.append(nn.MaxPool2d(module.kernel_size, module.stride, module.padding, module.dilation, module.return_indices, module.ceil_mode))
                elif isinstance(module, nn.AdaptiveAvgPool2d):
                    feature_modules.append(nn.AdaptiveAvgPool2d(module.output_size))
                else:
                    raise TypeError(f"Unsupported CNN feature module for SSB V5: {type(module).__name__}")

        # DenseCNN classifier is Sequential[Flatten, Linear, ReLU, ...].
        master_linears = [m for m in self.master.classifier if isinstance(m, nn.Linear)]
        final_conv_out = selected_by_conv[-1][2]
        full_final_channels = selected_by_conv[-1][0].out_channels
        first_linear = master_linears[0]
        spatial_area = first_linear.in_features // full_final_channels
        offsets = torch.arange(spatial_area, device=device)
        in_idx = (final_conv_out[:, None] * spatial_area + offsets[None, :]).reshape(-1)

        child_linears = []
        for i, master_linear in enumerate(master_linears):
            is_output = i == len(master_linears) - 1
            out_idx = (
                torch.arange(master_linear.out_features, device=device)
                if is_output
                else _choose_indices(master_linear.out_features, self.keep_ratio, device)
            )
            child_linear = _copy_linear(master_linear, out_idx, in_idx)
            child_linears.append(child_linear)
            self._maps.append(LayerMap("linear", master_linear, child_linear, out_idx, in_idx))
            in_idx = out_idx
        return ChildCNN(feature_modules, child_linears)
