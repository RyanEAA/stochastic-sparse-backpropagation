"""SSB V6: informed fixed-size structured child training.

V6 retains V5's dense master, physically smaller dense child, gather/scatter
mapping, and master-owned Adam state. At fixed intervals it performs a dense
scoring backward (without a dense optimizer update), ranks each hidden output
neuron/channel with a configured structured criterion and rebuilds the child
from the highest-scoring units. Optional Early-Bird detection freezes topology
after consecutive selected-unit masks become stable.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Dict

import torch
from torch import nn

from algorithms.ssb.v5 import StructuredChildModelV5


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def structured_gradient_l2(weight_gradient: torch.Tensor) -> torch.Tensor:
    """Return one L2 score per Linear output neuron or Conv2d output channel."""
    if weight_gradient.ndim < 2:
        raise ValueError("A structured weight gradient must have at least two dimensions.")
    return weight_gradient.detach().flatten(start_dim=1).norm(p=2, dim=1)


def structured_weight_l2(weight: torch.Tensor) -> torch.Tensor:
    """Return one weight-magnitude score per structured output unit."""
    if weight.ndim < 2:
        raise ValueError("A structured weight tensor must have at least two dimensions.")
    return weight.detach().flatten(start_dim=1).norm(p=2, dim=1)


def structured_taylor(weight: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """First-order Taylor score: sum(abs(weight * gradient)) per output unit."""
    if weight.shape != gradient.shape or weight.ndim < 2:
        raise ValueError("weight and gradient must have the same shape and at least two dimensions.")
    return (weight.detach() * gradient.detach()).abs().flatten(start_dim=1).sum(dim=1)


def selected_unit_mask_distance(previous_maps, current_maps) -> float:
    """Fraction of previously selected hidden units replaced in the new child."""
    if len(previous_maps) != len(current_maps):
        raise ValueError("Topology snapshots must contain the same number of hidden layers.")
    changed = selected = 0
    for previous, current in zip(previous_maps, current_maps):
        previous_set = set(previous.detach().cpu().tolist())
        current_set = set(current.detach().cpu().tolist())
        selected += len(previous_set)
        changed += len(previous_set - current_set)
    return changed / selected if selected else 0.0


def topk_structured_indices(scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Choose exact top-k score indices, sorted into master order for slicing."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty 1D tensor.")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1].")
    keep = max(1, min(scores.numel(), round(scores.numel() * keep_ratio)))
    if keep == scores.numel():
        return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
    # stable=True makes equal-score selection deterministic (lowest master index first).
    ranked = torch.argsort(scores, descending=True, stable=True)[:keep]
    return ranked.sort().values


def gradient_retention_indices(scores: torch.Tensor, retention: float) -> torch.Tensor:
    """Choose the smallest top-ranked set retaining a fraction of gradient energy.

    ``scores`` are L2 norms, so squaring them gives the sum of squared gradients
    contributed by each structured unit.
    """
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty 1D tensor.")
    if not 0 < retention <= 1:
        raise ValueError("gradient_retention must be in (0, 1].")
    energy = scores.detach().square()
    total_energy = energy.sum()
    if not torch.isfinite(total_energy) or total_energy <= 0:
        return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
    ranked = torch.argsort(energy, descending=True, stable=True)
    cumulative = energy.index_select(0, ranked).cumsum(0) / total_energy
    keep = int(torch.searchsorted(
        cumulative, torch.tensor(retention, device=scores.device, dtype=cumulative.dtype)
    ).item()) + 1
    return ranked[:keep].sort().values


class GradientSelectedChildModelV6(StructuredChildModelV5):
    """V5-style child whose hidden structured units are selected by importance scores."""

    is_v6_gradient_selected = True

    def __init__(
        self,
        master: nn.Module,
        keep_ratio: float,
        score_refresh_steps: int = 100,
        selection_mode: str = "fixed",
        gradient_retention: float = 0.90,
        selection_method: str = "gradient_l2",
        early_bird: bool = False,
        stability_window: int = 5,
        stability_threshold: float = 0.10,
    ):
        if score_refresh_steps < 1:
            raise ValueError("score_refresh_steps must be >= 1.")
        if selection_mode not in {"fixed", "gradient_retention"}:
            raise ValueError("selection_mode must be 'fixed' or 'gradient_retention'.")
        if not 0 < gradient_retention <= 1:
            raise ValueError("gradient_retention must be in (0, 1].")
        if selection_method not in {"random", "gradient_l2", "weight_l2", "taylor"}:
            raise ValueError(
                "selection_method must be random, gradient_l2, weight_l2, or taylor."
            )
        if stability_window < 1:
            raise ValueError("stability_window must be >= 1.")
        if not 0 <= stability_threshold <= 1:
            raise ValueError("stability_threshold must be in [0, 1].")
        # V6 controls refreshes through scoring events rather than V5's post-step path.
        super().__init__(master, keep_ratio=keep_ratio, refresh_steps=score_refresh_steps)
        self.score_refresh_steps = int(score_refresh_steps)
        self.selection_mode = selection_mode
        self.gradient_retention = float(gradient_retention)
        self.selection_method = selection_method
        self.early_bird = bool(early_bird)
        self.stability_window = int(stability_window)
        self.stability_threshold = float(stability_threshold)
        self._importance: Dict[nn.Module, torch.Tensor] = {}
        self._last_scored_topology = None
        self.mask_distance_history = deque(maxlen=self.stability_window)
        self.last_mask_distance = float("nan")
        self.topology_frozen = False
        self.topology_freeze_step = 0
        self.topology_freeze_scoring_event = 0
        self.scoring_event_count = 0
        self.selector_scoring_time_s = 0.0
        self.dense_scoring_time_s = 0.0
        self.child_rebuild_time_s = 0.0
        self.last_dense_scoring_time_s = 0.0
        self.last_selector_scoring_time_s = 0.0
        self.last_child_rebuild_time_s = 0.0

    def _select_out_indices(self, master_layer, *, is_output, device):
        total = master_layer.out_features if isinstance(master_layer, nn.Linear) else master_layer.out_channels
        if is_output:
            return torch.arange(total, device=device, dtype=torch.long)
        scores = self._importance.get(master_layer)
        if scores is None:
            # A deterministic placeholder child permits optimizer construction before
            # the first batch; the first batch is scored before it is trained.
            scores = torch.zeros(total, device=device)
        scores = scores.to(device)
        if self.selection_mode == "gradient_retention":
            return gradient_retention_indices(scores, self.gradient_retention)
        return topk_structured_indices(scores, self.keep_ratio)

    def _score_master(self) -> None:
        importance = {}
        for module in self.master.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if self.selection_method == "random":
                    total = (
                        module.out_features
                        if isinstance(module, nn.Linear)
                        else module.out_channels
                    )
                    importance[module] = torch.rand(total, device=module.weight.device)
                    continue
                if self.selection_method == "weight_l2":
                    importance[module] = structured_weight_l2(module.weight)
                    continue
                if module.weight.grad is None:
                    raise RuntimeError(f"Dense scoring produced no gradient for {type(module).__name__}.")
                if self.selection_method == "taylor":
                    importance[module] = structured_taylor(module.weight, module.weight.grad)
                else:
                    importance[module] = structured_gradient_l2(module.weight.grad)
        if not importance:
            raise TypeError("SSB V6 master has no supported Linear or Conv2d layers.")
        self._importance = importance

    def scoring_due(self) -> bool:
        return (
            not self.topology_frozen
            and self.optimizer_step_count % self.score_refresh_steps == 0
        )

    def _hidden_topology_snapshot(self):
        return [mapping.out_idx.detach().cpu().clone() for mapping in self._maps[:-1]]

    def _update_early_bird_state(self) -> None:
        current = self._hidden_topology_snapshot()
        if self._last_scored_topology is not None:
            distance = selected_unit_mask_distance(self._last_scored_topology, current)
            self.last_mask_distance = distance
            self.mask_distance_history.append(distance)
        self._last_scored_topology = current
        if (
            self.early_bird
            and len(self.mask_distance_history) == self.stability_window
            and max(self.mask_distance_history) <= self.stability_threshold
        ):
            self.topology_frozen = True
            self.topology_freeze_step = self.optimizer_step_count
            self.topology_freeze_scoring_event = self.scoring_event_count + 1

    def score_and_refresh(self, x, y, criterion, optimizer, learning_rate: float):
        """Score structured units and rebuild the selected child when due."""
        if not self.scoring_due():
            return optimizer, False

        # Preserve the latest child update and moments before using the dense master.
        if optimizer is not None:
            self.sync_optimizer_state_to_master(optimizer)
        self.sync_child_to_master()
        device = x.device
        _synchronize(device)
        scoring_start = time.perf_counter()
        dense_scoring_elapsed = 0.0
        self.master.zero_grad(set_to_none=True)
        if self.selection_method not in {"random", "weight_l2"}:
            dense_start = time.perf_counter()
            dense_loss = criterion(self.master(x), y)
            dense_loss.backward()
            _synchronize(device)
            dense_scoring_elapsed = time.perf_counter() - dense_start
        self._score_master()
        _synchronize(device)
        scoring_elapsed = time.perf_counter() - scoring_start
        self.master.zero_grad(set_to_none=True)

        _synchronize(device)
        rebuild_start = time.perf_counter()
        self.refresh_child()
        self._update_early_bird_state()
        new_optimizer = self.make_optimizer(learning_rate)
        _synchronize(device)
        rebuild_elapsed = time.perf_counter() - rebuild_start

        self.scoring_event_count += 1
        self.last_dense_scoring_time_s = dense_scoring_elapsed
        self.last_selector_scoring_time_s = scoring_elapsed
        self.last_child_rebuild_time_s = rebuild_elapsed
        self.selector_scoring_time_s += scoring_elapsed
        self.dense_scoring_time_s += dense_scoring_elapsed
        self.child_rebuild_time_s += rebuild_elapsed
        return new_optimizer, True

    def after_optimizer_step(self, optimizer, learning_rate: float):
        """Persist V5 Adam state; the next due scoring event owns topology refresh."""
        self.sync_optimizer_state_to_master(optimizer)
        self.optimizer_step_count += 1
        self._steps_since_refresh = self.optimizer_step_count % self.score_refresh_steps
        return optimizer

    def active_structured_units(self) -> int:
        return sum(int(mapping.out_idx.numel()) for mapping in self._maps[:-1])

    def total_structured_units(self) -> int:
        return sum(
            int(mapping.master.out_features if mapping.kind == "linear" else mapping.master.out_channels)
            for mapping in self._maps[:-1]
        )

    def effective_keep_ratio(self) -> float:
        total = self.total_structured_units()
        return self.active_structured_units() / total if total else 1.0

    def importance_statistics(self):
        hidden = []
        output_modules = set()
        if hasattr(self.master, "net"):
            linears = [m for m in self.master.net if isinstance(m, nn.Linear)]
            output_modules.update(linears[-1:])
        elif hasattr(self.master, "classifier"):
            linears = [m for m in self.master.classifier if isinstance(m, nn.Linear)]
            output_modules.update(linears[-1:])
        for module, scores in self._importance.items():
            if module not in output_modules:
                hidden.append(scores.flatten())
        if not hidden:
            return 0.0, 0.0, 0.0
        values = torch.cat(hidden)
        return float(values.min()), float(values.mean()), float(values.max())
