"""SSB V7: fast selected-child training with optional hybrid dense updates."""
from __future__ import annotations

from collections.abc import Sequence
import time

import torch
from torch import nn, optim

from algorithms.ssb.v6 import GradientSelectedChildModelV6
from algorithms.ssb.v6.gradient_selected_child import _synchronize


class OptimizedSelectedChildModelV7(GradientSelectedChildModelV6):
    """Avoid V6's full master Adam scatter after every optimizer update.

    Child parameters and Adam moments remain authoritative while the topology is
    unchanged. ``score_and_refresh`` already synchronizes both to the master
    immediately before a scoring/rebuild event, so per-step synchronization is
    redundant. Once Early-Bird freezes the topology, it is never needed again.
    """

    is_v7_optimized = True

    def __init__(
        self,
        *args,
        dense_correction_steps: int = 0,
        layer_keep_ratios: Sequence[float] | None = None,
        early_bird_min_events: int = 0,
        early_bird_min_steps: int = 0,
        **kwargs,
    ):
        if dense_correction_steps < 0:
            raise ValueError("dense_correction_steps must be >= 0.")
        if early_bird_min_events < 0 or early_bird_min_steps < 0:
            raise ValueError("Early-Bird minimums must be >= 0.")
        super().__init__(*args, **kwargs)
        self.dense_correction_steps = int(dense_correction_steps)
        self.early_bird_min_events = int(early_bird_min_events)
        self.early_bird_min_steps = int(early_bird_min_steps)
        self.sparse_steps_since_dense_correction = 0
        self.dense_correction_count = 0

        hidden_layers = self._hidden_master_layers()
        if layer_keep_ratios is None:
            self.layer_keep_ratios = None
            self._layer_keep_ratio = {}
        else:
            ratios = tuple(float(ratio) for ratio in layer_keep_ratios)
            if (
                len(ratios) == 3
                and hasattr(self.master, "features")
                and hasattr(self.master, "classifier")
                and len(hidden_layers) != 3
            ):
                conv_count = sum(
                    isinstance(module, nn.Conv2d)
                    for module in self.master.features.net
                )
                linear_count = len(hidden_layers) - conv_count
                ratios = (
                    ratios[0],
                    *((ratios[1],) * max(0, conv_count - 1)),
                    *((ratios[2],) * linear_count),
                )
            if len(ratios) != len(hidden_layers):
                raise ValueError(
                    f"layer_keep_ratios has {len(ratios)} values, but this model has "
                    f"{len(hidden_layers)} hidden structured layers."
                )
            if any(ratio <= 0 or ratio > 1 for ratio in ratios):
                raise ValueError("Every layer keep ratio must be in (0, 1].")
            self.layer_keep_ratios = ratios
            self._layer_keep_ratio = dict(zip(hidden_layers, ratios))

    def _hidden_master_layers(self):
        if hasattr(self.master, "net"):
            layers = [module for module in self.master.net if isinstance(module, nn.Linear)]
            return layers[:-1]
        convs = [
            module for module in self.master.features.net
            if isinstance(module, nn.Conv2d)
        ]
        linears = [
            module for module in self.master.classifier
            if isinstance(module, nn.Linear)
        ]
        return convs + linears[:-1]

    def _select_out_indices(self, master_layer, *, is_output, device):
        if is_output or self.selection_mode == "gradient_retention":
            return super()._select_out_indices(
                master_layer, is_output=is_output, device=device
            )
        layer_ratio = self._layer_keep_ratio.get(master_layer)
        if layer_ratio is None:
            return super()._select_out_indices(
                master_layer, is_output=is_output, device=device
            )
        from algorithms.ssb.v6.gradient_selected_child import topk_structured_indices

        total = (
            master_layer.out_features
            if isinstance(master_layer, nn.Linear)
            else master_layer.out_channels
        )
        scores = self._importance.get(master_layer)
        if scores is None:
            scores = torch.zeros(total, device=device)
        return topk_structured_indices(scores.to(device), layer_ratio)

    def _update_early_bird_state(self) -> None:
        enabled = self.early_bird
        next_event = self.scoring_event_count + 1
        if (
            next_event < self.early_bird_min_events
            or self.optimizer_step_count < self.early_bird_min_steps
        ):
            self.early_bird = False
        try:
            super()._update_early_bird_state()
        finally:
            self.early_bird = enabled

    def after_optimizer_step(self, optimizer, learning_rate: float):
        self.optimizer_step_count += 1
        self.sparse_steps_since_dense_correction += 1
        self._steps_since_refresh = self.optimizer_step_count % self.score_refresh_steps
        return optimizer

    def dense_correction_due(self) -> bool:
        return (
            self.dense_correction_steps > 0
            and self.sparse_steps_since_dense_correction
            >= self.dense_correction_steps
        )

    def _copy_optimizer_state_to_master_store(self, optimizer) -> None:
        """Persist a full-master Adam optimizer into V5's master-shaped store."""
        with torch.no_grad():
            for parameter in self.master.parameters():
                source = optimizer.state.get(parameter)
                if not source or "exp_avg" not in source:
                    continue
                target = self._master_state_for(parameter)
                step = source.get("step", 0.0)
                target["step"] = float(step.item() if torch.is_tensor(step) else step)
                target["exp_avg"].copy_(source["exp_avg"])
                target["exp_avg_sq"].copy_(source["exp_avg_sq"])

    def make_master_optimizer(self, learning_rate: float):
        optimizer = optim.Adam(self.master.parameters(), lr=learning_rate)
        for parameter in self.master.parameters():
            state = self._master_state_for(parameter)
            optimizer.state[parameter] = {
                "step": torch.tensor(state["step"], device=parameter.device),
                "exp_avg": state["exp_avg"].clone(),
                "exp_avg_sq": state["exp_avg_sq"].clone(),
            }
        return optimizer

    def prepare_sparse_after_warmup(self, master_optimizer) -> None:
        """Keep dense warm-up weights/moments and force scoring on the next batch."""
        self._copy_optimizer_state_to_master_store(master_optimizer)
        self.child = None
        self._maps = []
        self.optimizer_step_count = 0
        self.sparse_steps_since_dense_correction = 0
        self.topology_frozen = False
        self.mask_distance_history.clear()
        self._last_scored_topology = None

    def prepare_dense_correction(self, child_optimizer, learning_rate: float):
        """Synchronize child state and return a hydrated full-master optimizer."""
        self.sync_optimizer_state_to_master(child_optimizer)
        self.sync_child_to_master()
        return self.make_master_optimizer(learning_rate)

    def finish_dense_correction(self, master_optimizer, learning_rate: float):
        """Persist a dense update, optionally rescore, and rebuild the child."""
        self._copy_optimizer_state_to_master_store(master_optimizer)
        device = next(self.master.parameters()).device
        selector_elapsed = 0.0
        if not self.topology_frozen:
            _synchronize(device)
            selector_start = time.perf_counter()
            self._score_master()
            _synchronize(device)
            selector_elapsed = time.perf_counter() - selector_start
            # The dense optimizer now owns the newest active weights. Prevent
            # refresh_child() from scattering the pre-correction child over them.
            self.child = None
            rebuild_start = time.perf_counter()
            self.refresh_child()
            self._update_early_bird_state()
            self.scoring_event_count += 1
        else:
            # Reuse the frozen importance/topology while gathering updated weights.
            self.child = None
            _synchronize(device)
            rebuild_start = time.perf_counter()
            self.refresh_child()
        new_optimizer = self.make_optimizer(learning_rate)
        _synchronize(device)
        rebuild_elapsed = time.perf_counter() - rebuild_start
        self.master.zero_grad(set_to_none=True)
        self.optimizer_step_count += 1
        self.sparse_steps_since_dense_correction = 0
        self.dense_correction_count += 1
        self.last_selector_scoring_time_s = selector_elapsed
        self.last_dense_scoring_time_s = 0.0
        self.last_child_rebuild_time_s = rebuild_elapsed
        self.selector_scoring_time_s += selector_elapsed
        self.child_rebuild_time_s += rebuild_elapsed
        return new_optimizer
