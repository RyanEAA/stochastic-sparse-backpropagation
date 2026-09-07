"""SSB V5.1: dense forward prediction with a structured sparse backward surrogate.

Forward values come from the full dense master network. During training, autograd is
attached only to a physically smaller V5-style structured child. The returned tensor
uses a straight-through/surrogate bridge so its numerical logits equal the dense
master logits while gradients flow through the child graph only.

Because the next batch's prediction uses the dense master, active child weights are
scattered into the master after every optimizer step. Adam moments remain master-owned
and persistent exactly as in V5.
"""
from __future__ import annotations

import torch

from algorithms.ssb.v5 import StructuredChildModelV5


class StructuredBackwardModelV51(StructuredChildModelV5):
    """Full dense forward; physically smaller structured child for parameter gradients."""

    is_v51_structured_backward = True

    def forward(self, x):
        if not self.training:
            return self.master(x)
        if self.child is None:
            raise RuntimeError("Structured backward child has not been initialized.")

        # Dense prediction is the actual forward value and carries no dense autograd graph.
        with torch.no_grad():
            dense_logits = self.master(x)

        # This auxiliary child execution constructs only the sparse backward graph.
        child_logits = self.child(x)

        # Numerically identical to dense_logits, but d(output)/d(child_logits) = 1.
        return dense_logits + (child_logits - child_logits.detach())

    def after_optimizer_step(self, optimizer, learning_rate: float):
        """Persist child update into dense master every step; resample every N steps."""
        self.sync_optimizer_state_to_master(optimizer)
        # V5.1 must make the updated parameters visible to the next dense forward.
        self.sync_child_to_master()

        self.optimizer_step_count += 1
        self._steps_since_refresh = self.optimizer_step_count % self.refresh_steps
        if self.optimizer_step_count % self.refresh_steps != 0:
            return optimizer

        # Child values are already synchronized. refresh_child() will safely sync again,
        # sample a fresh topology from the updated master, and reset local refresh state.
        self.refresh_child()
        return self.make_optimizer(learning_rate)
