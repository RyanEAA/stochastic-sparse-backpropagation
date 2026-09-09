"""SSB V7: V6 selection with refresh-only master-state synchronization."""
from algorithms.ssb.v6 import GradientSelectedChildModelV6


class OptimizedSelectedChildModelV7(GradientSelectedChildModelV6):
    """Avoid V6's full master Adam scatter after every optimizer update.

    Child parameters and Adam moments remain authoritative while the topology is
    unchanged. ``score_and_refresh`` already synchronizes both to the master
    immediately before a scoring/rebuild event, so per-step synchronization is
    redundant. Once Early-Bird freezes the topology, it is never needed again.
    """

    is_v7_optimized = True

    def after_optimizer_step(self, optimizer, learning_rate: float):
        self.optimizer_step_count += 1
        self._steps_since_refresh = self.optimizer_step_count % self.score_refresh_steps
        return optimizer
