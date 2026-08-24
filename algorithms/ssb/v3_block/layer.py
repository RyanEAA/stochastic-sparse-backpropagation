import math

import torch
import torch.nn as nn

from algorithms.ssb.v3.function import SparseLinearFunctionV3


class BlockSparseLinearV3(nn.Module):
    """V3 SSB with contiguous block sparsity in forward and backward.

    Output neurons are partitioned into aligned contiguous blocks. During
    training, each block is independently selected with probability
    ``keep_ratio``. Only selected output rows participate in the forward
    matrix multiplication, and the same selected blocks participate in
    backward through ``SparseLinearFunctionV3``.

    Evaluation remains dense, matching neuron-level V3 semantics.
    """

    def __init__(self, in_features, out_features, keep_ratio=0.2, block_size=32):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in the range (0, 1].")
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer.")

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.keep_ratio = keep_ratio
        self.block_size = int(block_size)

    def _sample_mask(self):
        out_features = self.weight.size(0)
        num_blocks = math.ceil(out_features / self.block_size)
        block_mask = torch.rand(num_blocks, device=self.weight.device) < self.keep_ratio
        return block_mask.repeat_interleave(self.block_size)[:out_features]

    def forward(self, x):
        if not self.training:
            return x @ self.weight.t() + self.bias

        active_mask = self._sample_mask()
        return SparseLinearFunctionV3.apply(x, self.weight, self.bias, active_mask)
