from models.common.mlp import SparseMLP
from ..config import INPUT_FEATURES, HIDDEN_DIMS, NUM_CLASSES

def build(*, sparse_linear_cls, keep_ratio, **_):
    return SparseMLP(INPUT_FEATURES, HIDDEN_DIMS, NUM_CLASSES, sparse_linear_cls, keep_ratio)
