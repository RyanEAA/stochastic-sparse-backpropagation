from models.common.mlp import PrunedMLP
from ..config import INPUT_FEATURES, HIDDEN_DIMS, NUM_CLASSES

def build(*, keep_ratio, **_):
    return PrunedMLP(INPUT_FEATURES, HIDDEN_DIMS, NUM_CLASSES, keep_ratio)
