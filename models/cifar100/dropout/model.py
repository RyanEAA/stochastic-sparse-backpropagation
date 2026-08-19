from models.common.mlp import DropoutMLP
from ..config import INPUT_FEATURES, HIDDEN_DIMS, NUM_CLASSES

def build(*, keep_ratio, **_):
    return DropoutMLP(INPUT_FEATURES, HIDDEN_DIMS, NUM_CLASSES, keep_ratio)
