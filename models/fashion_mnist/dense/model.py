from models.common.mlp import DenseMLP
from ..config import INPUT_FEATURES, HIDDEN_DIMS, NUM_CLASSES

def build(**_):
    return DenseMLP(INPUT_FEATURES, HIDDEN_DIMS, NUM_CLASSES)
