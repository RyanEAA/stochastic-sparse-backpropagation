from models.common.builders import build_pruning
from .. import config

def build(*, keep_ratio, architecture="mlp", **_):
    return build_pruning(config, keep_ratio, architecture)
