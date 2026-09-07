from models.common.builders import build_dropout
from .. import config

def build(*, keep_ratio, architecture="mlp", **_):
    return build_dropout(config, keep_ratio, architecture)
