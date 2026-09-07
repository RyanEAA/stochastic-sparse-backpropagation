from models.common.builders import build_dense
from .. import config

def build(*, architecture="mlp", **_):
    return build_dense(config, architecture)
