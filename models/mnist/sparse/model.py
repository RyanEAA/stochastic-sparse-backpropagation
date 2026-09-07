from models.common.builders import build_sparse
from .. import config

def build(*, sparse_linear_cls, keep_ratio, architecture="mlp", sparse_layer_kwargs=None, **_):
    return build_sparse(config, sparse_linear_cls, keep_ratio, architecture, sparse_layer_kwargs)
