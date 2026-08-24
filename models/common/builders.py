from .mlp import DenseMLP, DropoutMLP, PrunedMLP, SparseMLP
from .cnn import DenseCNN, DropoutCNN, PrunedCNN, SparseCNN


def build_dense(config, architecture="mlp"):
    if architecture == "mlp":
        return DenseMLP(config.INPUT_FEATURES, config.HIDDEN_DIMS, config.NUM_CLASSES)
    if architecture == "cnn":
        return DenseCNN(
            config.INPUT_CHANNELS,
            config.INPUT_SIZE,
            config.CNN_CHANNELS,
            config.CNN_CLASSIFIER_HIDDEN,
            config.NUM_CLASSES,
            config.CNN_POOLED_SIZE,
        )
    raise ValueError(f"Unknown architecture {architecture!r}")


def build_dropout(config, keep_ratio, architecture="mlp"):
    if architecture == "mlp":
        return DropoutMLP(config.INPUT_FEATURES, config.HIDDEN_DIMS, config.NUM_CLASSES, keep_ratio)
    if architecture == "cnn":
        return DropoutCNN(
            config.INPUT_CHANNELS,
            config.INPUT_SIZE,
            config.CNN_CHANNELS,
            config.CNN_CLASSIFIER_HIDDEN,
            config.NUM_CLASSES,
            keep_ratio,
            config.CNN_POOLED_SIZE,
        )
    raise ValueError(f"Unknown architecture {architecture!r}")


def build_pruning(config, keep_ratio, architecture="mlp"):
    if architecture == "mlp":
        return PrunedMLP(config.INPUT_FEATURES, config.HIDDEN_DIMS, config.NUM_CLASSES, keep_ratio)
    if architecture == "cnn":
        return PrunedCNN(
            config.INPUT_CHANNELS,
            config.INPUT_SIZE,
            config.CNN_CHANNELS,
            config.CNN_CLASSIFIER_HIDDEN,
            config.NUM_CLASSES,
            keep_ratio,
            config.CNN_POOLED_SIZE,
        )
    raise ValueError(f"Unknown architecture {architecture!r}")


def build_sparse(
    config,
    sparse_linear_cls,
    keep_ratio,
    architecture="mlp",
    sparse_layer_kwargs=None,
):
    if architecture == "mlp":
        return SparseMLP(
            config.INPUT_FEATURES,
            config.HIDDEN_DIMS,
            config.NUM_CLASSES,
            sparse_linear_cls,
            keep_ratio,
            sparse_layer_kwargs=sparse_layer_kwargs,
        )
    if architecture == "cnn":
        return SparseCNN(
            config.INPUT_CHANNELS,
            config.INPUT_SIZE,
            config.CNN_CHANNELS,
            config.CNN_CLASSIFIER_HIDDEN,
            config.NUM_CLASSES,
            sparse_linear_cls,
            keep_ratio,
            config.CNN_POOLED_SIZE,
            sparse_layer_kwargs=sparse_layer_kwargs,
        )
    raise ValueError(f"Unknown architecture {architecture!r}")
