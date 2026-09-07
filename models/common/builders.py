from .mlp import DenseMLP, DropoutMLP, PrunedMLP, SparseMLP
from .cnn import DenseCNN, DropoutCNN, PrunedCNN, SparseCNN
from algorithms.ssb.v4 import StructuredChildModel
from algorithms.ssb.v5 import StructuredChildModelV5
from algorithms.ssb.v5_1 import StructuredBackwardModelV51


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


def build_structured_child(config, keep_ratio, architecture="mlp", refresh_steps=100):
    """Build SSB V4: full dense master + physically smaller structured child."""
    master = build_dense(config, architecture)
    return StructuredChildModel(master, keep_ratio=keep_ratio, refresh_steps=refresh_steps)


def build_structured_child_v5(config, keep_ratio, architecture="mlp", refresh_steps=100):
    """Build SSB V5: V4 structured child with master-owned persistent Adam state."""
    master = build_dense(config, architecture)
    return StructuredChildModelV5(master, keep_ratio=keep_ratio, refresh_steps=refresh_steps)


def build_structured_backward_v51(config, keep_ratio, architecture="mlp", refresh_steps=100):
    """Build SSB V5.1: full dense forward + structured smaller backward surrogate."""
    master = build_dense(config, architecture)
    return StructuredBackwardModelV51(master, keep_ratio=keep_ratio, refresh_steps=refresh_steps)
