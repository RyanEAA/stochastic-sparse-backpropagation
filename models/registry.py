import importlib

from algorithms.ssb import get_sparse_linear
from data.loaders import SUPPORTED_DATASETS

AVAILABLE_DATASETS = SUPPORTED_DATASETS
AVAILABLE_MODELS = (
    "dense",
    "dropout",
    "pruning",
    "ssb-v0",
    "ssb-v1",
    "ssb-v2",
    "ssb-v3",
    "ssb-v1-block",
    "ssb-v2-block",
    "ssb-v3-block",
)
AVAILABLE_ARCHITECTURES = ("mlp", "cnn")


def build_model(
    dataset: str,
    model: str,
    keep_ratio: float = 1.0,
    architecture: str = "mlp",
    block_size: int = 32,
):
    dataset = dataset.lower().replace('-', '_')
    model = model.lower()
    architecture = architecture.lower()
    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {', '.join(AVAILABLE_DATASETS)}")
    if architecture not in AVAILABLE_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. Available: {', '.join(AVAILABLE_ARCHITECTURES)}"
        )

    kwargs = {"architecture": architecture}
    if model == "dense":
        return importlib.import_module(f"models.{dataset}.dense.model").build(**kwargs)
    if model == "dropout":
        return importlib.import_module(f"models.{dataset}.dropout.model").build(
            keep_ratio=keep_ratio, **kwargs
        )
    if model == "pruning":
        return importlib.import_module(f"models.{dataset}.pruning.model").build(
            keep_ratio=keep_ratio, **kwargs
        )
    if model.startswith("ssb-"):
        layer = get_sparse_linear(model)
        sparse_layer_kwargs = {"block_size": block_size} if model in {"ssb-v1-block", "ssb-v2-block", "ssb-v3-block"} else None
        return importlib.import_module(f"models.{dataset}.sparse.model").build(
            sparse_linear_cls=layer,
            keep_ratio=keep_ratio,
            sparse_layer_kwargs=sparse_layer_kwargs,
            **kwargs,
        )
    raise ValueError(f"Unknown model {model!r}. Available: {', '.join(AVAILABLE_MODELS)}")
