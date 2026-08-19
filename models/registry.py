import importlib
from algorithms.ssb import get_sparse_linear
from data.loaders import SUPPORTED_DATASETS

AVAILABLE_DATASETS = SUPPORTED_DATASETS
AVAILABLE_MODELS = ("dense", "dropout", "pruning", "ssb-v0", "ssb-v1", "ssb-v2")

def build_model(dataset: str, model: str, keep_ratio: float = 1.0):
    dataset = dataset.lower().replace('-', '_')
    model = model.lower()
    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {', '.join(AVAILABLE_DATASETS)}")
    if model == "dense":
        return importlib.import_module(f"models.{dataset}.dense.model").build()
    if model == "dropout":
        return importlib.import_module(f"models.{dataset}.dropout.model").build(keep_ratio=keep_ratio)
    if model == "pruning":
        return importlib.import_module(f"models.{dataset}.pruning.model").build(keep_ratio=keep_ratio)
    if model.startswith("ssb-"):
        layer = get_sparse_linear(model)
        return importlib.import_module(f"models.{dataset}.sparse.model").build(sparse_linear_cls=layer, keep_ratio=keep_ratio)
    raise ValueError(f"Unknown model {model!r}. Available: {', '.join(AVAILABLE_MODELS)}")
