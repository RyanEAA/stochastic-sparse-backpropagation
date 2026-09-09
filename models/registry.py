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
    "ssb-v4",
    "ssb-v5",
    "ssb-v5.1",
    "ssb-v6",
    "ssb-v7",
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
    child_refresh_steps: int = 100,
    score_refresh_steps: int = 100,
    v6_selection_mode: str = "fixed",
    v6_gradient_retention: float = 0.90,
    v6_selection_method: str = "gradient_l2",
    v6_early_bird: bool = False,
    v6_stability_window: int = 5,
    v6_stability_threshold: float = 0.10,
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
    if model in {"ssb-v4", "ssb-v5", "ssb-v5.1", "ssb-v6", "ssb-v7"}:
        config = importlib.import_module(f"models.{dataset}.config")
        from models.common.builders import build_structured_child, build_structured_child_v5, build_structured_backward_v51, build_gradient_selected_v6, build_optimized_selected_v7
        if model in {"ssb-v6", "ssb-v7"}:
            builder = build_optimized_selected_v7 if model == "ssb-v7" else build_gradient_selected_v6
            return builder(
                config, keep_ratio=keep_ratio, architecture=architecture,
                score_refresh_steps=score_refresh_steps,
                selection_mode=v6_selection_mode,
                gradient_retention=v6_gradient_retention,
                selection_method=v6_selection_method,
                early_bird=v6_early_bird,
                stability_window=v6_stability_window,
                stability_threshold=v6_stability_threshold,
            )
        builder = build_structured_backward_v51 if model == "ssb-v5.1" else (build_structured_child_v5 if model == "ssb-v5" else build_structured_child)
        return builder(
            config, keep_ratio=keep_ratio, architecture=architecture,
            refresh_steps=child_refresh_steps,
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
