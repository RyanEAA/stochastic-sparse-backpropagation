from argparse import Namespace
from pathlib import Path

from master_train import build_jobs


def _args():
    return Namespace(
        architectures=["cnn"],
        models=["dense", "ssb-v7"],
        keep_ratios=[0.2],
        block_sizes=[32],
        child_refresh_steps=[25],
        score_refresh_steps=[25, 100],
        v6_keep_ratio=0.2,
        v6_selection_methods=["random", "gradient_l2", "weight_l2", "taylor"],
        v7_hybrid_configs=[
            "layerwise", "layerwise_warmup", "layerwise_correction", "layerwise_hybrid"
        ],
        v7_target_parameter_ratio=0.2,
        runs=1,
        results_dir=Path("results/test-grid"),
        protocol_version="protocol-test",
    )


def test_v7_factorial_grid_has_one_dense_and_unique_output_directories():
    jobs = build_jobs(_args(), ["mnist"])
    assert len(jobs) == 33
    assert sum(job[2] == "dense" for job in jobs) == 1
    output_directories = [job[-1] for job in jobs]
    assert len(output_directories) == len(set(output_directories))


def test_six_dataset_screening_grid_has_expected_run_count():
    datasets = ["mnist", "fashion_mnist", "kmnist", "cifar10", "cifar100", "svhn"]
    jobs = build_jobs(_args(), datasets)
    assert len(jobs) == 198
    assert sum(job[2] == "dense" for job in jobs) == 6
