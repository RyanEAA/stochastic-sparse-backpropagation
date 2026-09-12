"""Run meaningful experiment grids while keeping raw results separated by architecture.

Grid rules:
- dense: one run per (dataset, architecture, seed)
- dropout/pruning/neuron-level SSB: sweep keep ratio
- block SSB: sweep keep ratio x block size

SSB V0 is historical and intentionally excluded from the default model list. It can
still be requested explicitly with ``--models ssb-v0 ...``.

Means/std are NOT calculated here; run summarize_results.py after raw runs finish.
"""
import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

from data.loaders import SUPPORTED_DATASETS, normalize_dataset_name
from models import AVAILABLE_MODELS, AVAILABLE_ARCHITECTURES
from train import PROTOCOL_VERSION

# Five-point sweep used for the clean scaling benchmark.
DEFAULT_RATIOS = [1.0, 0.8, 0.6, 0.4, 0.2]
DEFAULT_BLOCK_SIZES = [8, 16, 32, 64, 128]
DEFAULT_MODELS = [
    "dense",
    "dropout",
    "pruning",
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
]
BLOCK_MODELS = {"ssb-v1-block", "ssb-v2-block", "ssb-v3-block"}
V7_HYBRID_CONFIGS = (
    "pure",
    "warmup",
    "correction",
    "warmup_correction",
    "layerwise",
    "layerwise_warmup",
    "layerwise_correction",
    "layerwise_hybrid",
)


def keep_dir(ratio):
    return f"keep_{str(ratio).replace('.', '_')}"


def validate_args(args):
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be >= 1")
    if args.patience < 1:
        raise ValueError("--patience must be >= 1")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be >= 0")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.subset < 0:
        raise ValueError("--subset must be >= 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if any(ratio <= 0 or ratio > 1 for ratio in args.keep_ratios):
        raise ValueError("all --keep-ratios must satisfy 0 < ratio <= 1")
    if any(size < 1 for size in args.block_sizes):
        raise ValueError("all --block-sizes must be >= 1")
    if any(step < 1 for step in args.child_refresh_steps):
        raise ValueError("all --child-refresh-steps must be >= 1")
    if any(step < 1 for step in args.score_refresh_steps):
        raise ValueError("all --score-refresh-steps must be >= 1")
    if not 0 < args.v6_gradient_retention <= 1:
        raise ValueError("--v6-gradient-retention must be in (0, 1]")
    if not 0 < args.v6_keep_ratio <= 1:
        raise ValueError("--v6-keep-ratio must be in (0, 1]")
    if args.v6_stability_window < 1:
        raise ValueError("--v6-stability-window must be >= 1")
    if not 0 <= args.v6_stability_threshold <= 1:
        raise ValueError("--v6-stability-threshold must be in [0, 1]")
    if args.v7_warmup_epochs < 0:
        raise ValueError("--v7-warmup-epochs must be >= 0")
    if args.v7_correction_steps < 1:
        raise ValueError("--v7-correction-steps must be >= 1")
    if args.v7_early_bird_min_events < 0 or args.v7_early_bird_min_steps < 0:
        raise ValueError("V7 Early-Bird minimums must be >= 0")
    if any(ratio <= 0 or ratio > 1 for ratio in args.v7_layer_keep_ratios):
        raise ValueError("all --v7-layer-keep-ratios must satisfy 0 < ratio <= 1")
    if args.v7_target_parameter_ratio is not None and not 0 < args.v7_target_parameter_ratio <= 1:
        raise ValueError("--v7-target-parameter-ratio must be in (0, 1]")


def build_jobs(args, datasets):
    jobs = []

    for dataset in datasets:
        for architecture in args.architectures:
            for model in args.models:
                if model == "dense":
                    # Dense has no keep-ratio/block-size/refresh dimension.
                    configurations = [(1.0, 0, 0, "none", "none")]
                elif model in BLOCK_MODELS:
                    configurations = [
                        (ratio, block_size, 0, "none", "none")
                        for ratio in args.keep_ratios
                        for block_size in args.block_sizes
                    ]
                elif model in {"ssb-v4", "ssb-v5", "ssb-v5.1"}:
                    configurations = [
                        (ratio, 0, refresh_steps, "none", "none")
                        for ratio in args.keep_ratios
                        for refresh_steps in args.child_refresh_steps
                    ]
                elif model == "ssb-v6":
                    configurations = [
                        (args.v6_keep_ratio, 0, score_steps, selector, "none")
                        for score_steps in args.score_refresh_steps
                        for selector in args.v6_selection_methods
                    ]
                elif model == "ssb-v7":
                    configurations = [
                        (args.v6_keep_ratio, 0, score_steps, selector, hybrid)
                        for score_steps in args.score_refresh_steps
                        for selector in args.v6_selection_methods
                        for hybrid in args.v7_hybrid_configs
                    ]
                else:
                    configurations = [(ratio, 0, 0, "none", "none") for ratio in args.keep_ratios]

                for ratio, block_size, refresh_steps, selector, hybrid in configurations:
                    for seed in range(1, args.runs + 1):
                        model_root = (
                            args.results_dir
                            / dataset
                            / model
                            / f"architecture_{architecture}"
                            / args.protocol_version
                        )

                        if model == "dense":
                            leaf = f"seed_{seed:02d}"
                        elif model in BLOCK_MODELS:
                            leaf = (
                                f"{keep_dir(ratio)}/block_{block_size}/seed_{seed:02d}"
                            )
                        elif model in {"ssb-v4", "ssb-v5", "ssb-v5.1"}:
                            leaf = f"{keep_dir(ratio)}/refresh_{refresh_steps}/seed_{seed:02d}"
                        elif model in {"ssb-v6", "ssb-v7"}:
                            hybrid_dir = f"/hybrid_{hybrid}" if model == "ssb-v7" else ""
                            budget_dir = (
                                f"/parameter_budget_{str(args.v7_target_parameter_ratio).replace('.', '_')}"
                                if model == "ssb-v7" and args.v7_target_parameter_ratio is not None
                                else ""
                            )
                            leaf = f"selector_{selector}/{keep_dir(ratio)}{budget_dir}/score_refresh_{refresh_steps}{hybrid_dir}/seed_{seed:02d}"
                        else:
                            leaf = f"{keep_dir(ratio)}/seed_{seed:02d}"

                        jobs.append(
                            (
                                dataset,
                                architecture,
                                model,
                                ratio,
                                block_size,
                                refresh_steps,
                                selector,
                                hybrid,
                                seed,
                                model_root / leaf,
                            )
                        )

    return jobs


def print_plan(jobs, args):
    counts = Counter()
    for _, _, model, _, _, _, _, _, _, _ in jobs:
        if model == "dense":
            counts["dense"] += 1
        elif model in BLOCK_MODELS:
            counts["block_ssb"] += 1
        elif model in {"dropout", "pruning"}:
            counts["baselines"] += 1
        elif model in {"ssb-v4", "ssb-v5", "ssb-v5.1", "ssb-v6", "ssb-v7"}:
            counts["structured_child"] += 1
        else:
            counts["neuron_ssb"] += 1

    print("Experiment plan")
    print(f"  datasets:       {len(set(job[0] for job in jobs))}")
    print(f"  architectures:  {len(set(job[1] for job in jobs))}")
    print(f"  seeds/config:   {args.runs}")
    print(f"  dense runs:     {counts['dense']}")
    print(f"  baseline runs:  {counts['baselines']}")
    print(f"  neuron SSB:     {counts['neuron_ssb']}")
    print(f"  structured V4/5/5.1/6/7: {counts['structured_child']}")
    print(f"  block SSB:      {counts['block_ssb']}")
    print(f"Planned experiments: {len(jobs)}")

    if "ssb-v0" not in args.models:
        print("  note: ssb-v0 is historical and excluded unless explicitly requested")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(SUPPORTED_DATASETS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=AVAILABLE_MODELS,
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(AVAILABLE_ARCHITECTURES),
        choices=AVAILABLE_ARCHITECTURES,
    )
    parser.add_argument(
        "--keep-ratios", nargs="+", type=float, default=DEFAULT_RATIOS
    )
    parser.add_argument(
        "--block-sizes", nargs="+", type=int, default=DEFAULT_BLOCK_SIZES
    )
    parser.add_argument("--child-refresh-steps", nargs="+", type=int, default=[1, 10, 25, 100], help="For V4/V5 only: resample the structured child exactly every N optimizer steps.")
    parser.add_argument("--score-refresh-steps", nargs="+", type=int, default=[10, 25, 100], help="For V6 only: dense-score and rebuild the gradient-ranked child every N optimizer steps.")
    parser.add_argument("--v6-gradient-retention", type=float, default=0.90, help="Dynamic V6 retains this fraction of per-layer squared gradient energy.")
    parser.add_argument("--v6-keep-ratio", type=float, default=0.20, help="Fixed V6 child width ratio (independent of other model sweeps).")
    parser.add_argument("--v6-selection-methods", nargs="+", choices=["random", "gradient_l2", "weight_l2", "taylor"], default=["weight_l2", "taylor"])
    parser.add_argument("--v6-disable-early-bird", action="store_true", help="Keep refreshing V6 masks for the whole run.")
    parser.add_argument("--v6-stability-window", type=int, default=5)
    parser.add_argument("--v6-stability-threshold", type=float, default=0.10)
    parser.add_argument("--v7-hybrid-configs", nargs="+", choices=V7_HYBRID_CONFIGS, default=["pure"], help="Named V7 accuracy/speed configurations to compare without creating an unintended Cartesian grid.")
    parser.add_argument("--v7-warmup-epochs", type=int, default=1, help="Dense warm-up used by V7 warmup/hybrid configurations.")
    parser.add_argument("--v7-correction-steps", type=int, default=25, help="Sparse updates between dense corrective batches in V7 correction/hybrid configurations.")
    parser.add_argument("--v7-layer-keep-ratios", nargs="+", type=float, default=[1.0, 0.5, 0.2], help="Forward-order hidden-layer ratios for layerwise_hybrid; default matches the current two-conv/one-hidden CNN.")
    parser.add_argument("--v7-target-parameter-ratio", type=float, default=None, help="Calibrate the V7 layer profile per architecture so the structured child is as close as possible to this child/master parameter ratio.")
    parser.add_argument("--v7-early-bird-min-events", type=int, default=10)
    parser.add_argument("--v7-early-bird-min-steps", type=int, default=0)
    parser.add_argument("--timing-detail", choices=["basic", "full"], default="basic")
    parser.add_argument("--record-batch-metrics", action="store_true")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stop-at-convergence", action="store_true", help="Use validation-loss early stopping instead of a fixed epoch count.")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned grid without launching train.py")
    args = parser.parse_args()

    validate_args(args)
    datasets = [normalize_dataset_name(dataset) for dataset in args.datasets]
    jobs = build_jobs(args, datasets)
    print_plan(jobs, args)
    if args.dry_run:
        return

    for index, (dataset, architecture, model, ratio, block_size, refresh_steps, selector, hybrid, seed, output_dir) in enumerate(jobs, start=1):
        required = [
            output_dir / "epochs.csv",
            output_dir / "batches.csv",
            output_dir / "metadata.json",
        ]
        if not args.force and all(path.exists() for path in required):
            print(f"[{index}/{len(jobs)}] skip {output_dir}")
            continue

        command = [
            sys.executable,
            "train.py",
            "--dataset", dataset,
            "--model", model,
            "--architecture", architecture,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--max-epochs", str(args.max_epochs),
            "--patience", str(args.patience),
            "--min-delta", str(args.min_delta),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--subset", str(args.subset),
            "--device", args.device,
            "--data-dir", args.data_dir,
            "--num-workers", str(args.num_workers),
            "--validation-fraction", str(args.validation_fraction),
            "--split-seed", str(args.split_seed),
            "--protocol-version", args.protocol_version,
            "--experiment-tag", args.experiment_tag,
            "--timing-detail", args.timing_detail,
            "--output-dir", str(output_dir),
        ]

        if args.stop_at_convergence:
            command.append("--stop-at-convergence")
        if args.record_batch_metrics:
            command.append("--record-batch-metrics")

        # Pass only parameters that are meaningful for this model family.
        if model != "dense":
            command.extend(["--keep-ratio", str(ratio)])
        if model in BLOCK_MODELS:
            command.extend(["--block-size", str(block_size)])
        if model in {"ssb-v4", "ssb-v5", "ssb-v5.1"}:
            command.extend(["--child-refresh-steps", str(refresh_steps)])
        if model in {"ssb-v6", "ssb-v7"}:
            command.extend([
                "--score-refresh-steps", str(refresh_steps),
                "--v6-selection-mode", "fixed",
                "--v6-gradient-retention", str(args.v6_gradient_retention),
                "--v6-selection-method", selector,
                "--v6-stability-window", str(args.v6_stability_window),
                "--v6-stability-threshold", str(args.v6_stability_threshold),
            ])
            if not args.v6_disable_early_bird:
                command.append("--v6-early-bird")
        if model == "ssb-v7":
            command.extend(["--v7-hybrid-config-label", hybrid])
        if model == "ssb-v7" and hybrid != "pure":
            if hybrid in {
                "warmup", "warmup_correction", "layerwise_warmup",
                "layerwise_hybrid",
            }:
                command.extend(["--v7-dense-warmup-epochs", str(args.v7_warmup_epochs)])
            if hybrid in {
                "correction", "warmup_correction", "layerwise_correction",
                "layerwise_hybrid",
            }:
                command.extend(["--v7-dense-correction-steps", str(args.v7_correction_steps)])
            if hybrid in {
                "layerwise", "layerwise_warmup", "layerwise_correction",
                "layerwise_hybrid",
            }:
                command.append("--v7-layer-keep-ratios")
                command.extend(str(ratio) for ratio in args.v7_layer_keep_ratios)
            command.extend([
                "--v7-early-bird-min-events", str(args.v7_early_bird_min_events),
                "--v7-early-bird-min-steps", str(args.v7_early_bird_min_steps),
            ])
        if model == "ssb-v7" and args.v7_target_parameter_ratio is not None:
            command.extend([
                "--v7-target-parameter-ratio", str(args.v7_target_parameter_ratio)
            ])

        print()
        print(f"[{index}/{len(jobs)}] {' '.join(command)}")
        subprocess.run(command, check=True)

    print()
    print("All requested raw runs complete. Run summarize_results.py afterwards.")


if __name__ == "__main__":
    main()
