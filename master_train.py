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
    "ssb-v1-block",
    "ssb-v2-block",
    "ssb-v3-block",
]
BLOCK_MODELS = {"ssb-v1-block", "ssb-v2-block", "ssb-v3-block"}


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


def build_jobs(args, datasets):
    jobs = []

    for dataset in datasets:
        for architecture in args.architectures:
            for model in args.models:
                if model == "dense":
                    # Dense has no keep-ratio/block-size/refresh dimension.
                    configurations = [(1.0, 0, 0)]
                elif model in BLOCK_MODELS:
                    configurations = [
                        (ratio, block_size, 0)
                        for ratio in args.keep_ratios
                        for block_size in args.block_sizes
                    ]
                elif model in {"ssb-v4", "ssb-v5", "ssb-v5.1"}:
                    configurations = [
                        (ratio, 0, refresh_steps)
                        for ratio in args.keep_ratios
                        for refresh_steps in args.child_refresh_steps
                    ]
                elif model == "ssb-v6":
                    configurations = [
                        (1.0, 0, score_steps)
                        for score_steps in args.score_refresh_steps
                    ]
                else:
                    configurations = [(ratio, 0, 0) for ratio in args.keep_ratios]

                for ratio, block_size, refresh_steps in configurations:
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
                        elif model == "ssb-v6":
                            retention = str(args.v6_gradient_retention).replace('.', '_')
                            leaf = f"gradient_retention_{retention}/score_refresh_{refresh_steps}/seed_{seed:02d}"
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
                                seed,
                                model_root / leaf,
                            )
                        )

    return jobs


def print_plan(jobs, args):
    counts = Counter()
    for _, _, model, _, _, _, _, _ in jobs:
        if model == "dense":
            counts["dense"] += 1
        elif model in BLOCK_MODELS:
            counts["block_ssb"] += 1
        elif model in {"dropout", "pruning"}:
            counts["baselines"] += 1
        elif model in {"ssb-v4", "ssb-v5", "ssb-v5.1", "ssb-v6"}:
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
    print(f"  structured V4/5/5.1/6: {counts['structured_child']}")
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

    for index, (dataset, architecture, model, ratio, block_size, refresh_steps, seed, output_dir) in enumerate(jobs, start=1):
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
            "--protocol-version", args.protocol_version,
            "--experiment-tag", args.experiment_tag,
            "--output-dir", str(output_dir),
        ]

        if args.stop_at_convergence:
            command.append("--stop-at-convergence")

        # Pass only parameters that are meaningful for this model family.
        if model != "dense":
            command.extend(["--keep-ratio", str(ratio)])
        if model in BLOCK_MODELS:
            command.extend(["--block-size", str(block_size)])
        if model in {"ssb-v4", "ssb-v5", "ssb-v5.1"}:
            command.extend(["--child-refresh-steps", str(refresh_steps)])
        if model == "ssb-v6":
            command.extend([
                "--score-refresh-steps", str(refresh_steps),
                "--v6-selection-mode", "gradient_retention",
                "--v6-gradient-retention", str(args.v6_gradient_retention),
            ])

        print()
        print(f"[{index}/{len(jobs)}] {' '.join(command)}")
        subprocess.run(command, check=True)

    print()
    print("All requested raw runs complete. Run summarize_results.py afterwards.")


if __name__ == "__main__":
    main()
