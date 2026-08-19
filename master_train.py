"""Run many datasets/models while keeping raw results separated.

Dense runs once per seed. Keep-ratio models run once per
(dataset, model, keep_ratio, seed). Means/std are NOT calculated here.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from data.loaders import SUPPORTED_DATASETS
from models import AVAILABLE_MODELS

DEFAULT_RATIOS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

def keep_dir(ratio):
    return f"keep_{str(ratio).replace('.', '_')}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(SUPPORTED_DATASETS))
    parser.add_argument("--models", nargs="+", default=["dense", "dropout", "pruning", "ssb-v0", "ssb-v1", "ssb-v2"], choices=AVAILABLE_MODELS)
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jobs = []
    for dataset in args.datasets:
        for model in args.models:
            ratios = [1.0] if model == "dense" else args.keep_ratios
            for ratio in ratios:
                for seed in range(1, args.runs + 1):
                    leaf = f"seed_{seed:02d}" if model == "dense" else f"{keep_dir(ratio)}/seed_{seed:02d}"
                    output_dir = args.results_dir / dataset / model / leaf
                    jobs.append((dataset, model, ratio, seed, output_dir))

    print(f"Planned experiments: {len(jobs)}")
    for index, (dataset, model, ratio, seed, output_dir) in enumerate(jobs, start=1):
        if not args.force and (output_dir / "epochs.csv").exists() and (output_dir / "batches.csv").exists():
            print(f"[{index}/{len(jobs)}] skip {output_dir}")
            continue

        command = [
            sys.executable, "train.py",
            "--dataset", dataset,
            "--model", model,
            "--keep-ratio", str(ratio),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--subset", str(args.subset),
            "--device", args.device,
            "--data-dir", args.data_dir,
            "--output-dir", str(output_dir),
        ]
        print()
        print(f"[{index}/{len(jobs)}] {' '.join(command)}")
        subprocess.run(command, check=True)

    print()
    print("All requested raw runs complete. Run summarize_results.py afterwards.")

if __name__ == "__main__":
    main()
