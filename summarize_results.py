import argparse
from pathlib import Path

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    root = args.results_dir / args.dataset if args.dataset else args.results_dir
    epoch_files = list(root.rglob("epochs.csv"))
    batch_files = list(root.rglob("batches.csv"))
    if not epoch_files or not batch_files:
        raise SystemExit(f"No results found under {root}")

    epochs = pd.concat((pd.read_csv(path) for path in epoch_files), ignore_index=True)
    batches = pd.concat((pd.read_csv(path) for path in batch_files), ignore_index=True)
    keys = ["dataset", "model", "keep_ratio", "seed"]

    final_epoch = epochs.sort_values("epoch").groupby(keys, as_index=False).tail(1)
    per_run_batches = batches.groupby(keys, as_index=False).agg(
        mean_backward_ms=("backward_time_s", lambda s: s.mean() * 1000.0),
        mean_memory_mb=("memory_bytes", lambda s: s.mean() / (1024 ** 2)),
    )
    per_run = final_epoch.merge(per_run_batches, on=keys, how="inner")

    summary = per_run.groupby(["dataset", "model", "keep_ratio"], as_index=False).agg(
        runs=("seed", "count"),
        val_accuracy_mean=("val_accuracy", "mean"),
        val_accuracy_std=("val_accuracy", "std"),
        train_accuracy_mean=("train_accuracy", "mean"),
        train_accuracy_std=("train_accuracy", "std"),
        backward_ms_mean=("mean_backward_ms", "mean"),
        backward_ms_std=("mean_backward_ms", "std"),
        memory_mb_mean=("mean_memory_mb", "mean"),
        memory_mb_std=("mean_memory_mb", "std"),
        epoch_time_mean=("epoch_time_s", "mean"),
        epoch_time_std=("epoch_time_s", "std"),
    )

    output = root / "summary.csv"
    summary.to_csv(output, index=False)
    per_run.to_csv(root / "per_run.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
