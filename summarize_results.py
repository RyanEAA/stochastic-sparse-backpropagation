import argparse
from pathlib import Path

import pandas as pd


def add_compatibility_columns(frame):
    defaults = {
        "architecture": "historical_mlp",
        "protocol_version": "historical",
        "block_size": 0,
        "child_refresh_steps": 0,
        "run_id": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    frame["block_size"] = frame["block_size"].fillna(0).astype(int)
    frame["child_refresh_steps"] = frame["child_refresh_steps"].fillna(0).astype(int)
    if "forward_time_s" not in frame.columns:
        frame["forward_time_s"] = float("nan")
    return frame


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

    epochs = add_compatibility_columns(
        pd.concat((pd.read_csv(path) for path in epoch_files), ignore_index=True)
    )
    batches = add_compatibility_columns(
        pd.concat((pd.read_csv(path) for path in batch_files), ignore_index=True)
    )

    run_keys = [
        "dataset",
        "model",
        "architecture",
        "protocol_version",
        "keep_ratio",
        "block_size",
        "child_refresh_steps",
        "seed",
    ]
    experiment_keys = run_keys[:-1]

    final_epoch = epochs.sort_values("epoch").groupby(run_keys, as_index=False, dropna=False).tail(1)
    per_run_batches = batches.groupby(run_keys, as_index=False, dropna=False).agg(
        mean_forward_ms=("forward_time_s", lambda values: values.mean() * 1000.0),
        mean_backward_ms=("backward_time_s", lambda values: values.mean() * 1000.0),
        mean_memory_mb=("memory_bytes", lambda values: values.mean() / (1024 ** 2)),
    )
    per_run = final_epoch.merge(per_run_batches, on=run_keys, how="inner")

    summary = per_run.groupby(experiment_keys, as_index=False, dropna=False).agg(
        runs=("seed", "count"),
        val_accuracy_mean=("val_accuracy", "mean"),
        val_accuracy_std=("val_accuracy", "std"),
        train_accuracy_mean=("train_accuracy", "mean"),
        train_accuracy_std=("train_accuracy", "std"),
        forward_ms_mean=("mean_forward_ms", "mean"),
        forward_ms_std=("mean_forward_ms", "std"),
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
