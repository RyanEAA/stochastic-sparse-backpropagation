import argparse
from pathlib import Path

import pandas as pd


def add_compatibility_columns(frame):
    defaults = {
        "architecture": "historical_mlp",
        "protocol_version": "historical",
        "block_size": 0,
        "child_refresh_steps": 0,
        "score_refresh_steps": 0,
        "v6_gradient_retention": float("nan"),
        "v6_selection_mode": "none",
        "v6_selection_method": "none",
        "v6_early_bird": False,
        "v6_stability_window": 0,
        "v6_stability_threshold": float("nan"),
        "run_id": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    frame["block_size"] = frame["block_size"].fillna(0).astype(int)
    frame["child_refresh_steps"] = frame["child_refresh_steps"].fillna(0).astype(int)
    frame["score_refresh_steps"] = frame["score_refresh_steps"].fillna(0).astype(int)
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
        "score_refresh_steps",
        "v6_gradient_retention",
        "v6_selection_mode",
        "v6_selection_method",
        "v6_early_bird",
        "v6_stability_window",
        "v6_stability_threshold",
        "seed",
    ]
    experiment_keys = run_keys[:-1]

    # For convergence experiments, report model quality at the best validation-loss epoch
    # while separately measuring the full cost paid until the stopping point.
    best_indices = epochs.groupby(run_keys, dropna=False)["val_loss"].idxmin()
    selected_epoch = epochs.loc[best_indices].copy()
    selected_epoch = selected_epoch.rename(columns={"epoch": "best_epoch"})

    run_training = epochs.groupby(run_keys, as_index=False, dropna=False).agg(
        epochs_completed=("epoch", "max"),
        total_training_time_s=("epoch_time_s", "sum"),
        mean_epoch_time_s=("epoch_time_s", "mean"),
    )
    batch_aggs = {
        "mean_forward_ms": ("forward_time_s", lambda values: values.mean() * 1000.0),
        "mean_backward_ms": ("backward_time_s", lambda values: values.mean() * 1000.0),
        "mean_memory_mb": ("memory_bytes", lambda values: values.mean() / (1024 ** 2)),
    }
    if "child_refreshed" in batches.columns:
        batch_aggs["child_refresh_events"] = ("child_refreshed", "sum")
    if "gradient_scoring_event" in batches.columns:
        batch_aggs["gradient_scoring_events"] = ("gradient_scoring_event", "sum")
        batch_aggs["dense_scoring_time_s"] = ("dense_scoring_time_s", "sum")
        if "selector_scoring_time_s" in batches.columns:
            batch_aggs["selector_scoring_time_s"] = ("selector_scoring_time_s", "sum")
        batch_aggs["child_rebuild_time_s"] = ("child_rebuild_time_s", "sum")
    if "topology_frozen" in batches.columns:
        batch_aggs["topology_frozen"] = ("topology_frozen", "max")
        batch_aggs["topology_freeze_step"] = ("topology_freeze_step", "max")
    if "mask_distance" in batches.columns:
        batch_aggs["mask_distance_mean"] = ("mask_distance", "mean")
    if "effective_keep_ratio" in batches.columns:
        batch_aggs["effective_keep_ratio_mean"] = ("effective_keep_ratio", "mean")
        batch_aggs["effective_keep_ratio_min"] = ("effective_keep_ratio", "min")
        batch_aggs["effective_keep_ratio_max"] = ("effective_keep_ratio", "max")
    per_run_batches = batches.groupby(run_keys, as_index=False, dropna=False).agg(**batch_aggs)
    per_run = selected_epoch.merge(run_training, on=run_keys, how="inner").merge(per_run_batches, on=run_keys, how="inner")

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
        epoch_time_mean=("mean_epoch_time_s", "mean"),
        epoch_time_std=("mean_epoch_time_s", "std"),
        best_epoch_mean=("best_epoch", "mean"),
        best_epoch_std=("best_epoch", "std"),
        epochs_completed_mean=("epochs_completed", "mean"),
        epochs_completed_std=("epochs_completed", "std"),
        total_training_time_mean=("total_training_time_s", "mean"),
        total_training_time_std=("total_training_time_s", "std"),
        **({
            "child_refresh_events_mean": ("child_refresh_events", "mean"),
            "child_refresh_events_std": ("child_refresh_events", "std"),
        } if "child_refresh_events" in per_run.columns else {}),
        **({
            "gradient_scoring_events_mean": ("gradient_scoring_events", "mean"),
            "dense_scoring_time_mean": ("dense_scoring_time_s", "mean"),
            **({"selector_scoring_time_mean": ("selector_scoring_time_s", "mean")} if "selector_scoring_time_s" in per_run.columns else {}),
            "child_rebuild_time_mean": ("child_rebuild_time_s", "mean"),
        } if "gradient_scoring_events" in per_run.columns else {}),
        **({
            "effective_keep_ratio_mean": ("effective_keep_ratio_mean", "mean"),
            "effective_keep_ratio_min": ("effective_keep_ratio_min", "mean"),
            "effective_keep_ratio_max": ("effective_keep_ratio_max", "mean"),
        } if "effective_keep_ratio_mean" in per_run.columns else {}),
        **({
            "topology_frozen_fraction": ("topology_frozen", "mean"),
            "topology_freeze_step_mean": ("topology_freeze_step", "mean"),
        } if "topology_frozen" in per_run.columns else {}),
        **({"mask_distance_mean": ("mask_distance_mean", "mean")} if "mask_distance_mean" in per_run.columns else {}),
    )

    output = root / "summary.csv"
    summary.to_csv(output, index=False)
    per_run.to_csv(root / "per_run.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
