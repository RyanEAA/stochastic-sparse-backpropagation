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
        "v7_dense_warmup_epochs": 0,
        "v7_dense_correction_steps": 0,
        "v7_layer_keep_ratios": "null",
        "v7_early_bird_min_events": 0,
        "v7_early_bird_min_steps": 0,
        "run_id": "",
        "timing_detail": "historical",
        "record_batch_metrics": True,
        "training_phase": "unknown",
        "child_val_loss": float("nan"),
        "child_val_accuracy": float("nan"),
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
        "v7_dense_warmup_epochs",
        "v7_dense_correction_steps",
        "v7_layer_keep_ratios",
        "v7_early_bird_min_events",
        "v7_early_bird_min_steps",
        "timing_detail",
        "record_batch_metrics",
        "seed",
    ]
    experiment_keys = run_keys[:-1]

    # For convergence experiments, report model quality at the best validation-loss epoch
    # while separately measuring the full cost paid until the stopping point.
    best_indices = epochs.groupby(run_keys, dropna=False)["val_loss"].idxmin()
    selected_epoch = epochs.loc[best_indices].copy()
    selected_epoch = selected_epoch.rename(columns={"epoch": "best_epoch"})

    final_indices = epochs.groupby(run_keys, dropna=False)["epoch"].idxmax()
    final_epoch = epochs.loc[
        final_indices,
        run_keys + [
            "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
            "child_val_loss", "child_val_accuracy",
        ],
    ].copy()
    final_epoch = final_epoch.rename(columns={
        "epoch": "final_epoch",
        "train_loss": "final_train_loss",
        "train_accuracy": "final_train_accuracy",
        "val_loss": "final_val_loss",
        "val_accuracy": "final_val_accuracy",
        "child_val_loss": "final_child_val_loss",
        "child_val_accuracy": "final_child_val_accuracy",
    })

    child_epochs = epochs.dropna(subset=["child_val_loss"])
    if child_epochs.empty:
        best_child_epoch = None
    else:
        child_best_indices = child_epochs.groupby(run_keys, dropna=False)["child_val_loss"].idxmin()
        best_child_epoch = child_epochs.loc[
            child_best_indices,
            run_keys + ["epoch", "child_val_loss", "child_val_accuracy"],
        ].copy().rename(columns={
            "epoch": "best_child_epoch",
            "child_val_loss": "best_child_val_loss",
            "child_val_accuracy": "best_child_val_accuracy",
        })

    epochs["dense_warmup_time_s"] = epochs["epoch_time_s"].where(
        epochs["training_phase"] == "dense_warmup", 0.0
    )
    epochs["sparse_hybrid_time_s"] = epochs["epoch_time_s"].where(
        epochs["training_phase"] == "sparse_or_hybrid", 0.0
    )
    epochs["dense_warmup_epoch"] = (epochs["training_phase"] == "dense_warmup").astype(int)
    epochs["sparse_hybrid_epoch"] = (epochs["training_phase"] == "sparse_or_hybrid").astype(int)

    run_training = epochs.groupby(run_keys, as_index=False, dropna=False).agg(
        epochs_completed=("epoch", "max"),
        total_training_time_s=("epoch_time_s", "sum"),
        mean_epoch_time_s=("epoch_time_s", "mean"),
        dense_warmup_time_s=("dense_warmup_time_s", "sum"),
        sparse_hybrid_time_s=("sparse_hybrid_time_s", "sum"),
        dense_warmup_epochs=("dense_warmup_epoch", "sum"),
        sparse_hybrid_epochs=("sparse_hybrid_epoch", "sum"),
    )
    batch_aggs = {
        "mean_forward_ms": ("forward_time_s", lambda values: values.mean() * 1000.0),
        "mean_backward_ms": ("backward_time_s", lambda values: values.mean() * 1000.0),
        "mean_memory_mb": ("memory_bytes", lambda values: values.mean() / (1024 ** 2)),
    }
    timing_columns = [
        "data_transfer_time_s", "selection_refresh_time_s", "zero_grad_time_s",
        "loss_time_s", "memory_measurement_time_s", "optimizer_step_time_s",
        "post_step_time_s", "instrumentation_time_s", "batch_wall_time_s",
        "unaccounted_time_s",
    ]
    for column in timing_columns:
        if column in batches.columns:
            batch_aggs[f"mean_{column.removesuffix('_time_s')}_ms"] = (
                column, lambda values: values.mean() * 1000.0
            )
    if "child_refreshed" in batches.columns:
        batch_aggs["child_refresh_events"] = ("child_refreshed", "sum")
    if "gradient_scoring_event" in batches.columns:
        batch_aggs["gradient_scoring_events"] = ("gradient_scoring_event", "sum")
        batch_aggs["dense_scoring_time_s"] = ("dense_scoring_time_s", "sum")
        if "selector_scoring_time_s" in batches.columns:
            batch_aggs["selector_scoring_time_s"] = ("selector_scoring_time_s", "sum")
        batch_aggs["child_rebuild_time_s"] = ("child_rebuild_time_s", "sum")
    if "dense_correction_event" in batches.columns:
        batch_aggs["dense_correction_events"] = ("dense_correction_event", "sum")
    if "dense_warmup_event" in batches.columns:
        batch_aggs["dense_warmup_batches"] = ("dense_warmup_event", "sum")
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
    per_run = (
        selected_epoch
        .merge(final_epoch, on=run_keys, how="inner")
        .merge(run_training, on=run_keys, how="inner")
        .merge(per_run_batches, on=run_keys, how="inner")
    )
    if best_child_epoch is not None:
        per_run = per_run.merge(best_child_epoch, on=run_keys, how="left")

    summary = per_run.groupby(experiment_keys, as_index=False, dropna=False).agg(
        runs=("seed", "count"),
        val_accuracy_mean=("val_accuracy", "mean"),
        val_accuracy_std=("val_accuracy", "std"),
        train_accuracy_mean=("train_accuracy", "mean"),
        train_accuracy_std=("train_accuracy", "std"),
        final_val_accuracy_mean=("final_val_accuracy", "mean"),
        final_val_accuracy_std=("final_val_accuracy", "std"),
        final_train_accuracy_mean=("final_train_accuracy", "mean"),
        final_train_accuracy_std=("final_train_accuracy", "std"),
        final_child_val_accuracy_mean=("final_child_val_accuracy", "mean"),
        final_child_val_accuracy_std=("final_child_val_accuracy", "std"),
        **({
            "child_val_accuracy_mean": ("child_val_accuracy", "mean"),
            "child_val_accuracy_std": ("child_val_accuracy", "std"),
        } if "child_val_accuracy" in per_run.columns else {}),
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
        dense_warmup_time_mean=("dense_warmup_time_s", "mean"),
        sparse_hybrid_time_mean=("sparse_hybrid_time_s", "mean"),
        dense_warmup_epochs_mean=("dense_warmup_epochs", "mean"),
        sparse_hybrid_epochs_mean=("sparse_hybrid_epochs", "mean"),
        **({
            "best_child_val_accuracy_mean": ("best_child_val_accuracy", "mean"),
            "best_child_val_accuracy_std": ("best_child_val_accuracy", "std"),
            "best_child_epoch_mean": ("best_child_epoch", "mean"),
        } if "best_child_val_accuracy" in per_run.columns else {}),
        **{
            output_name: (per_run_name, statistic)
            for column in timing_columns
            if (per_run_name := f"mean_{column.removesuffix('_time_s')}_ms") in per_run.columns
            for statistic, suffix in (("mean", "mean"), ("std", "std"))
            for output_name in (f"{column.removesuffix('_time_s')}_ms_{suffix}",)
        },
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
            "dense_correction_events_mean": ("dense_correction_events", "mean"),
            "dense_correction_events_std": ("dense_correction_events", "std"),
        } if "dense_correction_events" in per_run.columns else {}),
        **({
            "dense_warmup_batches_mean": ("dense_warmup_batches", "mean"),
            "dense_warmup_batches_std": ("dense_warmup_batches", "std"),
        } if "dense_warmup_batches" in per_run.columns else {}),
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
