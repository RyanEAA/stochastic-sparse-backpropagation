from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd


def config_label(row):
    model = str(row["model"])
    if model == "dense":
        return "dense"

    parts = [model]
    keep = row.get("keep_ratio", None)
    if pd.notna(keep):
        parts.append(f"k={float(keep):g}")

    block = row.get("block_size", 0)
    if pd.notna(block) and int(block) > 0:
        parts.append(f"b={int(block)}")

    refresh = row.get("child_refresh_steps", 0)
    if pd.notna(refresh) and int(refresh) > 0:
        parts.append(f"r={int(refresh)}")

    score_refresh = row.get("score_refresh_steps", 0)
    if pd.notna(score_refresh) and int(score_refresh) > 0:
        parts.append(f"r={int(score_refresh)}")

    selector = row.get("v6_selection_method", None)
    if pd.notna(selector) and str(selector) not in {"", "none"}:
        parts.append(str(selector).replace("_", "-"))

    warmup = row.get("v7_dense_warmup_epochs", 0)
    if pd.notna(warmup) and int(warmup) > 0:
        parts.append(f"warmup={int(warmup)}e")

    correction = row.get("v7_dense_correction_steps", 0)
    if pd.notna(correction) and int(correction) > 0:
        parts.append(f"dense/{int(correction)}")

    layer_ratios = row.get("v7_layer_keep_ratios", None)
    if pd.notna(layer_ratios) and str(layer_ratios) not in {"", "null", "None"}:
        compact = str(layer_ratios).replace(" ", "")
        parts.append(f"layers={compact}")

    return "\n".join(parts)


def bar_metric(df, mean, std, ylabel, title, path):
    plot_df = df.copy()
    plot_df["config_label"] = plot_df.apply(config_label, axis=1)
    plot_df = plot_df.sort_values(
        ["model", "keep_ratio", "block_size", "child_refresh_steps", "score_refresh_steps", "v6_selection_method", "v7_dense_warmup_epochs", "v7_dense_correction_steps", "v7_layer_keep_ratios"],
        na_position="last",
    )

    n = len(plot_df)
    width = max(10, min(40, 0.48 * n + 4))
    fig, ax = plt.subplots(figsize=(width, 7))
    x = list(range(n))
    values = plot_df[mean].to_numpy()
    errors = plot_df[std].fillna(0).to_numpy() if std in plot_df.columns else None

    ax.bar(x, values, yerr=errors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["config_label"], rotation=70, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    defaults = {
        "architecture": "historical_mlp",
        "protocol_version": "historical",
        "block_size": 0,
        "child_refresh_steps": 0,
        "score_refresh_steps": 0,
        "v6_selection_method": "none",
        "v7_dense_warmup_epochs": 0,
        "v7_dense_correction_steps": 0,
        "v7_layer_keep_ratios": "null",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    output = args.output_dir or args.summary.parent / "plots"
    output.mkdir(parents=True, exist_ok=True)

    group_keys = ["dataset", "architecture", "protocol_version"]
    for (dataset, architecture, protocol), group in df.groupby(group_keys, dropna=False):
        directory = output / str(dataset) / str(architecture) / str(protocol)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{dataset} / {architecture} / {protocol}"

        bar_metric(
            group,
            "val_accuracy_mean",
            "val_accuracy_std",
            "Validation accuracy",
            f"{prefix}: validation accuracy by configuration",
            directory / "validation_accuracy_bars.png",
        )
        if "child_val_accuracy_mean" in group.columns and group["child_val_accuracy_mean"].notna().any():
            child_group = group[group["child_val_accuracy_mean"].notna()]
            bar_metric(
                child_group,
                "child_val_accuracy_mean",
                "child_val_accuracy_std",
                "Child validation accuracy",
                f"{prefix}: trained child validation accuracy by configuration",
                directory / "child_validation_accuracy_bars.png",
            )
        if "final_val_accuracy_mean" in group.columns:
            bar_metric(
                group,
                "final_val_accuracy_mean",
                "final_val_accuracy_std",
                "Final validation accuracy",
                f"{prefix}: final-epoch validation accuracy by configuration",
                directory / "final_validation_accuracy_bars.png",
            )
        if "final_child_val_accuracy_mean" in group.columns and group["final_child_val_accuracy_mean"].notna().any():
            final_child_group = group[group["final_child_val_accuracy_mean"].notna()]
            bar_metric(
                final_child_group,
                "final_child_val_accuracy_mean",
                "final_child_val_accuracy_std",
                "Final child validation accuracy",
                f"{prefix}: final-epoch child accuracy by configuration",
                directory / "final_child_validation_accuracy_bars.png",
            )
        bar_metric(
            group,
            "epoch_time_mean",
            "epoch_time_std",
            "Epoch time (s)",
            f"{prefix}: wall-clock epoch time by configuration",
            directory / "epoch_time_bars.png",
        )
        if "total_training_time_mean" in group.columns:
            bar_metric(
                group,
                "total_training_time_mean",
                "total_training_time_std",
                "Total training time to stop (s)",
                f"{prefix}: wall-clock training time to convergence/stop",
                directory / "total_training_time_bars.png",
            )
        if "epochs_completed_mean" in group.columns:
            bar_metric(
                group,
                "epochs_completed_mean",
                "epochs_completed_std",
                "Epochs completed",
                f"{prefix}: epochs until convergence/stop",
                directory / "epochs_completed_bars.png",
            )
        bar_metric(
            group,
            "memory_mb_mean",
            "memory_mb_std",
            "Memory (MB)",
            f"{prefix}: measured memory by configuration",
            directory / "memory_bars.png",
        )

        # Keep forward/backward timing as diagnostic bar charts when present.
        if "forward_ms_mean" in group.columns:
            bar_metric(
                group,
                "forward_ms_mean",
                "forward_ms_std",
                "Forward time (ms)",
                f"{prefix}: forward time by configuration",
                directory / "forward_time_bars.png",
            )
        if "backward_ms_mean" in group.columns:
            bar_metric(
                group,
                "backward_ms_mean",
                "backward_ms_std",
                "Backward time (ms)",
                f"{prefix}: backward time by configuration",
                directory / "backward_time_bars.png",
            )

        for column, label in (
            ("optimizer_step", "Optimizer step"),
            ("post_step", "Post-step synchronization"),
            ("instrumentation", "Instrumentation"),
            ("selection_refresh", "Selection/refresh"),
            ("batch_wall", "Total batch wall-clock"),
            ("unaccounted", "Unaccounted batch time"),
        ):
            mean, std = f"{column}_ms_mean", f"{column}_ms_std"
            if mean in group.columns:
                bar_metric(
                    group, mean, std, f"{label} time (ms)",
                    f"{prefix}: {label.lower()} time by configuration",
                    directory / f"{column}_time_bars.png",
                )

    print(f"Bar plots saved under {output}")


if __name__ == "__main__":
    main()
