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

    return "\n".join(parts)


def bar_metric(df, mean, std, ylabel, title, path):
    plot_df = df.copy()
    plot_df["config_label"] = plot_df.apply(config_label, axis=1)
    plot_df = plot_df.sort_values(
        ["model", "keep_ratio", "block_size", "child_refresh_steps"],
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
        bar_metric(
            group,
            "epoch_time_mean",
            "epoch_time_std",
            "Epoch time (s)",
            f"{prefix}: wall-clock epoch time by configuration",
            directory / "epoch_time_bars.png",
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

    print(f"Bar plots saved under {output}")


if __name__ == "__main__":
    main()
