from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def variant_label(row):
    label = row["model"]
    if row.get("block_size", 0):
        label += f" block={int(row['block_size'])}"
    return label


def metric(df, mean, std, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    dense = df[df.model == "dense"]
    variants = df[df.model != "dense"].copy()
    if not variants.empty:
        variants["variant_label"] = variants.apply(variant_label, axis=1)
        for name, group in variants.groupby("variant_label"):
            group = group.sort_values("keep_ratio")
            ax.errorbar(
                group.keep_ratio,
                group[mean],
                yerr=group[std],
                marker="o",
                capsize=3,
                label=name,
            )
    if not dense.empty:
        ax.axhline(dense.iloc[0][mean], linestyle="--", label="dense")
    ax.set(xlabel="Keep ratio", ylabel=ylabel, title=title)
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    if "architecture" not in df.columns:
        df["architecture"] = "historical_mlp"
    if "protocol_version" not in df.columns:
        df["protocol_version"] = "historical"
    if "block_size" not in df.columns:
        df["block_size"] = 0

    output = args.output_dir or args.summary.parent / "plots"
    output.mkdir(parents=True, exist_ok=True)
    group_keys = ["dataset", "architecture", "protocol_version"]
    for (dataset, architecture, protocol), group in df.groupby(group_keys, dropna=False):
        directory = output / str(dataset) / str(architecture) / str(protocol)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{dataset} / {architecture} / {protocol}"
        metric(group, "val_accuracy_mean", "val_accuracy_std", "Validation accuracy", f"{prefix}: accuracy vs keep ratio", directory / "accuracy_vs_keep_ratio.png")
        if "forward_ms_mean" in group.columns:
            metric(group, "forward_ms_mean", "forward_ms_std", "Forward time (ms)", f"{prefix}: forward time vs keep ratio", directory / "forward_time_vs_keep_ratio.png")
        metric(group, "backward_ms_mean", "backward_ms_std", "Backward time (ms)", f"{prefix}: backward time vs keep ratio", directory / "backward_time_vs_keep_ratio.png")
        metric(group, "memory_mb_mean", "memory_mb_std", "Memory (MB)", f"{prefix}: memory vs keep ratio", directory / "memory_vs_keep_ratio.png")
        metric(group, "epoch_time_mean", "epoch_time_std", "Epoch time (s)", f"{prefix}: epoch time vs keep ratio", directory / "epoch_time_vs_keep_ratio.png")
    print(f"Plots saved under {output}")


if __name__ == "__main__":
    main()
