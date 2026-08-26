from pathlib import Path
import subprocess
import pandas as pd


SSB_VERSIONS = [
    "v0",
    "v1",
    "v2",
]

KEEP_RATIOS = [
    1.0,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
]

RUNS_PER_CONFIG = 20
EPOCHS = 3
BATCH_SIZE = 128

OUTPUT_DIR = Path(
    "mnist_experiments"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


def run_experiment(
    ssb_version,
    keep_ratio,
    run_number,
):
    seed = run_number

    ratio_name = (
        str(keep_ratio)
        .replace(".", "_")
    )

    run_dir = (
        OUTPUT_DIR
        / ssb_version
        / f"keep_{ratio_name}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        run_dir
        / f"run_{run_number:02d}.csv"
    )

    iter_csv = output_csv.with_name(
        output_csv.stem
        + "_iters.csv"
    )

    # Resume support.
    if (
        output_csv.exists()
        and iter_csv.exists()
    ):
        print(
            f"Skipping existing: "
            f"{ssb_version} "
            f"keep={keep_ratio} "
            f"run={run_number}"
        )

        return output_csv

    command = [
        "python",
        "train_compare_mnist.py",

        "--ssb-version",
        ssb_version,

        "--keep-ratio",
        str(keep_ratio),

        "--epochs",
        str(EPOCHS),

        "--batch-size",
        str(BATCH_SIZE),

        "--seed",
        str(seed),

        "--out-csv",
        str(output_csv),
    ]

    print()
    print("=" * 80)

    print(
        f"SSB {ssb_version.upper()} "
        f"| KEEP {keep_ratio} "
        f"| RUN {run_number}/{RUNS_PER_CONFIG} "
        f"| SEED {seed}"
    )

    print("=" * 80)

    subprocess.run(
        command,
        check=True,
    )

    return output_csv


def build_summary(
    result_files,
):
    rows = []

    for (
        ssb_version,
        keep_ratio,
        run_number,
        csv_path,
    ) in result_files:

        epoch_df = pd.read_csv(
            csv_path
        )

        iter_path = csv_path.with_name(
            csv_path.stem
            + "_iters.csv"
        )

        iter_df = pd.read_csv(
            iter_path
        )

        dense_epochs = epoch_df[
            epoch_df["model"]
            == "dense"
        ]

        sparse_epochs = epoch_df[
            epoch_df["model"]
            == "sparse"
        ]

        dense_iters = iter_df[
            iter_df["model"]
            == "dense"
        ]

        sparse_iters = iter_df[
            iter_df["model"]
            == "sparse"
        ]

        final_dense = (
            dense_epochs.iloc[-1]
        )

        final_sparse = (
            sparse_epochs.iloc[-1]
        )

        dense_bw = (
            dense_iters[
                "bw_time_s"
            ].mean()
        )

        sparse_bw = (
            sparse_iters[
                "bw_time_s"
            ].mean()
        )

        dense_mem = (
            dense_iters[
                "peak_mem_bytes"
            ].mean()
        )

        sparse_mem = (
            sparse_iters[
                "peak_mem_bytes"
            ].mean()
        )

        rows.append({
            "ssb_version":
                ssb_version,

            "keep_ratio":
                keep_ratio,

            "run":
                run_number,

            "dense_final_train_acc":
                final_dense[
                    "train_acc"
                ],

            "sparse_final_train_acc":
                final_sparse[
                    "train_acc"
                ],

            "dense_final_val_acc":
                final_dense[
                    "val_acc"
                ],

            "sparse_final_val_acc":
                final_sparse[
                    "val_acc"
                ],

            "dense_final_val_loss":
                final_dense[
                    "val_loss"
                ],

            "sparse_final_val_loss":
                final_sparse[
                    "val_loss"
                ],

            "dense_mean_backward_ms":
                dense_bw * 1000,

            "sparse_mean_backward_ms":
                sparse_bw * 1000,

            "backward_speedup":
                dense_bw
                / sparse_bw,

            "dense_mean_memory_mb":
                dense_mem
                / (1024 ** 2),

            "sparse_mean_memory_mb":
                sparse_mem
                / (1024 ** 2),

            "memory_savings_percent":
                (
                    (dense_mem - sparse_mem)
                    / dense_mem
                    * 100
                ),
        })

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUTPUT_DIR
        / "all_runs_summary.csv",
        index=False,
    )

    return summary


def build_aggregate_summary(
    summary,
):
    aggregate = (
        summary
        .groupby(
            [
                "ssb_version",
                "keep_ratio",
            ]
        )
        .agg(
            runs=(
                "run",
                "count",
            ),

            sparse_val_acc_mean=(
                "sparse_final_val_acc",
                "mean",
            ),

            sparse_val_acc_std=(
                "sparse_final_val_acc",
                "std",
            ),

            dense_val_acc_mean=(
                "dense_final_val_acc",
                "mean",
            ),

            dense_val_acc_std=(
                "dense_final_val_acc",
                "std",
            ),

            sparse_backward_ms_mean=(
                "sparse_mean_backward_ms",
                "mean",
            ),

            sparse_backward_ms_std=(
                "sparse_mean_backward_ms",
                "std",
            ),

            dense_backward_ms_mean=(
                "dense_mean_backward_ms",
                "mean",
            ),

            dense_backward_ms_std=(
                "dense_mean_backward_ms",
                "std",
            ),

            speedup_mean=(
                "backward_speedup",
                "mean",
            ),

            speedup_std=(
                "backward_speedup",
                "std",
            ),

            sparse_memory_mb_mean=(
                "sparse_mean_memory_mb",
                "mean",
            ),

            sparse_memory_mb_std=(
                "sparse_mean_memory_mb",
                "std",
            ),

            dense_memory_mb_mean=(
                "dense_mean_memory_mb",
                "mean",
            ),

            dense_memory_mb_std=(
                "dense_mean_memory_mb",
                "std",
            ),

            memory_savings_mean=(
                "memory_savings_percent",
                "mean",
            ),

            memory_savings_std=(
                "memory_savings_percent",
                "std",
            ),
        )
        .reset_index()
    )

    aggregate.to_csv(
        OUTPUT_DIR
        / "aggregate_summary.csv",
        index=False,
    )

    return aggregate


def main():
    result_files = []

    total_runs = (
        len(SSB_VERSIONS)
        * len(KEEP_RATIOS)
        * RUNS_PER_CONFIG
    )

    current = 0

    for ssb_version in SSB_VERSIONS:

        for keep_ratio in KEEP_RATIOS:

            for run_number in range(
                1,
                RUNS_PER_CONFIG + 1,
            ):
                current += 1

                print(
                    f"\nExperiment "
                    f"{current}/{total_runs}"
                )

                path = run_experiment(
                    ssb_version,
                    keep_ratio,
                    run_number,
                )

                result_files.append(
                    (
                        ssb_version,
                        keep_ratio,
                        run_number,
                        path,
                    )
                )

    print()
    print("=" * 80)
    print(
        "ALL RUNS COMPLETE"
    )
    print("=" * 80)

    summary = build_summary(
        result_files
    )

    aggregate = (
        build_aggregate_summary(
            summary
        )
    )

    print()
    print(
        aggregate.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()