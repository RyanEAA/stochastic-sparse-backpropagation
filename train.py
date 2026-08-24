"""Run one dataset/model experiment and save raw measurements plus run metadata."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data.loaders import build_loaders, normalize_dataset_name, SUPPORTED_DATASETS
from models import build_model, AVAILABLE_MODELS, AVAILABLE_ARCHITECTURES
from training.runtime import (
    set_seed,
    initialize_model_parameters,
    get_device,
    synchronize,
    memory_bytes,
    git_commit,
)

PROTOCOL_VERSION = "protocol-v2"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def make_run_id(config):
    stable = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode()).hexdigest()[:16]


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    loss_sum = correct = total = 0
    batches = []
    start = time.perf_counter()
    for batch_index, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        forward_start = time.perf_counter()
        output = model(x)
        synchronize(device)
        forward_time = time.perf_counter() - forward_start
        loss = criterion(output, y)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        backward_start = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward_time = time.perf_counter() - backward_start
        memory = memory_bytes(device)
        optimizer.step()

        n = y.size(0)
        loss_sum += loss.item() * n
        batch_correct = (output.argmax(1) == y).sum().item()
        correct += batch_correct
        total += n
        batches.append(
            dict(
                epoch=epoch,
                batch=batch_index,
                forward_time_s=forward_time,
                backward_time_s=backward_time,
                memory_bytes=memory,
                batch_accuracy=batch_correct / n,
            )
        )
    synchronize(device)
    return loss_sum / total, correct / total, time.perf_counter() - start, batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        synchronize(device)
        forward_start = time.perf_counter()
        output = model(x)
        synchronize(device)
        forward_time = time.perf_counter() - forward_start
        loss = criterion(output, y)
        n = y.size(0)
        loss_sum += loss.item() * n
        correct += (output.argmax(1) == y).sum().item()
        total += n
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help=f"One of: {', '.join(SUPPORTED_DATASETS)}")
    parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS)
    parser.add_argument("--architecture", default="mlp", choices=AVAILABLE_ARCHITECTURES)
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    if args.model != "dense" and not 0 < args.keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1].")
    if args.model in {"ssb-v1-block", "ssb-v2-block", "ssb-v3-block"} and args.block_size <= 0:
        raise ValueError("block_size must be positive for block SSB variants.")

    set_seed(args.seed)
    device = get_device(args.device)
    train_loader, val_loader = build_loaders(
        dataset, args.batch_size, args.seed, args.subset, args.data_dir, args.num_workers
    )
    model = build_model(
        dataset,
        args.model,
        args.keep_ratio,
        architecture=args.architecture,
        block_size=args.block_size,
    )
    initialize_model_parameters(model, args.seed)
    model = model.to(device)
    set_seed(args.seed)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    total_parameters, trainable_parameters = parameter_counts(model)

    identity = {
        "dataset": dataset,
        "model": args.model,
        "architecture": args.architecture,
        "keep_ratio": 1.0 if args.model == "dense" else args.keep_ratio,
        "block_size": args.block_size if args.model in {"ssb-v1-block", "ssb-v2-block", "ssb-v3-block"} else 0,
        "seed": args.seed,
        "protocol_version": args.protocol_version,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "subset": args.subset,
    }
    run_id = make_run_id(identity)
    metadata = {
        **identity,
        "run_id": run_id,
        "experiment_tag": args.experiment_tag,
        "optimizer": "Adam",
        "criterion": "CrossEntropyLoss",
        "initialization": "xavier_uniform_weights_zero_bias",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "requested_device": args.device,
        "resolved_device": str(device),
        "num_workers": args.num_workers,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "memory_metric": "cuda_peak_allocated_after_backward" if device.type == "cuda" else "process_rss_after_backward",
        "cnn_scope": "dense_conv_backbone_ssb_linear_classifier" if args.architecture == "cnn" and args.model.startswith("ssb-") else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(
        f"dataset={dataset} model={args.model} architecture={args.architecture} "
        f"keep_ratio={identity['keep_ratio']} block_size={identity['block_size']} "
        f"seed={args.seed} device={device} run_id={run_id}"
    )

    epoch_rows = []
    batch_rows = []
    csv_base = {
        "run_id": run_id,
        "dataset": dataset,
        "model": args.model,
        "architecture": args.architecture,
        "protocol_version": args.protocol_version,
        "keep_ratio": identity["keep_ratio"],
        "block_size": identity["block_size"],
        "seed": args.seed,
    }
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, epoch_time, batches = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} time={epoch_time:.2f}s"
        )
        epoch_rows.append(
            {
                **csv_base,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_time_s": epoch_time,
            }
        )
        for row in batches:
            batch_rows.append({**csv_base, **row})

    common_fields = [
        "run_id",
        "dataset",
        "model",
        "architecture",
        "protocol_version",
        "keep_ratio",
        "block_size",
        "seed",
    ]
    write_csv(
        args.output_dir / "epochs.csv",
        epoch_rows,
        common_fields
        + ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "epoch_time_s"],
    )
    write_csv(
        args.output_dir / "batches.csv",
        batch_rows,
        common_fields + ["epoch", "batch", "forward_time_s", "backward_time_s", "memory_bytes", "batch_accuracy"],
    )


if __name__ == "__main__":
    main()
