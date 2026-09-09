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


def train_epoch(model, loader, optimizer, criterion, device, epoch, learning_rate, global_step_start=0):
    model.train()
    loss_sum = correct = total = 0
    batches = []
    start = time.perf_counter()
    global_step = global_step_start
    for batch_index, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        refresh_count_before = int(getattr(model, "refresh_count", 0))
        topology_before = model.topology_signature() if hasattr(model, "topology_signature") else ""
        scoring_event = False
        selector_scoring_time_s = 0.0
        dense_scoring_time_s = 0.0
        child_rebuild_time_s = 0.0
        if getattr(model, "is_v6_gradient_selected", False):
            optimizer, scoring_event = model.score_and_refresh(
                x, y, criterion, optimizer, learning_rate
            )
            if scoring_event:
                selector_scoring_time_s = model.last_selector_scoring_time_s
                dense_scoring_time_s = model.last_dense_scoring_time_s
                child_rebuild_time_s = model.last_child_rebuild_time_s
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
        global_step += 1
        if getattr(model, "is_v5_structured_child", False):
            optimizer = model.after_optimizer_step(optimizer, learning_rate)
        elif hasattr(model, "after_optimizer_step") and model.after_optimizer_step():
            # V4 creates a new physically smaller child; its Parameters are new objects.
            # Rebuild Adam for the new child. This reset is recorded in metadata and is
            # intentionally part of the V4 baseline.
            optimizer = optim.Adam(model.training_parameters(), lr=learning_rate)
        refresh_count_after = int(getattr(model, "refresh_count", 0))
        topology_after = model.topology_signature() if hasattr(model, "topology_signature") else ""
        child_refreshed = refresh_count_after > refresh_count_before
        importance_min, importance_mean, importance_max = (
            model.importance_statistics()
            if getattr(model, "is_v6_gradient_selected", False)
            else (0.0, 0.0, 0.0)
        )

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
                global_step=global_step,
                child_refreshed=int(child_refreshed),
                child_refresh_count=refresh_count_after,
                child_topology_before=topology_before,
                child_topology_after=topology_after,
                gradient_scoring_event=int(scoring_event),
                gradient_scoring_event_count=int(getattr(model, "scoring_event_count", 0)),
                selector_scoring_time_s=selector_scoring_time_s,
                dense_scoring_time_s=dense_scoring_time_s,
                child_rebuild_time_s=child_rebuild_time_s,
                active_structured_units=(model.active_structured_units() if getattr(model, "is_v6_gradient_selected", False) else 0),
                total_structured_units=(model.total_structured_units() if getattr(model, "is_v6_gradient_selected", False) else 0),
                effective_keep_ratio=(model.effective_keep_ratio() if getattr(model, "is_v6_gradient_selected", False) else 1.0),
                importance_min=importance_min,
                importance_mean=importance_mean,
                importance_max=importance_max,
                mask_distance=float(getattr(model, "last_mask_distance", float("nan"))),
                topology_frozen=int(getattr(model, "topology_frozen", False)),
                topology_freeze_step=int(getattr(model, "topology_freeze_step", 0)),
                topology_freeze_scoring_event=int(getattr(model, "topology_freeze_scoring_event", 0)),
            )
        )
    synchronize(device)
    return loss_sum / total, correct / total, time.perf_counter() - start, batches, optimizer, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    if hasattr(model, "sync_child_to_master"):
        model.sync_child_to_master()
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
    parser.add_argument("--child-refresh-steps", type=int, default=100)
    parser.add_argument("--score-refresh-steps", type=int, default=100, help="For V6: run dense gradient scoring and rebuild the top-k child every N optimizer steps.")
    parser.add_argument("--v6-selection-mode", choices=["fixed", "gradient_retention"], default="fixed")
    parser.add_argument("--v6-gradient-retention", type=float, default=0.90, help="For dynamic V6: smallest child retaining this fraction of per-layer squared gradient energy.")
    parser.add_argument("--v6-selection-method", choices=["gradient_l2", "weight_l2", "taylor"], default="gradient_l2", help="Structured-unit score used by V6.")
    parser.add_argument("--v6-early-bird", action="store_true", help="Freeze the V6 topology once recent mask distances are stable.")
    parser.add_argument("--v6-stability-window", type=int, default=5, help="Number of consecutive mask distances used by Early-Bird.")
    parser.add_argument("--v6-stability-threshold", type=float, default=0.10, help="Maximum mask replacement fraction considered stable.")
    parser.add_argument("--epochs", type=int, default=3, help="Fixed epoch count when --stop-at-convergence is not used.")
    parser.add_argument("--stop-at-convergence", action="store_true", help="Stop when validation loss has not improved by --min-delta for --patience epochs.")
    parser.add_argument("--max-epochs", type=int, default=100, help="Safety cap when --stop-at-convergence is enabled.")
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience in validation epochs.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation-loss decrease counted as improvement.")
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
    if args.model in {"ssb-v4", "ssb-v5", "ssb-v5.1"} and args.child_refresh_steps <= 0:
        raise ValueError("child_refresh_steps must be positive for structured-child variants.")
    if args.model == "ssb-v6" and args.score_refresh_steps <= 0:
        raise ValueError("score_refresh_steps must be positive for SSB-V6.")
    if args.model == "ssb-v6" and not 0 < args.v6_gradient_retention <= 1:
        raise ValueError("v6_gradient_retention must be in (0, 1].")
    if args.v6_stability_window < 1:
        raise ValueError("v6_stability_window must be >= 1.")
    if not 0 <= args.v6_stability_threshold <= 1:
        raise ValueError("v6_stability_threshold must be in [0, 1].")
    if args.stop_at_convergence and args.max_epochs < 1:
        raise ValueError("max_epochs must be >= 1.")
    if args.stop_at_convergence and args.patience < 1:
        raise ValueError("patience must be >= 1.")
    if args.min_delta < 0:
        raise ValueError("min_delta must be >= 0.")

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
        child_refresh_steps=args.child_refresh_steps,
        score_refresh_steps=args.score_refresh_steps,
        v6_selection_mode=args.v6_selection_mode,
        v6_gradient_retention=args.v6_gradient_retention,
        v6_selection_method=args.v6_selection_method,
        v6_early_bird=args.v6_early_bird,
        v6_stability_window=args.v6_stability_window,
        v6_stability_threshold=args.v6_stability_threshold,
    )
    initialize_model_parameters(model, args.seed)
    model = model.to(device)
    set_seed(args.seed)
    if hasattr(model, "refresh_child"):
        model.refresh_child()

    optimizer_parameters = model.training_parameters() if hasattr(model, "training_parameters") else model.parameters()
    if getattr(model, "is_v5_structured_child", False):
        optimizer = model.make_optimizer(args.lr)
    else:
        optimizer = optim.Adam(optimizer_parameters, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    total_parameters, trainable_parameters = parameter_counts(model)

    identity = {
        "dataset": dataset,
        "model": args.model,
        "architecture": args.architecture,
        "keep_ratio": 1.0 if args.model == "dense" else args.keep_ratio,
        "block_size": args.block_size if args.model in {"ssb-v1-block", "ssb-v2-block", "ssb-v3-block"} else 0,
        "child_refresh_steps": args.child_refresh_steps if args.model in {"ssb-v4", "ssb-v5", "ssb-v5.1"} else 0,
        "score_refresh_steps": args.score_refresh_steps if args.model == "ssb-v6" else 0,
        "v6_selection_mode": args.v6_selection_mode if args.model == "ssb-v6" else None,
        "v6_gradient_retention": args.v6_gradient_retention if args.model == "ssb-v6" and args.v6_selection_mode == "gradient_retention" else None,
        "v6_selection_method": args.v6_selection_method if args.model == "ssb-v6" else None,
        "v6_early_bird": bool(args.v6_early_bird) if args.model == "ssb-v6" else False,
        "v6_stability_window": args.v6_stability_window if args.model == "ssb-v6" else 0,
        "v6_stability_threshold": args.v6_stability_threshold if args.model == "ssb-v6" else None,
        "seed": args.seed,
        "protocol_version": args.protocol_version,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "stop_at_convergence": bool(args.stop_at_convergence),
        "max_epochs": args.max_epochs if args.stop_at_convergence else args.epochs,
        "patience": args.patience if args.stop_at_convergence else 0,
        "min_delta": args.min_delta if args.stop_at_convergence else 0.0,
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
        "cnn_scope": (
            "structured_dense_child_conv_and_classifier" if args.architecture == "cnn" and args.model in {"ssb-v4", "ssb-v5", "ssb-v6"}
            else "dense_forward_structured_backward_conv_and_classifier" if args.architecture == "cnn" and args.model == "ssb-v5.1"
            else "dense_conv_backbone_ssb_linear_classifier" if args.architecture == "cnn" and args.model.startswith("ssb-")
            else None
        ),
        "structured_child_optimizer_state_policy": (
            "adam_state_resets_when_child_is_resampled" if args.model == "ssb-v4"
            else "master_owned_persistent_adam_moments" if args.model in {"ssb-v5", "ssb-v5.1", "ssb-v6"}
            else None
        ),
        "structured_child_master_parameters": model.master_parameter_count() if hasattr(model, "master_parameter_count") else None,
        "structured_child_initial_child_parameters": model.child_parameter_count() if hasattr(model, "child_parameter_count") else None,
        "forward_backward_policy": (
            "dense_forward_structured_sparse_backward" if args.model == "ssb-v5.1"
            else "structured_child_forward_and_backward_with_periodic_dense_gradient_scoring" if args.model == "ssb-v6"
            else "structured_child_forward_and_backward" if args.model in {"ssb-v4", "ssb-v5"}
            else "dense_standard" if args.model == "dense" else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(
        f"dataset={dataset} model={args.model} architecture={args.architecture} "
        f"keep_ratio={identity['keep_ratio']} block_size={identity['block_size']} "
        f"child_refresh_steps={identity['child_refresh_steps']} "
        f"score_refresh_steps={identity['score_refresh_steps']} "
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
        "child_refresh_steps": identity["child_refresh_steps"],
        "score_refresh_steps": identity["score_refresh_steps"],
        "v6_selection_mode": identity["v6_selection_mode"],
        "v6_gradient_retention": identity["v6_gradient_retention"],
        "v6_selection_method": identity["v6_selection_method"],
        "v6_early_bird": identity["v6_early_bird"],
        "v6_stability_window": identity["v6_stability_window"],
        "v6_stability_threshold": identity["v6_stability_threshold"],
        "seed": args.seed,
    }
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = "fixed_epochs"
    epoch_limit = args.max_epochs if args.stop_at_convergence else args.epochs
    training_wall_start = time.perf_counter()

    for epoch in range(1, epoch_limit + 1):
        train_loss, train_accuracy, epoch_time, batches, optimizer, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.lr, global_step
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        improved = val_loss < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} time={epoch_time:.2f}s "
            f"best_epoch={best_epoch} no_improve={epochs_without_improvement}"
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
                "is_best_val_loss": int(improved),
                "best_epoch_so_far": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        for row in batches:
            batch_rows.append({**csv_base, **row})

        if args.stop_at_convergence and epochs_without_improvement >= args.patience:
            stop_reason = "validation_loss_patience"
            print(f"Convergence stop at epoch {epoch}: no validation-loss improvement > {args.min_delta:g} for {args.patience} epochs.")
            break
    else:
        if args.stop_at_convergence:
            stop_reason = "max_epochs"

    total_training_wall_time_s = time.perf_counter() - training_wall_start
    metadata.update({
        "epochs_completed": len(epoch_rows),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stop_reason": stop_reason,
        "total_training_wall_time_s": total_training_wall_time_s,
        "converged_by_patience": stop_reason == "validation_loss_patience",
        "gradient_scoring_events": int(getattr(model, "scoring_event_count", 0)),
        "dense_scoring_time_s": float(getattr(model, "dense_scoring_time_s", 0.0)),
        "selector_scoring_time_s": float(getattr(model, "selector_scoring_time_s", 0.0)),
        "child_rebuild_time_s": float(getattr(model, "child_rebuild_time_s", 0.0)),
        "final_effective_keep_ratio": float(model.effective_keep_ratio()) if getattr(model, "is_v6_gradient_selected", False) else None,
        "topology_frozen": bool(getattr(model, "topology_frozen", False)),
        "topology_freeze_step": int(getattr(model, "topology_freeze_step", 0)),
        "topology_freeze_scoring_event": int(getattr(model, "topology_freeze_scoring_event", 0)),
    })
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    common_fields = [
        "run_id",
        "dataset",
        "model",
        "architecture",
        "protocol_version",
        "keep_ratio",
        "block_size",
        "child_refresh_steps",
        "score_refresh_steps",
        "v6_selection_mode",
        "v6_gradient_retention",
        "v6_selection_method",
        "v6_early_bird",
        "v6_stability_window",
        "v6_stability_threshold",
        "seed",
    ]
    write_csv(
        args.output_dir / "epochs.csv",
        epoch_rows,
        common_fields
        + ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "epoch_time_s", "is_best_val_loss", "best_epoch_so_far", "epochs_without_improvement"],
    )
    write_csv(
        args.output_dir / "batches.csv",
        batch_rows,
        common_fields + ["epoch", "batch", "global_step", "forward_time_s", "backward_time_s", "memory_bytes", "batch_accuracy", "child_refreshed", "child_refresh_count", "child_topology_before", "child_topology_after", "gradient_scoring_event", "gradient_scoring_event_count", "selector_scoring_time_s", "dense_scoring_time_s", "child_rebuild_time_s", "active_structured_units", "total_structured_units", "effective_keep_ratio", "importance_min", "importance_mean", "importance_max", "mask_distance", "topology_frozen", "topology_freeze_step", "topology_freeze_scoring_event"],
    )


if __name__ == "__main__":
    main()
