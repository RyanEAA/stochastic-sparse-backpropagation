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


def train_epoch(
    model, loader, optimizer, criterion, device, epoch, learning_rate,
    global_step_start=0, detailed_timing=False, record_batch_metrics=False,
    master_training=False,
):
    model.train()
    loss_sum = correct = total = 0
    batches = []
    start = time.perf_counter()
    global_step = global_step_start
    # MPS does not support float64. Training losses are float32, so accumulating
    # them in float32 keeps this path portable across MPS, CUDA, and CPU.
    loss_accumulator = torch.zeros((), device=device, dtype=torch.float32)
    correct_accumulator = torch.zeros((), device=device, dtype=torch.long)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (x, y) in enumerate(loader):
        batch_wall_start = time.perf_counter()
        if detailed_timing:
            synchronize(device)
        transfer_start = time.perf_counter()
        x, y = x.to(device), y.to(device)
        if detailed_timing:
            synchronize(device)
        data_transfer_time_s = time.perf_counter() - transfer_start if detailed_timing else 0.0
        optimized_instrumentation = bool(getattr(model, "is_v7_optimized", False))
        refresh_count_before = int(getattr(model, "refresh_count", 0))
        scoring_due = bool(
            getattr(model, "is_v6_gradient_selected", False) and model.scoring_due()
        )
        topology_before = (
            model.topology_signature()
            if hasattr(model, "topology_signature") and (not optimized_instrumentation or scoring_due)
            else ""
        )
        scoring_event = False
        dense_correction_event = bool(
            not master_training
            and hasattr(model, "dense_correction_due")
            and model.dense_correction_due()
        )
        selection_refresh_time_s = 0.0
        selector_scoring_time_s = 0.0
        dense_scoring_time_s = 0.0
        child_rebuild_time_s = 0.0
        if dense_correction_event:
            if detailed_timing:
                synchronize(device)
            selection_start = time.perf_counter()
            optimizer = model.prepare_dense_correction(optimizer, learning_rate)
            if detailed_timing:
                synchronize(device)
            selection_refresh_time_s = time.perf_counter() - selection_start if detailed_timing else 0.0
        elif getattr(model, "is_v6_gradient_selected", False) and not master_training:
            if detailed_timing:
                synchronize(device)
            selection_start = time.perf_counter()
            optimizer, scoring_event = model.score_and_refresh(
                x, y, criterion, optimizer, learning_rate
            )
            if detailed_timing:
                synchronize(device)
            selection_refresh_time_s = time.perf_counter() - selection_start if detailed_timing else 0.0
            if scoring_event:
                selector_scoring_time_s = model.last_selector_scoring_time_s
                dense_scoring_time_s = model.last_dense_scoring_time_s
                child_rebuild_time_s = model.last_child_rebuild_time_s
        if detailed_timing:
            synchronize(device)
        zero_grad_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if detailed_timing:
            synchronize(device)
        zero_grad_time_s = time.perf_counter() - zero_grad_start if detailed_timing else 0.0
        synchronize(device)
        forward_start = time.perf_counter()
        output = model.master(x) if (master_training or dense_correction_event) else model(x)
        synchronize(device)
        forward_time = time.perf_counter() - forward_start
        loss_start = time.perf_counter()
        loss = criterion(output, y)
        if detailed_timing:
            synchronize(device)
        loss_time_s = time.perf_counter() - loss_start if detailed_timing else 0.0
        if detailed_timing:
            synchronize(device)
        backward_start = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward_time = time.perf_counter() - backward_start
        memory_start = time.perf_counter()
        memory = memory_bytes(device)
        memory_measurement_time_s = time.perf_counter() - memory_start if detailed_timing else 0.0
        if detailed_timing:
            synchronize(device)
        optimizer_start = time.perf_counter()
        optimizer.step()
        if detailed_timing:
            synchronize(device)
        optimizer_step_time_s = time.perf_counter() - optimizer_start if detailed_timing else 0.0
        global_step += 1
        if detailed_timing:
            synchronize(device)
        post_step_start = time.perf_counter()
        if dense_correction_event:
            optimizer = model.finish_dense_correction(optimizer, learning_rate)
            scoring_event = model.last_correction_scoring_event
            selector_scoring_time_s = model.last_selector_scoring_time_s
            child_rebuild_time_s = model.last_child_rebuild_time_s
        elif master_training:
            pass
        elif getattr(model, "is_v5_structured_child", False):
            optimizer = model.after_optimizer_step(optimizer, learning_rate)
        elif hasattr(model, "after_optimizer_step") and model.after_optimizer_step():
            # V4 creates a new physically smaller child; its Parameters are new objects.
            # Rebuild Adam for the new child. This reset is recorded in metadata and is
            # intentionally part of the V4 baseline.
            optimizer = optim.Adam(model.training_parameters(), lr=learning_rate)
        if detailed_timing:
            synchronize(device)
        post_step_time_s = time.perf_counter() - post_step_start if detailed_timing else 0.0

        instrumentation_start = time.perf_counter()
        refresh_count_after = int(getattr(model, "refresh_count", 0))
        topology_after = (
            model.topology_signature()
            if hasattr(model, "topology_signature") and (not optimized_instrumentation or scoring_event)
            else ""
        )
        child_refreshed = refresh_count_after > refresh_count_before
        importance_min, importance_mean, importance_max = (
            model.importance_statistics()
            if getattr(model, "is_v6_gradient_selected", False)
            and (not optimized_instrumentation or scoring_event)
            else (0.0, 0.0, 0.0)
        )

        n = y.size(0)
        batch_correct_tensor = (output.argmax(1) == y).sum()
        loss_accumulator += loss.detach().to(torch.float32) * n
        correct_accumulator += batch_correct_tensor
        if record_batch_metrics:
            loss_sum += loss.item() * n
            batch_correct = batch_correct_tensor.item()
            correct += batch_correct
            batch_accuracy = batch_correct / n
        else:
            batch_accuracy = float("nan")
        total += n
        if detailed_timing:
            synchronize(device)
        instrumentation_time_s = time.perf_counter() - instrumentation_start
        row = dict(
                epoch=epoch,
                batch=batch_index,
                forward_time_s=forward_time,
                backward_time_s=backward_time,
                data_transfer_time_s=data_transfer_time_s,
                selection_refresh_time_s=selection_refresh_time_s,
                zero_grad_time_s=zero_grad_time_s,
                loss_time_s=loss_time_s,
                memory_measurement_time_s=memory_measurement_time_s,
                optimizer_step_time_s=optimizer_step_time_s,
                post_step_time_s=post_step_time_s,
                instrumentation_time_s=instrumentation_time_s,
                detailed_timing=int(detailed_timing),
                memory_bytes=memory,
                batch_accuracy=batch_accuracy,
                global_step=global_step,
                child_refreshed=int(child_refreshed),
                child_refresh_count=refresh_count_after,
                child_topology_before=topology_before,
                child_topology_after=topology_after,
                gradient_scoring_event=int(scoring_event),
                dense_correction_event=int(dense_correction_event),
                dense_warmup_event=int(master_training),
                gradient_scoring_event_count=int(getattr(model, "scoring_event_count", 0)),
                selector_scoring_time_s=selector_scoring_time_s,
                dense_scoring_time_s=dense_scoring_time_s,
                child_rebuild_time_s=child_rebuild_time_s,
                active_structured_units=(model.active_structured_units() if getattr(model, "is_v6_gradient_selected", False) else 0),
                total_structured_units=(model.total_structured_units() if getattr(model, "is_v6_gradient_selected", False) else 0),
                effective_keep_ratio=(model.effective_keep_ratio() if getattr(model, "is_v6_gradient_selected", False) else 1.0),
                effective_parameter_ratio=(model.effective_parameter_ratio() if getattr(model, "is_v6_gradient_selected", False) else 1.0),
                importance_min=importance_min,
                importance_mean=importance_mean,
                importance_max=importance_max,
                mask_distance=float(getattr(model, "last_mask_distance", float("nan"))),
                topology_frozen=int(getattr(model, "topology_frozen", False)),
                topology_freeze_step=int(getattr(model, "topology_freeze_step", 0)),
                topology_freeze_scoring_event=int(getattr(model, "topology_freeze_scoring_event", 0)),
            )
        if detailed_timing:
            synchronize(device)
        batch_wall_time_s = time.perf_counter() - batch_wall_start if detailed_timing else 0.0
        accounted = sum((
            data_transfer_time_s, selection_refresh_time_s, zero_grad_time_s,
            forward_time, loss_time_s, backward_time, memory_measurement_time_s,
            optimizer_step_time_s, post_step_time_s, instrumentation_time_s,
        ))
        row["batch_wall_time_s"] = batch_wall_time_s
        row["unaccounted_time_s"] = max(0.0, batch_wall_time_s - accounted)
        batches.append(row)
    synchronize(device)
    if not record_batch_metrics:
        loss_sum = float(loss_accumulator.item())
        correct = int(correct_accumulator.item())
    return loss_sum / total, correct / total, time.perf_counter() - start, batches, optimizer, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, target="default"):
    if target == "child":
        if getattr(model, "child", None) is None:
            raise RuntimeError("Cannot evaluate an uninitialized structured child.")
        evaluation_model = model.child
        evaluation_model.eval()
    else:
        if target != "master_no_sync" and hasattr(model, "sync_child_to_master"):
            model.sync_child_to_master()
        model.eval()
        evaluation_model = model
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        synchronize(device)
        forward_start = time.perf_counter()
        output = evaluation_model(x)
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
    parser.add_argument("--v6-selection-method", choices=["random", "gradient_l2", "weight_l2", "taylor"], default="gradient_l2", help="Structured-unit score used by V6/V7; random is a matched selector control.")
    parser.add_argument("--v6-early-bird", action="store_true", help="Freeze the V6 topology once recent mask distances are stable.")
    parser.add_argument("--v6-stability-window", type=int, default=5, help="Number of consecutive mask distances used by Early-Bird.")
    parser.add_argument("--v6-stability-threshold", type=float, default=0.10, help="Maximum mask replacement fraction considered stable.")
    parser.add_argument("--v7-dense-warmup-epochs", type=int, default=0, help="For V7: train the full master for this many initial epochs.")
    parser.add_argument("--v7-dense-correction-steps", type=int, default=0, help="For V7: replace one batch with a dense update after this many sparse updates; 0 disables corrections.")
    parser.add_argument("--v7-layer-keep-ratios", nargs="+", type=float, default=None, help="For V7 fixed selection: one keep ratio per hidden conv/linear layer, in forward order.")
    parser.add_argument("--v7-target-parameter-ratio", type=float, default=None, help="For V7: scale the layer profile to the closest realizable child/master parameter ratio.")
    parser.add_argument("--v7-hybrid-config-label", default="none", help="Descriptive V7 strategy label recorded by the grid orchestrator.")
    parser.add_argument("--v7-early-bird-min-events", type=int, default=0, help="Do not freeze a V7 mask before this many scoring events.")
    parser.add_argument("--v7-early-bird-min-steps", type=int, default=0, help="Do not freeze a V7 mask before this many sparse-phase optimizer steps.")
    parser.add_argument("--timing-detail", choices=["basic", "full"], default="basic", help="Full adds synchronized per-stage profiling; basic minimizes benchmark overhead.")
    parser.add_argument("--record-batch-metrics", action="store_true", help="Record per-batch accuracy; disabled by default to avoid a device synchronization every batch.")
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
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of the training split reserved for early-stopping validation.")
    parser.add_argument("--split-seed", type=int, default=2026, help="Fixed seed for the train/validation split; keep matched across methods and run seeds.")
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
    if args.model in {"ssb-v6", "ssb-v7"} and args.score_refresh_steps <= 0:
        raise ValueError("score_refresh_steps must be positive for SSB-V6/V7.")
    if args.model in {"ssb-v6", "ssb-v7"} and not 0 < args.v6_gradient_retention <= 1:
        raise ValueError("v6_gradient_retention must be in (0, 1].")
    if args.v6_stability_window < 1:
        raise ValueError("v6_stability_window must be >= 1.")
    if not 0 <= args.v6_stability_threshold <= 1:
        raise ValueError("v6_stability_threshold must be in [0, 1].")
    if args.v7_dense_warmup_epochs < 0:
        raise ValueError("v7_dense_warmup_epochs must be >= 0.")
    if args.v7_dense_correction_steps < 0:
        raise ValueError("v7_dense_correction_steps must be >= 0.")
    if args.v7_early_bird_min_events < 0 or args.v7_early_bird_min_steps < 0:
        raise ValueError("V7 Early-Bird minimums must be >= 0.")
    if args.v7_layer_keep_ratios and any(
        ratio <= 0 or ratio > 1 for ratio in args.v7_layer_keep_ratios
    ):
        raise ValueError("v7_layer_keep_ratios must all be in (0, 1].")
    if args.v7_target_parameter_ratio is not None and not 0 < args.v7_target_parameter_ratio <= 1:
        raise ValueError("v7_target_parameter_ratio must be in (0, 1].")
    if args.model != "ssb-v7" and (
        args.v7_dense_warmup_epochs
        or args.v7_dense_correction_steps
        or args.v7_layer_keep_ratios is not None
        or args.v7_target_parameter_ratio is not None
        or args.v7_early_bird_min_events
        or args.v7_early_bird_min_steps
    ):
        raise ValueError("V7 hybrid options require --model ssb-v7.")
    if args.stop_at_convergence and args.max_epochs < 1:
        raise ValueError("max_epochs must be >= 1.")
    if args.stop_at_convergence and args.patience < 1:
        raise ValueError("patience must be >= 1.")
    if args.min_delta < 0:
        raise ValueError("min_delta must be >= 0.")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")

    set_seed(args.seed)
    device = get_device(args.device)
    train_loader, val_loader, test_loader = build_loaders(
        dataset, args.batch_size, args.seed, args.subset, args.data_dir, args.num_workers,
        validation_fraction=args.validation_fraction, split_seed=args.split_seed,
        include_test=True,
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
        v7_dense_correction_steps=args.v7_dense_correction_steps,
        v7_layer_keep_ratios=args.v7_layer_keep_ratios,
        v7_target_parameter_ratio=args.v7_target_parameter_ratio,
        v7_early_bird_min_events=args.v7_early_bird_min_events,
        v7_early_bird_min_steps=args.v7_early_bird_min_steps,
    )
    initialize_model_parameters(model, args.seed)
    model = model.to(device)
    set_seed(args.seed)
    if hasattr(model, "refresh_child"):
        model.refresh_child()

    optimizer_parameters = model.training_parameters() if hasattr(model, "training_parameters") else model.parameters()
    if args.model == "ssb-v7" and args.v7_dense_warmup_epochs > 0:
        optimizer = model.make_master_optimizer(args.lr)
    elif getattr(model, "is_v5_structured_child", False):
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
        "score_refresh_steps": args.score_refresh_steps if args.model in {"ssb-v6", "ssb-v7"} else 0,
        "v6_selection_mode": args.v6_selection_mode if args.model in {"ssb-v6", "ssb-v7"} else None,
        "v6_gradient_retention": args.v6_gradient_retention if args.model in {"ssb-v6", "ssb-v7"} and args.v6_selection_mode == "gradient_retention" else None,
        "v6_selection_method": args.v6_selection_method if args.model in {"ssb-v6", "ssb-v7"} else None,
        "v6_early_bird": bool(args.v6_early_bird) if args.model in {"ssb-v6", "ssb-v7"} else False,
        "v6_stability_window": args.v6_stability_window if args.model in {"ssb-v6", "ssb-v7"} else 0,
        "v6_stability_threshold": args.v6_stability_threshold if args.model in {"ssb-v6", "ssb-v7"} else None,
        "v7_dense_warmup_epochs": args.v7_dense_warmup_epochs if args.model == "ssb-v7" else 0,
        "v7_dense_correction_steps": args.v7_dense_correction_steps if args.model == "ssb-v7" else 0,
        "v7_layer_keep_ratios": args.v7_layer_keep_ratios if args.model == "ssb-v7" else None,
        "v7_target_parameter_ratio": args.v7_target_parameter_ratio if args.model == "ssb-v7" else None,
        "v7_hybrid_config": args.v7_hybrid_config_label if args.model == "ssb-v7" else "none",
        "v7_early_bird_min_events": args.v7_early_bird_min_events if args.model == "ssb-v7" else 0,
        "v7_early_bird_min_steps": args.v7_early_bird_min_steps if args.model == "ssb-v7" else 0,
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
        "validation_fraction": args.validation_fraction,
        "split_seed": args.split_seed,
        "timing_detail": args.timing_detail,
        "record_batch_metrics": bool(args.record_batch_metrics),
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
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "memory_metric": "cuda_peak_allocated_after_backward" if device.type == "cuda" else "process_rss_after_backward",
        "cnn_scope": (
            "structured_dense_child_conv_and_classifier" if args.architecture == "cnn" and args.model in {"ssb-v4", "ssb-v5", "ssb-v6", "ssb-v7"}
            else "dense_forward_structured_backward_conv_and_classifier" if args.architecture == "cnn" and args.model == "ssb-v5.1"
            else "dense_conv_backbone_ssb_linear_classifier" if args.architecture == "cnn" and args.model.startswith("ssb-")
            else None
        ),
        "structured_child_optimizer_state_policy": (
            "adam_state_resets_when_child_is_resampled" if args.model == "ssb-v4"
            else "master_owned_persistent_adam_moments" if args.model in {"ssb-v5", "ssb-v5.1", "ssb-v6", "ssb-v7"}
            else None
        ),
        "structured_child_master_parameters": model.master_parameter_count() if hasattr(model, "master_parameter_count") else None,
        "structured_child_initial_child_parameters": model.child_parameter_count() if hasattr(model, "child_parameter_count") else None,
        "structured_child_initial_parameter_ratio": model.effective_parameter_ratio() if hasattr(model, "effective_parameter_ratio") else None,
        "v7_calibrated_layer_keep_ratios": list(model.layer_keep_ratios) if args.model == "ssb-v7" and model.layer_keep_ratios is not None else None,
        "forward_backward_policy": (
            "dense_forward_structured_sparse_backward" if args.model == "ssb-v5.1"
            else "structured_child_forward_and_backward_with_periodic_dense_gradient_scoring" if args.model in {"ssb-v6", "ssb-v7"}
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
        "v7_dense_warmup_epochs": identity["v7_dense_warmup_epochs"],
        "v7_dense_correction_steps": identity["v7_dense_correction_steps"],
        "v7_layer_keep_ratios": json.dumps(identity["v7_layer_keep_ratios"]),
        "v7_target_parameter_ratio": identity["v7_target_parameter_ratio"],
        "v7_hybrid_config": identity["v7_hybrid_config"],
        "v7_early_bird_min_events": identity["v7_early_bird_min_events"],
        "v7_early_bird_min_steps": identity["v7_early_bird_min_steps"],
        "seed": args.seed,
        "validation_fraction": identity["validation_fraction"],
        "split_seed": identity["split_seed"],
        "timing_detail": args.timing_detail,
        "record_batch_metrics": bool(args.record_batch_metrics),
    }
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = "fixed_epochs"
    epoch_limit = args.max_epochs if args.stop_at_convergence else args.epochs
    training_wall_start = time.perf_counter()

    for epoch in range(1, epoch_limit + 1):
        master_training = bool(
            args.model == "ssb-v7" and epoch <= args.v7_dense_warmup_epochs
        )
        if (
            args.model == "ssb-v7"
            and args.v7_dense_warmup_epochs > 0
            and epoch == args.v7_dense_warmup_epochs + 1
        ):
            model.prepare_sparse_after_warmup(optimizer)
            optimizer = None
        train_loss, train_accuracy, epoch_time, batches, optimizer, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.lr, global_step,
            detailed_timing=args.timing_detail == "full",
            record_batch_metrics=args.record_batch_metrics,
            master_training=master_training,
        )
        child_val_loss = child_val_accuracy = float("nan")
        if args.model == "ssb-v7" and not master_training:
            child_val_loss, child_val_accuracy = evaluate(
                model, val_loader, criterion, device, target="child"
            )
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
            target="master_no_sync" if master_training else "default",
        )

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
            f"child_val_acc={child_val_accuracy:.4f} "
            f"best_epoch={best_epoch} no_improve={epochs_without_improvement}"
        )
        epoch_rows.append(
            {
                **csv_base,
                "epoch": epoch,
                "training_phase": "dense_warmup" if master_training else "sparse_or_hybrid",
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "child_val_loss": child_val_loss,
                "child_val_accuracy": child_val_accuracy,
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
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    child_test_loss = child_test_accuracy = float("nan")
    if args.model == "ssb-v7" and getattr(model, "child", None) is not None:
        child_test_loss, child_test_accuracy = evaluate(
            model, test_loader, criterion, device, target="child"
        )
    epoch_rows[-1].update({
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
        "final_child_test_loss": child_test_loss,
        "final_child_test_accuracy": child_test_accuracy,
    })
    for row in epoch_rows[:-1]:
        row.update({
            "final_test_loss": float("nan"),
            "final_test_accuracy": float("nan"),
            "final_child_test_loss": float("nan"),
            "final_child_test_accuracy": float("nan"),
        })
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
        "final_effective_parameter_ratio": float(model.effective_parameter_ratio()) if getattr(model, "is_v6_gradient_selected", False) else None,
        "topology_frozen": bool(getattr(model, "topology_frozen", False)),
        "topology_freeze_step": int(getattr(model, "topology_freeze_step", 0)),
        "topology_freeze_scoring_event": int(getattr(model, "topology_freeze_scoring_event", 0)),
        "dense_correction_count": int(getattr(model, "dense_correction_count", 0)),
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
        "final_child_test_loss": child_test_loss,
        "final_child_test_accuracy": child_test_accuracy,
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
        "v7_dense_warmup_epochs",
        "v7_dense_correction_steps",
        "v7_layer_keep_ratios",
        "v7_target_parameter_ratio",
        "v7_hybrid_config",
        "v7_early_bird_min_events",
        "v7_early_bird_min_steps",
        "seed",
        "validation_fraction",
        "split_seed",
        "timing_detail",
        "record_batch_metrics",
    ]
    write_csv(
        args.output_dir / "epochs.csv",
        epoch_rows,
        common_fields
        + ["epoch", "training_phase", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "child_val_loss", "child_val_accuracy", "final_test_loss", "final_test_accuracy", "final_child_test_loss", "final_child_test_accuracy", "epoch_time_s", "is_best_val_loss", "best_epoch_so_far", "epochs_without_improvement"],
    )
    write_csv(
        args.output_dir / "batches.csv",
        batch_rows,
        common_fields + ["epoch", "batch", "global_step", "forward_time_s", "backward_time_s", "detailed_timing", "data_transfer_time_s", "selection_refresh_time_s", "zero_grad_time_s", "loss_time_s", "memory_measurement_time_s", "optimizer_step_time_s", "post_step_time_s", "instrumentation_time_s", "batch_wall_time_s", "unaccounted_time_s", "memory_bytes", "batch_accuracy", "child_refreshed", "child_refresh_count", "child_topology_before", "child_topology_after", "gradient_scoring_event", "dense_correction_event", "dense_warmup_event", "gradient_scoring_event_count", "selector_scoring_time_s", "dense_scoring_time_s", "child_rebuild_time_s", "active_structured_units", "total_structured_units", "effective_keep_ratio", "effective_parameter_ratio", "importance_min", "importance_mean", "importance_max", "mask_distance", "topology_frozen", "topology_freeze_step", "topology_freeze_scoring_event"],
    )


if __name__ == "__main__":
    main()
