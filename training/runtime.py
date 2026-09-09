"""Shared runtime helpers for training and tests."""
import random
import subprocess
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn

PROCESS = psutil.Process()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_model_parameters(model, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in model.modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, nn.Parameter) and weight.ndim in (2, 4):
                nn.init.xavier_uniform_(weight)
                bias = getattr(module, "bias", None)
                if isinstance(bias, nn.Parameter):
                    nn.init.zeros_(bias)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def memory_bytes(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return PROCESS.memory_info().rss


def git_commit(root="."):
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(root)), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
