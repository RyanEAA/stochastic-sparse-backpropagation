import random, time
import numpy as np
import psutil
import torch

PROCESS = psutil.Process()

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device(name="auto"):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"): torch.mps.synchronize()

def memory_bytes(device):
    if device.type == "cuda": return torch.cuda.max_memory_allocated(device)
    return PROCESS.memory_info().rss
