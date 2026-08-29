"""Device selection and mixed-precision helpers.

The notebooks hardcoded `os.environ["CUDA_VISIBLE_DEVICES"] = "2"` (and 0, and 1,
inconsistently across files) and unconditionally used CUDA autocast. That breaks
on any machine without CUDA. Here device choice is explicit and AMP follows it.
"""
from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Pick a compute device.

    spec: "auto" | "cuda" | "cuda:N" | "mps" | "cpu"
    """
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_autocast(device: torch.device):
    """CUDA autocast where supported, a no-op elsewhere.

    Entering `autocast('cuda')` on a non-CUDA device is not merely useless --
    it raises. MPS has no working autocast implementation for these ops.
    """
    if device.type == "cuda":
        return torch.autocast("cuda")
    return nullcontext()


def make_scaler(device: torch.device) -> torch.amp.GradScaler:
    """GradScaler that is inert unless CUDA is actually in use.

    Kept (rather than dropped) on non-CUDA devices so the surrounding
    scale/unscale/step calls need no branching.
    """
    return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


def empty_cache(device: torch.device) -> None:
    """Release cached blocks between iterations.

    MPS's allocator does not reliably free memory between iterations for these
    3D models; without this it creeps past the watermark and OOMs partway
    through an epoch. Observed empirically on a 16GB machine.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return str(device)
