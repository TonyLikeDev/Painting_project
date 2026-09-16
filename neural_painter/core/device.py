"""Device selection shared by every module.

Replaces the five module-level ``device`` globals of the original code with one
explicit function, so a caller can always say which backend a result came from.
"""
from __future__ import annotations

import platform
from typing import Any

import torch


def get_device(preference: str | torch.device | None = None) -> torch.device:
    """Return the torch device to use.

    ``preference`` may be ``None`` / ``"auto"`` (prefer CUDA, then Apple MPS, then
    CPU) or an explicit ``"cuda"``, ``"cuda:1"``, ``"mps"`` or ``"cpu"``. An explicit
    request for an unavailable backend raises ``RuntimeError`` instead of silently
    falling back, so benchmark rows are never mislabelled.
    """
    if isinstance(preference, torch.device):
        return preference
    if preference in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(preference)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if dev.type == "mps" and not _mps_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    return dev


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def device_summary(device: torch.device | None = None) -> dict[str, Any]:
    """Collect the facts every experiment log must record (RESEARCH_PLAN.md, section 3.6)."""
    device = device or get_device()
    info: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if device.type == "cuda":
        idx = device.index or 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["vram_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / 2**30, 2)
    return info


def synchronize(device: torch.device) -> None:
    """Block until queued kernels finish, so wall-clock timings are honest on every backend."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float | None:
    """Peak allocated memory since the last reset, in GB (``None`` when the backend cannot report it)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    if device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        return torch.mps.driver_allocated_memory() / 2**30
    return None
