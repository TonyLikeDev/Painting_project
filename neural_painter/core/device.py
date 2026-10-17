"""Device selection shared by every module.

Replaces the five module-level ``device`` globals of the original code with one
explicit function, so a caller can always say which backend a result came from.
"""
from __future__ import annotations

import platform
import sys
import threading
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


def set_deterministic(enabled: bool = True) -> None:
    """Make two runs of one seed give the same numbers on CUDA (it changes nothing on the CPU, which already does).

    cuDNN may pick convolution algorithms whose floating-point summation order differs from call to call, and a painting is
    a chaotic computation (500 optimizer steps through clamps and max-pools), so without this two runs of the same seed end in
    visibly different paintings (the PSNR of one image varied by up to 0.1 dB in a measured pair; the strokes were not equal).
    ``cudnn.deterministic`` fixes the algorithm choice.
    """
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cudnn.benchmark = False


def disable_power_throttling() -> bool:
    """Opt this process out of Windows' power throttling ("EcoQoS"); a no-op returning ``False`` on other systems.
    See :func:`set_power_throttling`."""
    return set_power_throttling(False)


def set_power_throttling(enabled: bool) -> bool:
    """Opt this process out of (``enabled=False``) or into (``True``, for measuring it) Windows' power throttling ("EcoQoS");
    a no-op returning ``False`` on other systems.

    Windows 11 throttles a busy process it does not consider in the foreground (a script started from a background shell, an
    IDE or an agent): its threads are moved to the efficiency cores at a low clock. On the i5-13400F of the desktop this made a
    CPU optimizer step of the light renderer 3 to 4 times slower once the first large step had run, and on ``cuda`` a
    progressive painting took 21 s instead of 12 s and the 512 px rasterization 2.7 s instead of 1.3 s (Week 7, report
    Section 4.4). Every script that paints or times calls :func:`disable_power_throttling` first.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        state = State(1, 0x1, 0x1 if enabled else 0x0)  # control EXECUTION_SPEED, and turn it on or off
        return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)))  # 4: ProcessPowerThrottling
    except (AttributeError, OSError):  # an older Windows without the call: nothing to opt out of
        return False


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def device_summary(device: torch.device | None = None) -> dict[str, Any]:
    """Collect the facts every experiment log must record (RESEARCH_PLAN.md, section 3.6)."""
    device = device or get_device()
    info: dict[str, Any] = {
        "device": str(device),
        "torch": str(torch.__version__),  # a TorchVersion (str subclass), which YAML cannot represent
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
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
    """CUDA: peak allocated memory since the last reset, in GB. MPS has no peak counter: this is the memory the Metal driver
    holds *now*. ``None`` on the CPU. For the peak of a piece of code on any backend use :class:`PeakMemory`."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    if device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        return torch.mps.driver_allocated_memory() / 2**30
    return None


def _memory_now(device: torch.device) -> int:
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    import psutil

    return int(psutil.Process().memory_info().rss)


class PeakMemory:
    """``with PeakMemory(device) as mem: ...`` then ``mem.gb``: how far memory rose above its level at the start of the block.

    * ``cuda``: the allocator's exact peak of tensor memory (``torch.cuda.max_memory_allocated`` after a reset);
    * ``mps``: the largest ``torch.mps.current_allocated_memory()`` (tensor memory) seen by a thread that samples it every
      ``interval`` seconds, because MPS keeps no peak counter, so a peak shorter than the interval can be missed;
    * ``cpu``: the largest resident set size of the process (``psutil``) seen by the same sampling thread, which counts every
      allocation of the process, not only tensors.

    The three are therefore the working memory of the code inside the block, not of the whole process; what was allocated
    before it (the renderer's weights, the CUDA context) is not counted.
    """

    def __init__(self, device: torch.device, interval: float = 0.005) -> None:
        self.device, self.interval = device, interval
        self.gb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self._peak = max(self._peak, _memory_now(self.device))

    def __enter__(self) -> "PeakMemory":
        synchronize(self.device)
        if self.device.type == "cuda":
            self._start = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            self._start = self._peak = _memory_now(self.device)
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        synchronize(self.device)
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device)
        else:
            self._stop.set()
            self._thread.join()
            peak = max(self._peak, _memory_now(self.device))
        self.gb = max(0, peak - self._start) / 2**30
