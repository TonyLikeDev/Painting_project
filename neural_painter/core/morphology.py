"""Grey-scale morphology on batched tensors.

Vectorized replacement for the original ``Erosion2d`` / ``Dilation2d`` modules,
which looped over channels with ``nn.Unfold`` and mutated their input in place.
``max_pool2d`` pads with ``-inf``, so border pixels are handled exactly like the
original's ``+-1e9`` constant padding: the window simply ignores the outside.

Used by the painter engine (Week 5) to mimic the rasterizer's inference-time
dilate-foreground / erode-alpha step on the neural renderer's outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dilation2d(x: torch.Tensor, m: int = 1) -> torch.Tensor:
    """Max over a ``(2m+1) x (2m+1)`` window, same spatial size. ``x`` is ``(N, C, H, W)``."""
    if m <= 0:
        return x
    return F.max_pool2d(x, kernel_size=2 * m + 1, stride=1, padding=m)


def erosion2d(x: torch.Tensor, m: int = 1) -> torch.Tensor:
    """Min over a ``(2m+1) x (2m+1)`` window, same spatial size."""
    if m <= 0:
        return x
    return -F.max_pool2d(-x, kernel_size=2 * m + 1, stride=1, padding=m)


class Dilation2d(nn.Module):
    def __init__(self, m: int = 1) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dilation2d(x, self.m)


class Erosion2d(nn.Module):
    def __init__(self, m: int = 1) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return erosion2d(x, self.m)
