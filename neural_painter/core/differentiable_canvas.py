"""Differentiable alpha compositing of stroke batches onto a canvas.

The whole method rests on one equation (PLAN.md, section 2.2)::

    C_k = A_k * F_k + (1 - A_k) * C_{k-1}

Because every operation is a tensor product, gradients flow from the pixel loss
on ``C_k`` back through ``F_k`` and ``A_k`` (the neural renderer's outputs) to the
stroke parameters. Week 3 provides the compositing functions; the painter engine
(Week 5) drives them.
"""
from __future__ import annotations

import torch

from .procedural_rasterizer import CanvasColor, canvas_rgb


def blank_canvas(
    batch: int,
    size: int | tuple[int, int],
    color: CanvasColor = "black",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``(batch, 3, H, W)`` canvas filled with ``color``."""
    h, w = (size, size) if isinstance(size, int) else size
    rgb = torch.as_tensor(canvas_rgb(color), dtype=dtype, device=device).view(1, 3, 1, 1)
    return rgb.expand(batch, 3, h, w).clone()


def composite(canvas: torch.Tensor, foreground: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """One compositing step. ``alpha`` may have 1 or 3 channels; all inputs broadcast to ``(B, 3, H, W)``."""
    return foreground * alpha + canvas * (1.0 - alpha)


def composite_sequence(canvas: torch.Tensor, foregrounds: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """Composite ``N`` strokes in order.

    ``foregrounds`` is ``(B, N, 3, H, W)``, ``alphas`` is ``(B, N, 1 or 3, H, W)``.
    Strokes are applied front to back (index 0 first), matching the original
    ``_forward_pass`` loop. The loop is over strokes, not pixels, so it stays
    differentiable and cheap.
    """
    if foregrounds.dim() != 5 or alphas.dim() != 5:
        raise ValueError("foregrounds and alphas must be (B, N, C, H, W)")
    for i in range(foregrounds.shape[1]):
        canvas = composite(canvas, foregrounds[:, i], alphas[:, i])
    return canvas
