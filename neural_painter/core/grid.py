"""Grid splitting and stitching (Week 3).

An image is cut into an ``m x m`` grid of equal square blocks so that every block
can be painted by its own batch of strokes (the original ``img2patches`` /
``patches2img``). Blocks are ordered row-major: block ``k`` sits at row
``k // m`` and column ``k % m``. The same convention is used by
:func:`block_to_global`, which converts stroke coordinates optimized inside a
block back to whole-image coordinates (the original ``_normalize_strokes``).

The coarse-to-fine *loop* that uses these utilities is built in Week 7
(``pipeline/progressive_painter.py``); only the schedule helper lives here.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .stroke_models import BrushSpec, get_brush

Box = tuple[int, int, int, int]


# ----- tensor grid ops (differentiable) ----------------------------------------------------
def split_grid(x: torch.Tensor, m: int) -> torch.Tensor:
    """``(B, C, H, W)`` or ``(C, H, W)`` -> ``(B * m * m, C, H // m, W // m)``, row-major blocks.

    ``H`` and ``W`` must be divisible by ``m``. A pure reshape / permute, so gradients pass through.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
    b, c, h, w = x.shape
    if h % m or w % m:
        raise ValueError(f"image size {(h, w)} is not divisible by m = {m}")
    hb, wb = h // m, w // m
    x = x.reshape(b, c, m, hb, m, wb).permute(0, 2, 4, 1, 3, 5)  # (B, my, mx, C, hb, wb)
    return x.reshape(b * m * m, c, hb, wb)


def merge_grid(patches: torch.Tensor, m: int) -> torch.Tensor:
    """Inverse of :func:`split_grid`: ``(B * m * m, C, hb, wb)`` -> ``(B, C, hb * m, wb * m)``."""
    n, c, hb, wb = patches.shape
    if n % (m * m):
        raise ValueError(f"{n} patches cannot form a batch of {m}x{m} grids")
    b = n // (m * m)
    x = patches.reshape(b, m, m, c, hb, wb).permute(0, 3, 1, 4, 2, 5)  # (B, C, my, hb, mx, wb)
    return x.reshape(b, c, hb * m, wb * m)


def grid_boxes(height: int, width: int, m: int) -> list[Box]:
    """Pixel boxes ``(left, top, right, bottom)`` of the ``m x m`` blocks in row-major order."""
    hb, wb = height // m, width // m
    return [(x * wb, y * hb, (x + 1) * wb, (y + 1) * hb) for y in range(m) for x in range(m)]


# ----- numpy convenience (same semantics as the original helpers) --------------------------
def img2patches(img: np.ndarray, m: int, patch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
    """``(H, W, 3)`` float image -> ``(m * m, 3, s, s)`` tensor after resizing to ``(m * s, m * s)``.

    Like the original this *stretches* the image to a square; feed a pre-cropped
    square image (``image_io.preprocess(..., mode="crop")``) to avoid distortion.
    """
    img = cv2.resize(np.asarray(img, dtype=np.float32), (m * patch_size, m * patch_size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    patches = split_grid(t, m)
    return patches.to(device) if device is not None else patches


def patches2img(patches: torch.Tensor, m: int, to_numpy: bool = True):
    """``(m * m, 3, s, s)`` -> ``(H, W, 3)`` float32 numpy image (or ``(1, 3, H, W)`` tensor)."""
    merged = merge_grid(patches, m)
    if to_numpy:
        return merged[0].detach().cpu().numpy().transpose(1, 2, 0)
    return merged


# ----- stroke coordinate transform --------------------------------------------------------
def block_to_global(params, spec: str | BrushSpec, m: int):
    """Map stroke parameters optimized inside grid blocks to whole-image coordinates.

    ``params`` is ``(m * m, n, d)`` (numpy or torch). For block ``k`` at row ``r = k // m``
    and column ``c = k % m``: ``x -> c / m + x / m``, ``y -> r / m + y / m``, sizes ``-> size / m``.
    Relative Bezier control points (``x1, y1``) are left untouched, as in the original
    ``_normalize_strokes``. Returns a new array / tensor.
    """
    spec = get_brush(spec)
    xs, ys, rs = spec.x_indices, spec.y_indices, spec.size_indices
    n_blocks = params.shape[0]
    if n_blocks != m * m:
        raise ValueError(f"expected {m * m} blocks for m = {m}, got {n_blocks}")
    if hasattr(params, "detach"):
        v = params.detach().clone()
        k = torch.arange(n_blocks, device=v.device, dtype=v.dtype)
        y_bias = (torch.div(k, m, rounding_mode="floor") / m).view(-1, 1, 1)
        x_bias = ((k % m) / m).view(-1, 1, 1)
        xs_t, ys_t, rs_t = (torch.as_tensor(i, device=v.device) for i in (xs, ys, rs))
        v[:, :, ys_t] = y_bias + v[:, :, ys_t] / m
        v[:, :, xs_t] = x_bias + v[:, :, xs_t] / m
        v[:, :, rs_t] = v[:, :, rs_t] / m
        return v
    v = np.array(params, dtype=np.float32, copy=True)
    k = np.arange(n_blocks)
    y_bias = ((k // m) / m).reshape(-1, 1, 1).astype(np.float32)
    x_bias = ((k % m) / m).reshape(-1, 1, 1).astype(np.float32)
    v[:, :, ys] = y_bias + v[:, :, ys] / m
    v[:, :, xs] = x_bias + v[:, :, xs] / m
    v[:, :, rs] = v[:, :, rs] / m
    return v


def flatten_blocks(params: np.ndarray, order: np.ndarray | None = None) -> np.ndarray:
    """``(m * m, n, d)`` -> ``(n * m * m, d)`` interleaving blocks stroke by stroke.

    Stroke ``j`` of every block is emitted before stroke ``j + 1`` of any block, so a
    time-lapse fills the whole image evenly. ``order`` optionally permutes the blocks
    (the original shuffles them with an unseeded ``random.shuffle``; pass a permutation
    from a seeded generator instead).
    """
    v = np.asarray(params, dtype=np.float32)
    if order is not None:
        v = v[np.asarray(order)]
    return v.transpose(1, 0, 2).reshape(-1, v.shape[-1])


# ----- schedule ---------------------------------------------------------------------------
def strokes_per_block(max_strokes: int, max_divide: int) -> int:
    """Original ``stroke_parser``: the stroke budget is shared equally among all blocks of all levels."""
    total_blocks = sum(i * i for i in range(1, max_divide + 1))
    return int(max_strokes / total_blocks)


@dataclass(frozen=True)
class GridLevel:
    m: int
    strokes_per_block: int
    patch_size: int

    @property
    def blocks(self) -> int:
        return self.m * self.m

    @property
    def strokes(self) -> int:
        return self.blocks * self.strokes_per_block

    @property
    def image_size(self) -> int:
        return self.m * self.patch_size


def progressive_schedule(max_strokes: int, max_divide: int, patch_size: int) -> list[GridLevel]:
    """Levels ``1x1, 2x2, ..., MxM`` with the original's equal per-block budget."""
    if max_divide < 1:
        raise ValueError("max_divide must be >= 1")
    spb = strokes_per_block(max_strokes, max_divide)
    if spb < 1:
        raise ValueError(f"{max_strokes} strokes are too few for max_divide = {max_divide} (need at least {sum(i * i for i in range(1, max_divide + 1))})")
    return [GridLevel(m, spb, patch_size) for m in range(1, max_divide + 1)]


# ----- figures ---------------------------------------------------------------------------
def draw_grid_lines(img: np.ndarray, m: int, color=(1.0, 0.0, 0.0), thickness: int = 2) -> np.ndarray:
    """Copy of ``img`` with the ``m x m`` block borders drawn on top (for report figures)."""
    out = np.array(img, dtype=np.float32, copy=True)
    h, w = out.shape[:2]
    col = tuple(float(c) for c in color)
    for i in range(1, m):
        cv2.line(out, (i * w // m, 0), (i * w // m, h - 1), col, thickness)
        cv2.line(out, (0, i * h // m), (w - 1, i * h // m), col, thickness)
    return out
