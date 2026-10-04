"""Error-map guided stroke initialization (Week 5).

Implements section 2.8.1 of ``report/chapters/02_theory.md`` and the sampler of the original
``painter.stroke_sampler`` / ``renderer.random_stroke_params_sampler``:

1. the error map of a block is ``E = sum_c |I - C|``, box-blurred with a kernel of ``size / 8`` pixels and raised to
   the 4th power, so that strokes are placed where the canvas is wrong, and mostly where it is *very* wrong;
2. the map is upsampled (bilinear) to ``resolution`` pixels (the original: 512), turned into a probability and one
   pixel is drawn from it, giving the stroke centre ``(cx, cy)``;
3. the target colour at that pixel and the brush-specific size / angle / opacity ranges give the initial parameters
   (``ProceduralRasterizer.initial_params_at``).

Differences from the original: the error map is computed for the whole batch of blocks on the device (the original
looped over blocks on the CPU), the blur is :func:`box_blur` (identical to ``cv2.blur``, checked in the tests), the
draw is an inverse-CDF lookup on a float64 cumulative sum (the original called ``np.random.choice`` over 262,144
categories per block and stroke), and everything uses one explicit ``numpy.random.Generator`` so a run is
reproducible from a seed.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def box_blur(x: torch.Tensor, ksize: int) -> torch.Tensor:
    """Normalized box filter on ``(B, C, H, W)``, equal to ``cv2.blur(img, (ksize, ksize))`` for every kernel size.

    OpenCV centres the window with an integer anchor ``ksize // 2`` and reflects the border without repeating the edge
    pixel (``BORDER_REFLECT_101``), so an even kernel such as the 4 pixels used for 32 px blocks is asymmetric:
    ``ksize // 2`` pixels before the centre and ``ksize - ksize // 2 - 1`` after.
    """
    if ksize <= 1:
        return x
    before, after = ksize // 2, ksize - ksize // 2 - 1
    padded = F.pad(x, (before, after, before, after), mode="reflect")
    return F.avg_pool2d(padded, ksize, stride=1)


def error_map(target: torch.Tensor, canvas: torch.Tensor, blur_frac: float = 1 / 8, power: float = 4.0) -> torch.Tensor:
    """Unnormalized sampling weights ``(B, 1, H, W)``: ``box_blur(sum_c |target - canvas|, H * blur_frac) ** power``."""
    err = (target - canvas).abs().sum(dim=1, keepdim=True).detach()
    return box_blur(err, int(err.shape[-1] * blur_frac)) ** power


class StrokeSampler:
    """Draws one new stroke per block from the error map of the current canvas.

    ``rasterizer`` supplies the brush (``initial_params_at`` and the YAML size / angle / opacity ranges). ``rng`` seeds
    every random choice; when given it replaces the rasterizer's own generator. ``resolution`` is the side of the grid
    the error map is upsampled to before drawing (512, as in the original, so centres fall on multiples of 1/512).
    """

    def __init__(self, rasterizer, rng: int | np.random.Generator | None = None, resolution: int = 512) -> None:
        self.rasterizer = rasterizer
        if rng is not None:
            rasterizer.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.rng: np.random.Generator = rasterizer.rng
        self.resolution = int(resolution)

    def positions(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Draw one centre ``(cy, cx)`` in ``[0, 1)`` per block from ``(B, H, W)`` non-negative sampling weights."""
        r = self.resolution
        cy, cx = np.empty(len(weights)), np.empty(len(weights))
        for i, w in enumerate(weights):
            p = cv2.resize(np.ascontiguousarray(w, dtype=np.float32), (r, r)).astype(np.float64)
            p[p < 0] = 0.0
            if not p.any():  # a perfect canvas: every place is as good as any other
                p = np.ones_like(p)
            cdf = np.cumsum(p.ravel())
            index = min(int(np.searchsorted(cdf, self.rng.random() * cdf[-1], side="right")), p.size - 1)
            cy[i], cx[i] = (index // r) / r, (index % r) / r
        return cy, cx

    def sample(self, target: torch.Tensor, canvas: torch.Tensor, uniform: bool = False) -> np.ndarray:
        """New initial parameters ``(B, d)`` for the blocks of ``target`` and ``canvas`` (both ``(B, 3, S, S)``).

        ``uniform=True`` draws the centres uniformly over the block instead of from the error map (the colour is still the
        target's at the centre). The original never does this; it exists for the ablation of the initialization.
        """
        if uniform:
            weights = np.ones((target.shape[0], *target.shape[-2:]), dtype=np.float32)
        else:
            weights = error_map(target, canvas)[:, 0].cpu().numpy()
        cy, cx = self.positions(weights)
        size = target.shape[-1]
        image = target.detach().permute(0, 2, 3, 1).cpu().numpy()  # (B, S, S, 3), RGB
        out = np.empty((len(cy), self.rasterizer.dim), dtype=np.float32)
        for i in range(len(cy)):
            color = image[i, int(cy[i] * size), int(cx[i] * size), :]  # nearest pixel, as the original
            out[i] = self.rasterizer.initial_params_at(cx[i], cy[i], color)
        return out
