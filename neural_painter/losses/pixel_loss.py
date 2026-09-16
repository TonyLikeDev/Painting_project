"""Pixel reconstruction losses (Week 5; implemented early because it is small).

``PixelLoss`` reproduces the original ``loss.PixelLoss`` (mean of ``|C - I|^p``)
and adds the Charbonnier variant from PLAN.md section 2.3::

    L = mean( sqrt((C - I)^2 + eps^2) )

which is smooth at zero and behaves like L1 elsewhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PixelLoss(nn.Module):
    """Mean pixel error between a canvas and its target.

    ``kind`` is ``"l1"``, ``"l2"`` or ``"charbonnier"``. ``ignore_color=True``
    compares luminance (channel mean) only, as the original style-transfer mode does.
    """

    def __init__(self, kind: str = "l1", eps: float = 1e-3) -> None:
        super().__init__()
        if kind not in ("l1", "l2", "charbonnier"):
            raise ValueError("kind must be 'l1', 'l2' or 'charbonnier'")
        self.kind = kind
        self.eps = float(eps)

    def forward(self, canvas: torch.Tensor, target: torch.Tensor, ignore_color: bool = False) -> torch.Tensor:
        if ignore_color:
            canvas = canvas.mean(dim=1)
            target = target.mean(dim=1)
        diff = canvas - target
        if self.kind == "l1":
            return diff.abs().mean()
        if self.kind == "l2":
            return (diff * diff).mean()
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


def psnr(canvas: torch.Tensor, target: torch.Tensor, pixel_max: float = 1.0) -> torch.Tensor:
    """Batch PSNR in dB (original ``utils.cpt_batch_psnr``)."""
    mse = ((canvas - target) ** 2).mean()
    return 20 * torch.log10(pixel_max / torch.sqrt(mse))
