"""Neural renderer G_phi (Week 4).

Planned interface (RESEARCH_PLAN.md, Week 4):

* :class:`NeuralRenderer` - FusionNet-style generator. ``forward(params)`` takes
  ``(N, d)`` stroke parameters in ``[0, 1]`` and returns ``(foreground, alpha)``,
  both ``(N, 3, S, S)`` in ``[0, 1]``, where ``S`` is 128 (full) or 32 (light).
  Internally: a shape decoder (MLP + PixelShuffle, fed the ``shape_dim`` first
  parameters) predicts the mask, a colour decoder (transposed convolutions, fed
  all parameters) predicts the RGB patch, and the fusion step returns
  ``colour * mask`` and ``alpha * mask`` (``alpha`` forced to 1 for the oil brush).
* :func:`load_original_checkpoint` - adapter that maps the original
  ``zou-fusion-net`` / ``zou-fusion-net-light`` ``last_ckpt.pt`` state dicts onto
  :class:`NeuralRenderer` so both renderers share one interface.

See ``report/notes/original_code/networks.md`` for the layer table of the original.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..core.stroke_models import BrushSpec, get_brush


class NeuralRenderer(nn.Module):
    """Differentiable stroke renderer. Implemented in Week 4."""

    def __init__(self, brush: str | BrushSpec, light: bool = False) -> None:
        super().__init__()
        self.spec = get_brush(brush)
        self.light = light
        self.out_size = int(self.spec.renderer["out_size_light" if light else "out_size_full"])
        raise NotImplementedError("NeuralRenderer is scheduled for Week 4 (see ROADMAP.md)")

    def forward(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError


def load_original_checkpoint(path: str | Path, brush: str | BrushSpec, light: bool = False) -> NeuralRenderer:
    """Build a :class:`NeuralRenderer` from an original ``last_ckpt.pt``. Implemented in Week 4."""
    raise NotImplementedError("checkpoint adapter is scheduled for Week 4 (see ROADMAP.md)")
