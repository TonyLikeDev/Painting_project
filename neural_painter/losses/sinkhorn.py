"""Entropy-regularized optimal transport (Sinkhorn) loss (Week 5).

Planned interface, matching the theory in ``report/chapters/02_theory.md`` section 2.6
and the original ``pytorch_batch_sinkhorn.py``:

* :func:`sinkhorn_loss(x, y, mass_x, mass_y, epsilon, n_iter)` - batched, log-domain
  Sinkhorn on point clouds ``x, y`` of shape ``(B, n, 2)`` with masses ``(B, n)``;
  returns the mean transport cost ``<pi, C>`` with ``C_ij = |x_i - y_j|^2``.
* :class:`SinkhornLoss(epsilon=0.01, n_iter=5, size=24, normalize=False)`` - takes a
  canvas and a target ``(B, 3, H, W)``, area-downsamples both to ``size x size``,
  uses pixel intensities of one colour channel as masses (random channel per call
  in the original; ``channel="mean"`` will be offered as a deterministic option) and
  returns the Sinkhorn cost. ``normalize=True`` returns the debiased
  ``2 W(x, y) - W(x, x) - W(y, y)``.

Device is taken from the inputs; no module-level globals.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def sinkhorn_loss(x: torch.Tensor, y: torch.Tensor, mass_x: torch.Tensor, mass_y: torch.Tensor, epsilon: float = 0.01, n_iter: int = 5) -> torch.Tensor:  # pragma: no cover - stub
    raise NotImplementedError("sinkhorn_loss is scheduled for Week 5 (see ROADMAP.md)")


class SinkhornLoss(nn.Module):  # pragma: no cover - stub
    def __init__(self, epsilon: float = 0.01, n_iter: int = 5, size: int = 24, normalize: bool = False, channel: str = "random") -> None:
        super().__init__()
        raise NotImplementedError("SinkhornLoss is scheduled for Week 5 (see ROADMAP.md)")
