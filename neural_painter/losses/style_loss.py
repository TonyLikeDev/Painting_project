"""Gram-matrix style loss on VGG features (Week 9, stretch).

Planned interface: :class:`StyleLoss(transfer_mode)` with ``transfer_mode`` 0
(colour only: blocks relu1_2 and relu2_2) or 1 (colour and texture: four blocks),
computing ``sum_l || G(phi_l(C)) - G(phi_l(I_style)) ||_F^2`` with
``G = F F^T / (C H W)``, exactly as ``loss.VGGStyleLoss`` in the original code.
"""
from __future__ import annotations

import torch.nn as nn


class StyleLoss(nn.Module):  # pragma: no cover - stub
    def __init__(self, transfer_mode: int = 1) -> None:
        super().__init__()
        raise NotImplementedError("StyleLoss is scheduled for Week 9 (stretch goal, see ROADMAP.md)")
