"""Pretrained VGG feature extractor for perceptual and style losses (Week 9, stretch).

Planned interface: :class:`VGGFeatures` wraps ``torchvision.models.vgg16`` with
ImageNet normalization and returns the activations of a configurable list of
blocks (``relu1_2``, ``relu2_2``, ``relu3_3``, ``relu4_3``). Max pooling is
replaced by average pooling for the style loss, as in the original ``loss.py``.
Uses the modern ``weights=VGG16_Weights.IMAGENET1K_V1`` API instead of the
deprecated ``pretrained=True``.
"""
from __future__ import annotations

import torch.nn as nn


class VGGFeatures(nn.Module):  # pragma: no cover - stub
    def __init__(self, blocks: tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3"), avg_pool: bool = True) -> None:
        super().__init__()
        raise NotImplementedError("VGGFeatures is scheduled for Week 9 (stretch goal, see ROADMAP.md)")
