"""Single-block stroke optimization (Week 5).

Planned interface:

* :class:`PainterEngine(renderer, losses, config, device)`` - holds the stroke
  parameters of ``B`` blocks as three leaf tensors (shape, colour, alpha), runs
  RMSprop / Adam steps against the target patches, clamps shape parameters to
  ``[0.1, 0.9]`` and colour / alpha to ``[0, 1]``, composites strokes with
  ``core.differentiable_canvas.composite_sequence`` after
  ``core.morphology`` dilate / erode, and reports loss and PSNR through a
  progress callback. Saves the optimized strokes as ``.npz``.
* Loss: ``beta_l1 * PixelLoss + beta_ot * SinkhornLoss`` (Ablation A toggles the second term).
"""
from __future__ import annotations


class PainterEngine:  # pragma: no cover - stub
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("PainterEngine is scheduled for Week 5 (see ROADMAP.md)")
