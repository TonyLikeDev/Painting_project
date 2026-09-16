"""Train the neural renderer on synthetic strokes (Week 4).

Planned interface (RESEARCH_PLAN.md, Week 4):

* :class:`SyntheticStrokeDataset` - an ``IterableDataset`` that draws uniform
  stroke parameters and rasterizes them on the fly with
  ``core.procedural_rasterizer.ProceduralRasterizer(train=True)``; no disk storage.
  Yields ``(params, foreground, alpha)``.
* :func:`train(config)`` - Adam, L1 + SSIM (or cosine) loss on foreground and alpha,
  device from ``core.device.get_device``, checkpoints every epoch, validation PSNR /
  SSIM on a fixed held-out set of 1,000 strokes, CSV log of the training curve.
  Batch size 64 on the RTX 3070, halve on out-of-memory.

CLI: ``python -m neural_painter.pipeline.train_renderer --brush oil --light --epochs 100``.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - stub
    raise NotImplementedError("renderer training is scheduled for Week 4 (see ROADMAP.md)")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
