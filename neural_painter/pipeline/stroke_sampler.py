"""Error-map guided stroke initialization (Week 5).

Planned interface, matching ``report/chapters/02_theory.md`` section 2.8:

* :func:`error_map(target, canvas, blur_frac=1/8, power=4)`` - ``E = sum_c |I - C|``
  per block, box-blurred with kernel ``W * blur_frac`` and raised to ``power``, then
  normalized to a probability map. Batched on tensors (the original looped over
  blocks with OpenCV on the CPU).
* :class:`StrokeSampler(rasterizer, rng)`` - samples a pixel from the probability
  map per block, reads the target colour there, and builds the brush-specific
  initial parameter vector through
  ``ProceduralRasterizer.initial_params_at(cx, cy, color)``.
"""
from __future__ import annotations


def error_map(target, canvas, blur_frac: float = 1 / 8, power: float = 4.0):  # pragma: no cover - stub
    raise NotImplementedError("error_map is scheduled for Week 5 (see ROADMAP.md)")


class StrokeSampler:  # pragma: no cover - stub
    def __init__(self, rasterizer, rng=None) -> None:
        raise NotImplementedError("StrokeSampler is scheduled for Week 5 (see ROADMAP.md)")
