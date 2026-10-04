"""Entropy-regularized optimal transport (Sinkhorn) loss (Week 5).

Implements section 2.6 of ``report/chapters/02_theory.md`` and reproduces the original
``pytorch_batch_sinkhorn.py`` + ``loss.SinkhornLoss`` (see ``tests/test_sinkhorn.py`` for the
number-for-number comparison).

* :func:`sinkhorn_loss` - batched Sinkhorn on point clouds ``x, y`` of shape ``(B, n, 2)`` with
  masses ``(B, n)``; returns the mean transport cost ``<pi, C>`` with ``C_ij = |x_i - y_j|^2``.
* :class:`SinkhornLoss` - takes a canvas and a target ``(B, 3, H, W)``, area-downsamples both to
  ``size x size`` when they are larger, uses the intensities of one colour channel as masses and returns
  the Sinkhorn cost. ``normalize=True`` returns the debiased ``2 W(x, y) - W(x, x) - W(y, y)``.

Why it exists: when a stroke does not overlap the region it should paint, the pixel loss has no gradient with
respect to the stroke position, because the pixels it covers are all equally wrong. Transport cost still falls as
mass moves towards the target, so the stroke is pulled in.

Two numerical modes for the log-sum-exp inside the Sinkhorn updates:

``"reference"`` (default)
    ``log(sum(exp(A)) + 1e-6)`` exactly as the original, so results are comparable with the 2021 code. The
    ``1e-6`` floors the update where a row of the kernel has underflowed.
``"exact"``
    ``torch.logsumexp``, the textbook log-domain update, stable for any input.

Differences from the original, none of which changes a result on ordinary inputs: no module-level device (it is
taken from the inputs), the mass is clamped out of place instead of through ``.data`` (the original silently
modified the canvas it was given), the batch dimension is never squeezed, the random colour channel comes from a
seedable generator, and the cost matrix is computed once as ``|x|^2 + |y|^2 - 2 x y^T``.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

LSE_MODES = ("reference", "exact")


def area_matrix(n_in: int, n_out: int, like: torch.Tensor) -> torch.Tensor:
    """``(n_out, n_in)`` averaging matrix of ``F.interpolate(mode="area")``: row ``i`` averages inputs
    ``floor(i * n_in / n_out)`` to ``ceil((i + 1) * n_in / n_out) - 1``."""
    m = torch.zeros(n_out, n_in, dtype=like.dtype)
    for i in range(n_out):
        start, end = (i * n_in) // n_out, ((i + 1) * n_in + n_out - 1) // n_out
        m[i, start:end] = 1.0 / (end - start)
    return m.to(like.device)


def area_resize(x: torch.Tensor, size: int) -> torch.Tensor:
    """Area (adaptive average) resize of ``(B, C, H, W)`` to ``size x size`` as two matrix products.

    Numerically the same as ``F.interpolate(x, [size, size], mode="area")``, but it also runs on Apple MPS, whose
    adaptive average pooling rejects input sizes that the output size does not divide (32 -> 24 is one).
    """
    h, w = x.shape[-2:]
    return area_matrix(h, size, x) @ x @ area_matrix(w, size, x).T


def cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances ``|x_i - y_j|^2`` for ``x`` ``(B, n, k)`` and ``y`` ``(B, m, k)``, shape ``(B, n, m)``."""
    sq = (x * x).sum(-1).unsqueeze(2) + (y * y).sum(-1).unsqueeze(1) - 2.0 * torch.bmm(x, y.transpose(1, 2))
    return sq.clamp_min(0.0)  # rounding can leave -1e-8 for coincident points


def _normalized_mass(mass: torch.Tensor | None, batch: int, n: int, like: torch.Tensor) -> torch.Tensor:
    if mass is None:  # equal weights
        return torch.full((batch, n), 1.0 / n, dtype=like.dtype, device=like.device)
    mass = mass.clamp(min=0.0, max=1e9) + 1e-9
    return mass / mass.sum(dim=-1, keepdim=True)


def _lse(a: torch.Tensor, dim: int, mode: str) -> torch.Tensor:
    if mode == "exact":
        return torch.logsumexp(a, dim=dim, keepdim=True)
    return torch.log(torch.exp(a).sum(dim, keepdim=True) + 1e-6)  # the original; the 1e-6 prevents log(0)


def sinkhorn_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mass_x: torch.Tensor | None = None,
    mass_y: torch.Tensor | None = None,
    epsilon: float = 0.01,
    n_iter: int = 5,
    lse: str = "reference",
) -> torch.Tensor:
    """Sinkhorn transport cost between two weighted point clouds, averaged over the batch.

    ``x`` ``(B, n, k)`` and ``y`` ``(B, m, k)`` are the locations, ``mass_x`` ``(B, n)`` and ``mass_y`` ``(B, m)`` the
    weights (normalized inside; ``None`` means equal weights). ``epsilon`` is the entropic regularization and
    ``n_iter`` the number of Sinkhorn iterations; the original uses 5, far from convergence but enough for a useful
    gradient direction. Differentiable with respect to both masses.
    """
    if lse not in LSE_MODES:
        raise ValueError(f"lse must be one of {LSE_MODES}, got {lse!r}")
    batch, n, m = x.shape[0], x.shape[1], y.shape[1]
    cost = cost_matrix(x, y)
    mu = _normalized_mass(mass_x, batch, n, x)
    nu = _normalized_mass(mass_y, batch, m, y)
    log_mu, log_nu = mu.log(), nu.log()

    def modified_cost(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # M_ij = (-c_ij + u_i + v_j) / epsilon
        return (-cost + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    u, v = torch.zeros_like(mu), torch.zeros_like(nu)
    for _ in range(n_iter):
        u = epsilon * (log_mu - _lse(modified_cost(u, v), 2, lse).squeeze(2)) + u
        v = epsilon * (log_nu - _lse(modified_cost(u, v).transpose(1, 2), 2, lse).squeeze(2)) + v
    plan = torch.exp(modified_cost(u, v))  # transport plan diag(a) K diag(b)
    return torch.sum(plan * cost, dim=[1, 2]).mean()


def sinkhorn_divergence(
    x: torch.Tensor, y: torch.Tensor, mass_x=None, mass_y=None, epsilon: float = 0.01, n_iter: int = 5, lse: str = "reference"
) -> torch.Tensor:
    """Debiased ``2 W(x, y) - W(x, x) - W(y, y)``; zero when the two measures are the same."""
    wxy = sinkhorn_loss(x, y, mass_x, mass_y, epsilon, n_iter, lse)
    wxx = sinkhorn_loss(x, x, mass_x, mass_x, epsilon, n_iter, lse)
    wyy = sinkhorn_loss(y, y, mass_y, mass_y, epsilon, n_iter, lse)
    return 2 * wxy - wxx - wyy


class SinkhornLoss(nn.Module):
    """Sinkhorn cost between a canvas and its target, treating one colour channel as a mass over the pixel grid.

    Parameters
    ----------
    epsilon, n_iter: entropic regularization and iteration count (original defaults 0.01 and 5).
    size:            canvases larger than ``size`` are area-downsampled to ``size x size`` first (original: 24), which
                     keeps the ``n x n`` cost matrix at 576 x 576.
    normalize:       use the debiased divergence instead of the plain cost.
    channel:         ``"random"`` picks one of R, G, B per call (the original, to save time), ``"mean"`` uses the mean
                     of the three (deterministic), or an integer 0 to 2.
    lse:             ``"reference"`` or ``"exact"``, see the module docstring.
    seed:            seeds the random channel choice, so a run is reproducible.
    """

    def __init__(self, epsilon: float = 0.01, n_iter: int = 5, size: int = 24, normalize: bool = False,
                 channel: str | int = "random", lse: str = "reference", seed: int | None = None) -> None:
        super().__init__()
        if not (channel in ("random", "mean") or channel in (0, 1, 2)):
            raise ValueError(f"channel must be 'random', 'mean' or 0, 1, 2, got {channel!r}")
        if lse not in LSE_MODES:
            raise ValueError(f"lse must be one of {LSE_MODES}, got {lse!r}")
        self.epsilon, self.n_iter, self.size, self.normalize, self.channel, self.lse = epsilon, n_iter, int(size), normalize, channel, lse
        self._rng = random.Random(seed)
        self._grids: dict[tuple, torch.Tensor] = {}

    def _grid(self, h: int, w: int, like: torch.Tensor) -> torch.Tensor:
        """``(1, h*w, 2)`` pixel coordinates ``(row / h, col / w)`` in row-major order."""
        key = (h, w, like.device, like.dtype)
        if key not in self._grids:
            rows = torch.arange(h, dtype=like.dtype, device=like.device).view(-1, 1).expand(h, w) / h
            cols = torch.arange(w, dtype=like.dtype, device=like.device).view(1, -1).expand(h, w) / w
            self._grids[key] = torch.stack([rows, cols], dim=-1).reshape(1, h * w, 2)
        return self._grids[key]

    def _masses(self, image: torch.Tensor, channel) -> torch.Tensor:
        plane = image.mean(dim=1, keepdim=True) if channel == "mean" else image[:, [channel]]
        return plane.reshape(image.shape[0], -1)

    def forward(self, canvas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = target.shape
        if h > self.size:
            canvas, target = area_resize(canvas, self.size), area_resize(target, self.size)
            h = w = self.size
        channel = self._rng.randrange(3) if self.channel == "random" else self.channel
        grid = self._grid(h, w, target).expand(batch, -1, -1)
        mass_canvas, mass_target = self._masses(canvas, channel), self._masses(target, channel)
        fn = sinkhorn_divergence if self.normalize else sinkhorn_loss
        return fn(grid, grid, mass_canvas, mass_target, epsilon=self.epsilon, n_iter=self.n_iter, lse=self.lse)
