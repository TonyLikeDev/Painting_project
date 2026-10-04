"""Sinkhorn loss: agreement with the original code, and the property it exists for."""
from __future__ import annotations

import pytest
import torch

from neural_painter.losses.sinkhorn import SinkhornLoss, area_resize, cost_matrix, sinkhorn_divergence, sinkhorn_loss


def grid(size: int, batch: int) -> torch.Tensor:
    """``(batch, size*size, 2)`` pixel coordinates (row / size, col / size), the layout the loss uses."""
    return SinkhornLoss()._grid(size, size, torch.zeros(1)).expand(batch, -1, -1).clone()


def masses(batch: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, n, generator=g) ** 2 + 0.01


def blob(cy, cx, size: int = 24, sigma: float = 0.04) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) / size
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))


# ----- agreement with the original ------------------------------------------------------
@pytest.mark.parametrize("epsilon,n_iter", [(0.01, 5), (0.05, 10)])
def test_matches_original_sinkhorn_loss_and_gradient(original_repo, monkeypatch, epsilon, n_iter):
    import pytorch_batch_sinkhorn as original

    monkeypatch.setattr(original, "device", torch.device("cpu"))
    x = grid(12, 3)
    mass_x, mass_y = masses(3, 144, 0), masses(3, 144, 1)
    ref_x = mass_x.clone().requires_grad_(True)
    ours_x = mass_x.clone().requires_grad_(True)
    ref = original.sinkhorn_loss(x, x, epsilon, n_iter, ref_x, mass_y.clone())
    ours = sinkhorn_loss(x, x, ours_x, mass_y, epsilon, n_iter)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-8)
    (g_ref,) = torch.autograd.grad(ref, ref_x)
    (g_ours,) = torch.autograd.grad(ours, ours_x)
    torch.testing.assert_close(g_ours, g_ref, rtol=1e-3, atol=1e-8)  # float32: the original builds the cost by differences, we by a product
    assert float(g_ours.abs().max()) > 0.0, "vacuous gradient comparison"


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("channel", [0, 1, 2])
def test_matches_original_sinkhorn_loss_module(original_repo, monkeypatch, normalize, channel):
    import loss as original_loss
    import pytorch_batch_sinkhorn as original

    monkeypatch.setattr(original, "device", torch.device("cpu"))
    monkeypatch.setattr(original_loss, "device", torch.device("cpu"))
    monkeypatch.setattr(original_loss.random, "randint", lambda a, b: channel)  # the original picks a random channel
    g = torch.Generator().manual_seed(3)
    canvas = torch.rand(2, 3, 32, 32, generator=g)  # larger than 24, so the area downsampling is exercised
    target = torch.rand(2, 3, 32, 32, generator=g)
    ref = original_loss.SinkhornLoss(epsilon=0.01, niter=5, normalize=normalize)(canvas.clone(), target.clone())
    ours = SinkhornLoss(epsilon=0.01, n_iter=5, normalize=normalize, channel=channel)(canvas, target)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-7)


# ----- properties ------------------------------------------------------------------------
def test_divergence_vanishes_for_identical_measures_and_the_plain_cost_does_not():
    x = grid(12, 2)
    m = masses(2, 144, 5)
    assert abs(float(sinkhorn_divergence(x, x, m, m))) < 1e-6
    assert float(sinkhorn_loss(x, x, m, m)) > 1e-4  # entropic bias: the plain cost of a measure with itself is positive
    other = masses(2, 144, 6)
    assert float(sinkhorn_divergence(x, x, m, other)) > 0.0


def test_pixel_loss_is_blind_to_a_distant_blob_but_transport_cost_is_not():
    """The reason the loss exists: with no overlap the pixel loss has no gradient on position; transport cost pulls the blob in."""
    target = blob(0.65, 0.65)[None, None].repeat(1, 3, 1, 1)
    centre = torch.tensor([0.3, 0.3], requires_grad=True)
    canvas = blob(centre[0], centre[1])[None, None].repeat(1, 3, 1, 1)
    assert float((canvas * target).max()) < 1e-6, "the two blobs must not overlap"

    (grad_pixel,) = torch.autograd.grad((canvas - target).abs().mean(), centre, retain_graph=True)
    (grad_transport,) = torch.autograd.grad(SinkhornLoss(channel="mean")(canvas, target), centre)
    assert float(grad_transport.abs().max()) > 1e-4
    assert float(grad_pixel.abs().max()) < 1e-3 * float(grad_transport.abs().max())
    assert bool((grad_transport < 0).all()), "descending the loss must move the blob towards the target (up and to the right)"
    step = centre.detach() - 0.02 * grad_transport / grad_transport.norm()
    moved = blob(step[0], step[1])[None, None].repeat(1, 3, 1, 1)
    assert float(SinkhornLoss(channel="mean")(moved, target)) < float(SinkhornLoss(channel="mean")(canvas, target))


def test_exact_mode_agrees_on_dense_masses_and_stays_finite_on_sparse_ones():
    x = grid(12, 2)
    dense_x, dense_y = masses(2, 144, 7), masses(2, 144, 8)
    ref = sinkhorn_loss(x, x, dense_x, dense_y, lse="reference")
    exact = sinkhorn_loss(x, x, dense_x, dense_y, lse="exact")
    torch.testing.assert_close(exact, ref, rtol=1e-3, atol=1e-6)

    sparse = torch.zeros(2, 144)
    sparse[:, 0] = 1.0  # all the mass in one corner pixel
    other = torch.zeros(2, 144)
    other[:, -1] = 1.0
    sparse.requires_grad_(True)
    value = sinkhorn_loss(x, x, sparse, other, lse="exact")
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(sparse.grad).all()


def test_mass_scale_does_not_matter_and_a_batch_of_one_works():
    x = grid(8, 1)
    m1, m2 = masses(1, 64, 9), masses(1, 64, 10)
    torch.testing.assert_close(sinkhorn_loss(x, x, 10 * m1, 3 * m2), sinkhorn_loss(x, x, m1, m2), rtol=1e-4, atol=1e-8)
    assert sinkhorn_loss(x, x, None, None).shape == ()  # equal weights, scalar result


@pytest.mark.parametrize("size_in", [32, 48, 100, 128])
def test_area_resize_equals_torch_area_interpolation(size_in):
    """Including sizes the output does not divide (32 -> 24), which Apple MPS cannot pool but this can."""
    x = torch.rand(2, 3, size_in, size_in, generator=torch.Generator().manual_seed(size_in))
    torch.testing.assert_close(area_resize(x, 24), torch.nn.functional.interpolate(x, [24, 24], mode="area"), rtol=1e-5, atol=1e-6)
    assert area_resize(x, size_in).allclose(x)  # identity when the size is unchanged


def test_cost_matrix_is_the_squared_distance():
    g = torch.Generator().manual_seed(11)
    a, b = torch.rand(2, 5, 2, generator=g), torch.rand(2, 7, 2, generator=g)
    direct = ((a.unsqueeze(2) - b.unsqueeze(1)) ** 2).sum(-1)
    torch.testing.assert_close(cost_matrix(a, b), direct, rtol=0, atol=1e-6)
    assert float(cost_matrix(a, a).diagonal(dim1=1, dim2=2).abs().max()) < 1e-6


def test_channel_modes_seeding_and_validation():
    g = torch.Generator().manual_seed(12)
    canvas, target = torch.rand(2, 3, 16, 16, generator=g), torch.rand(2, 3, 16, 16, generator=g)
    mean_a = SinkhornLoss(channel="mean")(canvas, target)
    assert torch.equal(mean_a, SinkhornLoss(channel="mean")(canvas, target))  # deterministic
    seeded = [SinkhornLoss(channel="random", seed=4)(canvas, target) for _ in range(2)]
    assert torch.equal(seeded[0], seeded[1])  # the same seed picks the same channel
    loss_fn = SinkhornLoss(channel="random", seed=0)
    values = {round(float(loss_fn(canvas, target)), 7) for _ in range(30)}
    assert len(values) > 1, "the random channel must actually vary between calls"
    with pytest.raises(ValueError, match="channel"):
        SinkhornLoss(channel="blue")
    with pytest.raises(ValueError, match="lse"):
        SinkhornLoss(lse="fast")
    with pytest.raises(ValueError, match="lse"):
        sinkhorn_loss(grid(4, 1), grid(4, 1), lse="fast")
    assert SinkhornLoss(size=24)(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)).shape == ()  # no downsampling below the size
