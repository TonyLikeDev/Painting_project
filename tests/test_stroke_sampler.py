"""Stroke sampler: the error map and the blur against OpenCV, and the sampling distribution against the original."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from neural_painter.core.procedural_rasterizer import ProceduralRasterizer
from neural_painter.pipeline.stroke_sampler import StrokeSampler, box_blur, error_map

BRUSHES = ["oilpaintbrush", "watercolor", "markerpen", "rectangle"]


@pytest.mark.parametrize("size", [32, 128])
@pytest.mark.parametrize("ksize", [1, 2, 3, 4, 5, 8, 16])
def test_box_blur_equals_opencv(size, ksize):
    img = np.random.default_rng(size + ksize).random((size, size), dtype=np.float32)
    ours = box_blur(torch.from_numpy(img)[None, None], ksize)[0, 0].numpy()
    reference = cv2.blur(img, (ksize, ksize)) if ksize > 1 else img
    np.testing.assert_allclose(ours, reference, rtol=0, atol=2e-6)


@pytest.mark.parametrize("size", [32, 128])
def test_error_map_is_the_original_formula(size):
    g = np.random.default_rng(0)
    target = g.random((3, 3, size, size), dtype=np.float32)
    canvas = g.random((3, 3, size, size), dtype=np.float32)
    ours = error_map(torch.from_numpy(target), torch.from_numpy(canvas))
    ks = int(size / 8)  # 4 px for 32 px blocks, 16 px for 128 px blocks
    for i in range(3):
        reference = cv2.blur(np.abs(target[i] - canvas[i]).sum(0), (ks, ks)) ** 4
        np.testing.assert_allclose(ours[i, 0].numpy(), reference, rtol=1e-4, atol=1e-8)
    assert ours.shape == (3, 1, size, size) and float(ours.min()) >= 0.0


def sampler(brush: str, seed: int = 0, resolution: int = 512) -> StrokeSampler:
    return StrokeSampler(ProceduralRasterizer(brush, 32, "black", rng=seed), resolution=resolution)


def test_strokes_land_where_the_canvas_is_wrong_and_take_the_target_colour():
    size = 32
    target = torch.zeros(1, 3, size, size)
    target[:, 0, 24:30, 2:8] = 1.0  # a red patch in the bottom-left; the blank canvas is wrong only there
    canvas = torch.zeros_like(target)
    s = sampler("oilpaintbrush", seed=1)
    strokes = np.stack([s.sample(target, canvas)[0] for _ in range(300)])
    cx, cy = strokes[:, 0], strokes[:, 1]
    inside = (cx > 0.0) & (cx < 0.42) & (cy > 0.6) & (cy < 1.0)  # the patch, widened by the blur
    assert inside.mean() > 0.95
    on_patch = (cx * size >= 2) & (cx * size < 8) & (cy * size >= 24) & (cy * size < 30)
    assert on_patch.any() and np.allclose(strokes[on_patch][:, 5:8], [1.0, 0.0, 0.0])  # r0, g0, b0 read from the target
    assert strokes.min() >= 0.0 and strokes.max() <= 1.0 and strokes.dtype == np.float32


def test_a_perfect_canvas_falls_back_to_uniform_positions():
    img = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    s = sampler("watercolor", seed=3)
    pos = np.stack([s.sample(img, img.clone())[0][:2] for _ in range(400)])  # x0, y0
    assert abs(pos[:, 0].mean() - 0.5) < 0.06 and abs(pos[:, 1].mean() - 0.5) < 0.06
    assert pos.std(axis=0).min() > 0.2


def test_uniform_initialization_ignores_the_error_map_but_still_reads_the_target_colour():
    size = 32
    target = torch.zeros(1, 3, size, size)
    target[:, 0, 24:30, 2:8] = 1.0  # the only wrong place on a blank canvas: a red patch in the bottom-left
    canvas = torch.zeros_like(target)
    s = sampler("oilpaintbrush", seed=1)
    guided = np.stack([s.sample(target, canvas)[0] for _ in range(300)])
    flat = np.stack([s.sample(target, canvas, uniform=True)[0] for _ in range(300)])
    assert guided[:, 0].mean() < 0.3  # the error map puts the centres on the patch
    assert abs(flat[:, 0].mean() - 0.5) < 0.06 and abs(flat[:, 1].mean() - 0.5) < 0.06  # uniform: mean 0.5 ...
    assert 0.24 < flat[:, 0].std() < 0.34 and 0.24 < flat[:, 1].std() < 0.34  # ... and standard deviation 0.289
    on_patch = (flat[:, 0] * size >= 2) & (flat[:, 0] * size < 8) & (flat[:, 1] * size >= 24) & (flat[:, 1] * size < 30)
    off_patch = ~on_patch
    assert on_patch.any() and np.allclose(flat[on_patch][:, 5:8], [1.0, 0.0, 0.0]) and np.allclose(flat[off_patch][:, 5:8], 0.0)


@pytest.mark.parametrize("brush", BRUSHES)
def test_parameters_are_valid_and_follow_the_brush(brush):
    g = torch.Generator().manual_seed(4)
    target, canvas = torch.rand(4, 3, 32, 32, generator=g), torch.rand(4, 3, 32, 32, generator=g)
    s = sampler(brush)
    params = s.sample(target, canvas)
    assert params.shape == (4, s.rasterizer.dim) and params.min() >= 0.0 and params.max() <= 1.0
    spec = s.rasterizer.spec
    lo, hi = spec.sampler.get("size_range", [0.1, 0.25])
    sizes = params[:, spec.size_indices]
    assert (sizes >= lo - 1e-6).all() and (sizes <= hi + 1e-6).all()


def test_same_seed_same_strokes_and_different_seed_differs():
    g = torch.Generator().manual_seed(5)
    target, canvas = torch.rand(3, 3, 32, 32, generator=g), torch.rand(3, 3, 32, 32, generator=g)
    a = np.stack([sampler("oilpaintbrush", 7).sample(target, canvas) for _ in range(2)])
    assert np.array_equal(a[0], a[1])
    assert not np.array_equal(a[0], sampler("oilpaintbrush", 8).sample(target, canvas))


def test_distribution_matches_the_original_sampler(original_repo):
    """Same error map, many draws: the centre distribution of ours and of the original ``random_stroke_params_sampler``."""
    size = 32
    g = np.random.default_rng(6)
    target = g.random((size, size, 3), dtype=np.float32)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[:, : size // 2] = target[:, : size // 2]  # only the right half is wrong, with a smooth gradient of error
    weights = error_map(torch.from_numpy(target).permute(2, 0, 1)[None], torch.from_numpy(canvas).permute(2, 0, 1)[None])[0, 0].numpy()

    original = original_repo.Renderer(renderer="oilpaintbrush")
    np.random.seed(0)
    ref = []
    for _ in range(500):
        original.random_stroke_params_sampler(err_map=weights.copy(), img=target)
        ref.append(original.stroke_params[:2])
    ref = np.array(ref)

    s = sampler("oilpaintbrush", seed=9)
    t, c = torch.from_numpy(target).permute(2, 0, 1)[None], torch.from_numpy(canvas).permute(2, 0, 1)[None]
    ours = np.stack([s.sample(t, c)[0][:2] for _ in range(500)])
    np.testing.assert_allclose(ours.mean(axis=0), ref.mean(axis=0), atol=0.04)
    np.testing.assert_allclose(ours.std(axis=0), ref.std(axis=0), atol=0.04)
    assert ours[:, 0].mean() > 0.6, "vacuous comparison: the wrong half is on the right"
