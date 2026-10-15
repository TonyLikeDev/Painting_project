"""How new strokes start, per brush (Week 6): the marker pen's straight first segment and the per-material start ranges."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_painter.core.procedural_rasterizer import INIT_BOUNDS, ProceduralRasterizer
from neural_painter.core.stroke_models import get_brush
from neural_painter.models.neural_renderer import load_original_checkpoint, original_checkpoint_path
from neural_painter.pipeline.painter_engine import PainterConfig, PainterEngine


def chord(p: np.ndarray) -> float:
    return float(np.hypot(p[4] - p[0], p[5] - p[1]))


@pytest.mark.parametrize("brush", ["markerpen", "watercolor"])
def test_a_zero_chord_is_the_week_5_dot_draw_for_draw(brush):
    """chord = 0 must reproduce the Week 5 painter's start: all three Bezier points on the centre, and the same random
    draws (alpha first, then the two radii) and no others, so that every later stroke of a seeded run is unchanged."""
    r = ProceduralRasterizer(brush, 32, rng=11)
    p = r.initial_params_at(0.3, 0.6, (0.1, 0.2, 0.3), chord=0)
    np.testing.assert_allclose(p[:6], [0.3, 0.6] * 3, atol=1e-7)
    s, g = get_brush(brush).sampler, np.random.default_rng(11)
    alpha = g.uniform(*s["alpha_range"])
    radii = g.uniform(*s["size_range"], 2)
    np.testing.assert_allclose(p[6:8], radii, atol=1e-7)
    assert p[-1] == pytest.approx(alpha, abs=1e-7)
    assert r.rng.bit_generator.state == g.bit_generator.state  # no extra draw: the next stroke is unchanged too


@pytest.mark.parametrize("length", [0.1, 0.15, 0.2])
def test_a_positive_chord_starts_a_straight_segment_of_that_length_inside_the_block(length):
    r = ProceduralRasterizer("markerpen", 32, rng=3)
    lo, hi = INIT_BOUNDS
    for cx, cy in [(0.5, 0.5), (0.0, 0.0), (0.99, 0.02), (0.3, 0.97)]:
        p = r.initial_params_at(cx, cy, (1.0, 0.5, 0.0), chord=length)
        assert chord(p) == pytest.approx(length, abs=1e-5)
        assert (p[[0, 1, 4, 5]] >= lo - 1e-6).all() and (p[[0, 1, 4, 5]] <= hi + 1e-6).all()
        np.testing.assert_allclose(p[2:4], 0.5)  # control point mid-chord
        np.testing.assert_allclose(p[8:11], (1.0, 0.5, 0.0))  # colour still the target's


def test_the_centre_moves_only_as_far_as_each_end_needs():
    """Near the bottom border a stroke keeps its x where it was sampled; its y moves just enough for both ends to fit."""
    lo, hi = INIT_BOUNDS
    for seed in range(20):
        p = ProceduralRasterizer("markerpen", 32, rng=seed).initial_params_at(0.5, 0.02, (0, 0, 0), chord=0.4)
        mx, my = (p[0] + p[4]) / 2, (p[1] + p[5]) / 2
        assert mx == pytest.approx(0.5, abs=1e-6)  # 0.5 +- 0.2 fits in [0.1, 0.9] whatever the direction
        assert min(p[1], p[5]) == pytest.approx(lo, abs=1e-6)  # pushed up only to the bound
        p = ProceduralRasterizer("markerpen", 32, rng=seed).initial_params_at(0.45, 0.55, (0, 0, 0), chord=0.4)
        assert ((p[0] + p[4]) / 2, (p[1] + p[5]) / 2) == pytest.approx((0.45, 0.55), abs=1e-6)  # inside: not moved at all


def test_start_options_are_checked_and_follow_the_bounds():
    r = ProceduralRasterizer("markerpen", 32, rng=0)
    with pytest.raises(ValueError, match="longer"):
        r.initial_params_at(0.5, 0.5, (0, 0, 0), chord=0.9)
    with pytest.raises(ValueError, match=">= 0"):
        r.initial_params_at(0.5, 0.5, (0, 0, 0), chord=-0.1)
    with pytest.raises(ValueError, match="Bezier"):
        ProceduralRasterizer("oilpaintbrush", 32, rng=0).initial_params_at(0.5, 0.5, (0, 0, 0), chord=0.2)
    p = r.initial_params_at(0.0, 0.0, (0, 0, 0), chord=0.2, bounds=(0.2, 0.8))
    assert min(p[[0, 1, 4, 5]]) >= 0.2 - 1e-6 and max(p[[0, 1, 4, 5]]) <= 0.8 + 1e-6


def test_directions_vary_and_follow_the_seed():
    def angles(seed):
        r = ProceduralRasterizer("markerpen", 32, rng=seed)
        ps = [r.initial_params_at(0.5, 0.5, (0, 0, 0), chord=0.15) for _ in range(50)]
        return np.array([np.arctan2(p[5] - p[1], p[4] - p[0]) for p in ps])

    a = angles(1)
    assert np.array_equal(a, angles(1)) and not np.array_equal(a, angles(2))
    assert a.std() > 0.5  # spread over directions, not one fixed orientation


def test_the_rasterizer_draws_a_started_marker_stroke_but_not_the_original_dot():
    r = ProceduralRasterizer("markerpen", 32, train=True, rng=5)
    dot = r.initial_params_at(0.5, 0.5, (1, 1, 1), chord=0)
    seg = r.initial_params_at(0.5, 0.5, (1, 1, 1), chord=0.15)
    assert r.render_stroke(dot)[1].max() == 0.0
    assert r.render_stroke(seg)[1].max() > 0.5


@pytest.mark.parametrize("brush", ["oilpaintbrush", "rectangle", "watercolor", "markerpen"])
def test_start_ranges_can_be_overridden_per_call_and_none_keeps_the_yaml(brush):
    spec = get_brush(brush)
    p = ProceduralRasterizer(brush, 32, rng=2).initial_params_at(0.5, 0.5, (0, 0, 0), size_range=(0.4, 0.45), alpha_range=(0.5, 0.5))
    assert (p[spec.size_indices] >= 0.4 - 1e-6).all() and (p[spec.size_indices] <= 0.45 + 1e-6).all()
    assert p[-1] == pytest.approx(0.5)
    kept = ProceduralRasterizer(brush, 32, rng=2).initial_params_at(0.5, 0.5, (0, 0, 0), size_range=None, alpha_range=None)
    default = ProceduralRasterizer(brush, 32, rng=2).initial_params_at(0.5, 0.5, (0, 0, 0))
    np.testing.assert_array_equal(kept, default)


@pytest.fixture
def marker_light():
    path = original_checkpoint_path("markerpen", light=True)
    if not path.is_file():
        pytest.skip(f"{path} has not been downloaded")
    return load_original_checkpoint(path, "markerpen", light=True)


def test_the_marker_pen_paints_only_when_its_strokes_start_with_a_length(marker_light):
    """The Week 5 finding and its fix, end to end: with the original's dot start the light marker renderer leaves the canvas
    blank (no stroke ever renders, so none gets a gradient); with a 0.15 start the canvas moves towards the target."""
    g = torch.Generator().manual_seed(0)
    target = torch.rand(2, 3, 1, 1, generator=g).expand(2, 3, 32, 32).contiguous()  # two flat-colour blocks
    blank = torch.zeros_like(target)
    start_psnr = float(-10 * torch.log10(((blank - target) ** 2).mean()))

    def paint(length):
        cfg = PainterConfig(strokes_per_block=4, iters_per_stroke=20, init_chord=length, seed=0)
        return PainterEngine(marker_light, cfg, device="cpu").paint(target, canvas_color="black")

    assert paint(0.0).psnr[-1] < start_psnr + 0.5
    assert paint(0.15).psnr[-1] > start_psnr + 3.0
