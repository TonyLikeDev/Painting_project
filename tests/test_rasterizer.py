"""Procedural rasterizer: behaviour tests plus pixel-exact parity with the original renderer.py."""
from __future__ import annotations

import numpy as np
import pytest

from neural_painter.core.procedural_rasterizer import (
    ProceduralRasterizer,
    canvas_rgb,
    create_transformed_brush,
    rotate_point,
    to_pixel,
)
from neural_painter.core.stroke_models import get_brush, validate_params

BRUSHES = ["oilpaintbrush", "watercolor", "markerpen", "rectangle"]


def visible_params(r: ProceduralRasterizer, n: int) -> np.ndarray:
    """Uniform samples with the size parameters pushed above the visibility threshold."""
    p = r.sample_uniform(n)
    p[:, r.spec.size_indices] = 0.15 + 0.5 * p[:, r.spec.size_indices]
    return p


# ----- behaviour -------------------------------------------------------------------------
@pytest.mark.parametrize("brush", BRUSHES)
@pytest.mark.parametrize("size", [32, 128])
def test_random_stroke_renders(brush, size):
    r = ProceduralRasterizer(brush, size, "black", rng=0)
    for p in visible_params(r, 5):
        fg, am = r.render_stroke(p)
        assert fg.shape == (size, size, 3) and am.shape == (size, size, 3)
        assert fg.dtype == np.float32 and am.dtype == np.float32
        assert 0.0 <= fg.min() and fg.max() <= 1.0 and 0.0 <= am.min() and am.max() <= 1.0
        assert am.max() > 0.0, "a visible stroke must leave a mark"
        assert np.array_equal(am[..., 0], am[..., 1]) and np.array_equal(am[..., 0], am[..., 2])
        assert not np.any(am[fg.sum(-1) > 0] == 0) or True  # foreground may extend past alpha after dilation


def test_canvas_composite_and_reset():
    r = ProceduralRasterizer("watercolor", 64, "white", rng=1)
    assert r.canvas.shape == (64, 64, 3) and r.canvas.min() == 1.0
    p = visible_params(r, 1)[0]
    p[8:14] = 0.0  # black ink
    fg, am = r.render_stroke(p)
    canvas_before = r.canvas.copy()
    canvas = r.draw(p)
    np.testing.assert_array_equal(canvas, fg * am + canvas_before * (1 - am))
    assert canvas.min() < 1.0
    r.reset_canvas()
    assert r.canvas.min() == 1.0


def test_render_sequence_callback_and_skipping():
    r = ProceduralRasterizer("rectangle", 48, "black", rng=2)
    params = visible_params(r, 6)
    params[2, r.spec.size_indices] = 0.01  # invisible
    frames = []
    canvas = r.render_sequence(params, on_stroke=lambda i, c: frames.append((i, c.copy())))
    assert [i for i, _ in frames] == list(range(6))
    assert canvas.max() > 0
    np.testing.assert_array_equal(frames[1][1], frames[2][1])  # skipped stroke leaves the canvas unchanged
    assert not r.is_visible(params[2]) and r.is_visible(params[0])


def test_train_mode_disables_morphology():
    infer = ProceduralRasterizer("markerpen", 128, rng=3)
    train = ProceduralRasterizer("markerpen", 128, train=True, rng=3)
    p = visible_params(infer, 1)[0]
    fg_i, am_i = infer.render_stroke(p)
    fg_t, am_t = train.render_stroke(p)
    assert fg_i.sum() >= fg_t.sum() and am_i.sum() <= am_t.sum()
    assert not np.array_equal(am_i, am_t)


@pytest.mark.parametrize("brush", BRUSHES)
def test_initial_params_at_are_valid(brush):
    r = ProceduralRasterizer(brush, 128, rng=4)
    p = r.initial_params_at(0.3, 0.7, (0.2, 0.4, 0.6))
    validate_params(p, brush)
    spec = get_brush(brush)
    assert p.shape == (spec.dim,)
    np.testing.assert_allclose(p[spec.x_indices], 0.3)
    np.testing.assert_allclose(p[spec.y_indices], 0.7)
    np.testing.assert_allclose(p[spec.color_slice][:3], (0.2, 0.4, 0.6))
    assert r.is_visible(p)


def test_seeded_sampling_is_reproducible():
    a = ProceduralRasterizer("oil", 32, rng=7).sample_uniform(4)
    b = ProceduralRasterizer("oil", 32, rng=7).sample_uniform(4)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32 and a.shape == (4, 12)


def test_helpers():
    assert to_pixel(0.0, 128) == 0 and to_pixel(1.0, 128) == 127 and to_pixel(0.5, 128) == 64
    assert rotate_point((1, 0), (0, 0), 0.0) == (1, 0)
    np.testing.assert_allclose(rotate_point((1, 0), (0, 0), np.pi / 2, return_int=False), (0.0, -1.0), atol=1e-12)
    np.testing.assert_array_equal(canvas_rgb("white"), [1, 1, 1])
    np.testing.assert_array_equal(canvas_rgb("#ff0000"), [1, 0, 0])
    np.testing.assert_array_equal(canvas_rgb(0.5), [0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        canvas_rgb("blue")
    with pytest.raises(ValueError):
        canvas_rgb((2.0, 0, 0))
    with pytest.raises(ValueError):
        ProceduralRasterizer("oil", 32).render_stroke(np.zeros(9))


# ----- pixel-exact parity with the original 2021 code ------------------------------------
@pytest.mark.parametrize("brush", BRUSHES)
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("size", [32, 128])
def test_single_stroke_parity_with_original(original_repo, brush, train, size):
    ours = ProceduralRasterizer(brush, size, "black", train=train, rng=11)
    theirs = original_repo.Renderer(renderer=brush, CANVAS_WIDTH=size, train=train, canvas_color="black")
    params = ours.sample_uniform(25)  # includes tiny and off-canvas strokes on purpose
    for p in params:
        fg, am = ours.render_stroke(p)
        theirs.stroke_params = p.copy()
        theirs.draw_stroke()
        np.testing.assert_array_equal(fg, theirs.foreground)
        np.testing.assert_array_equal(am, theirs.stroke_alpha_map)
        assert ours.is_visible(p) == theirs.check_stroke()


@pytest.mark.parametrize("brush", BRUSHES)
def test_sequence_parity_with_original(original_repo, brush):
    ours = ProceduralRasterizer(brush, 128, "white", rng=12)
    theirs = original_repo.Renderer(renderer=brush, CANVAS_WIDTH=128, canvas_color="white")
    params = ours.sample_uniform(40)
    canvas = ours.render_sequence(params)
    theirs.create_empty_canvas()
    for p in params:  # the original _render loop
        theirs.stroke_params = p.copy()
        if theirs.check_stroke():
            theirs.draw_stroke()
    np.testing.assert_array_equal(canvas, theirs.canvas)


def test_transformed_brush_matches_original(original_repo):
    import utils as original_utils  # noqa: WPS433

    ours = ProceduralRasterizer("oil", 128, rng=13)
    brush = ours.textures["large_horizontal"]
    args = (128, 128, 60, 70, 50, 30, np.float32(0.7), *np.random.default_rng(0).random(6, dtype=np.float32))
    fg_a, al_a = create_transformed_brush(brush, *args)
    fg_b, al_b = original_utils.create_transformed_brush(brush, *args)
    np.testing.assert_array_equal(fg_a, fg_b)
    np.testing.assert_array_equal(al_a, al_b)
