from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neural_painter.core import grid
from neural_painter.core.image_io import to_tensor

BRUSHES = ["oilpaintbrush", "watercolor", "markerpen", "rectangle"]


@pytest.mark.parametrize("m", [1, 2, 4])
def test_split_merge_roundtrip(m):
    x = torch.rand(2, 3, 64, 64)
    patches = grid.split_grid(x, m)
    assert patches.shape == (2 * m * m, 3, 64 // m, 64 // m)
    assert torch.equal(grid.merge_grid(patches, m), x)
    single = grid.split_grid(x[0], m)
    assert single.shape[0] == m * m


def test_block_order_is_row_major():
    x = torch.arange(16.0).view(1, 1, 4, 4)
    patches = grid.split_grid(x, 2)
    assert patches[0].flatten().tolist() == [0, 1, 4, 5]
    assert patches[1].flatten().tolist() == [2, 3, 6, 7]  # column 1, row 0
    assert patches[2].flatten().tolist() == [8, 9, 12, 13]  # row 1, column 0
    assert grid.grid_boxes(4, 4, 2) == [(0, 0, 2, 2), (2, 0, 4, 2), (0, 2, 2, 4), (2, 2, 4, 4)]
    with pytest.raises(ValueError):
        grid.split_grid(torch.zeros(1, 3, 10, 10), 3)


def test_img2patches_matches_split_grid(synthetic_image):
    patches = grid.img2patches(synthetic_image, 3, 32)
    assert patches.shape == (9, 3, 32, 32)
    resized = grid.cv2.resize(synthetic_image, (96, 96), interpolation=grid.cv2.INTER_AREA)
    expected = grid.split_grid(to_tensor(resized), 3)
    assert torch.equal(patches, expected)
    back = grid.patches2img(patches, 3)
    np.testing.assert_allclose(back, resized, atol=1e-7)
    assert grid.patches2img(patches, 3, to_numpy=False).shape == (1, 3, 96, 96)


def test_img2patches_with_linear_interpolation_is_the_originals(original_repo, synthetic_image):
    """The original shrinks with ``cv2.resize``'s default (bilinear); the default here averages. The option reproduces the original."""
    import utils as original_utils

    theirs = original_utils.img2patches(synthetic_image, 3, 32)
    ours = grid.img2patches(synthetic_image, 3, 32, interpolation=grid.cv2.INTER_LINEAR)
    assert torch.equal(ours, theirs)
    assert not torch.equal(grid.img2patches(synthetic_image, 3, 32), theirs)  # area averaging fits the strokes to a different target


def test_block_to_global_hand_checked():
    v = np.zeros((4, 1, 12), np.float32)
    v[3, 0, :5] = (0.5, 0.5, 0.4, 0.2, 0.9)  # block 3 = row 1, column 1 of a 2x2 grid
    g = grid.block_to_global(v, "oilpaintbrush", 2)
    np.testing.assert_allclose(g[3, 0, :5], (0.75, 0.75, 0.2, 0.1, 0.9))
    assert v[3, 0, 0] == 0.5, "input must not be modified"
    w = np.zeros((4, 1, 15), np.float32)
    w[1, 0, :8] = (0.2, 0.4, 0.5, 0.5, 0.6, 0.8, 0.3, 0.1)  # block 1 = row 0, column 1
    gw = grid.block_to_global(w, "watercolor", 2)
    np.testing.assert_allclose(gw[1, 0, :8], (0.6, 0.2, 0.5, 0.5, 0.8, 0.4, 0.15, 0.05))
    gt = grid.block_to_global(torch.from_numpy(w), "watercolor", 2)
    np.testing.assert_allclose(gt.numpy(), gw)
    with pytest.raises(ValueError):
        grid.block_to_global(w, "watercolor", 3)


@pytest.mark.parametrize("brush", BRUSHES)
@pytest.mark.parametrize("m", [1, 3])
def test_block_to_global_parity_with_original(original_painter, brush, m, rng):
    from neural_painter.core.stroke_models import get_brush

    d = get_brush(brush).dim
    v = rng.random((m * m, 4, d)).astype(np.float32)
    fake_self = SimpleNamespace(rderr=SimpleNamespace(renderer=brush), m_grid=m)
    theirs = original_painter.PainterBase._normalize_strokes(fake_self, torch.from_numpy(v.copy()))
    ours = grid.block_to_global(v, brush, m)
    np.testing.assert_allclose(ours, theirs, atol=1e-7)


def test_flatten_blocks_interleaves_strokes():
    v = np.arange(2 * 3 * 1, dtype=np.float32).reshape(2, 3, 1)  # 2 blocks, 3 strokes, d = 1
    flat = grid.flatten_blocks(v)
    assert flat.flatten().tolist() == [0, 3, 1, 4, 2, 5]
    assert grid.flatten_blocks(v, order=[1, 0]).flatten().tolist() == [3, 0, 4, 1, 5, 2]


def test_flatten_blocks_drops_empty_slots():
    v = np.arange(2 * 3 * 1, dtype=np.float32).reshape(2, 3, 1)
    active = np.array([[True, True, True], [True, False, False]])  # block 1 holds one stroke
    assert grid.flatten_blocks(v, active=active).flatten().tolist() == [0, 3, 1, 2]
    assert grid.flatten_blocks(v, order=[1, 0], active=active).flatten().tolist() == [3, 0, 1, 2]


def test_grid_levels_and_budgets():
    assert grid.grid_levels("progressive", 5) == [1, 2, 3, 4, 5]
    assert grid.grid_levels("fixed", 5) == [5] and grid.grid_levels("full", 5) == [1]
    with pytest.raises(ValueError):
        grid.grid_levels("spiral", 5)
    assert grid.strokes_per_block(500, [1, 2, 3, 4, 5]) == 9  # 500 // (1 + 4 + 9 + 16 + 25), the original's stroke_parser
    assert grid.strokes_per_block(500, [5]) == 20
    for total in (100, 300, 500, 1000):
        budgets = grid.level_budgets(total, [1, 2, 3, 4, 5])
        assert sum(budgets) == total  # exact, unlike the equal split (495 of 500)
        assert all(abs(n - total * m * m / 55) < 1 for n, m in zip(budgets, range(1, 6)))
    assert grid.level_budgets(500, [1, 2, 3, 4, 5]) == [9, 36, 82, 146, 227]


def test_draw_grid_lines(synthetic_image):
    out = grid.draw_grid_lines(synthetic_image, 4, color=(0.0, 1.0, 0.0), thickness=1)
    assert out.shape == synthetic_image.shape
    assert np.array_equal(out[120, 80], (0.0, 1.0, 0.0))  # vertical line at x = 80
    assert not np.array_equal(synthetic_image[120, 80], (0.0, 1.0, 0.0))
