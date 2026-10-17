"""Coarse-to-fine painting: allocation, levels, agreement with the fixed grid, the progress events (Week 7)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_painter.core.image_io import load_image
from neural_painter.core.procedural_rasterizer import ProceduralRasterizer
from neural_painter.models.neural_renderer import load_original_checkpoint, original_checkpoint_path
from neural_painter.pipeline import metrics
from neural_painter.pipeline.painter_engine import PainterConfig, PainterEngine, paint_fixed_grid
from neural_painter.pipeline.progressive_painter import allocate, paint_image

from test_optimizer import ROOT, ToyRenderer, two_colour_blocks


def config(**overrides) -> PainterConfig:
    overrides.setdefault("init_size_range", (0.1, 0.25))
    return PainterConfig(iters_per_stroke=4, seed=3, **overrides)


def half_image(size: int = 96) -> np.ndarray:
    """Left half black (the canvas colour), right half two colours: only the right half needs strokes."""
    img = np.zeros((size, size, 3), np.float32)
    img[: size // 2, size // 2 :] = (0.9, 0.2, 0.1)
    img[size // 2 :, size // 2 :] = (0.1, 0.3, 0.9)
    return img


def test_allocate_is_proportional_exact_and_capped():
    assert allocate([1, 1, 2], 8, 8).tolist() == [2, 2, 4]
    assert allocate([0, 0, 1, 3], 8, 5).tolist() == [0, 0, 3, 5]  # the cap moves strokes to the next largest weight
    assert allocate([0, 0, 0], 4, 2).tolist() == [2, 1, 1]  # a perfect canvas splits evenly
    counts = allocate(np.random.default_rng(0).random(25), 227, 19)
    assert counts.sum() == 227 and counts.max() <= 19
    with pytest.raises(ValueError):
        allocate([1, 1], 5, 2)


def test_fixed_mode_is_paint_fixed_grid():
    image = half_image()
    fixed = paint_fixed_grid(image, ToyRenderer(), config(strokes_per_block=3), grid=3, canvas_size=96, canvas_color="black", device="cpu")
    same = paint_image(image, ToyRenderer(), config(), mode="fixed", max_divide=3, strokes=27, canvas_size=96, canvas_color="black", device="cpu")
    np.testing.assert_array_equal(same.strokes, fixed.strokes)
    np.testing.assert_array_equal(same.image, fixed.image)
    np.testing.assert_array_equal(same.result.loss, fixed.result.loss)
    assert same.steps == len(fixed.result.loss) and len(same.levels) == 1


def test_progressive_levels_strokes_and_incremental_canvas():
    events = []
    paint = paint_image(half_image(), ToyRenderer(), config(), mode="progressive", max_divide=3, strokes=30, canvas_size=96,
                        canvas_color="black", device="cpu", callback=events.append)
    assert [lv.m for lv in paint.levels] == [1, 2, 3]
    assert [lv.counts.tolist() for lv in paint.levels] == [[2], [2] * 4, [2] * 9]  # 30 // 14 = 2 per block, as stroke_parser
    assert len(paint.strokes) == 28 and paint.steps == 3 * 2 * 4
    again = ProceduralRasterizer("rectangle", 96, "black").render_sequence(paint.strokes)
    np.testing.assert_array_equal(paint.image, again)  # drawing level by level = drawing every stroke from scratch
    done = [e.strokes_done for e in events]
    assert done == sorted(done) and done[-1] == 28 and all(e.strokes_total == 28 for e in events)
    assert [e.m for e in events] == [1, 1, 2, 2, 3, 3] and events[-1].step == paint.steps
    assert events[-1].preview().shape == (48, 48, 3)  # 3 x 3 blocks of 16 px


def test_error_allocation_spends_the_exact_budget_where_the_canvas_is_wrong():
    paint = paint_image(half_image(), ToyRenderer(), config(allocation="error"), mode="fixed", max_divide=4, strokes=30,
                        canvas_size=96, canvas_color="black", device="cpu")
    counts = paint.levels[0].counts.reshape(4, 4)
    assert counts.sum() == 30 and len(paint.strokes) == 30  # all of the budget (the equal split paints 16 x 1)
    assert counts[:, :2].sum() == 0 and counts[:, 2:].min() > 0  # the black half already matches the black canvas
    assert counts.max() <= 4  # at most twice the mean of 1.9 per block, rounded up
    xs = paint.strokes[:, 0]
    assert xs.min() >= 0.5  # every stroke lies in the right half


def test_a_block_without_strokes_keeps_its_canvas_and_slots_stay_transparent():
    eng = PainterEngine(ToyRenderer(), config(strokes_per_block=3), device="cpu")
    result = eng.paint(two_colour_blocks(3), canvas_color="black", strokes=np.array([3, 0, 1]))
    assert result.active.tolist() == [[True] * 3, [False] * 3, [True, False, False]]
    assert torch.count_nonzero(result.canvas[1]) == 0  # block 1 got no stroke: still the black canvas
    assert len(result.loss) == 3 * 4  # three anchors (the largest count) of four steps


def test_cosine_schedule_and_reduced_precision_change_the_run_and_options_are_checked():
    a = PainterEngine(ToyRenderer(), config(strokes_per_block=2), device="cpu").paint(two_colour_blocks(), canvas_color="black")
    b = PainterEngine(ToyRenderer(), config(strokes_per_block=2, lr_schedule="cosine"), device="cpu").paint(two_colour_blocks(), canvas_color="black")
    assert not np.array_equal(a.params, b.params)
    half = PainterEngine(ToyRenderer(), config(strokes_per_block=2, precision="bfloat16"), device="cpu").paint(two_colour_blocks(), canvas_color="black")
    assert half.loss.dtype == np.float32 and np.isfinite(half.loss).all()  # the toy has no layer autocast changes: see below
    for bad in ({"lr_schedule": "step"}, {"allocation": "random"}, {"precision": "int8"}):
        with pytest.raises(ValueError):
            PainterEngine(ToyRenderer(), config(**bad), device="cpu")
    with pytest.raises(ValueError):
        paint_image(half_image(), ToyRenderer(), config(), mode="progressive", max_divide=3, strokes=10, device="cpu")  # < 14 blocks


def test_progressive_oil_painting_end_to_end():
    path, image_path = original_checkpoint_path("oilpaintbrush", True), ROOT / "data" / "eval_set" / "apple.png"
    if not path.is_file() or not image_path.is_file():
        pytest.skip("pretrained light oil renderer or the evaluation image not present")
    renderer, image = load_original_checkpoint(path, "oilpaintbrush", True), load_image(image_path)
    paint = paint_image(image, renderer, PainterConfig(iters_per_stroke=5, seed=0), mode="progressive", max_divide=3, strokes=70, device="cpu")
    assert len(paint.strokes) == 70 and [lv.m for lv in paint.levels] == [1, 2, 3]
    blank = np.ones_like(image)  # the oil brush paints on white
    assert metrics.psnr(paint.image, image) > metrics.psnr(blank, image) + 3.0


def test_peak_memory_sees_a_temporary_allocation_and_throttling_opt_out_is_safe():
    import sys

    from neural_painter.core.device import PeakMemory, disable_power_throttling

    with PeakMemory(torch.device("cpu"), interval=0.001) as mem:
        x = torch.ones(64, 1024, 1024)  # 256 MB, touched
        x.add_(1.0)
        del x
    assert 0.15 < mem.gb < 2.0  # the sampling thread caught it (the rest of the process is not counted)
    assert disable_power_throttling() is (sys.platform == "win32")


def test_reduced_precision_runs_the_renderer_in_bfloat16():
    path = original_checkpoint_path("oilpaintbrush", True)
    if not path.is_file():
        pytest.skip("pretrained light oil renderer not present")
    renderer = load_original_checkpoint(path, "oilpaintbrush", True)
    seen = []
    renderer.register_forward_hook(lambda module, args, out: seen.append(out[0].dtype))
    runs = {p: PainterEngine(renderer, config(strokes_per_block=1, precision=p), device="cpu").paint(two_colour_blocks(2, 32), canvas_color="black")
            for p in ("float32", "bfloat16")}
    assert seen == [torch.float32] * 4 + [torch.bfloat16] * 4  # four steps each
    assert runs["bfloat16"].loss.dtype == np.float32 and not np.array_equal(runs["float32"].params, runs["bfloat16"].params)
