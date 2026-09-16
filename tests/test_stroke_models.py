from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from neural_painter.core import stroke_models as sm

BRUSHES = ["oilpaintbrush", "watercolor", "markerpen", "rectangle"]


@pytest.mark.parametrize("brush", BRUSHES)
def test_dataclass_fields_follow_yaml_layout(brush):
    spec = sm.get_brush(brush)
    cls = sm.stroke_class(brush)
    assert cls.brush == spec.name
    assert cls.names() == spec.param_names
    assert spec.param_names[-1] == "alpha"
    assert spec.shape_slice.stop == spec.shape_dim
    assert spec.alpha_slice == slice(spec.dim - 1, spec.dim)


@pytest.mark.parametrize("brush", BRUSHES)
def test_array_roundtrip(brush, rng):
    spec = sm.get_brush(brush)
    arr = rng.random(spec.dim).astype(np.float32)
    stroke = sm.stroke_from_array(brush, arr)
    np.testing.assert_allclose(stroke.to_array(), arr, rtol=0, atol=1e-7)
    assert stroke.to_dict()["alpha"] == pytest.approx(float(arr[-1]))
    batch = rng.random((7, spec.dim)).astype(np.float32)
    strokes = sm.strokes_from_array(brush, batch)
    assert len(strokes) == 7
    np.testing.assert_allclose(sm.strokes_to_array(strokes), batch, atol=1e-7)


def test_typed_stroke_validation():
    good = sm.OilStroke(0.5, 0.5, 0.2, 0.1, 0.3, 1, 0, 0, 0, 0, 1)
    assert good.alpha == 1.0 and isinstance(good.xc, float)
    with pytest.raises(ValueError):
        sm.OilStroke(1.5, 0.5, 0.2, 0.1, 0.3, 1, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError):
        sm.TapeStroke(0.5, 0.5, 0.2, 0.1, 0.3, 1, 0, 0, alpha=-0.01)
    with pytest.raises(ValueError):
        sm.MarkerStroke.from_array(np.zeros(11))
    moved = good.replace(xc=0.9)
    assert moved.xc == 0.9 and good.xc == 0.5
    assert dataclasses.is_dataclass(moved)


def test_validate_params_shape_range_and_nan():
    spec = sm.get_brush("watercolor")
    ok = sm.validate_params(np.zeros((3, 4, 15)), spec)
    assert ok.dtype == np.float32
    with pytest.raises(ValueError, match="last dimension"):
        sm.validate_params(np.zeros((3, 12)), spec)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sm.validate_params(np.full((2, 15), 1.2), spec)
    bad = np.zeros((2, 15), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sm.validate_params(bad, spec)
    # tolerance absorbs float noise just outside the box
    sm.validate_params(np.full((1, 15), 1.0 + 1e-7), spec)


def test_aliases_and_unknown_names():
    assert sm.get_brush("oil").name == "oilpaintbrush"
    assert sm.get_brush("Tape").name == "rectangle"
    assert sm.get_brush("marker").name == "markerpen"
    assert sm.get_brush(" ink ").name == "watercolor"
    assert sm.get_brush(sm.get_brush("oil")) is sm.get_brush("oil")
    with pytest.raises(ValueError, match="unknown brush"):
        sm.get_brush("crayon")


def test_split_and_join_numpy_and_torch(rng):
    spec = sm.get_brush("markerpen")
    arr = rng.random((5, 12)).astype(np.float32)
    s, c, a = spec.split(arr)
    assert s.shape == (5, 8) and c.shape == (5, 3) and a.shape == (5, 1)
    np.testing.assert_array_equal(spec.join(s, c, a), arr)
    t = torch.from_numpy(arr)
    ts, tc, ta = spec.split(t)
    assert torch.equal(spec.join(ts, tc, ta), t)
    assert list(spec.x_indices) == [0, 4] and list(spec.y_indices) == [1, 5] and list(spec.size_indices) == [6, 7]
    oil = sm.get_brush("oil")
    assert list(oil.x_indices) == [0] and list(oil.size_indices) == [2, 3]


def test_clip_params_with_shape_margin():
    spec = sm.get_brush("rectangle")
    arr = np.array([[-0.5, 0.0, 1.0, 0.5, 2.0, 0.3, 0.3, 0.3, 1.5]], np.float32)
    clipped = sm.clip_params(arr, spec, shape_margin=0.1)
    np.testing.assert_allclose(clipped[0, :5], [0.1, 0.1, 0.9, 0.5, 0.9])
    np.testing.assert_allclose(clipped[0, 5:], [0.3, 0.3, 0.3, 1.0])
    t = sm.clip_params(torch.from_numpy(arr), spec, shape_margin=0.1)
    np.testing.assert_allclose(t.numpy(), clipped)
    assert arr[0, 0] == -0.5, "input must not be modified"


def test_spec_to_dict_roundtrip():
    spec = sm.get_brush("watercolor")
    again = sm.BrushSpec.from_dict(spec.to_dict())
    assert again == spec
