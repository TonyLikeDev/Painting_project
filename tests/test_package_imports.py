"""Week 2 exit criterion: the package skeleton imports cleanly and the configs match the parameter table."""
from __future__ import annotations

import importlib
import pkgutil

import neural_painter
from neural_painter.core.stroke_models import BRUSH_NAMES, get_brush

# RESEARCH_PLAN.md section 2.2: (dim, shape, colour, alpha)
PARAMETER_TABLE = {
    "oilpaintbrush": (12, 5, 6, 1),
    "watercolor": (15, 8, 6, 1),
    "markerpen": (12, 8, 3, 1),
    "rectangle": (9, 5, 3, 1),
}


def test_every_module_imports():
    names = [neural_painter.__name__]
    for info in pkgutil.walk_packages(neural_painter.__path__, prefix=neural_painter.__name__ + "."):
        names.append(info.name)
    for name in names:
        importlib.import_module(name)
    assert "neural_painter.core.procedural_rasterizer" in names
    assert "neural_painter.core.grid" in names


def test_version_string():
    assert neural_painter.__version__.count(".") == 2


def test_configs_match_parameter_table():
    assert set(BRUSH_NAMES) == set(PARAMETER_TABLE)
    for name, (d, s, c, a) in PARAMETER_TABLE.items():
        spec = get_brush(name)
        assert (spec.dim, spec.shape_dim, spec.color_dim, spec.alpha_dim) == (d, s, c, a), name
        assert spec.renderer["checkpoint_light"].endswith("_light")
        assert spec.default_canvas_color in ("black", "white")
