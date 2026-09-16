"""Core primitives: stroke models, procedural rasterizer, compositing, grid utilities, image I/O."""

from .device import get_device
from .stroke_models import (
    BRUSH_NAMES,
    BrushSpec,
    MarkerStroke,
    OilStroke,
    Stroke,
    TapeStroke,
    WatercolorStroke,
    get_brush,
    validate_params,
)
from .procedural_rasterizer import ProceduralRasterizer

__all__ = [
    "BRUSH_NAMES",
    "BrushSpec",
    "MarkerStroke",
    "OilStroke",
    "ProceduralRasterizer",
    "Stroke",
    "TapeStroke",
    "WatercolorStroke",
    "get_brush",
    "get_device",
    "validate_params",
]
