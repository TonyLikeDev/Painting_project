"""Parametric stroke models (Week 3).

A stroke is a parameter vector ``theta`` in ``[0, 1]^d``. What each entry means
depends on the brush; the layout (parameter names, dimension, shape / colour /
alpha split) is loaded from the YAML files in ``neural_painter/configs`` so that
the procedural rasterizer, the neural renderer and the optimizer share a single
definition.

Two views of the same data are provided:

* :class:`BrushSpec` - the layout of one brush type, used for batched ``numpy`` /
  ``torch`` arrays of shape ``(..., d)``.
* :class:`Stroke` dataclasses - one typed, validated stroke, convenient for tests,
  SVG export and the UI.

Parameter tables (RESEARCH_PLAN.md, section 2.2)::

    oilpaintbrush  xc yc w h theta | r0 g0 b0 r2 g2 b2 | alpha        d = 12 (5 / 6 / 1)
    rectangle      xc yc w h theta | r g b             | alpha        d = 9  (5 / 3 / 1)
    watercolor     x0 y0 x1 y1 x2 y2 radius0 radius2 | r0 g0 b0 r2 g2 b2 | alpha   d = 15 (8 / 6 / 1)
    markerpen      x0 y0 x1 y1 x2 y2 radius0 radius2 | r g b | alpha             d = 12 (8 / 3 / 1)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

BRUSH_CONFIG_FILES: dict[str, str] = {
    "oilpaintbrush": "oil_brush.yaml",
    "watercolor": "watercolor.yaml",
    "markerpen": "marker_pen.yaml",
    "rectangle": "tape.yaml",
}

BRUSH_ALIASES: dict[str, str] = {
    "oil": "oilpaintbrush",
    "oil_brush": "oilpaintbrush",
    "oil-brush": "oilpaintbrush",
    "oilpaint": "oilpaintbrush",
    "oil-paint": "oilpaintbrush",
    "ink": "watercolor",
    "watercolour": "watercolor",
    "marker": "markerpen",
    "marker_pen": "markerpen",
    "marker-pen": "markerpen",
    "tape": "rectangle",
    "rect": "rectangle",
    "8bit": "rectangle",
}

BRUSH_NAMES: tuple[str, ...] = tuple(BRUSH_CONFIG_FILES)


def canonical_brush_name(name: str) -> str:
    """Map a user-facing brush name or alias to the repository name used everywhere else."""
    key = str(name).strip().lower()
    key = BRUSH_ALIASES.get(key, key)
    if key not in BRUSH_CONFIG_FILES:
        raise ValueError(
            f"unknown brush {name!r}; choose one of {list(BRUSH_CONFIG_FILES)} "
            f"or an alias from {sorted(BRUSH_ALIASES)}"
        )
    return key


@dataclass(frozen=True)
class BrushSpec:
    """Layout of the parameter vector of one brush type (loaded from YAML)."""

    name: str
    display_name: str
    param_names: tuple[str, ...]
    shape_dim: int
    color_dim: int
    alpha_dim: int
    x_params: tuple[str, ...]
    y_params: tuple[str, ...]
    size_params: tuple[str, ...]
    default_canvas_color: str = "black"
    renderer: Mapping[str, Any] = field(default_factory=dict)
    sampler: Mapping[str, Any] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if self.shape_dim + self.color_dim + self.alpha_dim != self.dim:
            raise ValueError(
                f"{self.name}: shape_dim + color_dim + alpha_dim = "
                f"{self.shape_dim} + {self.color_dim} + {self.alpha_dim} != len(param_names) = {self.dim}"
            )
        if len(set(self.param_names)) != self.dim:
            raise ValueError(f"{self.name}: duplicate parameter names in {self.param_names}")
        for group in ("x_params", "y_params", "size_params"):
            for p in getattr(self, group):
                if p not in self.param_names:
                    raise ValueError(f"{self.name}: {group} entry {p!r} is not a parameter name")

    # ----- layout -------------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.param_names)

    def index(self, name: str) -> int:
        try:
            return self.param_names.index(name)
        except ValueError as exc:
            raise KeyError(f"{self.name}: no parameter named {name!r}") from exc

    def indices(self, names: Iterable[str]) -> np.ndarray:
        return np.array([self.index(n) for n in names], dtype=np.int64)

    @property
    def x_indices(self) -> np.ndarray:
        return self.indices(self.x_params)

    @property
    def y_indices(self) -> np.ndarray:
        return self.indices(self.y_params)

    @property
    def size_indices(self) -> np.ndarray:
        return self.indices(self.size_params)

    @property
    def shape_slice(self) -> slice:
        return slice(0, self.shape_dim)

    @property
    def color_slice(self) -> slice:
        return slice(self.shape_dim, self.shape_dim + self.color_dim)

    @property
    def alpha_slice(self) -> slice:
        return slice(self.shape_dim + self.color_dim, self.dim)

    def split(self, params):
        """Split ``(..., d)`` into ``(shape, color, alpha)`` views along the last axis (numpy or torch)."""
        return params[..., self.shape_slice], params[..., self.color_slice], params[..., self.alpha_slice]

    @staticmethod
    def join(shape, color, alpha):
        """Inverse of :meth:`split` for numpy arrays or torch tensors."""
        if hasattr(shape, "detach"):  # torch tensor
            import torch

            return torch.cat([shape, color, alpha], dim=-1)
        return np.concatenate([shape, color, alpha], axis=-1)

    # ----- construction -------------------------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "BrushSpec":
        layout = cfg.get("layout", {})
        return cls(
            name=str(cfg["name"]),
            display_name=str(cfg.get("display_name", cfg["name"])),
            param_names=tuple(str(p) for p in cfg["param_names"]),
            shape_dim=int(cfg["shape_dim"]),
            color_dim=int(cfg["color_dim"]),
            alpha_dim=int(cfg["alpha_dim"]),
            x_params=tuple(layout.get("x_params", ())),
            y_params=tuple(layout.get("y_params", ())),
            size_params=tuple(layout.get("size_params", ())),
            default_canvas_color=str(cfg.get("default_canvas_color", "black")),
            renderer=dict(cfg.get("renderer", {})),
            sampler=dict(cfg.get("sampler", {})),
            defaults=dict(cfg.get("defaults", {})),
            description=str(cfg.get("description", "")).strip(),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BrushSpec":
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls.from_dict(cfg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "param_names": list(self.param_names),
            "shape_dim": self.shape_dim,
            "color_dim": self.color_dim,
            "alpha_dim": self.alpha_dim,
            "layout": {
                "x_params": list(self.x_params),
                "y_params": list(self.y_params),
                "size_params": list(self.size_params),
            },
            "default_canvas_color": self.default_canvas_color,
            "renderer": dict(self.renderer),
            "sampler": dict(self.sampler),
            "defaults": dict(self.defaults),
        }


_SPEC_CACHE: dict[str, BrushSpec] = {}


def load_brush_spec(name: str, config_dir: str | Path = CONFIG_DIR) -> BrushSpec:
    """Load a brush spec from its YAML file (bypasses the cache)."""
    key = canonical_brush_name(name)
    path = Path(config_dir) / BRUSH_CONFIG_FILES[key]
    spec = BrushSpec.from_yaml(path)
    if spec.name != key:
        raise ValueError(f"{path} declares name {spec.name!r}, expected {key!r}")
    return spec


def get_brush(name: str | BrushSpec) -> BrushSpec:
    """Return the (cached) spec for a brush name, alias, or an existing spec."""
    if isinstance(name, BrushSpec):
        return name
    key = canonical_brush_name(name)
    if key not in _SPEC_CACHE:
        _SPEC_CACHE[key] = load_brush_spec(key)
    return _SPEC_CACHE[key]


def all_brushes() -> dict[str, BrushSpec]:
    return {n: get_brush(n) for n in BRUSH_NAMES}


# ----- validation ------------------------------------------------------------------------
def validate_params(params, spec: str | BrushSpec, *, tol: float = 1e-6, what: str = "params") -> np.ndarray:
    """Check that ``params`` has last dimension ``spec.dim`` and lies in ``[0, 1]``.

    Returns a ``float32`` numpy array (a view when possible). Raises ``ValueError``
    with a message that names the brush and the offending range.
    """
    spec = get_brush(spec)
    arr = np.asarray(params, dtype=np.float32)
    if arr.ndim == 0 or arr.shape[-1] != spec.dim:
        raise ValueError(
            f"{what}: expected last dimension {spec.dim} for brush {spec.name!r}, got shape {tuple(arr.shape)}"
        )
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what}: contains NaN or inf")
    lo, hi = float(arr.min()), float(arr.max())
    if lo < -tol or hi > 1.0 + tol:
        raise ValueError(f"{what}: values must lie in [0, 1], got range [{lo:.4g}, {hi:.4g}]")
    return arr


def clip_params(params, spec: str | BrushSpec | None = None, *, shape_margin: float = 0.0):
    """Clip parameters into the valid box (numpy or torch, returns a new object).

    ``shape_margin`` > 0 additionally keeps the *shape* parameters inside
    ``[margin, 1 - margin]``; the original optimizer used 0.1 so that strokes never
    collapse onto the patch border.
    """
    is_torch = hasattr(params, "detach")
    out = params.clamp(0.0, 1.0) if is_torch else np.clip(np.asarray(params, dtype=np.float32), 0.0, 1.0)
    if shape_margin > 0.0:
        if spec is None:
            raise ValueError("shape_margin requires a brush spec")
        s = get_brush(spec).shape_slice
        if is_torch:
            out = out.clone()
            out[..., s] = out[..., s].clamp(shape_margin, 1.0 - shape_margin)
        else:
            out[..., s] = np.clip(out[..., s], shape_margin, 1.0 - shape_margin)
    return out


# ----- typed single strokes --------------------------------------------------------------
@dataclass
class Stroke:
    """Base class of the typed stroke dataclasses. Field order == YAML ``param_names``."""

    brush: ClassVar[str] = ""

    def __post_init__(self) -> None:
        for f in fields(self):
            v = float(getattr(self, f.name))
            if not (0.0 <= v <= 1.0):  # also rejects NaN
                raise ValueError(f"{type(self).__name__}.{f.name} = {v!r} is outside [0, 1]")
            setattr(self, f.name, v)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @property
    def spec(self) -> BrushSpec:
        return get_brush(self.brush)

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.names()], dtype=np.float32)

    @classmethod
    def from_array(cls, arr) -> "Stroke":
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        names = cls.names()
        if arr.shape[0] != len(names):
            raise ValueError(f"{cls.__name__}: expected {len(names)} values, got {arr.shape[0]}")
        return cls(**{n: float(v) for n, v in zip(names, arr)})

    def to_dict(self) -> dict[str, float]:
        return {n: getattr(self, n) for n in self.names()}

    def replace(self, **changes: float) -> "Stroke":
        return dataclasses.replace(self, **changes)


@dataclass
class OilStroke(Stroke):
    """Oil paint brush: textured, rotated, scaled patch with a two-colour gradient."""

    brush: ClassVar[str] = "oilpaintbrush"
    xc: float
    yc: float
    w: float
    h: float
    theta: float
    r0: float
    g0: float
    b0: float
    r2: float
    g2: float
    b2: float
    alpha: float = 1.0


@dataclass
class TapeStroke(Stroke):
    """Coloured tape: solid rotated rectangle."""

    brush: ClassVar[str] = "rectangle"
    xc: float
    yc: float
    w: float
    h: float
    theta: float
    r: float
    g: float
    b: float
    alpha: float = 1.0


@dataclass
class WatercolorStroke(Stroke):
    """Watercolour: quadratic Bezier, disc pen, radius and colour gradients."""

    brush: ClassVar[str] = "watercolor"
    x0: float
    y0: float
    x1: float
    y1: float
    x2: float
    y2: float
    radius0: float
    radius2: float
    r0: float
    g0: float
    b0: float
    r2: float
    g2: float
    b2: float
    alpha: float = 1.0


@dataclass
class MarkerStroke(Stroke):
    """Marker pen: quadratic Bezier, tangent-aligned square pen, single colour."""

    brush: ClassVar[str] = "markerpen"
    x0: float
    y0: float
    x1: float
    y1: float
    x2: float
    y2: float
    radius0: float
    radius2: float
    r: float
    g: float
    b: float
    alpha: float = 1.0


STROKE_CLASSES: dict[str, type[Stroke]] = {
    "oilpaintbrush": OilStroke,
    "rectangle": TapeStroke,
    "watercolor": WatercolorStroke,
    "markerpen": MarkerStroke,
}


def stroke_class(brush: str | BrushSpec) -> type[Stroke]:
    return STROKE_CLASSES[get_brush(brush).name]


def stroke_from_array(brush: str | BrushSpec, arr) -> Stroke:
    return stroke_class(brush).from_array(arr)


def strokes_from_array(brush: str | BrushSpec, arr) -> list[Stroke]:
    """Convert an ``(N, d)`` array into a list of typed strokes."""
    spec = get_brush(brush)
    arr = validate_params(arr, spec).reshape(-1, spec.dim)
    cls = stroke_class(spec)
    return [cls.from_array(row) for row in arr]


def strokes_to_array(strokes: Iterable[Stroke]) -> np.ndarray:
    rows = [s.to_array() for s in strokes]
    if not rows:
        raise ValueError("no strokes given")
    return np.stack(rows, axis=0)
