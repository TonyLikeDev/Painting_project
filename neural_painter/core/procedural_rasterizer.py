"""Ground-truth CPU rasterizer (Week 3).

A faithful port of ``renderer.py`` and ``utils.create_transformed_brush`` from the
original repository, so that the strokes it produces are pixel-identical to the
ones the pretrained neural renderers were trained on (checked against the
original code in ``tests/test_rasterizer.py``). The differences are engineering
only: no module globals, explicit state, a seeded ``numpy.random.Generator``,
brush textures shipped inside the package, and a :meth:`ProceduralRasterizer.render_stroke`
method that returns the foreground / alpha pair without touching the canvas.

Rasterization here is intentionally *not* differentiable (OpenCV draws on integer
pixel coordinates). It is the teacher for the neural renderer in ``models`` and
the final high-resolution renderer for the output image and the time-lapse.

Conventions shared with the original code:

* all parameters are in ``[0, 1]``; ``to_pixel`` maps a coordinate to ``int(x * (W - 1) + 0.5)``;
* Bezier brushes: the control point is stored relative to the chord, radii are
  ``1 + r * W // 4`` pixels, 100 samples along ``t``;
* oil brush: ``w, h`` are ``1 + w * W`` pixels (full canvas scale), ``theta`` is ``pi * theta``;
* the foreground and the alpha map are drawn as ``uint8`` (so OpenCV anti-aliasing
  applies) and converted to ``float32`` in ``[0, 1]``; at inference (``train=False``)
  the foreground is dilated and the alpha map eroded with a 2x2 kernel to hide seams.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .stroke_models import BrushSpec, get_brush

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "brushes"

BEZIER_SAMPLES = 100
VISIBILITY_THRESHOLD = 0.025  # renderer.check_stroke: strokes thinner than this are not drawn
LARGE_TEXTURE_AREA_FRACTION = 0.1  # oil brush: switch to the large texture above this area fraction
MIN_MARKER_CHORD_PX = 4  # marker pen: skip strokes whose end points fall on (almost) the same pixel

OIL_TEXTURE_FILES = {
    "small_vertical": "brush_fromweb2_small_vertical.png",
    "small_horizontal": "brush_fromweb2_small_horizontal.png",
    "large_vertical": "brush_fromweb2_large_vertical.png",
    "large_horizontal": "brush_fromweb2_large_horizontal.png",
}

CanvasColor = str | float | Sequence[float]
FrameCallback = Callable[[int, np.ndarray], None]


# ----- small geometry helpers (ported one-to-one) ----------------------------------------
def to_pixel(x: float, width: int) -> int:
    """Normalized coordinate in ``[0, 1]`` to an integer pixel index (original ``_normalize``)."""
    return int(x * (width - 1) + 0.5)


def rotate_point(pt, center, theta, return_int: bool = True):
    """Rotate ``pt`` about ``center`` by ``theta`` radians (original ``utils.rotate_pt``)."""
    x, y = pt[0], pt[1]
    xc, yc = center[0], center[1]
    x_ = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta) + xc
    y_ = -1 * (x - xc) * np.sin(theta) + (y - yc) * np.cos(theta) + yc
    if return_int:
        return int(x_), int(y_)
    return x_, y_


def transformation_matrix(dx: float, dy: float, da: float) -> np.ndarray:
    """2x3 rotation (``da`` radians) + translation matrix (original ``build_transformation_matrix``)."""
    m = np.zeros((2, 3))
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy
    return m


def scale_matrix(sx: float, sy: float) -> np.ndarray:
    m = np.zeros((2, 3))
    m[0, 0] = sx
    m[1, 1] = sy
    return m


def compose(first: np.ndarray, then: np.ndarray) -> np.ndarray:
    """Return the 2x3 matrix of ``then o first`` (original ``update_transformation_matrix``)."""
    a = np.concatenate([first, np.zeros([1, 3])], axis=0)
    a[-1, -1] = 1
    b = np.concatenate([then, np.zeros([1, 3])], axis=0)
    b[-1, -1] = 1
    return np.matmul(b, a)[0:2, :]


def create_transformed_brush(brush, canvas_w, canvas_h, x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2):
    """Colour a grey-scale brush texture with a vertical gradient and warp it onto the canvas.

    Returns ``(foreground, alpha)`` as ``uint8`` ``(H, W, 3)`` arrays. Port of
    ``utils.create_transformed_brush``; the per-row colour loop is vectorized but
    evaluated in ``float32`` exactly like the original.
    """
    brush_alpha = np.stack([brush, brush, brush], axis=-1)
    brush_alpha = (brush_alpha > 0).astype(np.float32)
    brush_alpha = (brush_alpha * 255).astype(np.uint8)

    n_rows = brush.shape[0]
    t = np.arange(n_rows, dtype=np.float64) / n_rows
    one_minus_t = (1.0 - t).astype(np.float32)
    t32 = t.astype(np.float32)
    c0 = np.array([R0, G0, B0], dtype=np.float32)
    c2 = np.array([R2, G2, B2], dtype=np.float32)
    row_colors = one_minus_t[:, None] * c0[None, :] + t32[:, None] * c2[None, :]  # (rows, 3) float32
    colormap = np.ascontiguousarray(np.broadcast_to(row_colors[:, None, :], (n_rows, brush.shape[1], 3)))

    brush = np.expand_dims(brush, axis=-1).astype(np.float32) / 255.0
    brush = (brush * colormap * 255).astype(np.uint8)

    m = transformation_matrix(-brush.shape[1] / 2, -brush.shape[0] / 2, 0)
    m = compose(m, scale_matrix(sx=w / brush.shape[1], sy=h / brush.shape[0]))
    m = compose(m, transformation_matrix(0, 0, theta))
    m = compose(m, transformation_matrix(x0, y0, 0))

    brush = cv2.warpAffine(brush, m, (canvas_w, canvas_h), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)
    brush_alpha = cv2.warpAffine(
        brush_alpha, m, (canvas_w, canvas_h), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA
    )
    return brush, brush_alpha


def canvas_rgb(color: CanvasColor) -> np.ndarray:
    """Parse ``'black'``, ``'white'``, ``'#rrggbb'``, a grey level or an RGB triple into a float32 ``(3,)`` array."""
    if isinstance(color, str):
        key = color.strip().lower()
        if key == "black":
            return np.zeros(3, np.float32)
        if key == "white":
            return np.ones(3, np.float32)
        if key.startswith("#") and len(key) == 7:
            return np.array([int(key[i : i + 2], 16) / 255.0 for i in (1, 3, 5)], np.float32)
        raise ValueError(f"unknown canvas colour {color!r}")
    arr = np.asarray(color, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size != 3 or arr.min() < 0 or arr.max() > 1:
        raise ValueError(f"canvas colour must be a grey level or an RGB triple in [0, 1], got {color!r}")
    return arr


# ----- the rasterizer --------------------------------------------------------------------
class ProceduralRasterizer:
    """OpenCV rasterizer for one brush type on a square canvas.

    Parameters
    ----------
    brush:        brush name / alias / :class:`BrushSpec`.
    canvas_size:  side length ``W`` in pixels (128 for the full renderers, 32 for the light ones,
                  512 for final rendering).
    canvas_color: ``'black'``, ``'white'``, ``'#rrggbb'``, a grey level or an RGB triple.
    train:        ``True`` disables the inference-time dilate / erode so that training targets
                  are the raw anti-aliased shapes (as in the original ``StrokeDataset``).
    rng:          seed or ``numpy.random.Generator`` used by :meth:`sample_uniform`.
    """

    def __init__(
        self,
        brush: str | BrushSpec = "oilpaintbrush",
        canvas_size: int = 128,
        canvas_color: CanvasColor = "black",
        train: bool = False,
        rng: int | np.random.Generator | None = None,
        asset_dir: str | Path = ASSET_DIR,
    ) -> None:
        self.spec: BrushSpec = get_brush(brush)
        self.size = int(canvas_size)
        if self.size < 4:
            raise ValueError("canvas_size must be at least 4 pixels")
        self.train = bool(train)
        self.canvas_rgb = canvas_rgb(canvas_color)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.textures: dict[str, np.ndarray] = (
            self._load_textures(Path(asset_dir)) if self.spec.name == "oilpaintbrush" else {}
        )
        self.foreground: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.canvas: np.ndarray = self.blank_canvas()

    # ----- basic state ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dim(self) -> int:
        return self.spec.dim

    def blank_canvas(self) -> np.ndarray:
        return np.ones((self.size, self.size, 3), dtype=np.float32) * self.canvas_rgb

    def reset_canvas(self) -> np.ndarray:
        self.canvas = self.blank_canvas()
        return self.canvas

    @staticmethod
    def _load_textures(asset_dir: Path) -> dict[str, np.ndarray]:
        textures = {}
        for key, fname in OIL_TEXTURE_FILES.items():
            img = cv2.imread(str(asset_dir / fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"brush texture not found: {asset_dir / fname}")
            textures[key] = img
        return textures

    # ----- sampling ---------------------------------------------------------------------
    def sample_uniform(self, n: int | None = None) -> np.ndarray:
        """Uniform ``theta ~ U(0, 1)^d``: shape ``(d,)`` when ``n`` is None, else ``(n, d)``."""
        shape = (self.dim,) if n is None else (int(n), self.dim)
        return self.rng.random(shape, dtype=np.float32)

    def initial_params_at(self, cx: float, cy: float, color) -> np.ndarray:
        """Brush-specific initial parameters for a new stroke centred at ``(cx, cy)`` with ``color``.

        Port of the per-brush branches of the original ``random_stroke_params_sampler``:
        Bezier end points collapse onto the centre, sizes / alpha are drawn from the
        ranges in the YAML ``sampler`` section, and colours are copied from the target.
        """
        s = self.spec.sampler
        size_lo, size_hi = s.get("size_range", [0.1, 0.25])
        alpha_lo, alpha_hi = s.get("alpha_range", [0.8, 0.98])
        theta_lo, theta_hi = s.get("theta_range", [0.0, 1.0])
        rgb = [float(v) for v in np.asarray(color, dtype=np.float32).reshape(3)]
        alpha = [float(self.rng.uniform(alpha_lo, alpha_hi))]
        cx, cy = float(cx), float(cy)
        if self.name in ("markerpen", "watercolor"):
            xs = [cx, cy, cx, cy, cx, cy]
            radii = [float(v) for v in self.rng.uniform(size_lo, size_hi, 2)]
            colors = rgb if self.name == "markerpen" else rgb + rgb
            params = xs + radii + colors + alpha
        elif self.name in ("oilpaintbrush", "rectangle"):
            wh = [float(v) for v in self.rng.uniform(size_lo, size_hi, 2)]
            theta = [float(self.rng.uniform(theta_lo, theta_hi))]
            colors = rgb + rgb if self.name == "oilpaintbrush" else rgb
            params = [cx, cy] + wh + theta + colors + alpha
        else:  # pragma: no cover - guarded by get_brush
            raise NotImplementedError(self.name)
        return np.clip(np.array(params, dtype=np.float32), 0.0, 1.0)

    # ----- rendering --------------------------------------------------------------------
    def is_visible(self, params) -> bool:
        """``False`` for strokes whose largest size parameter is below 0.025 (original ``check_stroke``)."""
        p = np.asarray(params, dtype=np.float32)
        if self.name in ("markerpen", "watercolor"):
            r_ = max(p[6], p[7])
        else:  # oilpaintbrush, rectangle
            r_ = max(p[2], p[3])
        return bool(r_ > VISIBILITY_THRESHOLD)

    def render_stroke(self, params) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize one stroke.

        Returns ``(foreground, alpha)``: two ``float32`` ``(W, W, 3)`` arrays in ``[0, 1]``.
        The alpha map is replicated over the three channels, as in the original
        training data. The canvas is not modified.
        """
        p = np.asarray(params, dtype=np.float32).reshape(-1)
        if p.shape[0] != self.dim:
            raise ValueError(f"{self.name}: expected {self.dim} parameters, got {p.shape[0]}")
        if self.name == "watercolor":
            fg, am = self._draw_watercolor(p)
        elif self.name == "markerpen":
            fg, am = self._draw_markerpen(p)
        elif self.name == "oilpaintbrush":
            fg, am = self._draw_oilpaintbrush(p)
        elif self.name == "rectangle":
            fg, am = self._draw_rectangle(p)
        else:  # pragma: no cover
            raise NotImplementedError(self.name)
        if not self.train:
            fg = cv2.dilate(fg, np.ones([2, 2]))
            am = cv2.erode(am, np.ones([2, 2]))
        fg = np.array(fg, dtype=np.float32) / 255.0
        am = np.array(am, dtype=np.float32) / 255.0
        return fg, am

    @staticmethod
    def composite(foreground: np.ndarray, alpha: np.ndarray, canvas: np.ndarray) -> np.ndarray:
        """``C_k = A * F + (1 - A) * C_{k-1}``."""
        return foreground * alpha + canvas * (1 - alpha)

    def draw(self, params) -> np.ndarray:
        """Rasterize one stroke and composite it onto ``self.canvas``; returns the new canvas."""
        self.foreground, self.alpha = self.render_stroke(params)
        self.canvas = self.composite(self.foreground, self.alpha, self.canvas)
        return self.canvas

    def render_sequence(
        self,
        params,
        *,
        skip_invisible: bool = True,
        on_stroke: FrameCallback | None = None,
        reset: bool = True,
    ) -> np.ndarray:
        """Draw an ``(N, d)`` sequence of strokes in order and return the final canvas.

        ``on_stroke(i, canvas)`` is called after every stroke (also after skipped ones,
        so a caller writing a time-lapse gets exactly ``N`` frames, as the original
        ``_render`` did).
        """
        arr = np.asarray(params, dtype=np.float32).reshape(-1, self.dim)
        if reset:
            self.reset_canvas()
        for i, p in enumerate(arr):
            if not skip_invisible or self.is_visible(p):
                self.draw(p)
            if on_stroke is not None:
                on_stroke(i, self.canvas)
        return self.canvas

    # ----- per-brush drawing (uint8 stage, ported one-to-one) ----------------------------
    def _empty_layers(self) -> tuple[np.ndarray, np.ndarray]:
        fg = np.zeros((self.size, self.size, 3), dtype=np.uint8)  # uint8 for anti-aliasing
        am = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        return fg, am

    def _draw_watercolor(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W = self.size
        x0, y0, x1, y1, x2, y2, radius0, radius2 = p[0:8]
        R0, G0, B0, R2, G2, B2, ALPHA = p[8:]
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0, x1, x2 = to_pixel(x0, W), to_pixel(x1, W), to_pixel(x2, W)
        y0, y1, y2 = to_pixel(y0, W), to_pixel(y1, W), to_pixel(y2, W)
        radius0 = int(1 + radius0 * W // 4)
        radius2 = int(1 + radius2 * W // 4)
        fg, am = self._empty_layers()
        alpha = (float(ALPHA * 255), float(ALPHA * 255), float(ALPHA * 255))
        tmp = 1.0 / BEZIER_SAMPLES
        for i in range(BEZIER_SAMPLES):
            t = i * tmp
            x = int((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
            y = int((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
            radius = int((1 - t) * radius0 + t * radius2)
            color = (
                float((1 - t) * R0 * 255 + t * R2 * 255),
                float((1 - t) * G0 * 255 + t * G2 * 255),
                float((1 - t) * B0 * 255 + t * B2 * 255),
            )
            cv2.circle(fg, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(am, (x, y), radius, alpha, -1, lineType=cv2.LINE_AA)
        return fg, am

    def _draw_rectangle(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W = self.size
        x0, y0, w, h, theta = p[0:5]
        R0, G0, B0, ALPHA = p[5:]
        x0 = to_pixel(x0, W)
        y0 = to_pixel(y0, W)
        w = int(1 + w * W // 4)
        h = int(1 + h * W // 4)
        theta = np.pi * theta
        fg, am = self._empty_layers()
        color = (float(R0 * 255), float(G0 * 255), float(B0 * 255))
        alpha = (float(ALPHA * 255), float(ALPHA * 255), float(ALPHA * 255))
        ptc = (x0, y0)
        pt0 = rotate_point((x0 - w, y0 - h), ptc, theta)
        pt1 = rotate_point((x0 + w, y0 - h), ptc, theta)
        pt2 = rotate_point((x0 + w, y0 + h), ptc, theta)
        pt3 = rotate_point((x0 - w, y0 + h), ptc, theta)
        ppt = np.array([pt0, pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(fg, [ppt], color, lineType=cv2.LINE_AA)
        cv2.fillPoly(am, [ppt], alpha, lineType=cv2.LINE_AA)
        return fg, am

    def _draw_markerpen(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W = self.size
        x0, y0, x1, y1, x2, y2, radius, _ = p[0:8]
        R0, G0, B0, ALPHA = p[8:]
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0, x1, x2 = to_pixel(x0, W), to_pixel(x1, W), to_pixel(x2, W)
        y0, y1, y2 = to_pixel(y0, W), to_pixel(y1, W), to_pixel(y2, W)
        radius = int(1 + radius * W // 4)
        fg, am = self._empty_layers()
        if abs(x0 - x2) + abs(y0 - y2) < MIN_MARKER_CHORD_PX:  # too small, do not draw
            return fg, am
        color = (float(R0 * 255), float(G0 * 255), float(B0 * 255))
        alpha = (float(ALPHA * 255), float(ALPHA * 255), float(ALPHA * 255))
        tmp = 1.0 / BEZIER_SAMPLES
        for i in range(BEZIER_SAMPLES):
            t = i * tmp
            x = (1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2
            y = (1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2
            ptc = (x, y)
            dx = 2 * (t - 1) * x0 + 2 * (1 - 2 * t) * x1 + 2 * t * x2
            dy = 2 * (t - 1) * y0 + 2 * (1 - 2 * t) * y1 + 2 * t * y2
            theta = np.arctan2(dx, dy) - np.pi / 2
            pt0 = rotate_point((x - radius, y - radius), ptc, theta)
            pt1 = rotate_point((x + radius, y - radius), ptc, theta)
            pt2 = rotate_point((x + radius, y + radius), ptc, theta)
            pt3 = rotate_point((x - radius, y + radius), ptc, theta)
            ppt = np.array([pt0, pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(fg, [ppt], color, lineType=cv2.LINE_AA)
            cv2.fillPoly(am, [ppt], alpha, lineType=cv2.LINE_AA)
        return fg, am

    def _draw_oilpaintbrush(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W = self.size
        x0, y0, w, h, theta = p[0:5]
        R0, G0, B0, R2, G2, B2, ALPHA = p[5:]  # ALPHA is ignored: oil strokes are opaque
        x0 = to_pixel(x0, W)
        y0 = to_pixel(y0, W)
        w = int(1 + w * W)
        h = int(1 + h * W)
        theta = np.pi * theta
        if w * h / (W**2) > LARGE_TEXTURE_AREA_FRACTION:
            brush = self.textures["large_vertical"] if h > w else self.textures["large_horizontal"]
        else:
            brush = self.textures["small_vertical"] if h > w else self.textures["small_horizontal"]
        return create_transformed_brush(brush, W, W, x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2)
