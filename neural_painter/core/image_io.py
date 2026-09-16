"""Image input and preprocessing (Week 3).

Covers proposal section 3.1: load JPG / PNG (or pick a sample image), crop,
resize with or without keeping the aspect ratio, normalize to ``[0, 1]`` and
place the result on a device as a ``(1, 3, H, W)`` float tensor.

Conventions: images are ``numpy`` ``float32`` arrays of shape ``(H, W, 3)`` in
RGB order and ``[0, 1]``; tensors are ``(1, 3, H, W)``. The original code used
OpenCV's BGR order internally and converted on load; here everything is RGB.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
MAX_PIXELS = 40_000_000  # refuse accidental 100-megapixel uploads before they hit the GPU

ResizeMode = Literal["stretch", "crop", "pad"]
Box = tuple[int, int, int, int]  # (left, top, right, bottom), right / bottom exclusive


class UnsupportedImageError(ValueError):
    """The file is not one of the supported raster formats or cannot be decoded."""


class ImageTooLargeError(ValueError):
    """The image has more pixels than ``MAX_PIXELS``."""


# ----- loading and saving ----------------------------------------------------------------
def list_sample_images(directory: str | Path, suffixes: Iterable[str] = SUPPORTED_SUFFIXES) -> list[Path]:
    """Sorted list of image files directly inside ``directory`` (the sample-image picker)."""
    suffixes = {s.lower() for s in suffixes}
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def load_image(path: str | Path, *, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    """Load an image as ``float32`` RGB ``(H, W, 3)`` in ``[0, 1]``.

    EXIF orientation is applied (phone photos), alpha channels are dropped and
    grey-scale images are expanded to three channels.
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedImageError(f"unsupported image format {path.suffix!r} ({path.name}); use {sorted(SUPPORTED_SUFFIXES)}")
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.width * im.height > max_pixels:
                raise ImageTooLargeError(f"{path.name} has {im.width * im.height:,} pixels, more than the limit of {max_pixels:,}")
            arr = np.asarray(im.convert("RGB"))
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError(f"cannot decode {path}") from exc
    return arr.astype(np.float32) / 255.0


def save_image(path: str | Path, img: np.ndarray, *, quality: int = 95) -> Path:
    """Save a ``float32`` RGB image in ``[0, 1]`` (or ``uint8``) to PNG / JPG by suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = to_uint8(img)
    im = Image.fromarray(u8, mode="RGB")
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(path, quality=quality)
    else:
        im.save(path)
    return path


def ensure_float01(img: np.ndarray) -> np.ndarray:
    """Coerce ``uint8`` / float, grey / RGBA input into ``float32`` RGB ``(H, W, 3)`` in ``[0, 1]``."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ----- geometry ----------------------------------------------------------------------------
def crop(img: np.ndarray, box: Box) -> np.ndarray:
    """Crop ``(left, top, right, bottom)`` with bounds checking."""
    left, top, right, bottom = box
    h, w = img.shape[:2]
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError(f"crop box {box} is outside the image of size {(w, h)}")
    return img[top:bottom, left:right]


def center_crop_box(height: int, width: int, aspect: float = 1.0) -> Box:
    """Largest centred box with width / height == ``aspect`` that fits inside ``(height, width)``."""
    if aspect <= 0:
        raise ValueError("aspect must be positive")
    if width / height > aspect:  # too wide: cut the sides
        new_w = int(round(height * aspect))
        new_h = height
    else:  # too tall: cut top and bottom
        new_w = width
        new_h = int(round(width / aspect))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    return (left, top, left + new_w, top + new_h)


def center_crop(img: np.ndarray, aspect: float = 1.0) -> tuple[np.ndarray, Box]:
    box = center_crop_box(img.shape[0], img.shape[1], aspect)
    return crop(img, box), box


def resize(img: np.ndarray, size: int | tuple[int, int], interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize to ``size`` = side or ``(width, height)`` pixels. ``INTER_AREA`` matches the original code."""
    w, h = (size, size) if isinstance(size, int) else size
    if w <= 0 or h <= 0:
        raise ValueError("size must be positive")
    return np.ascontiguousarray(cv2.resize(img, (int(w), int(h)), interpolation=interpolation))


def pad_to_aspect(img: np.ndarray, aspect: float = 1.0, color: float | tuple[float, float, float] = 1.0) -> tuple[np.ndarray, Box]:
    """Letterbox ``img`` with ``color`` so that width / height == ``aspect``. Returns the box of the original content."""
    h, w = img.shape[:2]
    if w / h > aspect:
        new_w, new_h = w, int(round(w / aspect))
    else:
        new_w, new_h = int(round(h * aspect)), h
    out = np.empty((new_h, new_w, 3), dtype=np.float32)
    out[:] = np.asarray(color, dtype=np.float32).reshape(-1)[:3] if np.ndim(color) else float(color)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    out[top : top + h, left : left + w] = img
    return out, (left, top, left + w, top + h)


def fit(img: np.ndarray, size: int | tuple[int, int], mode: ResizeMode = "crop", pad_color=1.0) -> tuple[np.ndarray, Box | None]:
    """Bring ``img`` to ``size`` with one of three policies.

    - ``"stretch"``: plain resize, distorts the aspect ratio (what the original code does);
    - ``"crop"``: centre-crop to the target aspect ratio, then resize (no distortion, loses borders);
    - ``"pad"``: letterbox to the target aspect ratio, then resize (no distortion, adds borders).

    Returns the image and the crop / content box in the source image's pixel coordinates
    (``None`` for stretch).
    """
    w, h = (size, size) if isinstance(size, int) else size
    aspect = w / h
    if mode == "stretch":
        return resize(img, (w, h)), None
    if mode == "crop":
        cropped, box = center_crop(img, aspect)
        return resize(cropped, (w, h)), box
    if mode == "pad":
        padded, box = pad_to_aspect(img, aspect, pad_color)
        return resize(padded, (w, h)), box
    raise ValueError(f"unknown resize mode {mode!r}; use 'stretch', 'crop' or 'pad'")


# ----- tensors -----------------------------------------------------------------------------
def to_tensor(img: np.ndarray, device: torch.device | str | None = None) -> torch.Tensor:
    """``(H, W, 3)`` float image to a ``(1, 3, H, W)`` float32 tensor on ``device``."""
    arr = ensure_float01(img)
    t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0)
    return t.to(device) if device is not None else t


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """``(1, 3, H, W)`` or ``(3, H, W)`` tensor to a clipped ``float32`` ``(H, W, 3)`` image."""
    t = t.detach()
    if t.dim() == 4:
        if t.shape[0] != 1:
            raise ValueError("to_numpy expects a single image; use a loop for batches")
        t = t[0]
    return np.clip(t.float().cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)


# ----- end-to-end --------------------------------------------------------------------------
@dataclass
class PreprocessedImage:
    """Everything the painter needs about one input image."""

    tensor: torch.Tensor  # (1, 3, H, W) on the requested device
    image: np.ndarray  # (H, W, 3) float32 CPU copy of the same pixels
    source: str | None  # file path or None for in-memory input
    original_size: tuple[int, int]  # (width, height) before preprocessing
    aspect_ratio: float  # height / width of the original, as in the original code's input_aspect_ratio
    crop_box: Box | None  # region of the original that survived (None for stretch)
    mode: str

    @property
    def size(self) -> tuple[int, int]:
        """(height, width) of the preprocessed image."""
        return int(self.image.shape[0]), int(self.image.shape[1])

    def output_size(self, canvas_size: int, keep_aspect_ratio: bool = False) -> tuple[int, int]:
        """(width, height) of the final rendering, port of the original ``_render`` logic."""
        if not keep_aspect_ratio:
            return canvas_size, canvas_size
        if self.aspect_ratio < 1:
            return canvas_size, int(canvas_size * self.aspect_ratio)
        return int(canvas_size / self.aspect_ratio), canvas_size


def preprocess(
    source: str | Path | np.ndarray,
    size: int | tuple[int, int] = 512,
    mode: ResizeMode = "crop",
    device: torch.device | str | None = None,
    *,
    pad_color=1.0,
    max_pixels: int = MAX_PIXELS,
) -> PreprocessedImage:
    """Load (or accept) an image, fit it to ``size``, normalize, and move it to ``device``."""
    if isinstance(source, (str, Path)):
        img = load_image(source, max_pixels=max_pixels)
        src = str(source)
    else:
        img = ensure_float01(source)
        if img.shape[0] * img.shape[1] > max_pixels:
            raise ImageTooLargeError(f"image has {img.shape[0] * img.shape[1]:,} pixels, more than the limit of {max_pixels:,}")
        src = None
    h, w = img.shape[:2]
    fitted, box = fit(img, size, mode, pad_color)
    return PreprocessedImage(
        tensor=to_tensor(fitted, device),
        image=fitted,
        source=src,
        original_size=(w, h),
        aspect_ratio=h / w,
        crop_box=box,
        mode=mode,
    )
