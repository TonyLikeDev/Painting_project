from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from neural_painter.core import image_io as io


def test_png_roundtrip_is_exact(tmp_path, synthetic_image):
    path = io.save_image(tmp_path / "a.png", synthetic_image)
    loaded = io.load_image(path)
    assert loaded.dtype == np.float32 and loaded.shape == synthetic_image.shape
    np.testing.assert_array_equal(loaded, io.to_uint8(synthetic_image).astype(np.float32) / 255.0)
    np.testing.assert_array_equal(loaded[10, 20], (1.0, 0.0, 0.0))  # RGB order preserved


def test_jpeg_loads_and_rejects_bad_files(tmp_path, synthetic_image):
    path = io.save_image(tmp_path / "a.jpg", synthetic_image)
    loaded = io.load_image(path)
    assert loaded.shape == synthetic_image.shape and 0.0 <= loaded.min() and loaded.max() <= 1.0
    with pytest.raises(io.UnsupportedImageError):
        io.load_image(tmp_path / "a.txt")
    bogus = tmp_path / "b.png"
    bogus.write_text("not an image")
    with pytest.raises(io.UnsupportedImageError):
        io.load_image(bogus)
    with pytest.raises(io.ImageTooLargeError):
        io.load_image(path, max_pixels=100)


def test_exif_orientation_is_applied(tmp_path, synthetic_image):
    path = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    Image.fromarray(io.to_uint8(synthetic_image)).save(path, exif=exif)
    loaded = io.load_image(path)
    assert loaded.shape[:2] == (320, 240)


def test_grey_and_rgba_inputs_become_rgb():
    grey = np.full((8, 9), 0.25, np.float32)
    assert io.ensure_float01(grey).shape == (8, 9, 3)
    rgba = np.zeros((8, 9, 4), np.uint8)
    rgba[..., 3] = 255
    assert io.ensure_float01(rgba).shape == (8, 9, 3)
    with pytest.raises(ValueError):
        io.ensure_float01(np.zeros((8, 9, 2)))


def test_fit_modes(synthetic_image):
    stretched, box = io.fit(synthetic_image, 128, "stretch")
    assert stretched.shape == (128, 128, 3) and box is None
    cropped, box = io.fit(synthetic_image, 128, "crop")
    assert cropped.shape == (128, 128, 3) and box == (40, 0, 280, 240)
    padded, box = io.fit(synthetic_image, 128, "pad", pad_color=0.0)
    assert padded.shape == (128, 128, 3) and box == (0, 40, 320, 280)
    assert padded[0, 64].max() == 0.0 and padded[64, 64].max() > 0.0
    wide, box = io.fit(synthetic_image, (256, 128), "crop")
    assert wide.shape == (128, 256, 3) and box == (0, 40, 320, 200)
    with pytest.raises(ValueError):
        io.fit(synthetic_image, 128, "tile")


def test_crop_bounds():
    img = np.zeros((10, 12, 3), np.float32)
    assert io.crop(img, (2, 3, 7, 9)).shape == (6, 5, 3)
    with pytest.raises(ValueError):
        io.crop(img, (0, 0, 13, 5))


def test_tensor_conversion(synthetic_image):
    t = io.to_tensor(synthetic_image)
    assert t.shape == (1, 3, 240, 320) and t.dtype == torch.float32
    np.testing.assert_array_equal(io.to_numpy(t), synthetic_image)
    np.testing.assert_array_equal(io.to_numpy(t[0]), synthetic_image)
    with pytest.raises(ValueError):
        io.to_numpy(torch.zeros(2, 3, 4, 4))


def test_list_sample_images(tmp_path):
    for name in ("b.jpg", "a.png", "c.txt", "d.PNG"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    assert [p.name for p in io.list_sample_images(tmp_path)] == ["a.png", "b.jpg", "d.PNG"]
    assert io.list_sample_images(tmp_path / "missing") == []


def test_preprocess_from_array_and_path(tmp_path, synthetic_image):
    pre = io.preprocess(synthetic_image, 64, "crop")
    assert pre.tensor.shape == (1, 3, 64, 64) and pre.image.shape == (64, 64, 3)
    assert pre.original_size == (320, 240) and pre.aspect_ratio == pytest.approx(0.75)
    assert pre.crop_box == (40, 0, 280, 240) and pre.source is None
    assert pre.output_size(512) == (512, 512)
    assert pre.output_size(512, keep_aspect_ratio=True) == (512, 384)
    path = io.save_image(tmp_path / "in.png", synthetic_image)
    pre2 = io.preprocess(path, (96, 48), "stretch", device="cpu")
    assert pre2.tensor.shape == (1, 3, 48, 96) and pre2.source == str(path) and pre2.crop_box is None
    with pytest.raises(io.ImageTooLargeError):
        io.preprocess(synthetic_image, 64, max_pixels=10)
