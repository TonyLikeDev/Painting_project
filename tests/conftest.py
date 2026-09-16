"""Shared fixtures. The original 2021 code is optional: parity tests skip when it is absent."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = ROOT / "stylized-neural-painting"
BRUSHES = ["oilpaintbrush", "watercolor", "markerpen", "rectangle"]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture
def original_repo(monkeypatch):
    """Import the original ``renderer`` module, running from its directory (it uses relative brush paths)."""
    if not (ORIGINAL_DIR / "renderer.py").exists():
        pytest.skip("original stylized-neural-painting checkout not present")
    monkeypatch.chdir(ORIGINAL_DIR)
    monkeypatch.syspath_prepend(str(ORIGINAL_DIR))
    import renderer as original_renderer  # noqa: WPS433 (deliberate late import)

    return original_renderer


@pytest.fixture
def original_painter(original_repo):
    """The original ``painter`` module (heavier import: torchvision, networks)."""
    import painter as original_painter  # noqa: WPS433

    return original_painter


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_image(rng) -> np.ndarray:
    """240 x 320 RGB float image: smooth gradients plus a pure red pixel at row 10, column 20."""
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx / (w - 1), yy / (h - 1), 0.5 * np.ones((h, w))], axis=-1).astype(np.float32)
    img[10, 20] = (1.0, 0.0, 0.0)
    return img
