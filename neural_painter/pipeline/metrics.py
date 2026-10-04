"""Quality metrics of a finished painting against its target image (Week 5).

All inputs are ``float32`` RGB ``(H, W, 3)`` arrays in ``[0, 1]`` of the same size (the painting is rasterized at the
output size, the target is the frozen evaluation image).

* PSNR in dB from the mean squared error over all pixels and channels.
* SSIM: ``torchmetrics`` structural similarity, Gaussian window 11 x 11, sigma 1.5, data range 1; the same definition as
  the renderer-fidelity tables, so the numbers of the project are comparable.
* LPIPS (AlexNet): optional, because the backbone weights are not part of the ``lpips`` package and come from a download
  of about 230 MB; :class:`LPIPSMetric` refuses to start that download unless told to.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

ALEXNET_WEIGHTS = "alexnet-owt-7be5be79.pth"  # what torchvision's pretrained AlexNet downloads and caches


def _check(image: np.ndarray, target: np.ndarray) -> None:
    if image.shape != target.shape or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected two (H, W, 3) images of the same size, got {image.shape} and {target.shape}")


def psnr(image: np.ndarray, target: np.ndarray) -> float:
    _check(image, target)
    mse = float(np.mean((image.astype(np.float64) - target.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def ssim(image: np.ndarray, target: np.ndarray) -> float:
    _check(image, target)
    a, b = (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None] for x in (image, target))
    return float(structural_similarity_index_measure(a, b, data_range=1.0))


class LPIPSMetric:
    """LPIPS with the AlexNet backbone. Build it once and call it on many image pairs (lower is better)."""

    def __init__(self, device: str | torch.device = "cpu", allow_download: bool = False) -> None:
        if not allow_download and not self.available():
            raise RuntimeError(
                f"the AlexNet weights ({ALEXNET_WEIGHTS}, about 230 MB) are not in {self.cache_dir()}; LPIPS would "
                "download them. Pass allow_download=True to permit that."
            )
        import lpips

        self.device = torch.device(device)
        self.model = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    @staticmethod
    def cache_dir() -> Path:
        return Path(torch.hub.get_dir()) / "checkpoints"

    @classmethod
    def available(cls) -> bool:
        return (cls.cache_dir() / ALEXNET_WEIGHTS).is_file()

    @torch.no_grad()
    def __call__(self, image: np.ndarray, target: np.ndarray) -> float:
        _check(image, target)
        a, b = (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device) * 2.0 - 1.0 for x in (image, target))
        return float(self.model(a, b))


def image_metrics(image: np.ndarray, target: np.ndarray, lpips_metric: LPIPSMetric | None = None) -> dict[str, float | None]:
    """``{"psnr": ..., "ssim": ..., "lpips": ... or None}`` of ``image`` against ``target``."""
    return {
        "psnr": psnr(image, target),
        "ssim": ssim(image, target),
        "lpips": lpips_metric(image, target) if lpips_metric is not None else None,
    }
