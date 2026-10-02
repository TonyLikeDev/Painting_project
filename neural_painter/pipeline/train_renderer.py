"""Train the neural renderer on synthetic strokes (Week 4).

The renderer learns to imitate the procedural rasterizer. Uniform random stroke
parameters are rasterized on the fly (``core.procedural_rasterizer`` in ``train``
mode, at the renderer's native resolution, as the original ``StrokeDataset`` did)
and the network is fitted to the resulting foreground and alpha maps.

The default recipe is the original one (Adam 2e-4 with a step schedule, 100 x the
mean of the foreground and alpha MSE, 50,000 strokes per epoch, normal(0, 0.02)
initialisation) so that a retrained renderer is directly comparable to the 2021
checkpoint. ``--loss l1_ssim`` and ``--activation none`` are the variants.

Fidelity is measured on a fixed held-out set of strokes (default 1,000, always the
same for a given size) with :func:`evaluate`, so the pretrained and the retrained
renderers can be scored on identical inputs. PSNR is computed from the MSE over the
whole set, like the original's ``acc``; SSIM is the mean over strokes.

Every run writes ``experiments/<date>_train_<name>/`` (config, learning curve,
validation image grids, log, final metrics) and ``checkpoints/<name>/`` (``last.pt``
to resume, ``best.pt`` for painting). Checkpoints are git-ignored.

CLI: ``python -m neural_painter.pipeline.train_renderer --brush oil --light --epochs 100``
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from ..core.device import device_summary, get_device, synchronize
from ..core.image_io import save_image
from ..core.procedural_rasterizer import ProceduralRasterizer
from ..core.stroke_models import canonical_brush_name, get_brush
from ..models.neural_renderer import NeuralRenderer, OUTPUT_ACTIVATIONS, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = ROOT / "checkpoints"
EXPERIMENT_ROOT = ROOT / "experiments"
VALIDATION_SEED = 7_654_321  # fixed: every renderer is scored on the same held-out strokes
LOSSES = ("l2", "l1_ssim")


# ----- data ------------------------------------------------------------------------------
class SyntheticStrokes:
    """Batches of ``(params, foreground, alpha)`` rasterized on the fly; nothing is stored on disk.

    ``params`` is ``(B, d)`` drawn uniformly from ``[0, 1]^d``; ``foreground`` and ``alpha`` are ``(B, 3, S, S)``
    float32 in ``[0, 1]`` (alpha replicated over three channels, as in the original data). Iterating yields
    ``batches_per_epoch`` batches; every iteration is a new pass over fresh strokes.

    Rasterizing is OpenCV and NumPy work that releases the GIL, so a thread pool scales across cores *without* extra
    processes. That matters on Windows, where each DataLoader worker re-imports torch and commits over 1 GB, which
    runs a busy desktop out of commit memory. A batch is generated from the seed ``(seed, pass, batch index)``, so the
    same arguments always give the same strokes, whatever the number of threads. ``start_pass`` is the number of passes
    already used: a run resumed at epoch ``e`` starts at pass ``e - 1`` and so sees exactly the strokes an uninterrupted
    run would have seen, never a replay of the first epochs.
    """

    def __init__(
        self, brush: str, size: int, batch_size: int, batches_per_epoch: int, seed: int = 0, threads: int = 1, start_pass: int = 0
    ) -> None:
        self.brush = get_brush(brush).name
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.threads = max(1, int(threads))
        cv2.setNumThreads(0)  # parallelism comes from our own thread pool; OpenCV's would oversubscribe the cores
        self._rasterizer = ProceduralRasterizer(self.brush, self.size, "black", train=True)  # render_stroke() keeps no state
        self._pass = int(start_pass)

    def batch(self, pass_id: int, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch number ``index`` of pass ``pass_id`` as numpy arrays ``(B, d)``, ``(B, 3, S, S)``, ``(B, 3, S, S)``."""
        rng = np.random.default_rng([self.seed, pass_id, index])
        params = rng.random((self.batch_size, self._rasterizer.dim), dtype=np.float32)
        s = self.size
        foreground = np.empty((self.batch_size, 3, s, s), np.float32)
        alpha = np.empty((self.batch_size, 3, s, s), np.float32)
        for i, p in enumerate(params):
            fg, am = self._rasterizer.render_stroke(p)
            foreground[i] = fg.transpose(2, 0, 1)
            alpha[i] = am.transpose(2, 0, 1)
        return params, foreground, alpha

    def __iter__(self):
        pass_id, self._pass = self._pass, self._pass + 1
        n = self.batches_per_epoch
        if self.threads == 1:
            for b in range(n):
                yield tuple(torch.from_numpy(a) for a in self.batch(pass_id, b))
            return
        with ThreadPoolExecutor(self.threads) as pool:
            pending: deque = deque()
            submitted = 0
            while submitted < n or pending:
                while submitted < n and len(pending) < 2 * self.threads:  # keep the pool busy while the GPU trains
                    pending.append(pool.submit(self.batch, pass_id, submitted))
                    submitted += 1
                yield tuple(torch.from_numpy(a) for a in pending.popleft().result())


def make_validation_set(brush: str, size: int, n: int = 1000, seed: int = VALIDATION_SEED) -> dict[str, torch.Tensor]:
    """``n`` fixed random strokes with their rasterized foreground / alpha: ``params (n, d)``, ``foreground`` / ``alpha`` ``(n, 3, S, S)``."""
    rasterizer = ProceduralRasterizer(brush, size, "black", train=True, rng=np.random.default_rng([seed, size]))
    params, foregrounds, alphas = [], [], []
    for _ in range(n):
        p = rasterizer.sample_uniform()
        fg, am = rasterizer.render_stroke(p)
        params.append(p)
        foregrounds.append(fg.transpose(2, 0, 1))
        alphas.append(am.transpose(2, 0, 1))
    return {
        "params": torch.from_numpy(np.stack(params)),
        "foreground": torch.from_numpy(np.ascontiguousarray(np.stack(foregrounds))),
        "alpha": torch.from_numpy(np.ascontiguousarray(np.stack(alphas))),
    }


# ----- loss and metrics ------------------------------------------------------------------
def reconstruction_loss(kind: str, fg: torch.Tensor, alpha: torch.Tensor, gt_fg: torch.Tensor, gt_alpha: torch.Tensor) -> torch.Tensor:
    """``l2``: the original recipe, 100 * mean of the two MSEs. ``l1_ssim``: 10 * mean L1 + mean (1 - SSIM)."""
    if kind == "l2":
        return 100.0 * (F.mse_loss(fg, gt_fg) + F.mse_loss(alpha, gt_alpha)) / 2.0
    if kind == "l1_ssim":
        l1 = (F.l1_loss(fg, gt_fg) + F.l1_loss(alpha, gt_alpha)) / 2.0
        dssim = 1.0 - (ssim_fn(fg, gt_fg, data_range=1.0) + ssim_fn(alpha, gt_alpha, data_range=1.0)) / 2.0
        return 10.0 * l1 + dssim
    raise ValueError(f"loss must be one of {LOSSES}, got {kind!r}")


def _psnr(mse: torch.Tensor) -> float:
    return float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)))


@torch.no_grad()
def evaluate(model: NeuralRenderer, val: dict[str, torch.Tensor], device: torch.device, batch_size: int = 250) -> dict[str, float]:
    """Score ``model`` against the rasterized strokes in ``val``.

    Returns ``psnr_fg`` / ``psnr_alpha`` (dB, from the MSE over the whole set), their mean ``psnr_mean`` (the
    original code's ``acc``), and the per-stroke mean ``ssim_fg`` / ``ssim_alpha``.
    """
    was_training = model.training
    model.eval()
    n = val["params"].shape[0]
    sq_fg = sq_alpha = 0.0
    ssim_fg = ssim_alpha = 0.0
    ssim_device = torch.device("cpu") if device.type == "mps" else device  # keep a third-party metric off the Metal backend
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        params = val["params"][sl].to(device)
        gt_fg, gt_alpha = val["foreground"][sl].to(device), val["alpha"][sl].to(device)
        fg, alpha = model(params)
        k = params.shape[0]
        sq_fg += float(((fg - gt_fg) ** 2).mean()) * k
        sq_alpha += float(((alpha - gt_alpha) ** 2).mean()) * k
        ssim_fg += float(ssim_fn(fg.to(ssim_device), gt_fg.to(ssim_device), data_range=1.0)) * k
        ssim_alpha += float(ssim_fn(alpha.to(ssim_device), gt_alpha.to(ssim_device), data_range=1.0)) * k
    model.train(was_training)
    psnr_fg, psnr_alpha = _psnr(torch.tensor(sq_fg / n)), _psnr(torch.tensor(sq_alpha / n))
    return {
        "psnr_fg": psnr_fg,
        "psnr_alpha": psnr_alpha,
        "psnr_mean": (psnr_fg + psnr_alpha) / 2.0,
        "ssim_fg": ssim_fg / n,
        "ssim_alpha": ssim_alpha / n,
    }


@torch.no_grad()
def save_validation_grid(model: NeuralRenderer, val: dict[str, torch.Tensor], device: torch.device, path: Path, n: int = 8) -> None:
    """Rows: predicted foreground, rasterized foreground, predicted alpha, rasterized alpha for the first ``n`` strokes."""
    was_training = model.training
    model.eval()
    fg, alpha = model(val["params"][:n].to(device))
    model.train(was_training)
    rows = [fg.clamp(0, 1), val["foreground"][:n].to(device), alpha.clamp(0, 1), val["alpha"][:n].to(device)]
    grid = np.concatenate([np.concatenate(list(r.permute(0, 2, 3, 1).cpu().numpy()), axis=1) for r in rows], axis=0)
    if grid.shape[0] < 256:  # light renderers produce 32 px tiles: enlarge so the figure is readable
        scale = int(np.ceil(256 / grid.shape[0]))
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    save_image(path, np.clip(grid, 0.0, 1.0).astype(np.float32))


# ----- model setup -----------------------------------------------------------------------
def init_like_original(model: nn.Module, gain: float = 0.02) -> None:
    """pix2pix-style initialisation used by the original ``init_weights``: normal(0, 0.02), zero biases, BN scale normal(1, 0.02)."""
    for m in model.modules():
        name = m.__class__.__name__
        if hasattr(m, "weight") and (name.find("Conv") != -1 or name.find("Linear") != -1):
            nn.init.normal_(m.weight, 0.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)
        elif name.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight, 1.0, gain)
            nn.init.constant_(m.bias, 0.0)


@dataclass
class TrainConfig:
    brush: str = "oilpaintbrush"
    light: bool = False
    epochs: int = 100
    samples_per_epoch: int = 50_000
    batch_size: int = 64
    lr: float = 2e-4
    lr_step: int = 0  # 0 means epochs // 4 (the original: step 100 over 400 epochs)
    loss: str = "l2"
    activation: str = "sigmoid"
    device: str = "auto"
    threads: int = 8  # rasterizer threads feeding the GPU
    seed: int = 0
    val_size: int = 1000
    max_minutes: float = 0.0  # stop after the epoch that exceeds this many minutes; 0 = no limit
    vis_every: int = 10
    name: str = ""  # run name; default "<brush>[_light]"
    checkpoint_dir: str = ""  # default checkpoints/<name>
    experiments_dir: str = ""  # default experiments/
    resume: bool = False


def commit_info() -> dict[str, object]:
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:  # not a git checkout, or git missing
        return {"commit": "unknown", "dirty": None}


def new_experiment_dir(base: Path, run_name: str, kind: str = "train") -> Path:
    """``<base>/<date>_<kind>_<run_name>``; a numeric suffix is added rather than ever reusing a folder."""
    stem = f"{date.today().isoformat()}_{kind}_{run_name}"
    path, k = base / stem, 1
    while path.exists():
        k += 1
        path = base / f"{stem}-{k}"
    path.mkdir(parents=True)
    return path


CURVE_FIELDS = [
    "epoch", "train_loss", "psnr_fg", "psnr_alpha", "psnr_mean", "ssim_fg", "ssim_alpha", "lr", "epoch_seconds", "strokes_per_second",
]


# ----- training --------------------------------------------------------------------------
def train(cfg: TrainConfig, log: Callable[[str], None] = print) -> dict[str, object]:
    """Run one training job; returns the final metrics, timing and the paths of everything written."""
    spec = get_brush(cfg.brush)
    cfg.brush = spec.name
    if cfg.loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {cfg.loss!r}")
    if cfg.activation not in OUTPUT_ACTIVATIONS:
        raise ValueError(f"activation must be one of {OUTPUT_ACTIVATIONS}, got {cfg.activation!r}")
    run_name = cfg.name or spec.name + ("_light" if cfg.light else "")
    ckpt_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else CHECKPOINT_ROOT / run_name
    exp_base = Path(cfg.experiments_dir) if cfg.experiments_dir else EXPERIMENT_ROOT
    device = get_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input shapes: let cuDNN pick the fastest kernels
    torch.manual_seed(cfg.seed)

    model = NeuralRenderer(spec, light=cfg.light, output_activation=cfg.activation)
    init_like_original(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    lr_step = cfg.lr_step or max(1, cfg.epochs // 4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

    start_epoch, best_psnr, exp_dir = 1, float("-inf"), None
    last_path = ckpt_dir / "last.pt"
    if cfg.resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_psnr = int(state["epoch"]) + 1, float(state["best_psnr"])
        exp_dir = Path(state["experiment_dir"])
    if exp_dir is None:
        exp_dir = new_experiment_dir(exp_base, run_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "log.txt"

    def say(message: str) -> None:
        log(message)
        with open(log_path, "a", encoding="utf-8") as fh:  # open per message: nothing stays open if the run dies
            fh.write(message + "\n")

    if start_epoch == 1:
        info = {
            "run": run_name, "started": datetime.now().isoformat(timespec="seconds"), **commit_info(), **device_summary(device),
            "model": model.describe(), "config": asdict(cfg), "lr_step": lr_step,
        }
        (exp_dir / "config.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    say(f"== {run_name} on {device} | {spec.name} {'light' if cfg.light else 'full'} {model.out_size}px | "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params | loss {cfg.loss} | epochs {start_epoch}..{cfg.epochs}")
    say(f"   experiment: {exp_dir}\n   checkpoints: {ckpt_dir}")
    if start_epoch > cfg.epochs:  # a finished run: report it, never overwrite its final metrics with an empty loop
        say(f"nothing to do: {run_name} already has {start_epoch - 1} of {cfg.epochs} epochs")
        final = exp_dir / "final_metrics.json"
        return json.loads(final.read_text(encoding="utf-8")) if final.is_file() else {"run": run_name, "epochs_done": start_epoch - 1}

    val = make_validation_set(spec.name, model.out_size, cfg.val_size)
    loader = SyntheticStrokes(
        spec.name, model.out_size, cfg.batch_size, cfg.samples_per_epoch // cfg.batch_size, seed=cfg.seed, threads=cfg.threads,
        start_pass=start_epoch - 1,
    )

    curve_path = exp_dir / "curve.csv"
    new_curve = not curve_path.exists()
    started = time.perf_counter()
    metrics: dict[str, float] = {}
    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        lr_now = optimizer.param_groups[0]["lr"]
        running = torch.zeros((), device=device)
        n_batches = 0
        for params, gt_fg, gt_alpha in loader:
            params, gt_fg, gt_alpha = (t.to(device, non_blocking=True) for t in (params, gt_fg, gt_alpha))
            optimizer.zero_grad(set_to_none=True)
            fg, alpha = model(params)
            loss = reconstruction_loss(cfg.loss, fg, alpha, gt_fg, gt_alpha)
            loss.backward()
            optimizer.step()
            running += loss.detach()  # no .item() per step: it would stall the GPU pipeline
            n_batches += 1
        synchronize(device)
        scheduler.step()
        seconds = time.perf_counter() - t0
        train_loss = float(running) / max(1, n_batches)
        metrics = evaluate(model, val, device)
        row = {
            "epoch": epoch, "train_loss": train_loss, **metrics, "lr": lr_now, "epoch_seconds": seconds,
            "strokes_per_second": n_batches * cfg.batch_size / seconds,
        }
        with open(curve_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CURVE_FIELDS)
            if new_curve:
                writer.writeheader()
                new_curve = False
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        improved = metrics["psnr_mean"] > best_psnr
        if improved:
            best_psnr = metrics["psnr_mean"]
            save_checkpoint(ckpt_dir / "best.pt", model, epoch=epoch, metrics=metrics, config=asdict(cfg))
        save_checkpoint(
            last_path, model, epoch=epoch, metrics=metrics, best_psnr=best_psnr, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), experiment_dir=str(exp_dir), config=asdict(cfg),
        )
        if epoch == start_epoch or epoch % cfg.vis_every == 0 or epoch == cfg.epochs:
            save_validation_grid(model, val, device, exp_dir / f"val_epoch_{epoch:04d}.png")
        say(
            f"epoch {epoch:>3}/{cfg.epochs} | loss {train_loss:8.5f} | psnr fg {metrics['psnr_fg']:5.2f} alpha {metrics['psnr_alpha']:5.2f} "
            f"mean {metrics['psnr_mean']:5.2f}{'*' if improved else ' '} | ssim {metrics['ssim_fg']:.3f}/{metrics['ssim_alpha']:.3f} | "
            f"lr {lr_now:.1e} | {seconds:6.1f} s | {row['strokes_per_second']:7.0f} strokes/s"
        )
        if cfg.max_minutes and (time.perf_counter() - started) / 60.0 > cfg.max_minutes and epoch < cfg.epochs:
            say(f"stopping after epoch {epoch}: the {cfg.max_minutes:g} minute budget is used up (resume with --resume)")
            break

    with open(curve_path, newline="", encoding="utf-8") as fh:  # the whole run, across resumes
        train_seconds_total = sum(float(r["epoch_seconds"]) for r in csv.DictReader(fh))
    result = {
        "run": run_name, "device": str(device), "epochs_done": epoch, "best_psnr_mean": best_psnr, "last": metrics,
        "wall_seconds": time.perf_counter() - started, "train_seconds_total": train_seconds_total,
        "experiment_dir": str(exp_dir), "checkpoint_dir": str(ckpt_dir),
    }
    (exp_dir / "final_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    state = "paused (time budget)" if epoch < cfg.epochs else "done"
    say(f"== {state}: best psnr_mean {best_psnr:.2f} dB, {result['wall_seconds'] / 60:.1f} min this session, "
        f"{train_seconds_total / 60:.1f} min of training in total")
    return result


def lower_process_priority() -> None:
    """Below-normal priority (best effort), so the machine stays responsive while a long job trains. The rasterizer
    runs as threads of this process, so one setting covers all of the work."""
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # not installed, or not permitted: training is unaffected
        pass


def build_parser() -> argparse.ArgumentParser:
    d = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--brush", default="oil", help="oil, watercolor, marker, tape (or the repository names)")
    ap.add_argument("--light", action="store_true", help="the 32 px renderer (5.4 M parameters) instead of the 128 px one (18.1 M)")
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--samples-per-epoch", type=int, default=d.samples_per_epoch)
    ap.add_argument("--batch-size", type=int, default=d.batch_size, help="halve on out-of-memory")
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--lr-step", type=int, default=d.lr_step, help="epochs between x0.1 learning-rate drops (default: epochs // 4)")
    ap.add_argument("--loss", choices=LOSSES, default=d.loss)
    ap.add_argument("--activation", choices=OUTPUT_ACTIVATIONS, default=d.activation, help="output non-linearity of the new renderer")
    ap.add_argument("--device", default=d.device, help="auto, cuda, mps or cpu")
    ap.add_argument("--threads", type=int, default=min(10, max(1, (os.cpu_count() or 4) - 2)), help="rasterizer threads feeding the GPU")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--val-size", type=int, default=d.val_size)
    ap.add_argument("--max-minutes", type=float, default=d.max_minutes)
    ap.add_argument("--vis-every", type=int, default=d.vis_every)
    ap.add_argument("--name", default=d.name, help="run name (default: <brush>[_light])")
    ap.add_argument("--checkpoint-dir", default=d.checkpoint_dir)
    ap.add_argument("--experiments-dir", default=d.experiments_dir)
    ap.add_argument("--resume", action="store_true", help="continue from <checkpoint-dir>/last.pt")
    ap.add_argument("--low-priority", action="store_true", help="run at below-normal priority to keep the machine responsive")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.low_priority:
        lower_process_priority()
    cfg = TrainConfig(
        brush=canonical_brush_name(args.brush), light=args.light, epochs=args.epochs, samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size, lr=args.lr, lr_step=args.lr_step, loss=args.loss, activation=args.activation, device=args.device,
        threads=args.threads, seed=args.seed, val_size=args.val_size, max_minutes=args.max_minutes, vis_every=args.vis_every,
        name=args.name, checkpoint_dir=args.checkpoint_dir, experiments_dir=args.experiments_dir, resume=args.resume,
    )
    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
