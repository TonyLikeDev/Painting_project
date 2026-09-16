# Hardware and software profiles

Recorded for RESEARCH_PLAN.md section 3.7. Every experiment log names one of the
three profiles below. Update the software rows whenever the environment changes
and note the date.

| Profile | Machine | Compute device | CPU | RAM | Storage | OS | Python | PyTorch | Other | Recorded |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cuda` | Home desktop | NVIDIA GeForce RTX 3070, 8192 MiB VRAM, driver 595.79 (CUDA 13.2 runtime), WDDM | Intel Core i5-13400F, 10 cores / 16 threads, 2.5 GHz base | 31.8 GB | WD Blue SN580 500 GB NVMe (project on `G:`), 2 secondary disks | Windows 11 Pro 10.0.26200 | 3.11.9 (venv) | 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8 wheels | opencv-python 5.0.0, numpy 2.4.6, torchmetrics 1.9.0, lpips 0.1.4, gradio 6.27.0 | 2026-09-16 |
| `cpu` | Home desktop (same machine, `CUDA_VISIBLE_DEVICES=""`) | CPU only, 16 threads | same | same | same | same | same | same (CPU code path) | same | 2026-09-16 |
| `mps` | MacBook | Apple M4 GPU via Metal (MPS), unified memory | Apple M4 | 16 GB unified | internal SSD | macOS (Darwin 24.6) | 3.14.4 (venv) | 2.14.0, torchvision 0.29.0 | opencv-python 5.0.0, numpy 2.5.2 | 2026-09-05 |
| `cpu` (second data point) | MacBook (same machine, CPU code path) | Apple M4 CPU | Apple M4 | 16 GB | same | same | same | same | same | 2026-09-05 |

Notes

- The two machines run different PyTorch versions (2.11 on Windows, 2.14 on macOS). Both are 2.x and the
  code targets `torch>=2.0`; results tables state the profile, so this is a documented difference, not a
  problem. Pin both to the same version before the final Week 9 benchmark if a discrepancy shows up.
- The desktop GPU is shared with the Windows desktop (about 1 GB VRAM in use at idle), leaving roughly
  7 GB for training. Batch size 64 for the light renderer is expected to fit; halve on out-of-memory.
- `cpu` timings on the desktop are taken with the default thread count (16). Record `torch.get_num_threads()`
  in every CPU run (the baseline runner writes it to `device_cpu.txt`).
