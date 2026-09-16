# Note: `loss.py` (original repo)

## 1. Purpose

`loss.py` collects the four loss modules used to fit stroke parameters to a target image: the pixel loss, the Sinkhorn optimal-transport loss (wrapping `pytorch_batch_sinkhorn.py`), and two VGG-16 based losses (perceptual and Gram-matrix style). The painter always builds `PixelLoss` and `SinkhornLoss`; the style loss is built only by the style-transfer demo; the perceptual loss is never instantiated anywhere in the repo.

## 2. Key classes and functions

| Name | Inputs / outputs | What it does |
| :--- | :--- | :--- |
| `PixelLoss(p=1)` | `canvas, gt [B, 3, H, W]` to scalar | `mean(|canvas - gt| ** p)`. `p=1` in the painter (L1), `p=2` in the imitator (MSE). `ignore_color=True` averages over the channel axis first, so only luminance-like intensity is compared (used by style transfer). |
| `VGGPerceptualLoss(resize=True)` | two images to scalar | `torchvision.models.vgg16(pretrained=True)`, four feature blocks (`features[:4]`, `[4:9]`, `[9:16]`, `[16:23]`), ImageNet normalisation, bilinear resize to 224x224, sum of per-block L1 feature distances. **Unused.** |
| `VGGStyleLoss(transfer_mode, resize=True)` | `input, target` to scalar | Same VGG-16 with every `MaxPool2d` replaced by `AvgPool2d(2)`. `transfer_mode=0` ("colour only") keeps the first two blocks; any other mode keeps four blocks ("colour and texture"). Loss is the sum over blocks of the squared Frobenius distance between Gram matrices `G = F F^T / (C H W)`. Mean and std buffers are created directly on the module-level `device`. |
| `SinkhornLoss(epsilon=0.01, niter=5, normalize=False)` | `canvas, gt [B, 3, H, W]` to scalar | Downsamples both to **24x24** with `mode='area'` when `H > 24`, builds a normalised coordinate grid `[B, 576, 2]` with entries `(y/h, x/w)`, picks **one random colour channel** `i in {0, 1, 2}` per call, flattens that channel to a mass vector `[B, 576]`, and calls `sinkhorn_loss` (or `sinkhorn_normalized`, the Sinkhorn divergence `2 W_xy - W_xx - W_yy`, when `normalize=True`). |
| `SinkhornLoss._mesh_grids(batch_size, h, w)` | `[B, h*w, 2]` | Row coordinate first, then column, both divided by the dimension so the transport cost is scale-free. |

Effective hyper-parameters at painting time: `epsilon = 0.01`, `niter = 5`, `normalize = False`, `beta_ot = 0.1` (from `demo_prog.py`), and the loss is only added when `--with_ot_loss` is passed. Memory per Sinkhorn call is dominated by the cost matrix and transport plan, `B x 576 x 576` floats each, i.e. about 1.3 MB per batch element; at grid level 5 that is 25 elements.

## 3. Data flow

- `painter.PainterBase.__init__` instantiates `PixelLoss(p=1)` and `SinkhornLoss(0.01, 5, False)`; `painter._backward_x` sums `beta_L1 * pixel + beta_ot * sinkhorn` on `G_final_pred_canvas` versus `img_batch`, both of shape `[m_grid^2, 3, out_size, out_size]`.
- `painter.NeuralStyleTransfer` builds `VGGStyleLoss(transfer_mode)` and applies it to the stitched full canvas (`utils.patches2img(..., to_numpy=False)`) against a 128x128 blurred style image, together with the pixel loss in `ignore_color` mode.
- `imitator.Imitator` uses `PixelLoss(p=2)` twice per batch (foreground and alpha) for renderer training.
- `SinkhornLoss` delegates the actual algorithm to `pytorch_batch_sinkhorn.sinkhorn_loss` and `cost_matrix`.

## 4. Quirks, bugs and technical debt found

- The random channel choice in `SinkhornLoss.forward` uses the unseeded Python `random` module, so the OT loss is a different function on every step; this is a variance-reduction shortcut, not a documented design decision.
- Downsampling to a fixed 24x24 means the OT term sees the same resolution whether the renderer is the 128 px or the 32 px one; for the light renderer (32 px) the "downsampling" is actually to a size close to the input.
- The pixel intensities are used directly as mass, so a white canvas versus a dark target has very unbalanced masses; `sinkhorn_loss` renormalises each side to sum to one, which discards absolute brightness information.
- `torchvision.models.vgg16(pretrained=True)` is a deprecated argument; modern torchvision expects `weights=VGG16_Weights.IMAGENET1K_V1` and warns at import.
- `VGGPerceptualLoss` keeps `mean` and `std` on CPU and converts them with `type_as` in `forward`; `VGGStyleLoss` puts them on the global `device` at construction. Two conventions for the same thing.
- `VGGStyleLoss.gram_matrix` normalises by `C*H*W`, but the loss then uses `sum` rather than `mean` over the Gram entries, so its scale grows with the number of channels of deeper blocks.
- `ignore_color` in `PixelLoss` reduces to a mean over RGB, not to a proper luminance; fine for the demo but not a perceptual grey.
- A module-level `device` global is defined here too (fifth copy), so the file cannot be imported without touching CUDA/MPS availability.

## 5. What changes in the new `neural_painter` package

- `losses/pixel_loss.py`: `L1Loss`, `MSELoss`, and `CharbonnierLoss` (`sqrt(d^2 + eps^2)`) sharing one interface; an optional luminance mode using the Rec. 601 weights instead of a channel mean.
- `losses/sinkhorn.py`: a self-contained batched, log-domain Sinkhorn with configurable `epsilon`, `n_iter`, downsample size, and a `channel` policy (`'random'`, `'all'`, or `'luminance'`) driven by the config and a passed-in RNG; no device global, works on `cuda`, `mps`, and `cpu`; the Sinkhorn divergence is the default because it is zero when canvas equals target.
- `losses/style_loss.py`: Gram-matrix style loss built on `models/vgg_perceptual.py` with `weights=` enum, average pooling, a `mean` reduction, and per-layer weights from the config. Marked as a Week 9 stretch goal.
- All losses accept tensors on any device and are unit-tested in `tests/test_optimizer.py` (loss decreases over 50 Adam steps on a toy target, Sinkhorn gives a non-zero gradient for a non-overlapping stroke where the pixel loss gives none).
