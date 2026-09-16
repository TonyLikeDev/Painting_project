# Note: `networks.py` (original repo)

## 1. Purpose

`networks.py` defines the **neural renderer** $G_\phi$: a generator that maps one stroke parameter vector (shaped `[N, d, 1, 1]`) to an RGB foreground and an alpha map, plus several alternative generators kept for the paper's ablation (DCGAN, U-Net, HuangNet). Only `ZouFCNFusion` (128 px) and `ZouFCNFusionLight` (32 px) are used by the demos and shipped as checkpoints.

## 2. Key classes and functions

| Name | Inputs / outputs | What it does |
| :--- | :--- | :--- |
| `define_G(rdrr, netG, init_type='normal', init_gain=0.02, gpu_ids=[])` | `nn.Module` | Factory keyed by the strings `plain-dcgan`, `plain-unet`, `huang-net`, `zou-fusion-net`, `zou-fusion-net-light`; applies `init_weights` (normal, std 0.02, pix2pix style). The checkpoint overwrites these weights at load time anyway. |
| `DCGAN(rdrr, ngf=64)` | `[N, d, 1, 1]` to two `[N, 3, 128, 128]` | Six `ConvTranspose2d` + `BatchNorm2d` + `ReLU` stages, 1 to 4 to 8 to 16 to 32 to 64 to 128 px; the last layer has 6 channels split into (foreground, alpha). No output activation. |
| `DCGAN_32(rdrr, ngf=64)` | to `[N, 3, 32, 32]` | Four-stage variant, 1 to 4 to 8 to 16 to 32 px. |
| `PixelShuffleNet(input_nc)` | `[N, d_shape, 1, 1]` to `[N, 3, 128, 128]` | `squeeze`, FC 512 to 1024 to 2048 to 4096, reshape `[N, 16, 16, 16]`, then three (conv, conv, `PixelShuffle(2)`) stages 16 to 32 to 64 to 128 px ending in 3 channels. This is the **shape decoder** that produces the mask. |
| `PixelShuffleNet_32(input_nc)` | to `[N, 3, 32, 32]` | FC 512 to 1024 to 2048, reshape `[N, 8, 16, 16]`, one conv pair + `PixelShuffle(2)`. |
| `HuangNet(rdrr)` | to two `[N, 3, 128, 128]` | Same as `PixelShuffleNet` but on all `d` params with 6 output channels; the renderer from Learning to Paint (Huang et al. 2019). Not used by the demos. |
| `ZouFCNFusion(rdrr)` | `[N, d, 1, 1]` to `(color * mask, alpha * mask)` | `mask = PixelShuffleNet(d_shape)(x[:, :d_shape])`, `color, _ = DCGAN(x)`. The alpha scalar `x[:, [-1]]` is broadcast over the mask, **except** for `oilpaintbrush` (and `airbrush`) where it is forced to 1.0. `out_size = 128`. |
| `ZouFCNFusionLight(rdrr)` | same at 32 px | Same fusion with the `_32` sub-nets, `out_size = 32`. |
| `UNet / UnetGenerator / UnetSkipConnectionBlock` | `[N, d, 1, 1]` repeated to 128x128 | pix2pix U-Net with 7 down-samplings and 6 output channels. Ablation only. |
| `get_norm_layer`, `init_weights`, `init_net` | helpers | Copied from pix2pix/CycleGAN. `init_net` wraps in `DataParallel` when `gpu_ids` is non-empty (never used). |

Approximate parameter counts, estimated by hand from the layer sizes: the full fusion net is dominated by the 2048 to 4096 FC layer (about 8.4 M) and the two 512-channel transposed convolutions (about 6 M), roughly 18 M in total; the light net is roughly 5 M. Batch size at painting time is `m_grid^2 * (anchor_id + 1)` strokes, so up to 25 * 9 = 225 strokes per forward pass at grid level 5 with 500 strokes.

## 3. Data flow

- **Input** `x` is built in `painter._forward_pass` as `cat([x_ctt, x_color, x_alpha])` reshaped to `[N, d, 1, 1]`. Only the first `d_shape` entries reach the mask decoder; the colour decoder sees all `d`.
- **Output** foreground and alpha go through `morphology.Dilation2d` / `Erosion2d` (3x3) and then the sequential compositing loop in `painter._forward_pass`.
- **Training** pairs come from `utils.StrokeDataset`; `imitator.py` compares the foreground with `B` and the alpha with `ALPHA` using `PixelLoss(p=2)`.
- **Checkpoints** are dicts with the key `model_G_state_dict` (plus optimizer state and `best_val_acc`, a NumPy scalar, which is why `weights_only=False` is required on PyTorch 2.6+).

## 4. Quirks, bugs and technical debt found

- A module-level `device` global is recomputed here and in `painter.py`, `loss.py`, `pytorch_batch_sinkhorn.py`, `imitator.py`, and `demo_prog.py`; the fusion nets call `torch.tensor(1.0).to(device)` inside `forward`, so the model cannot be moved to a device other than the global.
- `x.squeeze()` in the shape decoder squeezes **all** singleton dims. For batch size 1 the input `[1, d_shape, 1, 1]` becomes 1-D `[d_shape]`; `nn.Linear` still accepts a 1-D input and the later `view(-1, ...)` recovers the batch, so it does not crash, but it is fragile and any hook expecting 2-D activations would break.
- No output activation: foreground and mask are unbounded; `painter` relies on clamping the inputs and on the training targets being in $[0,1]$.
- The colour decoder (`DCGAN`) also receives the shape parameters, and its second output (an alpha head) is computed and discarded, wasting a full 6-channel decode.
- `oilpaintbrush` alpha is hard-wired to 1.0 inside the network, so the optimizer's `x_alpha` for oil is a dead parameter that receives zero gradient and is still saved to the `.npz`.
- `HuangNet`, stand-alone `DCGAN`, and the U-Net are dead code paths for the shipped checkpoints; `matplotlib`, `numpy`, `lr_scheduler`, and `models` imports are unused.
- `BatchNorm2d` in the colour decoder means the output depends on `eval()` being called; the painter does call it, but validation inside the imitator shares running statistics with training.

## 5. What changes in the new `neural_painter` package

- `models/neural_renderer.py`: one `NeuralRenderer(nn.Module)` with `ShapeDecoder` (MLP + PixelShuffle) and `ColorDecoder` (transposed convolutions) sub-modules, `out_size` and the parameter layout read from the YAML config, a sigmoid on both heads, alpha handling (`opaque: true` for oil) as a config flag instead of a string test, and no device globals: the module is moved with `.to(device)` like any other.
- A **checkpoint adapter** (`load_original_checkpoint(path, brush)`) that maps the `huangnet.*` and `dcgan.main.*` state-dict keys onto the new names, so the pretrained 2021 weights and our retrained weights share one interface (Week 4).
- `models/vgg_perceptual.py`: the VGG feature extractor used by the style loss, separated from the renderer.
- `pipeline/train_renderer.py`: replaces `imitator.py` with on-the-fly data, $L_1$ + SSIM, optional AMP and `torch.compile`, and `cuda / mps / cpu` selection.
- `tests/test_renderer.py`: forward shape check, output range, and a finite-difference gradient check on a few stroke parameters.
