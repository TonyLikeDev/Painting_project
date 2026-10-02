# Week 4: Differentiable renderer and basic stroke model

Status mirror of `ROADMAP.md`. Date 2026-10-02. The desktop work is complete;
the MacBook validation run remains a user action.

## Build

- [x] Implemented `models/neural_renderer.py` for the full 128 px and light 32 px
  renderers: separate shape and colour decoders, fused as mask x colour.
- [x] Added an adapter for all eight 2021 checkpoints. The adapted network is
  bit-identical to the original network on the tested inputs, while loading is
  safe by default (`weights_only=True`); arbitrary pickle loading requires
  `trusted=True`.
- [x] Added a project checkpoint format and a common `load_renderer` loader for
  original and project-trained weights.
- [x] Implemented synthetic stroke training in
  `pipeline/train_renderer.py`: deterministic on-the-fly rasterization, threaded
  data generation, validation set, learning-curve CSV, atomic checkpoints and
  resume support.
- [x] Trained the light oil renderer for 100 epochs on the desktop RTX 3070.
- [x] Added renderer tests covering output shapes and ranges, alpha handling,
  finite-difference gradients, checkpoint equivalence, safe loading and round
  trips.
- [ ] Run `scripts/device_check.py` on the MacBook and record the MPS smoke-test
  results.

## Measure

- [x] Scored the eight original renderers and the retrained oil renderer on 1,000
  held-out strokes per brush and size in
  `2026-10-02_fidelity_week4`.
- [x] The eight original renderers reproduce the authors' stored validation
  accuracy within 0.25 dB on this protocol.
- [x] The retrained light oil renderer reached 24.16 dB mean PSNR versus 24.38 dB
  for the original, a difference of 0.22 dB and within the 1 dB target.
- [x] The retrained renderer reached foreground SSIM 0.944 and alpha SSIM 0.929,
  compared with 0.944 and 0.920 for the original.
- [x] Measured desktop light-renderer cost: 1.2 ms forward and 2.6 ms forward plus
  backward on CUDA, with 0.09 GB peak tensor memory.
- [x] Identified the procedural rasterizer as the training bottleneck: about
  2.1 ms per oil stroke on one core. Threaded generation reached about 1,600
  strokes per second during training; worker processes exhausted Windows commit
  memory and were not used.

## Write

- [x] Added the neural-renderer implementation section to
  `report/chapters/04_implementation.md`, including the architecture, checkpoint
  adapter, gradient validation, training recipe, fidelity tables, cost table and
  figures.
- [x] Saved the trained checkpoint at
  `checkpoints/oilpaintbrush_light_100ep/best.pt`.

## Exit criteria

- [x] Gradient test passes for every stroke parameter.
- [x] Oil renderer checkpoint saved.
- [x] Fidelity table completed for the original renderers and the retrained oil
  renderer.
- [~] Cross-device validation is incomplete until the MacBook MPS check is run.

## Notes and deviations

- The original recipe was retained for comparability: 50,000 synthetic strokes per
  epoch, batch size 64, Adam with learning rate $2 \times 10^{-4}$, and the loss
  $100 \cdot \tfrac{1}{2}(\mathrm{MSE}_F + \mathrm{MSE}_A)$. The project run used
  100 epochs rather than the original light renderer's 400 epochs.
- The oil opacity map is independent of the alpha parameter, but the foreground
  still depends on alpha weakly because the colour decoder receives the full
  parameter vector. The measured foreground effect is about 1% for the light
  renderer, so the painter keeps oil alpha fixed.
- The MacBook run is intentionally a short device and training smoke test, not a
  second full oil-renderer training run.