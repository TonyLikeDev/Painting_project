# Note: `pytorch_batch_sinkhorn.py` (original repo)

## 1. Purpose

`pytorch_batch_sinkhorn.py` is an 81-line batched implementation of the entropy-regularised optimal-transport (Sinkhorn) distance between two weighted point clouds, adapted from the SinkhornAutoDiff project by Gabriel Peyre. It is the numerical core behind `loss.SinkhornLoss`: the point clouds are the pixel-centre coordinates of the canvas and the target, and the point weights are the pixel intensities.

## 2. Key classes and functions

| Name | Inputs / outputs | What it does |
| :--- | :--- | :--- |
| `cost_matrix(x, y, p=2)` | `x [B, n, 2]`, `y [B, m, 2]` to `C [B, n, m]` | Pairwise cost $C_{ij} = \sum_k |x_{ik} - y_{jk}|^p$, i.e. squared Euclidean distance for `p=2`, computed with a broadcast subtraction `[B, n, 1, 2] - [B, 1, m, 2]`. For the 24x24 grids used by the painter, `n = m = 576`. |
| `sinkhorn_loss(x, y, epsilon, niter, mass_x=None, mass_y=None)` | coordinates and masses to scalar | Builds `C`, turns the masses into marginals $\mu, \nu$ (uniform `1/n` when a mass is `None`, otherwise clamped to `[0, 1e9]` in place through `.data`, plus `1e-9`, then normalised to sum to one), runs `niter` log-domain Sinkhorn updates, forms the transport plan $\pi = \exp(M(u, v))$ and returns `mean_B( sum_ij pi_ij C_ij )`. |
| `M(u, v)` (inner) | `[B, n, m]` | Log-domain kernel $M_{ij} = (-C_{ij} + u_i + v_j) / \epsilon$. |
| `lse(A)` (inner) | `[B, n, 1]` | `log(sum(exp(A), dim=2) + 1e-6)`; the `1e-6` guards against `log(0)` but is not a true numerically stable log-sum-exp (no max subtraction). |
| `sinkhorn_normalized(x, y, epsilon, niter, mass_x, mass_y)` | scalar | Sinkhorn divergence $2 W_{xy} - W_{xx} - W_{yy}$, which is zero when the two measures coincide. Costs three Sinkhorn solves. Not used by the demos (`normalize=False`). |

Algorithm as implemented (for each of `niter = 5` iterations):

```
u = eps * (log(mu) - lse(M(u, v)))            + u
v = eps * (log(nu) - lse(M(u, v)^T))          + v
```

then `pi = exp(M(u, v))` and `cost = sum(pi * C)`. With `epsilon = 0.01` and coordinates in `[0, 1]`, entries of `-C / eps` range down to about `-200`, so `exp` underflows to zero for far-apart pixel pairs; the `+ 1e-6` in `lse` is what keeps the update finite. Five iterations are far from convergence of the marginal constraints, but the gradient direction is already useful, which is all the painter needs.

## 3. Data flow

- Called only from `loss.SinkhornLoss.forward`, which supplies `x = y = mesh grid [B, 576, 2]` (identical coordinates for canvas and target) and `mass_x`, `mass_y` = one flattened colour channel of the 24x24 canvas and target.
- Gradients flow from `cost` back through `pi` (and through `mu` because `mass_x` depends on the canvas) into the neural renderer output and finally into the stroke parameters. Because `x` and `y` are constants, only the mass path carries gradient.
- The returned scalar is scaled by `beta_ot` (0.1) in `painter._backward_x`.

## 4. Quirks, bugs and technical debt found

- Docstring of `M(u, v)` contains `"$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"`; the `\e` is an invalid escape sequence and raises a `SyntaxWarning` on every import with Python 3.12+ (and a `DeprecationWarning` earlier). Harmless but noisy.
- `from torch.autograd import Variable` is imported and never used; `Variable` has been a no-op since PyTorch 0.4.
- Module-level `device` global (sixth copy in the repo); the uniform marginals and the normalised masses are moved with `.to(device)` even though they are already on the device of the inputs.
- `mass_x.data = torch.clamp(mass_x.data, ...)` mutates the caller's tensor storage in place through the `.data` back door, bypassing autograd. `mass_x` is a `reshape` of a channel slice of the canvas; because the slice was made with a list index (`canvas[:, [i]]`) it is a copy, so the canvas itself is not corrupted, but the pattern is fragile.
- `lse` is not a max-subtracted log-sum-exp; it relies on the `1e-6` floor rather than on `torch.logsumexp`, which exists and is exact.
- `err = 0.` and the `U, V = u, v` aliasing are leftovers from the original point-cloud script; there is no convergence check, so `niter` is always exhausted.
- `cost_matrix` builds a `[B, n, m, 2]` intermediate before the sum, which is the peak-memory term (`25 x 576 x 576 x 2` floats, about 66 MB, at grid level 5).
- The returned value is `mean` over the batch, while `PixelLoss` is a mean over all elements; the two terms therefore have different implicit scales, which is what `beta_ot = 0.1` compensates for empirically.

## 5. What changes in the new `neural_painter` package

- `losses/sinkhorn.py` replaces this file with one `sinkhorn_divergence(a, b, coords, epsilon, n_iter)` function and a `SinkhornLoss(nn.Module)` wrapper: `torch.logsumexp` for the updates, no `.data` mutation (masses are clamped with `clamp_min` in the graph), no device globals, an optional early stop on the marginal error, and the squared-distance cost computed as `|x|^2 + |y|^2 - 2 x y^T` to avoid the `[B, n, m, 2]` intermediate.
- The coordinate grid is cached per `(h, w, device)` instead of rebuilt every call.
- Configurable through `configs/*.yaml`: `epsilon`, `n_iter`, `downsample`, `channel` policy, `divergence: true|false`, and `weight` (the old `beta_ot`).
- `tests/test_optimizer.py` adds two unit checks: the divergence of a measure with itself is about zero, and a stroke placed far from its target receives a non-zero positional gradient from the OT term while the pixel-loss gradient on the position is zero. This is the empirical basis for RQ1 in `RESEARCH_PLAN.md`.
