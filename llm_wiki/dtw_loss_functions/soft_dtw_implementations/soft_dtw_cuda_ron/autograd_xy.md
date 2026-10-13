# `soft_dtw_cuda_ron/autograd_xy.py`

[← Index](../../../0_index.md) · [↑ ron backend](__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

`SoftDTWXYAutograd(torch.autograd.Function)` is the **fused** SoftDTW for squared Euclidean distance on CUDA. It takes the raw sequences `X (B,N,D)` and `Y (B,M,D)` and **never materializes the `B×N×M` distance matrix**: costs are recomputed on the fly inside the kernels. This is the memory-saving path (upstream claims about 98% memory saving, at some speed cost). [`module.SoftDTW`](module.md) selects it automatically on CUDA unless `fused=False`.

## `forward(ctx, X, Y, gamma, bandwidth)`

1. `out, R = softdtw_forward_cuda_fused_sqeuclid(X, Y, gamma, bw)` (in [cuda/launcher](cuda/launcher.md)), where `bw = -1.0` if `bandwidth is None`.
2. Saves `X`, `Y`, `R.detach()` and `gamma` on `ctx`. A `bandwidth <= 0` is normalized to `-1.0` on `ctx`.

## `backward(ctx, grad_output)`

1. `E = softdtw_backward_cuda_fused_sqeuclid(X, Y, R, gamma, bw)` has shape `(B,N,M)` and is computed in log space.
2. `E *= grad_output.view(-1,1,1)`.
3. Chain rule for `D_ij = ||x_i - y_j||²`:
   - `grad_X = 2·(X · ΣjE_ij − E @ Y)`
   - `grad_Y = 2·(Y · ΣiE_ij − Eᵀ @ X)`
4. Returns `(grad_X, grad_Y, None, None)`.

## Note

The forward pass forwards a `bandwidth <= 0` (when not `None`) unchanged to the kernel. The kernels treat any value `<= 0` as disabled (`if bandwidth > 0 and ...`), so the behavior is the same.

The backward pass still allocates `E` of shape `(B,N,M)`, so "fused" saves the `D` matrices (and the padded `D`) but not `E`.
