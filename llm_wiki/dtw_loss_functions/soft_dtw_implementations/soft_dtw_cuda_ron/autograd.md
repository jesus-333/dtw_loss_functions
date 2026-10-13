# `soft_dtw_cuda_ron/autograd.py`

[← Index](../../../0_index.md) · [↑ ron backend](__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

`SoftDTWAutograd(torch.autograd.Function)` is the **unfused** SoftDTW: it takes a precomputed distance matrix `D` of shape `(B, N, M)` and returns SoftDTW values of shape `(B,)`, with the gradient w.r.t. `D`. It is used by [`module.SoftDTW`](module.md) whenever fused mode is off or not possible (CPU, non-sqeuclidean distance, `fused=False`).

## `forward(ctx, D, gamma, bandwidth)`

1. `check_D(D)` (from [utils/checks](utils/checks.md)) validates the type, 3-D shape and dtype.
2. `gamma <= 0` raises `ValueError`.
3. `bandwidth` of `None` or `<= 0` becomes `-1.0` (the kernels' "disabled" sentinel).
4. Dispatch by device:
   - CUDA: `softdtw_forward_cuda(D, gamma, bw)`;
   - CPU: `softdtw_forward_cpu(D, gamma, bw)`.

   Both return `(out, R)`; see [cuda/launcher](cuda/launcher.md).
5. Saves `D` and `R.detach()`, plus `gamma` and `bandwidth` on `ctx`.

## `backward(ctx, grad_output)`

1. Computes `E = ∂out/∂D`:
   - CUDA: `softdtw_backward_cuda_log` (log-space);
   - CPU: `softdtw_backward_cpu`.
2. Returns `grad_output.view(-1,1,1) * E` as the gradient for `D`, and `None` for `gamma` and `bandwidth`.

Gradients reach `x` and `y` through autograd on `pairwise_distance` (the [distances](distances.md) module), which is ordinary torch code.
