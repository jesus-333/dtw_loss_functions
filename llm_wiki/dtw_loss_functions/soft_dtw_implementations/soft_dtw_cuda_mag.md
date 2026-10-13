# `dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_mag.py`

[← Index](../../0_index.md)

## Purpose

This is a vendored copy of **Mehran Maghoumi's `pytorch-softdtw-cuda`** (MIT license, header kept in the file), one of the most widely used PyTorch SoftDTW implementations. It is backend `'mag'` and the **default** for [`soft_dtw`](../soft_dtw.md) and [`block_dtw`](../block_dtw.md). Block-DTW was originally built on this implementation, which is why it was the first one included in the package.

The code is **identical to upstream**; only the docstrings were converted to numpydoc. Do not refactor it. If a change is needed, wrap or extend it from outside.

## Structure

### CUDA path (numba `@cuda.jit`)

- `compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R)` is the forward kernel.
  - **One CUDA block per batch element**, with `max(N, M)` threads.
  - It loops over the `2·max(N,M)-1` anti-diagonals and calls `cuda.syncthreads()` between passes.
  - Thread `tid` owns row `I = tid` and computes `R[i,j] = D[i-1,j-1] + softmin_γ(R[i-1,j-1], R[i-1,j], R[i,j-1])` with the max-shift log-sum-exp trick.
  - The diagonal-wavefront idea comes from "Developing a pattern discovery method in time series data and its GPU acceleration" (IEEE 8400444).
- `compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E)` is the reverse wavefront computing `E = ∂SoftDTW/∂D`. It works in **linear space** with `exp(...)` (no log-space stabilization, unlike `ron`).
- `_SoftDTWCUDA(torch.autograd.Function)`:
  - `forward` allocates `R` of shape `(B, N+2, M+2)` filled with `inf`, sets `R[:,0,0]=0`, launches the kernel and returns `R[:, -2, -2]`. It uses the legacy `torch.cuda.FloatTensor`.
  - `backward` pads `D`, sets the boundary conditions on `R`, runs the backward kernel and returns `grad_output * E` as the gradient w.r.t. `D`.

### CPU path (numba `@jit(nopython=True, parallel=True)`)

The CPU code is based on Kanru Hua's *pytorch-softdtw*, with batching and pruning added.

- `compute_softdtw(D, gamma, bandwidth)` and `compute_softdtw_backward(D_, R, gamma, bandwidth)` are the same recurrences with `prange` over the batch.
- `_SoftDTW(Function)` moves `D` to numpy, computes, and moves the result back to the original device.

### `class SoftDTW(torch.nn.Module)`: user-facing

- `__init__(use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None)`. `bandwidth=None` is stored as `0`, which means disabled.
- `_get_func_dtw(x, y)`:
  - asserts equal batch size and feature dimension;
  - **if `use_cuda` and either length > 1024, it prints a message and falls back to CPU** (CUDA block-size limit);
  - returns `_SoftDTWCUDA.apply` or `_SoftDTW.apply`.
- `_euclidean_dist_func(x, y)` gives the squared Euclidean pairwise distance `B x N x M`, computed by broadcasting (memory O(B·N·M·D)).
- `forward(X, Y)`:
  - With `normalize`, it uses the concatenation trick (`x=cat[X,X,Y]`, `y=cat[Y,X,Y]`), runs one batched call, and returns `xy - ½(xx+yy)`.
  - Otherwise it returns `func(dist_func(X,Y), gamma, bandwidth)`.
  - Output shape `B`.
- `timed_run` and `profile`, plus the `__main__` block, are upstream's CPU-vs-GPU self-test. They need CUDA.

## Gotchas

- `use_cuda=True` requires the inputs to already be on the GPU (`cuda.as_cuda_array` on a CPU tensor fails).
- The 1024 limit applies to sequence length. Block-DTW avoids it, since only `block_size` matters.
- `normalize=True` requires `X` and `Y` to have the same length (the concatenation needs it).
- The CPU path round-trips through numpy, so it is slow for big batches, and the first call pays numba JIT compilation.
- The linear-space backward can underflow or overflow for small `gamma`; the `ron` backend uses log-space instead.
- The module docstring example has a typo: `sdtw_config = {use_cuda : use_cuda}` (missing quotes around the key).
