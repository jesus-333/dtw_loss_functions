# `soft_dtw_cuda_ron/cuda/launcher.py`

[← Index](../../../../0_index.md) · [↑ ron backend](../__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

This is the host-side glue between the autograd `Function`s ([autograd](../autograd.md), [autograd_xy](../autograd_xy.md)) and the numba kernels ([kernels](kernels.md)). It:

- allocates the DP tables;
- sets the boundary conditions;
- chooses between the fast and tiled launch strategies;
- converts torch tensors with `cuda.as_cuda_array` (zero-copy).

It also contains the **CPU numba reference** implementation.

## Globals and helpers

- `TPB_LONG = 256`: threads per block for the tiled kernels.
- `_diag_bounds(p, N, M)` gives `(i_min, i_max)` of anti-diagonal `p`.
- `_threads_and_passes(N, M)` gives `(max(N,M), 2·max(N,M)−1)`.

## Fused (no `D` matrix), CUDA only

- `softdtw_forward_cuda_fused_sqeuclid(X, Y, gamma, bandwidth)`:
  - validates that the inputs are CUDA tensors of rank 3 with matching B and D;
  - `R = full(inf)` with `R[:,0,0]=0`;
  - launches `softdtw_forward_diag_sqeuclid_cuda` for `p in range(N+M−1)`;
  - returns `(R[:,-2,-2], R)`.
  - It **always uses the tiled path** (one launch per diagonal, no single-block fast path), which is partly why fused mode is slower.
- `softdtw_backward_cuda_fused_sqeuclid(X, Y, R, gamma, bandwidth)`:
  - checks the shape of `R`;
  - clones `R` into `R_work` with the boundary rows/cols set to `−inf` and `R_work[-1,-1] = R[-2,-2]`;
  - `logE = full(−inf)` with `logE[-1,-1] = 0`;
  - reverse diagonal loop;
  - returns `exp(logE[1:N+1, 1:M+1])`.

## Unfused (precomputed `D`)

- `softdtw_forward_cuda(D, gamma, bandwidth)`:
  - `max(N,M) ≤ 1024`: **fast path**, a single launch of `softdtw_forward_kernel[B, tpb]`;
  - otherwise: **tiled path**, one `softdtw_forward_diag_cuda` launch per diagonal. This removes `mag`'s 1024 limit.
- `softdtw_backward_cuda_log(D, R, gamma, bandwidth)`:
  - pads `D` to `D_pad`, sets the same boundary conditions on `R_work` and `logE`;
  - fast path `softdtw_backward_log_cuda`, or tiled path `softdtw_backward_log_diag_cuda` (where `bw <= 0` is coerced to `−1`);
  - returns `exp(logE)` cropped.

## CPU reference (numba `@jit(nopython=True, parallel=True)`)

- `_softdtw_forward_cpu_np` and `_softdtw_backward_cpu_np` use the same recurrences as `mag`'s CPU code (linear-space backward), with `prange` over the batch.
- `softdtw_forward_cpu(D, gamma, bw)` and `softdtw_backward_cpu(D, R, gamma, bw)` are the torch ↔ numpy wrappers. The backward wrapper calls `.copy()` on `R` so the in-place boundary edits do not corrupt the tensor saved by autograd.

## Performance notes

- The tiled paths do a Python loop over `N+M−1` diagonals, i.e. that many kernel launches per forward or backward. Launch overhead dominates for moderate lengths, which is why the fast path exists.
- Every call allocates fresh `R`/`logE` tensors of size `B·(N+2)·(M+2)`.
- The CPU path is linear-space, so it is less numerically robust than the CUDA log-space backward for small `gamma`.
