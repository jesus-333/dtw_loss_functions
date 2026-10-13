# `soft_dtw_cuda_ron/cuda/kernels.py`

[← Index](../../../../0_index.md) · [↑ ron backend](../__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

These are the numba `@cuda.jit` kernels for the `ron` backend. They are launched only by [`launcher.py`](launcher.md).

**Conventions:**

- `R` and `logE` are padded to `(B, N+2, M+2)`. Unpadded indices are `(i, j)`; padded ones are `(ip, jp) = (i+1, j+1)`.
- `bandwidth <= 0` (in practice `-1.0`) disables Sakoe-Chiba pruning.
- `softmin_γ(a,b,c) = −γ·(log Σ exp(−·/γ))`, computed with the max-shift trick.

## Two launch styles

1. **Single-block wavefront** (`softdtw_forward_kernel`, `softdtw_backward_log_cuda`, `softdtw_backward_kernel_legacy`):
   - grid `= B` blocks, `max(N,M)` threads per block;
   - loops over `n_passes = 2·max(N,M) − 1` anti-diagonals with `cuda.syncthreads()` between them;
   - the same scheme as `mag`; only valid when `max(N,M) ≤ 1024`.
2. **Per-diagonal tiled** (`*_diag_*` kernels):
   - one kernel launch **per anti-diagonal `p = i + j`**;
   - grid `(ceil(diag_len/256), B)`, so `blockIdx.y` is the batch index, with 256 threads (`TPB_LONG`);
   - each thread handles one cell `i = i_min + t`, `j = p − i`;
   - synchronization comes from kernel boundaries, so any length works.

## Kernels

| Kernel | Direction | Distance | Style | Notes |
|---|---|---|---|---|
| `softdtw_forward_kernel(D, gamma, bandwidth, max_i, max_j, n_passes, R)` | forward | precomputed `D (B,N,M)` | single-block | Same recurrence as `mag` |
| `softdtw_forward_diag_cuda(D, R, gamma, bandwidth, N, M, p)` | forward | precomputed `D` | tiled | Long-sequence path |
| `softdtw_forward_diag_sqeuclid_cuda(X, Y, R, gamma, bandwidth, N, M, D, p)` | forward | **fused**, computes `Σ_k (X[b,i,k]−Y[b,j,k])²` inline | tiled | Here `D` is the feature dimension, not a matrix |
| `softdtw_backward_log_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, logE)` | backward, **log-space** | padded `D (B,N+2,M+2)` | single-block | Uses `_logsumexp3` |
| `softdtw_backward_log_diag_cuda(Dp, R, logE, inv_gamma, bandwidth, N, M, p)` | backward, log-space | padded `D` | tiled | |
| `softdtw_backward_log_diag_sqeuclid_cuda(X, Y, R, logE, inv_gamma, bandwidth, N, M, D, p)` | backward, log-space | **fused** | tiled | Recomputes the costs of cells (i+1,j), (i,j+1), (i+1,j+1); at the border it uses cost 0 (masked by `R = −inf`) |
| `softdtw_backward_kernel_legacy(...)` | backward, linear-space | padded `D` | single-block | **Unused**; marked "numerically unsafe" baseline |
| `_logsumexp3(a,b,c)` | device fn | | | Returns `−inf` if all three inputs are `−inf` |

### Backward recurrence (log space)

```
la = (R[i+1,j] − R[i,j] − D[i+1,j]) / γ     (and similarly lb, lc)
logE[i,j] = logsumexp(logE[i+1,j]+la, logE[i,j+1]+lb, logE[i+1,j+1]+lc)
```

The result is the log of the expected alignment matrix `E = ∂SoftDTW/∂D`. `R[i,j] = +inf` (cells pruned or unreachable) is mapped to `−inf` before use.

## Subtle difference

`softdtw_forward_diag_cuda` prunes with `abs(ip − jp)`, while the other diagonal kernels use `abs(i − j)`. The two are equal, so there is no behavior difference.
