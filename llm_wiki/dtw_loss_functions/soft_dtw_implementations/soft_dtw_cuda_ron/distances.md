# `soft_dtw_cuda_ron/distances.py`

[← Index](../../../0_index.md) · [↑ ron backend](__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

This module provides the pairwise distance matrices for the **unfused** path of the `ron` backend.

## Functions

- `sqeuclidean(x, y)`:
  - takes `x (B,N,D)` and `y (B,M,D)` and returns `D (B,N,M)` with `D[b,i,j] = ||x[b,i] − y[b,j]||²`;
  - uses the expansion `‖x‖² + ‖y‖² − 2·x·yᵀ` with `torch.bmm`, so memory is O(B·N·M) rather than the O(B·N·M·D) of `mag`'s broadcasting;
  - applies `clamp_min(0)` to remove tiny negative values caused by rounding;
  - raises `ValueError` on wrong rank or on batch/feature mismatch.
- `pairwise_distance(x, y, *, dist)` dispatches on `dist.lower()`. The only names accepted are `"sqeuclidean"`, `"sq_euclidean"` and `"squared_euclidean"`; anything else raises `ValueError`. This is the `dist` key of `sdtw_config` in [`soft_dtw.py`](../../soft_dtw.md).

## Note

The expansion trick can lose precision when `x ≈ y` and the values have large magnitude (catastrophic cancellation). The fused kernels compute `(x−y)²` directly, so the fused and unfused paths can differ by a small amount numerically.
