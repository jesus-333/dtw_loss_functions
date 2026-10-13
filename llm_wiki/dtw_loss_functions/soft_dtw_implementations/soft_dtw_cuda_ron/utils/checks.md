# `soft_dtw_cuda_ron/utils/checks.py`

[← Index](../../../../0_index.md) · [↑ ron backend](../__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

`check_D(D)` validates the distance matrix at the start of [`SoftDTWAutograd.forward`](../autograd.md):

- raises `TypeError` if `D` is not a `torch.Tensor`;
- raises `ValueError` if `D.dim() != 3` (expects `(B, N, M)`);
- raises `TypeError` unless the dtype is `float32`, `float64`, `float16` or `bfloat16`;
- does nothing about non-contiguous tensors: the branch is a deliberate `pass`, because the launcher calls `.contiguous()` anyway.

## Note

`float16`/`bfloat16` pass this check, but the numba CPU path and `cuda.as_cuda_array` may not support `bfloat16`, and half-precision SoftDTW is numerically fragile. Prefer `float32`/`float64`. The fused path ([autograd_xy](../autograd_xy.md)) does not call this check.
