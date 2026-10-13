# `soft_dtw_cuda_ron/functional.py`

[← Index](../../../0_index.md) · [↑ ron backend](__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

This is a convenience functional API for the `ron` backend:

```python
softdtw(x, y, *, gamma=1.0, bandwidth=None, normalize=False, dist="sqeuclidean", fused=None) -> Tensor  # (B,)
```

It just builds `SoftDTW(gamma=..., bandwidth=..., normalize=..., dist=..., fused=...)` from [`module.py`](module.md) and calls it on `(x, y)`. It accepts `(B,N,D)`/`(B,M,D)` or unbatched `(N,D)`/`(M,D)` inputs.

It is exported as `soft_dtw_cuda_ron.softdtw`. It is **not used** by the `dtw_loss_functions` wrapper, which uses the module class instead. Building a new module per call is cheap (no parameters), so it is fine for one-off use.
