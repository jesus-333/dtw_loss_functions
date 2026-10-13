# `soft_dtw_cuda_ron/module.py`

[← Index](../../../0_index.md) · [↑ ron backend](__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

`SoftDTW(nn.Module)` is the user-facing class of the `ron` backend and the object that [`soft_dtw.py`](../../soft_dtw.md) instantiates.

## `SoftDTW.__init__(*, gamma=1.0, bandwidth=None, normalize=False, dist="sqeuclidean", fused=None)`

All arguments are **keyword-only**.

- `gamma` is cast to float; it raises `ValueError` if `<= 0` or not finite.
- `bandwidth`: `None` or `<= 0` is stored as `None` (disabled).
- `fused`:
  - `None` = auto;
  - `True` = require the fused kernel (raises if impossible);
  - `False` = always materialize `D`.

## `_use_fused(x, y)`

Fused mode is possible only when `dist` is one of `sqeuclidean`/`sq_euclidean`/`squared_euclidean`, **both tensors are CUDA**, and both are on the same device.

- `fused=True` when not possible: raises `ValueError`.
- `fused=False`: returns `False`.
- Otherwise: returns whether fused mode is possible. On CUDA with the default settings, **fused mode is used automatically**.

## `forward(x, y)`

1. Unbatched `(N, D)` inputs are unsqueezed to `(1, N, D)`; anything other than 3-D after that raises `ValueError`.
2. With `normalize=True`, `N == M` is required.
3. It checks matching feature dims and batch sizes, and non-empty sequences.
4. **Normalize:** `x_cat = cat[x, x, y]`, `y_cat = cat[y, x, y]`, one batched call, `split(bx)`, then `xy - 0.5*(xx + yy)`.
5. **Otherwise:**
   - fused: `SoftDTWXYAutograd.apply(x, y, gamma, bandwidth)`;
   - unfused: `D = pairwise_distance(x, y, dist)` followed by `SoftDTWAutograd.apply(D, gamma, bandwidth)`.
6. Output shape `(B,)`.

## Notes

- Device placement follows the inputs. There is no `use_cuda` flag, which is why `soft_dtw.py` does not pass one for `ron`.
- On CPU, the unfused numba path from `cuda/launcher.py` is used.
