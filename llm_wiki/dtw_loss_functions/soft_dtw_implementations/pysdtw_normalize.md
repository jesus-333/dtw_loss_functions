# `dtw_loss_functions/soft_dtw_implementations/pysdtw_normalize.py`

[← Index](../../0_index.md)

## Purpose

`pysdtw.SoftDTW` (the pip package by Antoine Loriette) has no `normalize` option. This file adds a **SoftDTW-divergence** variant, so that `soft_dtw(implementation='pysdtw', sdtw_config={'normalize': True})` works like the other backends. `pysdtw` itself is not vendored; it is listed in the `pyproject.toml` dependencies.

## API

### `class pysdtw_normalized(pysdtw.SoftDTW)`

- `__init__(use_cuda : bool, gamma=1, bandwidth=None, dist_func=None)` forwards everything to `pysdtw.SoftDTW.__init__`.
- `forward(x, y)` returns `self.sdtw(x, y) - self.sdtw(x, x) - self.sdtw(y, y)`.
  - `self.sdtw` is the attribute created by the parent `pysdtw.SoftDTW` that computes raw SoftDTW (this relies on a `pysdtw` internal attribute name).
  - Inputs `B x T x C`, output shape `B`.

Selected by [`soft_dtw.create_sdtw_function`](../soft_dtw.md) when `implementation == 'pysdtw' and normalize`.

## Gotchas (from reading the code)

1. **The formula differs from the other backends.**
   - The standard SoftDTW divergence (Blondel et al. 2021) is `SDTW(x,y) - ½·(SDTW(x,x) + SDTW(y,y))`. The `mag` and `ron` backends use that formula, and the docstring of `mag` says so explicitly.
   - This class subtracts the self-terms **without the ½**, so its values are shifted by `+½·(xx+yy)` relative to the others. The divergence is no longer zero when `x == y` (it equals `-SDTW(x,x)`).
   - The docstring documents the current formula, so changing it is a behavior change. Decide with the maintainer before "fixing" it.
2. It makes three separate calls instead of the concatenation trick used by `mag` and `ron` (`cat([x,x,y])` vs `cat([y,x,y])`), so it launches three kernels instead of one.
3. It depends on the private-ish attribute `pysdtw.SoftDTW.sdtw`. A `pysdtw` upgrade could break it, and the dependency is not version-pinned.
