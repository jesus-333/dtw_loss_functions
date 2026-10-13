# `dtw_loss_functions/soft_dtw.py`

[← Index](../0_index.md)

## Purpose

This module is the **single entry point for SoftDTW** in the package. Several SoftDTW implementations exist online with different APIs and strengths. The module wraps them in one `torch.nn.Module` (`soft_dtw`), so the backend can be swapped with a string and configured with a dict. [`block_dtw`](block_dtw.md) subclasses this class, so every backend is also available to Block-DTW.

## Backends

| `implementation` | Source | Code location | Devices |
|---|---|---|---|
| `'mag'` (default) | Mehran Maghoumi, *pytorch-softdtw-cuda* | vendored: [`soft_dtw_implementations/soft_dtw_cuda_mag.py`](soft_dtw_implementations/soft_dtw_cuda_mag.md) | CPU (numba) or CUDA (`use_cuda=True`, T ≤ 1024) |
| `'pysdtw'` | Antoine Loriette, *pysdtw* | pip dependency `pysdtw.SoftDTW`; normalized variant in [`pysdtw_normalize.py`](soft_dtw_implementations/pysdtw_normalize.md) | CPU or CUDA (`use_cuda`) |
| `'ron'` | Ron Shapira Weber (BGU-CS-VIL), *sdtw-cuda-torch* | vendored and auto-synced: [`soft_dtw_implementations/soft_dtw_cuda_ron/`](soft_dtw_implementations/soft_dtw_cuda_ron/__init__.md) | Device chosen from the input tensors; supports any length on CUDA (tiled path), fused memory-saving mode |

## `class soft_dtw(torch.nn.Module)`

### Constructor: `soft_dtw(implementation : str = 'mag', sdtw_config : dict = {})`

1. `set_implementation(implementation, reset_sdtw_function=False)` validates the name with `check_implementation` and stores it in `self.implementation`.
2. `set_sdtw_config(sdtw_config, reset_sdtw_function=False)` reads the config keys into attributes.
3. `self.sdtw_function = self.create_sdtw_function(sdtw_config)` builds the backend object.

### `sdtw_config` keys

| Key | Default | Applies to | Meaning |
|---|---|---|---|
| `gamma` | `1` | all | SoftDTW smoothing. Smaller values are closer to hard DTW. |
| `normalize` | `False` | all | Compute the SoftDTW **divergence** instead of the raw value. Different backends use different formulas (see Gotchas). |
| `bandwidth` | `None` | all | Sakoe-Chiba band; `None` means no pruning |
| `use_cuda` | `False` | `mag`, `pysdtw` | Use the CUDA kernel. The input tensors must already be on the GPU. |
| `dist_func` | `None` | `mag`, `pysdtw` | Callable `(x, y) -> D` of shape `B x N x M`. `None` gives squared Euclidean. |
| `dist` | `'sqeuclidean'` | `ron` | String name of the distance; only squared Euclidean is supported |
| `fused` | `None` | `ron` | `None` = auto, `True` = require fused kernel, `False` = never fused |

Keys that do not apply to the selected backend are silently ignored. This lets the same config dict be reused across backends; `test/sdtw_implementations.py` does exactly that.

### `forward(x, y)`

`return self.sdtw_function(x, y)`. Inputs are `B x T x C`; the output has shape `B` (one value per pair). The `ron` backend also accepts unbatched `(T, C)` inputs.

### Other methods

- `check_implementation(implementation)` raises `ValueError` unless the name is in `['mag', 'pysdtw', 'ron']`. To add a backend, update this list **and** `create_sdtw_function` **and** `set_sdtw_config`.
- `create_sdtw_function(sdtw_config)` dispatches on `self.implementation` (and on `self.normalize` for `pysdtw`, choosing `pysdtw_normalized` or `pysdtw.SoftDTW`). The `sdtw_config` argument is unused; the method reads the attributes set earlier.
- `set_sdtw_config(sdtw_config, reset_sdtw_function=True)` re-reads the config. The backend-specific attributes are set only for the current implementation.
- `set_implementation(implementation, reset_sdtw_function=True)` validates and stores the implementation.

## Gotchas and known issues (from reading the code)

1. **Broken reset path.** `set_sdtw_config(..., reset_sdtw_function=True)` calls `self.get_sdtw_function(sdtw_config)`, which does not exist; it should be `create_sdtw_function`. Calling `set_sdtw_config` after construction with the default argument should raise `AttributeError`. The constructor passes `False`, so it is not affected.
2. **`set_implementation` ignores `reset_sdtw_function`.** Changing the implementation after construction does not rebuild `self.sdtw_function`. Switching from `mag` to `ron` also leaves `self.dist` and `self.fused` unset until `set_sdtw_config` runs again.
3. **The divergence formula depends on the backend.** `mag` and `ron` return `xy - 0.5*(xx + yy)`. `pysdtw` with `normalize=True` returns `xy - xx - yy`. Values are not comparable across backends when `normalize=True`.
4. **Mutable default** `sdtw_config : dict = {}`. It is not mutated here, so this is harmless, but keep it that way.
5. **`mag` with `use_cuda=True` and T > 1024** prints a message and silently falls back to the CPU numba path (see the `mag` page). This interacts with Block-DTW: block length is what matters, not the full signal length.
6. `use_cuda` does not move tensors. The caller is responsible for the device.
