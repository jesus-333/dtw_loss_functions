# `soft_dtw_implementations/soft_dtw_cuda_ron/__init__.py`

[← Index](../../../0_index.md)

## Purpose

This is the public surface of the vendored **`sdtw-cuda-torch`** by BGU-CS-VIL (Ron Shapira Weber & Oren Freifeld, *"SoftDTW-CUDA-Torch: Memory-Efficient GPU-Accelerated Soft Dynamic Time Warping for PyTorch"*, arXiv 2602.17206). In this package it is backend `'ron'`. Upstream the package is called `softdtw_cuda`, which is why `BARYCENTERS.md` shows `from softdtw_cuda import ...`; here it is `dtw_loss_functions.soft_dtw_implementations.soft_dtw_cuda_ron`.

```python
from .functional import softdtw
from .module import SoftDTW
from .barycenters import softdtw_barycenter, softdtw_barycenter_cpu
__all__ = ["softdtw", "SoftDTW", "softdtw_barycenter", "softdtw_barycenter_cpu"]
```

[`soft_dtw.py`](../../soft_dtw.md) uses `soft_dtw_cuda_ron.SoftDTW(gamma=..., normalize=..., bandwidth=..., dist=..., fused=...)`.

## ⚠️ Auto-synced: do not edit

`.github/workflows/sync-sdtw-cuda-ron-automatic.yml` (cron `0 9 1 * *`, 09:00 on the 1st of every month) and `sync-sdtw-cuda-ron-manual.yml`:

1. clone `https://github.com/BGU-CS-VIL/sdtw-cuda-torch`;
2. `cp -r softdtw_cuda/. dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/`;
3. commit as `github-actions[bot]` with the message `chore: sync softdtw_cuda from sdtw-cuda-torch (<date>)` and push to `main`.

Any local edit to this folder will be overwritten. The copy also brings over upstream extras (`BARYCENTERS.md`, `py.typed`). Since the copy only adds or overwrites, files deleted upstream would linger here. After a sync, re-check that `module.SoftDTW`'s constructor signature still matches what `soft_dtw.create_sdtw_function` passes.

## Architecture

```
module.SoftDTW (nn.Module)          functional.softdtw (one-shot wrapper)     barycenters.softdtw_barycenter
        │ validates shapes, picks fused vs unfused, handles normalize
        ├── fused (CUDA + sqeuclidean) ──► autograd_xy.SoftDTWXYAutograd ──► cuda/launcher fused_* ──► cuda/kernels *_sqeuclid_cuda
        └── unfused ──► distances.pairwise_distance (D: B×N×M)
                        └► autograd.SoftDTWAutograd ──► cuda/launcher (CUDA fast/tiled or CPU numba) ──► cuda/kernels
                                     └ utils/checks.check_D
```

Compared with `mag`:

- **log-space backward**, which is numerically stable for small `gamma`;
- **no 1024-length limit**: a tiled per-anti-diagonal launch is used when `max(N,M) > 1024`;
- **fused mode**: never materializes `D`, which saves memory at some speed cost;
- input validation;
- only squared Euclidean distance (no custom `dist_func`).

## Pages

[module](module.md) · [functional](functional.md) · [autograd](autograd.md) · [autograd_xy](autograd_xy.md) · [distances](distances.md) · [barycenters](barycenters.md) · [cuda/\_\_init\_\_](cuda/__init__.md) · [cuda/kernels](cuda/kernels.md) · [cuda/launcher](cuda/launcher.md) · [utils/\_\_init\_\_](utils/__init__.md) · [utils/checks](utils/checks.md)
