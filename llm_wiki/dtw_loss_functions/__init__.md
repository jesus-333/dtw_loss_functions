# `dtw_loss_functions/__init__.py`

[← Index](../0_index.md)

## Purpose

This is the package root. It holds only the package docstring and metadata. It does **not** import any submodule, so users import modules explicitly:

```python
from dtw_loss_functions import soft_dtw, block_dtw, otw, benchmark
```

`from dtw_loss_functions import *` exposes nothing useful, and `dtw_loss_functions.soft_dtw` is not available until it has been imported.

## Content

- The docstring lists the public modules (`block_dtw`, `otw`, `soft_dtw`) and the subpackage `soft_dtw_implementations`. Sphinx renders it as the package landing page. `benchmark` is **not** listed even though it ships in the package; update the docstring if you add or rename modules.
- Metadata:
  - `__author__ = "Alberto (Jesus) Zancanaro"`
  - `__email__ = "alberto.zancanaro@uni.lu"`
  - `__version__ = "1.0.5"`, which **must match** `version` in `pyproject.toml` (bumped by hand; nothing syncs them)
  - `__license__ = "Apache-2.0"`

## Notes for development

- Importing `soft_dtw` (directly or through `block_dtw`) pulls in `pysdtw`, `numba` and the CUDA kernel modules at import time. Numba's `@cuda.jit` is lazy, so import works without a GPU, but `numba` must be installed.
- If you add a new loss, create `dtw_loss_functions/<name>.py`, list it in this docstring, add `docs/dtw_loss_functions.<name>.rst`, reference it in `docs/dtw_loss_functions.rst`, and add a page to this wiki.
