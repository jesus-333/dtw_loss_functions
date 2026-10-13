# `dtw_loss_functions/soft_dtw_implementations/__init__.py`

[← Index](../../0_index.md)

## Purpose

This is the subpackage that **collects third-party SoftDTW implementations**, so that [`soft_dtw.py`](../soft_dtw.md) can wrap them behind one API. The file contains only a docstring and metadata; it imports nothing. `soft_dtw.py` imports the members explicitly:

```python
from .soft_dtw_implementations import soft_dtw_cuda_mag, pysdtw_normalize, soft_dtw_cuda_ron
```

## Members

| Member | Kind | Origin | Editable? |
|---|---|---|---|
| [`soft_dtw_cuda_mag.py`](soft_dtw_cuda_mag.md) | vendored module | Maghoumi, *pytorch-softdtw-cuda* (MIT) | Only the docstrings were changed; keep the code identical to upstream |
| [`pysdtw_normalize.py`](pysdtw_normalize.md) | local extension | wraps the pip package `pysdtw` (Loriette) | Yes, project code |
| [`soft_dtw_cuda_ron/`](soft_dtw_cuda_ron/__init__.md) | vendored subpackage | BGU-CS-VIL *sdtw-cuda-torch* (Ron Shapira Weber) | **No.** Overwritten by the `sync-sdtw-cuda-ron-*` GitHub workflows |

## Metadata

`__version__ = "1.0.3"` is independent from (and older than) the package version `1.0.5`. The docstring asks users to cite the original paper of whichever implementation they use; `:cite:` keys are `soft_dtw_mag_1/2` and `soft_dtw_ron_1/2` in `docs/biblio.bib`, and the full BibTeX is in `citations.md`.

## Adding a new backend

1. Vendor the code in a new module or subpackage here (or add a pip dependency in `pyproject.toml`).
2. Extend `soft_dtw.check_implementation`, `set_sdtw_config` (backend-specific keys) and `create_sdtw_function`.
3. Add it to this docstring, the `soft_dtw` docstring, `README.md`, `citations.md`, `docs/biblio.bib`, the docs `.rst` files, and `test/sdtw_implementations.py` (`implementations_list`).
4. Add a wiki page for it.
