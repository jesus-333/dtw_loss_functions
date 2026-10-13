# Index: `dtw_loss_functions` LLM wiki

Every page in this wiki. Folder `dtw_loss_functions/` mirrors the Python package: one `.md` per `.py`, with the same name and path.

## Top level

- [0_quickstart.md](0_quickstart.md): purpose, history (Block-DTW → hvEEGNet → package), conventions, repo layout, known issues. **Read first.**
- [0_index.md](0_index.md): this file.

## `dtw_loss_functions/` (package)

- [\_\_init\_\_.md](dtw_loss_functions/__init__.md): package root; metadata and version (must match `pyproject.toml`)
- [soft_dtw.md](dtw_loss_functions/soft_dtw.md): `soft_dtw` wrapper class; backend selection (`mag`/`pysdtw`/`ron`) and `sdtw_config` keys
- [block_dtw.md](dtw_loss_functions/block_dtw.md): Block-DTW (`block_dtw`, `block_dtw_naive`, `block_dtw_optimized`), the project's original loss
- [otw.md](dtw_loss_functions/otw.md): Optimal Transport Warping loss
- [benchmark.md](dtw_loss_functions/benchmark.md): timing, saving, loading and plotting utilities

### `dtw_loss_functions/soft_dtw_implementations/`

- [\_\_init\_\_.md](dtw_loss_functions/soft_dtw_implementations/__init__.md): subpackage overview; how to add a backend
- [soft_dtw_cuda_mag.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_mag.md): Maghoumi's CUDA/CPU SoftDTW (backend `mag`, default; vendored)
- [pysdtw_normalize.md](dtw_loss_functions/soft_dtw_implementations/pysdtw_normalize.md): SoftDTW-divergence extension of `pysdtw` (backend `pysdtw`)

#### `.../soft_dtw_cuda_ron/` (backend `ron`; vendored, **auto-synced monthly, do not edit**)

- [\_\_init\_\_.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/__init__.md): public API, sync workflow, architecture diagram
- [module.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/module.md): `SoftDTW` nn.Module; fused/unfused choice and normalization
- [functional.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/functional.md): `softdtw(...)` functional API
- [autograd.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/autograd.md): `SoftDTWAutograd`, unfused (takes a `D` matrix)
- [autograd_xy.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/autograd_xy.md): `SoftDTWXYAutograd`, fused sqeuclidean (no `D` matrix)
- [distances.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/distances.md): `sqeuclidean`, `pairwise_distance`
- [barycenters.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/barycenters.md): SoftDTW barycenter averaging
- [cuda/\_\_init\_\_.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/cuda/__init__.md): empty subpackage marker
- [cuda/kernels.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/cuda/kernels.md): numba CUDA kernels (single-block vs per-diagonal, log-space backward)
- [cuda/launcher.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/cuda/launcher.md): host launch logic (fast vs tiled paths) and the CPU numba reference
- [utils/\_\_init\_\_.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/utils/__init__.md): empty subpackage marker
- [utils/checks.md](dtw_loss_functions/soft_dtw_implementations/soft_dtw_cuda_ron/utils/checks.md): `check_D` input validation

## Maintaining this wiki

- When a `.py` file is added, renamed or removed in `dtw_loss_functions/`, add, rename or remove the matching `.md` and update this index.
- After a `sync-sdtw-cuda-ron` workflow commit, re-check the `soft_dtw_cuda_ron/` pages against the new code.
- Record decisions and newly found pitfalls on the page of the file they concern, and summarize cross-cutting ones in [0_quickstart.md](0_quickstart.md).
