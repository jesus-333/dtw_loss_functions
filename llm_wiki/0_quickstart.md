# Quickstart: `dtw_loss_functions`

> LLM-facing wiki. It supports agents (and humans) doing future development on this repository.
> Start here, then use [0_index.md](0_index.md) to jump to the page for a specific Python file.
> The pages describe the code as it was when the wiki was written (2026-10-13, package version `1.0.5`). If a page and the code disagree, trust the code and update the page.

## What this project is

`dtw_loss_functions` (PyPI name in `pyproject.toml`: `dtw_loss_functions`; the README's install command says `dtw-loss-function`) is a PyTorch package that collects **differentiable DTW-inspired loss functions** for training neural networks on time series.

### Origin and history

- The project started as a **standalone release of Block-DTW**. Block-DTW was first developed while **training hvEEGNet** (a hierarchical VAE for EEG reconstruction), where full SoftDTW over long EEG signals was too slow. The idea was to publish only the block-wise loss.
- The repository was originally called `block-sdtw`. Its early code was research scripts: a CUDA SoftDTW, a simple MLP, and training/timing scripts (these still live under `scripts/`).
- Block-DTW is built on top of Mehran Maghoumi's CUDA SoftDTW, so that implementation was vendored too. Later the OTW loss was added. With three losses in the code base, the project was renamed and refactored into a proper Python package (`dtw_loss_functions`).
- Later work (around March 2026):
  - Several SoftDTW backends (`mag`, `pysdtw`, `ron`) were put behind a single wrapper configured by a dict.
  - `block_dtw` and `otw` were turned into `torch.nn.Module` subclasses.
  - A naive-vs-optimized Block-DTW comparison and a `benchmark` module were added.
- Reference paper for Block-DTW: Zancanaro, Badia, Cisotto, *"BlockDTW: Efficient and Scalable Similarity Search Algorithm for Healthcare-Focused Time-Series"*, IEEE CSCN 2025, doi `10.1109/CSCN67557.2025.11230700`.

## Loss functions provided

| Loss | Module | Idea |
|---|---|---|
| SoftDTW | [`soft_dtw`](dtw_loss_functions/soft_dtw.md) | Wrapper class `soft_dtw` that selects one of 3 backends: `mag`, `pysdtw`, `ron` |
| Block-DTW | [`block_dtw`](dtw_loss_functions/block_dtw.md) | Split the signal into non-overlapping time blocks of `block_size`, run SoftDTW on each block, sum the per-block values. Cost drops from O(T²) to O(T·block_size). |
| OTW | [`otw`](dtw_loss_functions/otw.md) | Optimal Transport Warping (Latorre et al., ICASSP 2023), built from windowed cumulative sums and a smooth-L1 term |

## Conventions every agent should know

- **Tensor layout is `B x T x C`** (batch, time, channels) for SoftDTW and Block-DTW. OTW is different: it expects `B x L` (or `L`).
- **SoftDTW and Block-DTW return one value per sample (shape `B`)**, not a scalar. The caller reduces it (e.g. `.mean()`).
- **Configuration uses a dict.** Every SoftDTW-based class takes `implementation: str` and `sdtw_config: dict`. Missing keys fall back to defaults (`gamma=1`, `normalize=False`, `bandwidth=None`, `use_cuda=False`, `dist='sqeuclidean'`, `fused=None`).
- **Class names are lowercase snake_case** (`soft_dtw`, `block_dtw`, `block_dtw_naive`, `block_dtw_optimized`, `otw`). The vendored third-party code keeps its own names (`SoftDTW`).
- **Code style:**
  - A space before the colon in signatures and control flow (`def f(x : int) :`, `if cond :`).
  - Keyword arguments written as `name = value`.
  - Sections separated by `# - - - - ...` banner comments.
  - numpydoc docstrings (Sphinx autodoc + `sphinxcontrib-bibtex` `:cite:` keys defined in `docs/biblio.bib`).
- **Vendored code:**
  - `soft_dtw_implementations/soft_dtw_cuda_ron/` is **overwritten automatically** by `.github/workflows/sync-sdtw-cuda-ron-*.yml` (monthly cron plus manual trigger), which copy `softdtw_cuda/` from `BGU-CS-VIL/sdtw-cuda-torch`. **Do not hand-edit those files**: local changes will be lost on the next sync. Wrap or extend the code from outside instead.
  - `soft_dtw_cuda_mag.py` is a verbatim copy of Maghoumi's code; only the docstrings were changed to numpydoc. Keep it that way.
  - `pysdtw` is **not** vendored. It is a pip dependency, extended by `pysdtw_normalize.py`.
- **Versions:** `pyproject.toml` and `dtw_loss_functions/__init__.py` both hold `1.0.5` and must be bumped together. `soft_dtw_implementations/__init__.py` has its own `__version__ = "1.0.3"`.

## Repository layout (outside the package)

| Path | Content |
|---|---|
| `dtw_loss_functions/` | The package (documented file by file in this wiki) |
| `test/` | Manual test scripts, not pytest. `block_dtw_implementation.py` checks naive vs optimized Block-DTW (shape, values, timing, CLI args). `sdtw_implementations.py` times the 3 SoftDTW backends. |
| `benchmark/` | `compute_*.py` / `plot_*.py` pairs that use the `benchmark` module (block naive vs optimized, SDTW vs block, SDTW backends), with plots from a Mac M4 Pro |
| `scripts/analysis_computation/`, `scripts/training/`, `scripts/other/` | Legacy research scripts from the `block-sdtw` era. Their READMEs say "to be refactored using the new package implementation". They are the source of the paper figures and the hvEEGNet/UCR experiments. |
| `Results/` | Saved `.npy` timing results (e.g. a Raspberry Pi comparison of standard SDTW, block SDTW and a Rust SDTW) |
| `docs/` | Sphinx sources (`*.rst`, `biblio.bib`). Deployed to GitHub Pages by `deploy-docs.yml` when `docs/**` changes on `main`. |
| `.github/workflows/python-publish.yml` | Builds with `hatchling` and publishes to PyPI on a GitHub release |
| `citations.md` | BibTeX for the package and for each SoftDTW backend |

## Minimal usage

```python
import torch
from dtw_loss_functions import soft_dtw, block_dtw, otw

x, y = torch.randn(5, 300, 1), torch.randn(5, 300, 1)

sdtw  = soft_dtw.soft_dtw(implementation = 'mag', sdtw_config = {'gamma' : 0.1})
bdtw  = block_dtw.block_dtw(block_size = 25, implementation = 'mag', sdtw_config = {'gamma' : 0.1})

loss_sdtw  = sdtw(x, y).mean()   # shape B -> scalar
loss_block = bdtw(x, y).mean()
```

## Dependencies

`torch`, `numba` (CPU JIT and CUDA kernels), `pysdtw`, `matplotlib` (benchmark plots), `toml`. Build backend: `hatchling`.

## Known issues found while writing this wiki

These come from reading the code. They have **not** been run (torch was not installed in the environment where the wiki was written). Details are on each file's page.

1. `otw.otw.__init__` never calls `super().__init__()`, so calling an `otw` instance should raise `AttributeError` ([otw.md](dtw_loss_functions/otw.md)).
2. `soft_dtw.set_sdtw_config(..., reset_sdtw_function=True)` calls `self.get_sdtw_function`, which does not exist; the method is `create_sdtw_function` ([soft_dtw.md](dtw_loss_functions/soft_dtw.md)).
3. The `normalize=True` divergence is inconsistent across backends:
   - `mag` and `ron` compute `xy - 0.5·(xx+yy)`.
   - `pysdtw_normalized` computes `xy - xx - yy` ([pysdtw_normalize.md](dtw_loss_functions/soft_dtw_implementations/pysdtw_normalize.md)).
4. `benchmark.repeat_inference` does not call `torch.cuda.synchronize()`, so CUDA timings may only measure kernel launch time ([benchmark.md](dtw_loss_functions/benchmark.md)).
