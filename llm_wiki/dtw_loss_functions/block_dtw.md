# `dtw_loss_functions/block_dtw.py`

[← Index](../0_index.md)

## Purpose

This module implements **Block-DTW**, the loss this project was originally created for. It was developed for hvEEGNet training on long EEG signals.

Block-DTW splits both signals along time into non-overlapping blocks of `block_size` samples. It computes SoftDTW only between **corresponding** blocks (block *k* of `x` against block *k* of `x_r`) and sums the results:

```
BlockDTW(x, x_r) = Σ_k SoftDTW(x[:, kL:(k+1)L, :], x_r[:, kL:(k+1)L, :])     (L = block_size)
```

- **Cost:** O(T·L) instead of O(T²) for full SoftDTW.
- **Trade-off:** warping is restricted to within a block, so no alignment path crosses block boundaries. A small `block_size` approaches a point-wise loss; `block_size = T` gives plain SoftDTW.
- **Paper:** *BlockDTW: Efficient and Scalable Similarity Search Algorithm for Healthcare-Focused Time-Series*, IEEE CSCN 2025.

All classes take `(block_size, implementation='mag', sdtw_config={})`. `implementation` and `sdtw_config` are passed unchanged to [`soft_dtw`](soft_dtw.md), so every SoftDTW backend and option (gamma, normalize, bandwidth, …) works per block.

## Classes

### `block_dtw(torch.nn.Module)`: the recommended entry point

- Holds **two** sub-modules, `block_dtw_naive` and `block_dtw_optimized` (so two backend instances are built).
- `forward(x, x_r)`:
  - if `x.shape[1] % block_size == 0`, it uses the optimized path;
  - otherwise it uses the naive path.
- Input `B x T x C`, output shape `B`.
- If you know `T` is always a multiple of `block_size`, use `block_dtw_optimized` directly to avoid the check (as the docstring recommends).

### `block_dtw_naive(soft_dtw)`: sequential version

- Loops over blocks with indices `idx_1 = i*L` and `idx_2 = (i+1)*L`. When `idx_2 >= T`, the last slice becomes `x[:, idx_1:, :]`, so a **shorter final block** absorbs any remainder.
- Calls `super().forward` (SoftDTW) per block and accumulates the `B`-shaped results.
- Works for any `T`. It costs one kernel launch per block, which is slow for many blocks.

### `block_dtw_optimized(soft_dtw)`: parallel version

- Folds the blocks into the batch dimension:
  - `virtual_batch_size = B * T / L`
  - `x.view(virtual_batch_size, L, -1)`, so one SoftDTW call processes all blocks
  - `.view(B, -1).sum(dim=1)` folds back to shape `B`
- This is correct for `B x T x C` row-major tensors, because consecutive `L` time steps of one sample are contiguous in memory.
- **Requirement, not checked:** `T % L == 0`. Otherwise `int()` truncation gives a wrong `virtual_batch_size` and `view` errors or misbehaves. The `block_dtw` wrapper guards against this.
- Uses `.view`, so **non-contiguous inputs** (e.g. after a `transpose`) raise an error. Call `.contiguous()` first.

## Equivalence and testing

- The naive and optimized versions should give the same values when `T % L == 0`. `test/block_dtw_implementation.py` checks shapes and values and also times both against full SoftDTW (CLI arguments: `--batch_size --time_samples --channels --block_size --n_repetitions --implementation --use_cuda --time_in_seconds`).
- `benchmark/compute_block_optimized_vs_naive.py` and `compute_sdtw_vs_block.py` produce the timing plots in `benchmark/MAC_M4_PRO/`.

## Gotchas

- With `implementation='mag'` and `use_cuda=True`, the 1024-sample CUDA limit applies to **`block_size`**, not to `T`. Block-DTW therefore runs on long signals on GPU where `mag` SoftDTW would fall back to CPU.
- With `normalize=True`, the result is a sum of per-block divergences, not the divergence of the full signal.
- The loss scales with the number of blocks (it is a sum, not a mean). If you compare runs with different `block_size`, keep this in mind (see `scripts/other/loss_vs_n_of_block.py`).
