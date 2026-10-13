# `soft_dtw_cuda_ron/cuda/__init__.py`

[← Index](../../../../0_index.md) · [↑ ron backend](../__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

This file marks `cuda/` as a subpackage. Its only content is the comment `# Intentionally minimal; kernels/launcher are imported by autograd.` It exports nothing. Import paths go explicitly through `.cuda.launcher` (used by [`autograd.py`](../autograd.md) and [`autograd_xy.py`](../autograd_xy.md)), and `launcher` imports from [`kernels`](kernels.md).

The subpackage holds both the CUDA kernels **and** the CPU numba reference (in `launcher.py`), despite its name.
