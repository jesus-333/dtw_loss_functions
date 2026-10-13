# `soft_dtw_cuda_ron/barycenters.py`

[← Index](../../../0_index.md) · [↑ ron backend](__init__.md) · ⚠️ vendored, auto-synced; do not edit

## Purpose

This module computes **SoftDTW barycenters** (time-series averages in DTW geometry; Cuturi & Blondel, ICML 2017, following the tslearn approach). It finds the series `z` that minimizes `Σ_i w_i · SoftDTW(z, X_i)` by gradient descent. Upstream documentation is in the sibling file `BARYCENTERS.md`, which is also vendored. This module is **not used** by the `dtw_loss_functions` wrapper; it is exported as an extra capability of the `ron` backend.

## `softdtw_barycenter(X, *, gamma=1.0, weights=None, max_iter=100, lr=0.1, init=None, device=None, verbose=False, fused=None, early_stopping=True, patience=10, tol=1e-5) -> (N, D)`

1. `X (B,N,D)` is moved to `device` (default: `X.device`).
2. Weights default to uniform and are normalized to sum to 1.
3. The initial barycenter is the weighted mean over the batch, unless `init (N,D)` is given.
4. `loss_fn = SoftDTW(gamma, normalize=False, fused=fused)` (see [module](module.md)).
5. Optimization:
   - Adam on the barycenter;
   - `CosineAnnealingLR(T_max=max_iter, eta_min=lr*0.1)`;
   - gradient clipping with `max_norm=1.0`.
6. Each iteration:
   - expands the barycenter to `(B,N,D)`;
   - computes `loss = Σ w·SoftDTW(bary, X)` (can be negative);
   - steps the optimizer.
7. Early stopping uses the absolute improvement `best − loss > tol`. It stops after `patience` iterations without improvement, **but only after `max_iter // 2` iterations**.
8. With `verbose`, it prints every 20 iterations; it calls `torch.cuda.synchronize()` whenever CUDA is available.
9. Returns the detached barycenter.

## `softdtw_barycenter_cpu(...)`

The same function with `device="cpu"`.

## Discrepancies with the docs (upstream)

- `BARYCENTERS.md` says the LR decays to `lr * 0.01`, but the code uses `eta_min = lr * 0.1`.
- The `BARYCENTERS.md` parameter table says `lr` defaults to 0.1, while its example comments show 0.01. The `softdtw_barycenter_cpu` docstring also says "Default: 0.01", but the actual default is `0.1`.
- `BARYCENTERS.md` links to `../examples/barycenter_example.py`, which exists only in the upstream repo, not in this package.
