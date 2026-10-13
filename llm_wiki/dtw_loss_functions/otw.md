# `dtw_loss_functions/otw.py`

[← Index](../0_index.md)

## Purpose

This module implements **OTW, Optimal Transport Warping** (Latorre, Liu, Sahoo, Hoi, *"OTW: Optimal Transport Warping for Time Series"*, ICASSP 2023; bib key `otw_paper` in `docs/biblio.bib`). OTW is a DTW alternative whose cost is linear in sequence length and parallelizable. It is built from cumulative sums of the difference signal, regularized with a smooth-L1 penalty. It was added to the package as a side result of testing; it does **not** depend on SoftDTW.

## API

### `class otw(torch.nn.Module)`

- `__init__(m=1, s=0.5, beta=1, reduction='mean')` stores the hyperparameters.
- `forward(x, y)` returns `otw_distance(x, y, self.m, self.s, self.beta, self.reduction)`.

### `otw_distance(x, y, m=1, s=0.5, beta=1, reduction='mean')`

It implements eq. (9) of the paper.

- **Input shape:** `B x L` (1-D inputs are unsqueezed to `1 x L`). **3-D `B x T x C` inputs raise `ValueError`**, unlike the SoftDTW/Block-DTW modules. For multichannel data, reshape or loop over channels.
- **`s` (window):**
  - must be > 0;
  - if `0 < s < 1`, it is converted to `int(s * L)`;
  - otherwise it is used as a number of time steps.
- **Computation:**
  - `term_1 = m * smooth_l1_loss(window_cumsum(x - y, s), beta, reduction)` is the "waste" (mass-mismatch) term weighted by `m`;
  - `term_2 = Σ_{i=0}^{L-2} smooth_l1_loss(window_cumsum(x[:, :i+1] - y[:, :i+1], s), beta, reduction)` is the transport term over all prefixes;
  - the function returns `term_1 + term_2`.
- **`reduction`** (`'mean'` / `'sum'` / `'none'`) is applied inside **each** term, so with `'mean'` the output is a scalar and with `'none'` it has shape `B`.

### `smooth_l1_loss(x, beta, reduction='mean')`

Huber-like penalty: `0.5*x²/beta` if `|x| < beta`, else `|x| - 0.5*beta`. Then the reduction is applied.

### `window_cumsum(x, s)`

Implements eq. (7). It returns `cumsum(x)[:, -1] - cumsum(x)[:, -(s+1)]`, i.e. the **sum of the last `s` elements** of each row (or the total sum if `s >= L`). Output shape `B`.

## Gotchas and known issues (from reading the code)

1. **`otw.__init__` never calls `super().__init__()`.** Setting plain float attributes works, but `nn.Module.__call__` needs the internal hook dicts, so calling `otw()(x, y)` should raise `AttributeError`. `.to()` and `.parameters()` are broken too. Use `otw_distance` directly until this is fixed (add `super().__init__()` as the first line).
2. **The loop in `term_2` is a Python loop over `L-1` prefixes**, each doing a full `cumsum`, so the cost is O(L²) in practice. This contradicts the paper's linear-time claim. It could be vectorized with a single cumsum, since prefix `i`'s windowed sum is a difference of two entries of the full cumsum.
3. A small fractional `s` (e.g. `s=0.001` with `L=100`) becomes `int(...) = 0`, which makes every `window_cumsum` zero. There is no check after the conversion.
4. A commented-out line `# beta = 100` in `otw_distance` hints that training was unstable with small `beta`.
