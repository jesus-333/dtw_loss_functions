# `dtw_loss_functions/benchmark.py`

[← Index](../0_index.md)

## Purpose

These are utilities to **time** the loss functions of this package over a grid of batch size (B), signal length (T) and channels (C), save the raw timings to disk, reload them, and plot them. They are used by the scripts in `benchmark/` (`compute_*.py` → `plot_*.py`); see `benchmark/README.md` for the Mac M4 Pro results.

## Pipeline

```
compute_benchmark  ──►  {path_save}/{loss_name}/B_{B}_T_{T}_C_{C}.npy   (n_repetitions timings, seconds)
load_benchmark_results ──► {loss_name: {"B_.._T_.._C_..": np.ndarray}}
get_array_to_plot      ──► (means, stds, labels) along one axis ('B' | 'T' | 'C')
plot_benchmark         ──► (fig, ax), optional savefig
```

## Functions

- **`compute_benchmark(B_list, T_list, C_list, loss_functions_to_use : dict, device, n_repetitions=100, path_save='benchmark', print_progress=False)`**
  - `loss_functions_to_use` maps a name to a loss **instance**; the name becomes the folder name.
  - For every (B, T, C) combination it builds `torch.rand(B, T, C)` inputs on `device` and times them with `repeat_inference`.
  - Existing files are overwritten.
- **`repeat_inference(x, x_r, loss_function, n_repetitions=100)`** returns a `np.ndarray` of wall-clock times. It runs under `torch.no_grad()`, so **only the forward pass is timed**.
- **`load_benchmark_results(path_results, loss_list=None)`**
  - Each sub-folder is one loss; the keys are file names without `.npy`.
  - Empty folders print a warning, but the empty dict entry is still created.
- **`get_array_to_plot(results_dict, loss_function_name, x_axis_variable, B_list, T_list, C_list)`**
  - Dispatches to the private helpers `__get_array_to_plot_T/_B/_C`.
  - For each combination of the two fixed variables, it returns the arrays of mean and std along the chosen axis, plus labels like `"name (B=.., C=..)"`.
  - Raises `ValueError` if a needed key is missing.
- **`get_info_from_filename(file_name)`** parses `B_x_T_y_C_z.npy` into `(B, T, C)` ints. It is currently unused (its call is commented out in the loader).
- **`plot_benchmark(plot_config, x_axis_variable_list, x_axis_variable_name, data_to_plot_mean, labels_to_plot, data_to_plot_std=None, color_dict=None)`**
  - Draws line plots with an optional ±std band.
  - `plot_config` keys: `figsize`, `fontsize`, `marker`, `markersize`, `linewidth`, `y_scale_log`, `y_lim`, `use_milliseconds`, `path_save`.
  - Missing keys are filled with defaults **in place**, so the caller's dict is mutated.

## Gotchas (from reading the code)

1. **No `torch.cuda.synchronize()`** around the timed call in `repeat_inference`. On CUDA, asynchronous kernels can make the measured times underestimate the real cost (mostly launch overhead). The backends that convert to numpy or call `.item()` synchronize implicitly, but the pure-CUDA paths (e.g. `ron`) may not. Add synchronization before trusting GPU numbers. On CPU/MPS-free runs (like the Mac benchmark) this does not matter.
2. `time.time()` is used rather than `time.perf_counter()`.
3. The first repetition includes numba JIT compilation, so ignore it or add a warm-up call.
4. In `plot_benchmark`, a `path_save` with no directory (e.g. `'plot.png'`) makes `os.makedirs('')` raise `FileNotFoundError`.
5. The docstring says the default `path_save` is `'benchmark/plot.png'`, but the code default is `None`.
6. The `color_dict` length check compares against `data_to_plot_mean` while the message mentions `labels_to_plot`. Harmless.
