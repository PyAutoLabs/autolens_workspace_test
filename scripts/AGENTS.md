# scripts/ — Integration Test Coverage

This document describes what each integration test script covers, what it asserts,
and where its JAX-specific responsibilities lie.

## Layout

`scripts/` mirrors the `autolens_workspace` dataset taxonomy: dataset-typed
folders `imaging/`, `interferometer/`, `point_source/`, `multi_galaxy/`,
`cluster/`, `multi_dataset/`, plus `misc/` for dataset-agnostic material (`aggregator/`, `database/`, `mass/`,
`mass_via_integral/`, `jax_assertions/`, `latent/`, `weak/`, `interop/`, the
`util.py` gradient helper, and loose tracer/profile/hessian tests). `gallery/`
and `profiling/` sit outside the taxonomy (external couplings).

Within each dataset folder, related scripts are grouped into **task
subfolders** so the dataset root holds only its modeling singletons:

- `jax_likelihood/` — the batched `fitness._vmap` likelihood-function tests
  (former `jax_likelihood_functions/` tree; `multipole` from `light_multipole/`).
- `jax_grad/` — the finite-difference gradient tests (former `jax_grad/` tree),
  which sit two levels deep and reach `../../misc/util.py` via a `sys.path` shim.
- `visualization/` — the `visualization*.py` and `modeling_visualization_*_jit.py`
  visualization tests (`multi_dataset/` strips the prefix to `visualization/imaging.py`
  and `visualization/interferometer.py`).
- `simulator/` — the dataset simulators; the loose `simulator*.py` bootstrap
  targets consolidate in beside `no_lens_light.py` / `with_lens_light.py` as
  `simple.py` / `dspl.py` (point_source keeps its `simulators/` dir name).
- `substructure/` — imaging's `test_*simulate*` / `test_scan_multiplane` trio
  (former `jax_substructure/` tree).
- `datacube/` — interferometer's former `*_datacube.py` scripts (suffix stripped).

The former `jax_likelihood_functions/`, `jax_grad/`, `jax_substructure/`,
`potential_correction/`, `model_composition/` and `light_multipole/` top-level
trees were dissolved into these per-dataset task subfolders. Dataset roots keep
their modeling singletons (`model_fit.py`, `convolution*.py`,
`subhalo_recovery*.py`, `nufft.py`, the `dataset_model_parity_*` tests, …).

## Codex / sandboxed runs

When running Python from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/imaging/model_fit.py
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

## Testing Philosophy

- Scripts run **without** `PYAUTO_TEST_MODE=1` — non-linear searches execute for
  real (using sampler limits like `n_like_max=300` to keep runtime short).
- The JAX likelihood-function scripts (now distributed across `imaging/`,
  `interferometer/`, `point_source/` and `multi_dataset/`) assert their `fitness._vmap`
  output against a hardcoded expected log-likelihood literal (`assert_allclose(np.array(result), <value>, rtol=1e-4)`).
  These literals are regression markers for the simulator + likelihood pipeline as a
  whole; if a deliberate simulator change shifts the value, regenerate the literal
  by running the script and pasting in the new `result` value. Don't replace these
  with relational `vmap ≈ NumPy-path` assertions — that would lose absolute regression
  detection.
- JAX tests follow the **three-step pattern** established in `hessian_jax.py`:
  1. NumPy path — assert correct autoarray return type with `np.ndarray` backing.
  2. JAX path outside JIT — assert same autoarray type but with `jax.Array` backing.
  3. JAX path inside `jax.jit` — extract `._array` at JIT boundary, assert numerical
     agreement with NumPy path via `npt.assert_allclose`.

---

## imaging/

### `imaging/simulator/no_lens_light.py` / `with_lens_light.py`
Simulate a realistic strong-lens CCD imaging dataset (FITS files + JSON tracer +
positions).  No assertions — output feeds other tests.

### `imaging/model_fit.py`
End-to-end parametric + pixelized source model-fit on simulated imaging.  Exercises
the full `AnalysisImaging → FitImaging → Tracer` pipeline with a Nautilus search.

### `imaging/convolution.py`
Tests PSF convolution of a simulated imaging dataset.

### `imaging/visualization/visualization.py`
Generates visualisation plots of imaging fits and tracers for all three source types
(parametric Sersic, rectangular pixelization, Delaunay pixelization).

`visualize_before_fit` runs once with the parametric source into the main
`visualization/` folder (dataset, positions, adapt images).  `visualize` then runs
for each source in its own subfolder (`parametric/`, `rectangular/`, `delaunay/`),
limited to `fit.png`, `tracer.png`, and (for pixelized sources) `inversion_0_0.png`
via `config_source/visualize/plots.yaml`.

---

## interferometer/

### `interferometer/simulator/*.py`
Simulate ALMA-style interferometer data.

### `interferometer/model_fit.py`
End-to-end model-fit on interferometer data using `FitInterferometer`.

### `interferometer/nufft.py`
Accuracy check of **nufftax** (the JAX-native NUFFT behind `TransformerNUFFT`,
https://github.com/GragasLab/nufftax) against `TransformerDFT`, the exact
direct Fourier transform. Originally the parity test for swapping pynufft for
nufftax; that swap shipped and pynufft has since been removed from
PyAutoArray, so the pynufft legs are gone and the DFT is the sole reference.

The script computes visibilities on (a) a 5x5 all-ones image, (b) a 256x256
lensed-Sersic image with real SMA uv coverage, (c) a mapping matrix (the
pixelization code path), and (d) the adjoint (visibilities -> image). It
asserts that **nufftax and the shipped `TransformerNUFFT` both match
`TransformerDFT` to machine precision**. The convention recipe (image flip,
frequency scaling, half-pixel parity-aware phase shift) is hard-coded in
helper functions at the top of the script and documented in the module
docstring.

Saves a residuals plot to `scripts/interferometer/images/nufft_residuals.png`.

Requires `nufftax` (`pip install nufftax`).

### `interferometer/visualization/visualization.py`
Generates visualisation plots of interferometer fits and tracers for all three source
types (parametric Sersic, rectangular pixelization, Delaunay pixelization).

`visualize_before_fit` runs once with the parametric source into the main
`visualization/` folder (subplot_dataset, positions, adapt images).  `visualize`
then runs for each source in its own subfolder (`parametric/`, `rectangular/`,
`delaunay/`), limited to `subplot_fit.png`, `subplot_tracer.png`, and (for pixelized
sources) `subplot_inversion_0.png` via `config_source/visualize/plots.yaml`.

---

## point_source/

### `point_source/simulators/point_source.py`
Simulate a lensed point-source (multiply-imaged quasar) dataset.

---

## JAX likelihood functions (in `imaging/` `interferometer/` `point_source/` `multi_dataset/`)

Scripts that test JAX can compute log-likelihood gradients and batch evaluations via
`jax.vmap` for various model types.  Each script builds a `Fitness` object and calls
`fitness._vmap(parameters)`.  These were consolidated out of the former
`jax_likelihood_functions/` tree into a `jax_likelihood/` task subfolder inside
the dataset folder each one exercises; the paths below are workspace-relative
under `scripts/`.  The former `jax_likelihood_functions/datacube/` variants live
in `interferometer/datacube/` (the `_datacube` suffix stripped now that the
folder carries it), keeping them clear of the `interferometer/jax_likelihood/`
likelihoods of the same base name.

| Script | Model type |
|---|---|
| `imaging/jax_likelihood/lp.py` | Light parametric (Sersic, Exponential) |
| `imaging/jax_likelihood/smbh.py` | Central `SMBH` point mass with FREE (traced) `mass` — regression cover for PyAutoGalaxy#553; non-SMBH components pinned to simulator truth because at prior medians the positive-only solver zeroes the source and the literal goes blind to source-plane mass |
| `imaging/jax_likelihood/mge.py` | Multi-Gaussian expansion |
| `imaging/jax_likelihood/delaunay.py` | Delaunay pixelization |
| `imaging/jax_likelihood/rectangular.py` | Rectangular pixelization |
| `imaging/jax_likelihood/mge_group.py` | MGE with extra galaxies |
| `interferometer/jax_likelihood/mge.py` | MGE for interferometry |
| `interferometer/jax_likelihood/rectangular.py` | Rectangular pixelization for interferometry |
| `interferometer/jax_likelihood/lp.py` | Parametric Sersic source for interferometry |
| `interferometer/jax_likelihood/delaunay.py` | Delaunay pixelization for interferometry |
| `interferometer/jax_likelihood/delaunay_mge.py` | Delaunay source + MGE lens for interferometry |
| `interferometer/jax_likelihood/rectangular_mge.py` | Rectangular source + MGE lens for interferometry |
| `interferometer/jax_likelihood/rectangular_dspl.py` | Rectangular source on double source plane (interferometry) |
| `interferometer/jax_likelihood/rectangular_sparse.py` | Rectangular pixelization via JAX sparse-operator NUFFT path |
| `point_source/jax_likelihood/point.py` | Point-source likelihood walkthrough (image-plane `FitPositionsImagePairAll`), plus the `jit(fit_from)` round-trip. The centre-free `FitPositionsImagePairAllSolved` block was dropped on #267 (33s of XLA compile on the PR gate); that coverage now lives only in `image_plane.py`, on the weekly workspace-smoke / release-integrate channels |
| `point_source/jax_likelihood/image_plane.py` | Point-source image-plane chi-squared (`FitPositionsImagePairAll`) + centre-free `FitPositionsImagePairAllSolved` / `FitPositionsImagePairRepeatSolved` variants |
| `point_source/jax_likelihood/source_plane.py` | Point-source source-plane chi-squared (`FitPositionsSource`) + centre-free `FitPositionsSourceSolved` — Path A JIT blocked by the fit-return pytree gap (`PyAutoPrompt/autolens/fit_point_pytree.md`), not the (already-fixed) xp-propagation bug |
| `point_source/jax_likelihood/fluxes_time_delays.py` | Point-source fluxes + time delays via the solved fit classes (`FitFluxesSolved`, `FitTimeDelaysSolved`) alongside `FitPositionsSourceSolved` |
| `multi_dataset/jax_likelihood/lp.py` | Parametric Sersic across g/r via `FactorGraphModel`; per-band source `ell_comps` (option B) |
| `multi_dataset/jax_likelihood/mge.py` | MGE source across g/r; per-band source MGE `ell_comps` (option B) |
| `multi_dataset/jax_likelihood/mge_group.py` | MGE + extra galaxies across g/r; per-band source MGE `ell_comps` (option B) |
| `multi_dataset/jax_likelihood/rectangular.py` | Rectangular pixelization across g/r; per-band `regularization.inner_coefficient` (option B) |
| `multi_dataset/jax_likelihood/delaunay.py` | Delaunay pixelization (Hilbert image-mesh) across g/r; per-band `regularization.inner_coefficient` (option B) |
| `multi_dataset/jax_likelihood/rectangular_mge.py` | MGE lens + rectangular source across g/r; per-band `regularization.inner_coefficient` (option B) |
| `*/jax_likelihood/rectangular*_rtu.py` | RTU (kernel-CDF) counterparts of the imaging/multi_dataset rectangular pin scripts — pure renames of the pre-split scripts, likelihood pins unchanged; kept so the RTU meshes stay pinned after the Bilinear default switch |
| `multi_dataset/jax_likelihood/delaunay_mge.py` | MGE lens + Delaunay source across g/r; per-band `regularization.inner_coefficient` (option B) |
| `multi_dataset/jax_likelihood/dataset_model.py` | Parametric Sersic across g/r with `al.DatasetModel.grid_offset` as a free 2D offset prior on every dataset after the first (band 0 stays at the fixed `(0.0, 0.0)` default) |
| `interferometer/datacube/rectangular.py` | 4-channel datacube via `FactorGraphModel`; `RectangularRTUAdaptDensity` + `reg.Adapt()` source. Identical channels — assertion at `4 × interferometer/rectangular` literal. Path A `jit(log_likelihood_function)` round-trip; Path B `TransformerNUFFT` cross-check |
| `interferometer/datacube/delaunay.py` | 4-channel datacube via `FactorGraphModel`; Delaunay source (Hilbert image-mesh, edge zeroing, `reg.AdaptSplit()`). Identical channels — assertion at `4 × interferometer/delaunay` literal. Path A + Path B (TransformerNUFFT cross-check) |

---

## misc/hessian_jax.py

Tests `LensCalc` hessian-derived lensing quantities (`hessian_from`,
`convergence_2d_via_hessian_from`, `shear_yx_2d_via_hessian_from`,
`magnification_2d_via_hessian_from`, `jacobian_from`, `tangential_eigen_value_from`,
`radial_eigen_value_from`) using the three-step JAX pattern on both irregular and
uniform grids.

This is the **reference implementation** for the JAX testing pattern — new JAX tests
follow the same style.

---

## misc/profiles_jit.py

Tests JAX JIT compilation of individual light and mass profile methods from
`autogalaxy.profiles`.  This is the lower-level complement to `hessian_jax.py` — it
targets the methods that are called internally by `LensCalc` and `Tracer`.

**Light profiles**: `lp.Sersic`, `lp.Exponential`, `lp.Gaussian`, `lp.DevVaucouleurs`
→ `image_2d_from`

**Mass profiles**: `mp.Isothermal`, `mp.PowerLaw`, `mp.NFW`, `mp.ExternalShear`, `mp.ExternalPotential`
→ `deflections_yx_2d_from`, `convergence_2d_from`

**Point-mass profiles**: `mp.PointMass`, `mp.SMBH` (regression cover for PyAutoGalaxy#553)
→ `deflections_yx_2d_from`, `potential_2d_from`, plus a raw-zeros `convergence_2d_from` check
(their convergence is undecorated and returns a raw zeros array by design). The traced-mass half
of #553 needs a free model parameter and lives in `imaging/jax_likelihood/smbh.py`.

Each method is tested on both `Grid2DIrregular` and `Grid2D.uniform`.
All three steps of the JAX pattern are applied.  NFW uses `rtol=1e-4` (looser) due
to its analytic JAX implementation.

---

## misc/tracer_multiplane.py

Tests multi-plane ray-tracing logic in the `Tracer` class using the NumPy path only.
All assertions are relational.

| Test | What it checks |
|---|---|
| No-mass tracer | Source-only Tracer returns grids identical to input |
| Two-plane deflection | Lens actually deflects the source-plane grid |
| Redshift order invariance | Galaxies listed in any order give same traced grids |
| Coplanar additivity | Two `IsothermalSph(R)` = one `IsothermalSph(2R)` |
| Three-plane system | Second intermediate lens changes source-plane grid |
| `plane_index_limit` | Early termination returns same grids for computed planes |
| Plane grouping | Co-redshift galaxies share a plane; `total_planes < len(galaxies)` |

---

## misc/tracer_jax.py

Tests that `Tracer` ray-tracing calculations produce identical results on the NumPy
and JAX paths, and compile correctly under `jax.jit`.  Uses the same two-plane and
three-plane tracer configurations as `tracer_multiplane.py`.

| Test | Method | Grid |
|---|---|---|
| 1–2 | `traced_grid_2d_list_from` NumPy vs JAX | irregular, 2p and 3p |
| 3–4 | `traced_grid_2d_list_from` inside `jax.jit` | irregular, 2p and 3p |
| 5–6 | `image_2d_from` NumPy vs JAX + inside JIT | irregular |
| 7–8 | `deflections_yx_2d_from` NumPy vs JAX + inside JIT | irregular |
| 9–10 | `convergence_2d_from` NumPy vs JAX + inside JIT | irregular |
| 11 | Three-plane `image_2d_from` + `deflections_yx_2d_from` | irregular |

The JIT tests extract `._array` from autoarray results at the JIT boundary so that
the output is a raw `jax.Array` list — a valid JAX pytree.

---

## misc/database/

See `misc/database/scrape/CLAUDE.md` for detail on the database scrape tests.
