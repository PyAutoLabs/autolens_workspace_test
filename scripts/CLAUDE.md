# scripts/ — Integration Test Coverage

This document describes what each integration test script covers, what it asserts,
and where its JAX-specific responsibilities lie.

## Codex / sandboxed runs

When running Python from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/imaging/model_fit.py
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

## Testing Philosophy

- Scripts run **without** `PYAUTO_TEST_MODE=1` — non-linear searches execute for
  real (using sampler limits like `n_like_max=300` to keep runtime short).
- `jax_likelihood_functions/` scripts assert their `fitness._vmap` output against a
  hardcoded expected log-likelihood literal (`assert_allclose(np.array(result), <value>, rtol=1e-4)`).
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

### `imaging/visualization.py`
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

### `interferometer/visualization.py`
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

## jax_likelihood_functions/

Scripts that test JAX can compute log-likelihood gradients and batch evaluations via
`jax.vmap` for various model types.  Each script builds a `Fitness` object and calls
`fitness._vmap(parameters)`.

| Script | Model type |
|---|---|
| `imaging/lp.py` | Light parametric (Sersic, Exponential) |
| `imaging/mge.py` | Multi-Gaussian expansion |
| `imaging/simulator.py` | Operated (PSF-convolved) light profiles |
| `imaging/delaunay.py` | Delaunay pixelization |
| `imaging/rectangular.py` | Rectangular pixelization |
| `imaging/mge_group.py` | MGE with extra galaxies |
| `interferometer/mge.py` | MGE for interferometry |
| `interferometer/rectangular.py` | Rectangular pixelization for interferometry |
| `interferometer/lp.py` | Parametric Sersic source for interferometry |
| `interferometer/delaunay.py` | Delaunay pixelization for interferometry |
| `interferometer/delaunay_mge.py` | Delaunay source + MGE lens for interferometry |
| `interferometer/rectangular_mge.py` | Rectangular source + MGE lens for interferometry |
| `interferometer/rectangular_dspl.py` | Rectangular source on double source plane (interferometry) |
| `interferometer/rectangular_sparse.py` | Rectangular pixelization via JAX sparse-operator NUFFT path |
| `point_source/point.py` | Point-source likelihood |
| `point_source/image_plane.py` | Point-source image-plane chi-squared (`FitPositionsImagePairAll`) |
| `point_source/source_plane.py` | Point-source source-plane chi-squared (`FitPositionsSource`) — JIT currently blocked |
| `multi/lp.py` | Parametric Sersic across g/r via `FactorGraphModel`; per-band source `ell_comps` (option B) |
| `multi/mge.py` | MGE source across g/r; per-band source MGE `ell_comps` (option B) |
| `multi/mge_group.py` | MGE + extra galaxies across g/r; per-band source MGE `ell_comps` (option B) |
| `multi/rectangular.py` | Rectangular pixelization across g/r; per-band `regularization.inner_coefficient` (option B) |
| `multi/delaunay.py` | Delaunay pixelization (Hilbert image-mesh) across g/r; per-band `regularization.inner_coefficient` (option B) |
| `multi/rectangular_mge.py` | MGE lens + rectangular source across g/r; per-band `regularization.inner_coefficient` (option B) |
| `multi/delaunay_mge.py` | MGE lens + Delaunay source across g/r; per-band `regularization.inner_coefficient` (option B) |

---

## hessian_jax.py

Tests `LensCalc` hessian-derived lensing quantities (`hessian_from`,
`convergence_2d_via_hessian_from`, `shear_yx_2d_via_hessian_from`,
`magnification_2d_via_hessian_from`, `jacobian_from`, `tangential_eigen_value_from`,
`radial_eigen_value_from`) using the three-step JAX pattern on both irregular and
uniform grids.

This is the **reference implementation** for the JAX testing pattern — new JAX tests
follow the same style.

---

## profiles_jit.py

Tests JAX JIT compilation of individual light and mass profile methods from
`autogalaxy.profiles`.  This is the lower-level complement to `hessian_jax.py` — it
targets the methods that are called internally by `LensCalc` and `Tracer`.

**Light profiles**: `lp.Sersic`, `lp.Exponential`, `lp.Gaussian`, `lp.DevVaucouleurs`
→ `image_2d_from`

**Mass profiles**: `mp.Isothermal`, `mp.PowerLaw`, `mp.NFW`, `mp.ExternalShear`
→ `deflections_yx_2d_from`, `convergence_2d_from`

Each method is tested on both `Grid2DIrregular` and `Grid2D.uniform`.
All three steps of the JAX pattern are applied.  NFW uses `rtol=1e-4` (looser) due
to its analytic JAX implementation.

---

## tracer_multiplane.py

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

## tracer_jax.py

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

## database/

See `database/scrape/CLAUDE.md` for detail on the database scrape tests.
