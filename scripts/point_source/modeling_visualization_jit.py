"""
End-to-end test: jit-cached visualization during a real Nautilus model-fit.
============================================================================

Exercises the full JAX visualization pipeline for the point-source analysis
path: ``AnalysisPoint(use_jax=True)`` with an ``Isothermal`` lens mass and
``PointFlux`` source (image-plane chi-squared via
``FitPositionsImagePairAll``).

This test runs in two parts:

Part 1 — **Caching probe.** Calls ``analysis.fit_for_visualization(instance)``
twice and asserts the second call is much faster than the first (confirming
the compiled function is cached on the analysis instance, not recompiled per
visualization call).

Part 2 — **Live Nautilus quick-update.** Runs a real (short) Nautilus fit.
The live search fires quick-update visualization every
``iterations_per_quick_update`` likelihood evaluations; we verify that
``fit.png`` lands on disk under the Nautilus output tree, proving the
JIT-cached ``fit_for_visualization`` fires correctly during the live
search callback.

This script deliberately opts in with
``AnalysisPoint(use_jax=True)``.
Default model-fit scripts elsewhere in the workspace leave the flag at
``False`` and are therefore untouched.
"""

import shutil
import time
from os import path
from pathlib import Path

import jax
import jax.numpy as jnp

from autolens import with_test_mode_segment

import autofit as af
import autolens as al


"""
__Dataset__
"""
dataset_path = Path("dataset") / "point_source" / "simple"

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/point_source/simulator.py"],
        check=True,
    )

dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)


"""
__Point Solver__
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)


"""
============================================================================
Part 1 — Caching probe
============================================================================

Model: Isothermal lens mass + PointFlux source. Same tight priors as the
other point-source scripts so the prior-median instance produces multiple
images. No free cosmology (breaks JIT via global-state distance caching).
"""
print("\n" + "=" * 72)
print("Part 1: Point-source caching probe")
print("=" * 72)

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.centre.centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.8)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

point_0 = af.Model(al.ps.PointFlux)
point_0.centre.centre_0 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)
point_0.centre.centre_1 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=True,
)

instance = model.instance_from_prior_medians()

t0 = time.perf_counter()
fit_1 = analysis.fit_for_visualization(instance)
jax.block_until_ready(fit_1.log_likelihood)
t1 = time.perf_counter()
compile_time = t1 - t0
print(f"First call (compile + run): {compile_time:.3f}s")
print(f"  log_likelihood leaf type: {type(fit_1.log_likelihood).__name__}")
assert isinstance(
    fit_1.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit_1.log_likelihood)}"

t0 = time.perf_counter()
fit_2 = analysis.fit_for_visualization(instance)
jax.block_until_ready(fit_2.log_likelihood)
t1 = time.perf_counter()
cached_time = t1 - t0
print(f"Second call (cached):       {cached_time:.3f}s")
print(f"Speedup:                    {compile_time / max(cached_time, 1e-9):.1f}x")

assert cached_time < compile_time * 0.5, (
    f"Cached call ({cached_time:.3f}s) not faster than compile "
    f"({compile_time:.3f}s) — JIT cache is not being hit."
)
print("PASS: Point-source jit-cached fit_for_visualization works and is reused.")


"""
__Visualization Sanity__

Phase D.1 rollout. SIE-tracer lensing-side check: non-empty tangential
critical curve + finite positive Einstein radius + warm-call < 100 ms.
Catches the silent-zero / cache-busting failure modes (PyAutoGalaxy
abd7b717, PyAutoFit #1280, PyAutoGalaxy #433).

No point-source-specific assertion on `fit.figure_of_merit` — the
script's prior-median position can legitimately give chi² = -inf
(positions outside the image-pair basin of the prior-median lens),
which would make the assertion flaky.
"""
import time as _sanity_time
import numpy as np  # not imported at module top in this script
from autogalaxy.operate.lens_calc import LensCalc as _SanityLensCalc

# Lensing-side sanity (SIE tracer, independent of the script's lens model).
_sanity_lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.2, ell_comps=(0.1, 0.0)),
)
_sanity_source = al.Galaxy(redshift=1.0)
_sanity_tracer = al.Tracer(galaxies=[_sanity_lens, _sanity_source])
_sanity_od = _SanityLensCalc.from_tracer(_sanity_tracer)

_sanity_t0 = _sanity_time.perf_counter()
_tc_list = (
    _sanity_od.tangential_critical_curve_list_via_zero_contour_from()
)  # cold: first call on fresh instance (JIT compile)
_sanity_cold_dt = _sanity_time.perf_counter() - _sanity_t0
assert len(_tc_list) > 0, (
    "no tangential critical curves returned by zero_contour — algorithmic "
    "regression (PyAutoGalaxy abd7b717 / PyAutoFit #1280 family)"
)
_er_sanity = _sanity_od.einstein_radius_via_zero_contour_from()
assert np.isfinite(float(_er_sanity)) and float(_er_sanity) > 0.0, (
    f"Einstein radius via zero_contour returned {_er_sanity} — should be "
    "finite and positive for the SIE sanity tracer (einstein_radius=1.2)"
)
print(
    f"  PASS Visualization Sanity (lensing): "
    f"{len(_tc_list)} tangential CC, einstein_radius={float(_er_sanity):.4f}"
)

_t0 = _sanity_time.perf_counter()
_sanity_od.tangential_critical_curve_list_via_zero_contour_from()  # warm (cached solver)
_warm_dt = _sanity_time.perf_counter() - _t0
# Hardware-independent guard: the warm (cached) call must be much faster than
# the cold JIT-compiling first call. A closure cache-busting regression
# (PyAutoGalaxy #433) recompiles the solver every call, so warm ~= cold (ratio
# near 1). An absolute millisecond budget instead false-positives on slower
# machines where the honest warm call legitimately exceeds it. Mirrors the
# compile-vs-cached ratio guard used elsewhere in this suite.
assert _warm_dt < _sanity_cold_dt * 0.5, (
    f"zero_contour warm call {_warm_dt * 1000:.1f} ms vs cold "
    f"{_sanity_cold_dt * 1000:.1f} ms (ratio {_warm_dt / _sanity_cold_dt:.2f}) — "
    "warm should be much faster than the cold compile; a ratio near 1 means the "
    "closure cache-busting bug from PyAutoGalaxy #433 has regressed"
)
print(
    f"  PASS Visualization Sanity (perf): warm {_warm_dt * 1000:.1f} ms vs "
    f"cold {_sanity_cold_dt * 1000:.1f} ms (ratio {_warm_dt / _sanity_cold_dt:.2f})"
)


"""
============================================================================
Part 2 — Live Nautilus quick-update
============================================================================

Rebuild the model fresh, create a separate analysis object, and run a short
Nautilus fit. The search fires
quick-update visualization every ``iterations_per_quick_update`` calls;
we assert that ``fit.png`` lands on disk under the Nautilus output tree.
"""
print("\n" + "=" * 72)
print("Part 2: Live Nautilus + jit-visualization for point source")
print("=" * 72)

mass2 = af.Model(al.mp.Isothermal)
mass2.centre.centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass2.centre.centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass2.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass2.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass2.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.8)

lens2 = af.Model(al.Galaxy, redshift=0.5, mass=mass2)

point_02 = af.Model(al.ps.PointFlux)
point_02.centre.centre_0 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)
point_02.centre.centre_1 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)

source2 = af.Model(al.Galaxy, redshift=1.0, point_0=point_02)

model2 = af.Collection(galaxies=af.Collection(lens=lens2, source=source2))


analysis_run = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=True,
)

output_root = Path("scripts") / "point_source" / "images" / "modeling_visualization_jit"
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True)

# Also clean the autofit search output so Nautilus performs live sampling
# instead of resuming from a cached samples.csv — without this the
# quick-update visualizer never fires on reruns.
output_search_root = (
    with_test_mode_segment(Path("output")) / output_root / "point_image_plane"
)
if output_search_root.exists():
    shutil.rmtree(output_search_root)

search = af.Nautilus(
    path_prefix=str(output_root),
    name="point_image_plane",
    n_live=50,
    n_like_max=1500,
    iterations_per_quick_update=500,
    number_of_cores=1,
)

print("Running Nautilus ...")
result = search.fit(model=model2, analysis=analysis_run)

# Nautilus writes quick-update images to output/<path_prefix>/<name>/<hash>/image/
# The lens quick-update visualizer writes fit_quick.png (via subplot_fit_quick).
produced_pngs = list(output_search_root.rglob("fit_quick.png"))
print(f"fit_quick.png files produced: {len(produced_pngs)}")
for p in produced_pngs:
    print(f"  {p}")
assert len(produced_pngs) > 0, (
    f"no fit_quick.png produced under {output_search_root} — "
    "quick-update visualization did not fire"
)
print(
    "\nPASS: jit-cached fit_for_visualization fires during Nautilus quick updates "
    f"for point source, fit_quick.png written."
)
