"""
Visualization JAX Pilot: Point Source Analysis
===============================================

Pilot for the JAX-backed visualization path on ``PointDataset``.

Goal
----
Run ``VisualizerPoint.visualize`` with ``use_jax=True`` on ``AnalysisPoint``.
Visualization now follows ``use_jax`` automatically — the point visualizer
dispatches through ``analysis.fit_for_visualization``, which lazily wraps
``fit_from`` in ``jax.jit``. To trace across that boundary the model and fit
return type must be JAX pytrees, so this script enables pytree registration
before constructing the model.

Scope
-----
- ``Isothermal`` lens mass + ``PointFlux`` source (image-plane chi-squared).
- Calls ``VisualizerPoint.visualize`` only (not ``visualize_before_fit``).
- Re-uses the ``simple/point_dataset_positions_only.json`` dataset.
- No ``try/except`` wrapper — failure surfaces immediately.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import autofit as af
import autolens as al
from autolens.point.model.visualizer import VisualizerPoint


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
__Model__

Tight priors centred on the true values so the prior-median instance
produces a sensible lens configuration (multiple images exist).
No free cosmology — cosmology distance caching breaks JIT.
"""
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


"""
__Analysis__

``use_jax=True`` turns on the JAX ``_xp`` path. Visualization now follows
``use_jax`` automatically via ``Analysis.fit_for_visualization``.
``title_prefix`` is passed through via PR #506's **kwargs fix.
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=True,
    title_prefix="JAX_PILOT",
)


"""
__Paths__
"""
image_path = Path("scripts") / "point_source" / "images" / "visualization_jax"
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)
output_path = image_path / "output"
output_path.mkdir(parents=True)
paths = SimpleNamespace(image_path=image_path, output_path=output_path)


"""
__Run visualize on the JAX-backed fit__
"""
instance = model.instance_from_prior_medians()

print("Running VisualizerPoint.visualize with use_jax=True ...")
VisualizerPoint.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)

print("Files in image_path:", list(image_path.iterdir()))
assert (
    image_path / "fit.png"
).exists(), f"fit.png was not produced. Files present: {list(image_path.iterdir())}"
print("PILOT SUCCEEDED — JAX-backed point-source visualization produced fit.png.")


"""
__Visualization Sanity__

Phase D.2.a rollout. SIE-tracer lensing-side check only — no
point-source-specific FoM assertion (prior-median position can
legitimately give chi² = -inf if outside the image-pair basin).
"""
import time as _sanity_time
import numpy as _sanity_np
from autogalaxy.operate.lens_calc import LensCalc as _SanityLensCalc

_sanity_lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.2, ell_comps=(0.1, 0.0)),
)
_sanity_source = al.Galaxy(redshift=1.0)
_sanity_tracer = al.Tracer(galaxies=[_sanity_lens, _sanity_source])
_sanity_od = _SanityLensCalc.from_tracer(_sanity_tracer)

_sanity_t0 = _sanity_time.perf_counter()
_tc_list = _sanity_od.tangential_critical_curve_list_via_zero_contour_from()  # cold: first call on fresh instance (JIT compile)
_sanity_cold_dt = _sanity_time.perf_counter() - _sanity_t0
assert len(_tc_list) > 0, (
    "no tangential critical curves returned by zero_contour — algorithmic "
    "regression (PyAutoGalaxy abd7b717 / PyAutoFit #1280 family)"
)
_er_sanity = _sanity_od.einstein_radius_via_zero_contour_from()
assert _sanity_np.isfinite(float(_er_sanity)) and float(_er_sanity) > 0.0, (
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
