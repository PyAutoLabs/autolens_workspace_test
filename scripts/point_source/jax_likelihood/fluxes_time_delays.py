"""
Func Grad: Point Source Fluxes + Time Delays (Solved)
========================================================

Test that JAX can compute the log-likelihood of a ``PointDataset`` that
carries fluxes and time delays in addition to positions, using the solved
(parameter-free) point-source fit classes throughout:
``al.FitPositionsSourceSolved`` for the source-plane position chi-squared,
``al.FitFluxesSolved`` for the flux chi-squared, and ``al.FitTimeDelaysSolved``
for the time-delay chi-squared — all evaluated against the same
analytically-solved source centre β* (Lombardi 2024, arXiv:2406.15280).

Because ``al.ps.PointSolved`` is parameter-free, the vmap model composes to
just the lens mass parameters plus a free ``cosmology.H0`` — H0 is kept free
because time delays are the observable that constrains it.

Full-pipeline JIT status
------------------------

Path A (``jax.jit(analysis.fit_from)``) goes through the same source-plane
fit as ``point_source/jax_likelihood/source_plane.py`` and is gated by the
same fit-return pytree gap: ``fit_from`` returns a bare ``PointSolver`` at
output component ``[1][1]``, tracked in
``PyAutoPrompt/autolens/fit_point_pytree.md``. When Path A JIT fails with
this ``TypeError`` the script prints a clear BLOCKER line and continues.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

import numpy as np
import jax.numpy as jnp
import jax
from pathlib import Path

import autofit as af
import autolens as al


"""
__Dataset__
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "point_source" / dataset_name
dataset_file = dataset_path / "point_dataset_with_fluxes_and_time_delays.json"

"""
__Dataset Auto-Simulation__

Guarded on this file specifically (not ``al.util.dataset.should_simulate``,
which only checks the directory): the directory already exists once
``simple.py``'s positions-only output is present, but the flux/time-delay
file is a second output of the same simulator added alongside it.
"""
if not dataset_file.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/point_source/simulators/simple.py"],
        check=True,
    )

dataset = al.from_json(file_path=dataset_file)

"""
__Point Solver__
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Model__

Mass priors copied from ``point_source/jax_likelihood/source_plane.py``. The
source is parameter-free (``al.ps.PointSolved``); cosmology stays free
(``H0``) because the time delays constrain it.
"""
mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.centre.centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.8)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

point_0 = af.Model(al.ps.PointSolved)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

cosmology = af.Model(al.cosmo.FlatLambdaCDM)
cosmology.H0 = af.UniformPrior(lower_limit=0.0, upper_limit=150.0)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), cosmology=cosmology
)

print(model.info)

"""
__Analysis__
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSourceSolved,
    fit_flux_cls=al.FitFluxesSolved,
    fit_time_delays_cls=al.FitTimeDelaysSolved,
)

from autofit.non_linear.fitness import Fitness
import time

batch_size = 1

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

parameters = np.zeros((batch_size, model.total_free_parameters))
for i in range(batch_size):
    parameters[i, :] = model.physical_values_from_prior_medians
parameters = jnp.array(parameters)

start = time.time()
print()
print(fitness._vmap(parameters))
print("JAX Time To VMAP + JIT Function", time.time() - start)

start = time.time()
print()
result = fitness._vmap(parameters)
print(result)
print("JAX Time Taken using VMAP:", time.time() - start)
print("JAX Time Taken per Likelihood:", (time.time() - start) / batch_size)

EXPECTED_VMAP_LOG_LIKELIHOOD_FLUXES_TIME_DELAYS = -137.67455417

np.testing.assert_allclose(
    np.array(result),
    EXPECTED_VMAP_LOG_LIKELIHOOD_FLUXES_TIME_DELAYS,
    rtol=1e-4,
    err_msg="point_source/fluxes_time_delays: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``analysis.fit_from``__

Free ``cosmology`` is dropped for this block (same ``UnexpectedTracerError``
caveat as ``image_plane.py`` / ``point.py``) even though ``H0`` is free in
the vmap model above.
"""

model_jit = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model_jit.instance_from_prior_medians()

analysis_np = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSourceSolved,
    fit_flux_cls=al.FitFluxesSolved,
    fit_time_delays_cls=al.FitTimeDelaysSolved,
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
log_likelihood_np = float(fit_np.log_likelihood)
print("NumPy fit.log_likelihood:", log_likelihood_np)

# This is the Path-A *cosmology-dropped* reference value (model_jit has no
# cosmology component, so AnalysisPoint falls back to its internal default
# rather than the vmap model's free H0=75 prior median) — it is NOT expected
# to match the vmap literal above; see the H0-matched parity check below.
EXPECTED_EAGER_LOG_LIKELIHOOD_FLUXES_TIME_DELAYS = -133.0115150576916

np.testing.assert_allclose(
    log_likelihood_np,
    EXPECTED_EAGER_LOG_LIKELIHOOD_FLUXES_TIME_DELAYS,
    rtol=1e-4,
    err_msg=(
        f"point_source/fluxes_time_delays: regression — eager (cosmology-dropped, "
        f"Path-A reference) log_likelihood drifted (got {log_likelihood_np}, "
        f"expected {EXPECTED_EAGER_LOG_LIKELIHOOD_FLUXES_TIME_DELAYS})"
    ),
)
print(
    f"Eager regression assertion PASSED (cosmology-dropped Path-A reference): "
    f"log_likelihood matches {EXPECTED_EAGER_LOG_LIKELIHOOD_FLUXES_TIME_DELAYS}"
)

"""
__NumPy-vs-JAX vmap parity (H0-matched)__

Time delays are directly and strongly H0-sensitive, so eager/vmap parity is
only meaningful when both paths share the same cosmology — unlike the
positions-only scripts (e.g. ``source_plane.py``), where dropping cosmology
in Path A is a no-op on the likelihood and parity holds incidentally. Here we
build a second eager fit from the *full* vmap model's prior-median instance
(cosmology included, H0=75) rather than reusing the cosmology-dropped
``instance`` above.
"""
instance_full = model.instance_from_prior_medians()

analysis_full_np = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSourceSolved,
    fit_flux_cls=al.FitFluxesSolved,
    fit_time_delays_cls=al.FitTimeDelaysSolved,
    use_jax=False,
)
fit_full_np = analysis_full_np.fit_from(instance=instance_full)
log_likelihood_full_np = float(fit_full_np.log_likelihood)
print("NumPy fit.log_likelihood (H0-matched, full model):", log_likelihood_full_np)

np.testing.assert_allclose(
    log_likelihood_full_np,
    float(result[0]),
    rtol=1e-4,
    err_msg="point_source/fluxes_time_delays: eager vs vmap parity mismatch (H0-matched)",
)
print("PASS: eager (H0-matched) vs vmap parity.")

analysis_jit = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSourceSolved,
    fit_flux_cls=al.FitFluxesSolved,
    fit_time_delays_cls=al.FitTimeDelaysSolved,
    use_jax=True,
)
fit_jit_fn = jax.jit(analysis_jit.fit_from)

full_pipeline_jits = False
try:
    fit = fit_jit_fn(instance)
    print("JIT fit.log_likelihood:", fit.log_likelihood)
    assert isinstance(
        fit.log_likelihood, jnp.ndarray
    ), f"expected jax.Array, got {type(fit.log_likelihood)}"
    np.testing.assert_allclose(float(fit.log_likelihood), log_likelihood_np, rtol=1e-4)
    full_pipeline_jits = True
    print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
except TypeError as e:
    print(
        "\nBLOCKER: fluxes/time-delays jit(fit_from) is gated by:\n"
        f"  {type(e).__name__}: {e}\n"
        "  fit_from returns a bare PointSolver at output component [1][1],\n"
        "  which is not pytree-registered under jax.jit (it goes through the\n"
        "  same source-plane fit as source_plane.py). Tracked in\n"
        "  PyAutoPrompt/autolens/fit_point_pytree.md.\n"
        "  Eager NumPy regression assertion still PASSED above."
    )
