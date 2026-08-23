"""
Func Grad: Point Source Source-Plane Chi-Squared
=================================================

Test that JAX can compute the log-likelihood of a ``PointDataset`` using the
**source-plane** chi-squared (``al.FitPositionsSource``).

Source-plane fitting traces each *observed* image-plane position back to the
source plane via the lens model, then computes a chi-squared between the
ray-traced positions and the model source position.  No image-plane solver
is required.

Full-pipeline JIT status
------------------------

The ``Grid2DIrregular.grid_2d_via_deflection_grid_from`` xp-propagation bug
that previously blocked Path A here was fixed in phase 2 (PyAutoArray#414).
The remaining blocker is a fit-return pytree gap: ``fit_from`` returns a
``PointSolver`` instance at output component ``[1][1]``, which is not a
valid JAX type under ``jax.jit`` — tracked in
``PyAutoPrompt/autolens/fit_point_pytree.md``.  When Path A JIT fails with
this ``TypeError`` the script prints a clear BLOCKER line and continues, so
the eager NumPy regression assertion is still exercised.  Once the pytree
registration lands, the JIT path will succeed without modifying this
script.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os

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

"""
__Dataset Cap Guard__

Guard the full-resolution regime this parity check depends on, BEFORE anything can
touch the dataset directory. ``dataset/point_source/simple`` is committed and
gitignore-allowlisted, and ``should_simulate`` deletes it outright when
``PYAUTO_SMALL_DATASETS=1`` is in force — it is JSON-only, so the ``SMALLDAT`` stamp
that spares capped FITS datasets cannot reach it, and a poisoned directory then
survives into the next full-regime run. Under the cap ``PointSolver.solve`` also
short-circuits to a fixed, model-independent position pair, so the pinned literal
below would be measuring nothing at all. Both were observed together as a parity
assert that failed with a *different* value on every run (PyAutoLens#710).

The ``ENV: jax full_datasets`` declaration at the top of this file is what keeps this
assertion true under the smoke and release profiles; this guard is what catches a
hand-rolled environment that overrides it. Checked on the env var rather than on grid
geometry (the ``interferometer/nufft.py`` approach) because a JSON dataset has no
capped shape to detect.
"""
assert os.environ.get("PYAUTO_SMALL_DATASETS") != "1", (
    "PYAUTO_SMALL_DATASETS=1 is set, but this script declares `ENV: jax full_datasets` "
    f"and pins an absolute likelihood literal. Under the cap the committed dataset at "
    f"{dataset_path} is deleted and re-simulated, and PointSolver.solve returns a fixed "
    "position pair that is identical for every lens model - the comparison below would "
    "measure nothing. Unset PYAUTO_SMALL_DATASETS, or run this script through the smoke "
    "profile, which honours the declaration."
)

"""
__Dataset Auto-Simulation__
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/point_source/simulators/simple.py"],
        check=True,
    )

dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)

"""
__Point Solver__

Source-plane chi-squared does not use the solver, but ``AnalysisPoint``
requires one — pass the standard JAX-friendly solver for consistency.
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
    fit_positions_cls=al.FitPositionsSource,
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

# Regression constant filled in on first run against the workspace_test
# seeded PointDataset (``scripts/point_source/simulators/simple.py``).
EXPECTED_VMAP_LOG_LIKELIHOOD_SOURCE_PLANE = -331481.25978149

np.testing.assert_allclose(
    np.array(result),
    EXPECTED_VMAP_LOG_LIKELIHOOD_SOURCE_PLANE,
    rtol=1e-4,
    err_msg="point_source/source_plane: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``analysis.fit_from``__

Wrapped in ``try/except TypeError`` — source-plane fitting's ``fit_from``
returns a bare ``PointSolver`` at output component ``[1][1]``, which is not
pytree-registered (the fit-return pytree gap tracked in
``PyAutoPrompt/autolens/fit_point_pytree.md``).  The eager NumPy
log-likelihood is still asserted for regression coverage.
"""


model_jit = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model_jit.instance_from_prior_medians()

analysis_np = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSource,
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
log_likelihood_np = float(fit_np.log_likelihood)
print("NumPy fit.log_likelihood:", log_likelihood_np)

EXPECTED_EAGER_LOG_LIKELIHOOD_SOURCE_PLANE = -331481.26508536364

np.testing.assert_allclose(
    log_likelihood_np,
    EXPECTED_EAGER_LOG_LIKELIHOOD_SOURCE_PLANE,
    rtol=1e-4,
    err_msg=(
        f"point_source/source_plane: regression — eager log_likelihood drifted "
        f"(got {log_likelihood_np}, expected {EXPECTED_EAGER_LOG_LIKELIHOOD_SOURCE_PLANE})"
    ),
)
print(
    f"Eager regression assertion PASSED: log_likelihood matches "
    f"{EXPECTED_EAGER_LOG_LIKELIHOOD_SOURCE_PLANE}"
)

analysis_jit = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSource,
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
    # fit_from returns a bare PointSolver instance at output component
    # [1][1], which is not pytree-registered — jax.jit cannot flatten the
    # return value. Tracked in PyAutoPrompt/autolens/fit_point_pytree.md.
    # (The previously-blocking Grid2DIrregular.grid_2d_via_deflection_grid_from
    # xp-propagation bug was fixed in phase 2, PyAutoArray#414.)
    print(
        "\nBLOCKER: source-plane jit(fit_from) is gated by:\n"
        f"  {type(e).__name__}: {e}\n"
        "  fit_from returns a bare PointSolver at output component [1][1],\n"
        "  which is not pytree-registered under jax.jit. Tracked in\n"
        "  PyAutoPrompt/autolens/fit_point_pytree.md.\n"
        "  Eager NumPy regression assertion still PASSED above."
    )


"""
__Model: Solved Source (Parameter-Free)__

``al.ps.PointSolved`` has zero free parameters — the source-plane position β* is
solved analytically (Lombardi 2024, arXiv:2406.15280) rather than fitted, so no
centre priors are set. ``al.FitPositionsSourceSolved`` performs the same
source-plane chi-squared as ``al.FitPositionsSource`` above but against the
analytic β* instead of a modelled centre.
"""

point_0_solved = af.Model(al.ps.PointSolved)

source_solved = af.Model(al.Galaxy, redshift=1.0, point_0=point_0_solved)

model_solved = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_solved), cosmology=cosmology
)

print(model_solved.info)

analysis_solved = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSourceSolved,
)

from autofit.non_linear.fitness import Fitness
import time

batch_size = 1

fitness_solved = Fitness(
    model=model_solved,
    analysis=analysis_solved,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

parameters_solved = np.zeros((batch_size, model_solved.total_free_parameters))
for i in range(batch_size):
    parameters_solved[i, :] = model_solved.physical_values_from_prior_medians
parameters_solved = jnp.array(parameters_solved)

start = time.time()
print()
print(fitness_solved._vmap(parameters_solved))
print("JAX Time To VMAP + JIT Function", time.time() - start)

start = time.time()
print()
result_solved = fitness_solved._vmap(parameters_solved)
print(result_solved)
print("JAX Time Taken using VMAP:", time.time() - start)
print("JAX Time Taken per Likelihood:", (time.time() - start) / batch_size)

EXPECTED_VMAP_LOG_LIKELIHOOD_SOURCE_PLANE_SOLVED = -94.70750993

np.testing.assert_allclose(
    np.array(result_solved),
    EXPECTED_VMAP_LOG_LIKELIHOOD_SOURCE_PLANE_SOLVED,
    rtol=1e-4,
    err_msg="point_source/source_plane: JAX vmap likelihood mismatch (solved)",
)


"""
__Path A: jit-wrap ``analysis.fit_from`` (Solved)__

Same narrowed ``except TypeError`` gate as the modelled-centre block above —
``fit_from`` returns a bare ``PointSolver`` at output component ``[1][1]``.
"""

model_solved_jit = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_solved)
)

instance_solved = model_solved_jit.instance_from_prior_medians()

analysis_solved_np = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSourceSolved,
    use_jax=False,
)
fit_solved_np = analysis_solved_np.fit_from(instance=instance_solved)
log_likelihood_solved_np = float(fit_solved_np.log_likelihood)
print("NumPy fit.log_likelihood (solved):", log_likelihood_solved_np)

EXPECTED_EAGER_LOG_LIKELIHOOD_SOURCE_PLANE_SOLVED = -94.70750992850252

np.testing.assert_allclose(
    log_likelihood_solved_np,
    EXPECTED_EAGER_LOG_LIKELIHOOD_SOURCE_PLANE_SOLVED,
    rtol=1e-4,
    err_msg=(
        f"point_source/source_plane: regression — eager log_likelihood (solved) "
        f"drifted (got {log_likelihood_solved_np}, expected "
        f"{EXPECTED_EAGER_LOG_LIKELIHOOD_SOURCE_PLANE_SOLVED})"
    ),
)
print(
    f"Eager regression assertion PASSED (solved): log_likelihood matches "
    f"{EXPECTED_EAGER_LOG_LIKELIHOOD_SOURCE_PLANE_SOLVED}"
)

# NumPy-vs-JAX vmap parity: the analytic solved fit removes the free-cosmology
# vs no-cosmology split as a large source of eager/vmap mismatch — check the
# two paths agree directly, not just against independent literals.
np.testing.assert_allclose(
    log_likelihood_solved_np,
    float(result_solved[0]),
    rtol=1e-4,
    err_msg="point_source/source_plane: solved eager vs vmap parity mismatch",
)

analysis_solved_jit = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSourceSolved,
    use_jax=True,
)
fit_solved_jit_fn = jax.jit(analysis_solved_jit.fit_from)

full_pipeline_jits_solved = False
try:
    fit_solved = fit_solved_jit_fn(instance_solved)
    print("JIT fit.log_likelihood (solved):", fit_solved.log_likelihood)
    assert isinstance(
        fit_solved.log_likelihood, jnp.ndarray
    ), f"expected jax.Array, got {type(fit_solved.log_likelihood)}"
    np.testing.assert_allclose(
        float(fit_solved.log_likelihood), log_likelihood_solved_np, rtol=1e-4
    )
    full_pipeline_jits_solved = True
    print("PASS: jit(fit_from) round-trip matches NumPy scalar (solved).")
except TypeError as e:
    print(
        "\nBLOCKER: source-plane jit(fit_from) (solved) is gated by:\n"
        f"  {type(e).__name__}: {e}\n"
        "  fit_from returns a bare PointSolver at output component [1][1],\n"
        "  which is not pytree-registered under jax.jit. Tracked in\n"
        "  PyAutoPrompt/autolens/fit_point_pytree.md.\n"
        "  Eager NumPy regression assertion still PASSED above."
    )
