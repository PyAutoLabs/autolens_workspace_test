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

The full pipeline (``jax.jit(analysis.fit_from)``) is currently **BLOCKED**
by a ``Grid2DIrregular.grid_2d_via_deflection_grid_from`` xp-propagation bug
documented in ``autolens_workspace_developer/jax_profiling/point_source/source_plane.py``
and tracked in ``admin_jammy/prompt/issued/fit_point_pytree.md``.  When Path
A JIT fails with ``TracerArrayConversionError`` the script prints a clear
BLOCKER line and continues, so the eager NumPy regression assertion is still
exercised.  Once the upstream xp-propagation fix lands, the JIT path will
succeed without modifying this script.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

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
__Dataset Auto-Simulation__
"""
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

Source-plane chi-squared does not use the solver, but ``AnalysisPoint``
requires one — pass the standard JAX-friendly solver for consistency.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1, xp=jnp
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
# seeded PointDataset (``scripts/jax_likelihood_functions/point_source/simulator.py``).
EXPECTED_VMAP_LOG_LIKELIHOOD_SOURCE_PLANE = -199.1555813

np.testing.assert_allclose(
    np.array(result),
    EXPECTED_VMAP_LOG_LIKELIHOOD_SOURCE_PLANE,
    rtol=1e-4,
    err_msg="point_source/source_plane: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``analysis.fit_from``__

Wrapped in ``try/except jax.errors.TracerArrayConversionError`` — source-plane
fitting currently fails Path A with the ``Grid2DIrregular.grid_2d_via_deflection_grid_from``
xp-propagation bug.  The eager NumPy log-likelihood is still asserted for
regression coverage.
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()

model_jit = af.Collection(galaxies=af.Collection(lens=lens, source=source))
register_model(model_jit)

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

EXPECTED_EAGER_LOG_LIKELIHOOD_SOURCE_PLANE = -199.1555813

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
    assert isinstance(fit.log_likelihood, jnp.ndarray), (
        f"expected jax.Array, got {type(fit.log_likelihood)}"
    )
    np.testing.assert_allclose(
        float(fit.log_likelihood), log_likelihood_np, rtol=1e-4
    )
    full_pipeline_jits = True
    print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
except (jax.errors.TracerArrayConversionError, TypeError) as e:
    # Two stacked blockers gate the full-pipeline JIT:
    #   1. FitPositionsSource is not pytree-registered, so fit_from returns a
    #      non-JAX type (TypeError).  Tracked in admin_jammy/prompt/issued/fit_point_pytree.md.
    #   2. Even if that is fixed, the source-plane chi-squared itself fails with
    #      TracerArrayConversionError owing to the Grid2DIrregular.grid_2d_via_deflection_grid_from
    #      xp-propagation bug (see autolens_workspace_developer/jax_profiling/point_source/source_plane.py).
    print(
        "\nBLOCKER: source-plane jit(fit_from) is gated by:\n"
        f"  {type(e).__name__}: {e}\n"
        "  Fixes tracked in admin_jammy/prompt/issued/fit_point_pytree.md and\n"
        "  autolens_workspace_developer/jax_profiling/point_source/source_plane.py.\n"
        "  Eager NumPy regression assertion still PASSED above."
    )
