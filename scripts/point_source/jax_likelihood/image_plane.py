"""
Func Grad: Point Source Image-Plane Chi-Squared
================================================

Test that JAX can compute the log-likelihood of a ``PointDataset`` using the
**image-plane** chi-squared (``al.FitPositionsImagePairAll``).

Image-plane fitting solves for the model multiple-image positions via the
``PointSolver`` (which JIT-traces a triangle-refinement loop), pairs each model
image with the closest observed image, and computes a chi-squared in
image-plane coordinates.

This variant is known to JIT end-to-end (see
``autolens_workspace_developer/jax_profiling/point_source/image_plane.py``),
so ``jax.jit(analysis.fit_from)`` succeeds without falling back to a prefix.

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
        [sys.executable, "scripts/point_source/simulators/simple.py"],
        check=True,
    )

dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)

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
    fit_positions_cls=al.FitPositionsImagePairAll,
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

np.testing.assert_allclose(
    np.array(result),
    -83.38049778,
    rtol=1e-4,
    err_msg="point_source/image_plane: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``analysis.fit_from``__

The Path A round-trip uses a model *without* free ``cosmology`` (same caveat
as ``point.py``): the cosmology distance calc caches intermediate values in
global state, triggering ``UnexpectedTracerError`` under ``jit``.
"""


model_jit = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model_jit.instance_from_prior_medians()

analysis_np = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=True,
)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(
    fit.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit.log_likelihood)}"
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")


"""
__Model: Solved Source (Parameter-Free)__

Swaps the source for parameter-free ``al.ps.PointSolved`` to exercise the
centre-free image-plane variants below.
"""

point_0_solved = af.Model(al.ps.PointSolved)

source_solved = af.Model(al.Galaxy, redshift=1.0, point_0=point_0_solved)

model_solved = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_solved), cosmology=cosmology
)

print(model_solved.info)

model_solved_jit = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_solved)
)
instance_solved = model_solved_jit.instance_from_prior_medians()

"""
__Analysis: FitPositionsImagePairAllSolved__
"""
analysis_all_solved = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAllSolved,
)

from autofit.non_linear.fitness import Fitness
import time

batch_size = 1

fitness_all_solved = Fitness(
    model=model_solved,
    analysis=analysis_all_solved,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

parameters_all_solved = np.zeros((batch_size, model_solved.total_free_parameters))
for i in range(batch_size):
    parameters_all_solved[i, :] = model_solved.physical_values_from_prior_medians
parameters_all_solved = jnp.array(parameters_all_solved)

start = time.time()
print()
print(fitness_all_solved._vmap(parameters_all_solved))
print("JAX Time To VMAP + JIT Function", time.time() - start)

start = time.time()
print()
result_all_solved = fitness_all_solved._vmap(parameters_all_solved)
print(result_all_solved)
print("JAX Time Taken using VMAP:", time.time() - start)
print("JAX Time Taken per Likelihood:", (time.time() - start) / batch_size)

EXPECTED_VMAP_LOG_LIKELIHOOD_IMAGE_PLANE_ALL_SOLVED = -82.33883111

np.testing.assert_allclose(
    np.array(result_all_solved),
    EXPECTED_VMAP_LOG_LIKELIHOOD_IMAGE_PLANE_ALL_SOLVED,
    rtol=1e-4,
    err_msg="point_source/image_plane: JAX vmap likelihood mismatch (all solved)",
)


"""
__Path A: jit-wrap ``analysis.fit_from`` (FitPositionsImagePairAllSolved)__

Expected to JIT end-to-end exactly like the modelled-centre ``FitPositionsImagePairAll``
block above — no try/except; a failure here is a real regression.
"""
analysis_all_solved_np = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAllSolved,
    use_jax=False,
)
fit_all_solved_np = analysis_all_solved_np.fit_from(instance=instance_solved)
print("NumPy fit.log_likelihood (all solved):", float(fit_all_solved_np.log_likelihood))

analysis_all_solved_jit = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAllSolved,
    use_jax=True,
)
fit_all_solved_jit_fn = jax.jit(analysis_all_solved_jit.fit_from)
fit_all_solved = fit_all_solved_jit_fn(instance_solved)

print("JIT fit.log_likelihood (all solved):", fit_all_solved.log_likelihood)
assert isinstance(
    fit_all_solved.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit_all_solved.log_likelihood)}"
np.testing.assert_allclose(
    float(fit_all_solved.log_likelihood), float(fit_all_solved_np.log_likelihood), rtol=1e-4
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar (all solved).")


"""
__Analysis: FitPositionsImagePairRepeatSolved__
"""
analysis_repeat_solved = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeatSolved,
)

fitness_repeat_solved = Fitness(
    model=model_solved,
    analysis=analysis_repeat_solved,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

parameters_repeat_solved = np.zeros((batch_size, model_solved.total_free_parameters))
for i in range(batch_size):
    parameters_repeat_solved[i, :] = model_solved.physical_values_from_prior_medians
parameters_repeat_solved = jnp.array(parameters_repeat_solved)

start = time.time()
print()
print(fitness_repeat_solved._vmap(parameters_repeat_solved))
print("JAX Time To VMAP + JIT Function", time.time() - start)

start = time.time()
print()
result_repeat_solved = fitness_repeat_solved._vmap(parameters_repeat_solved)
print(result_repeat_solved)
print("JAX Time Taken using VMAP:", time.time() - start)
print("JAX Time Taken per Likelihood:", (time.time() - start) / batch_size)

EXPECTED_VMAP_LOG_LIKELIHOOD_IMAGE_PLANE_REPEAT_SOLVED = -89.71129442

np.testing.assert_allclose(
    np.array(result_repeat_solved),
    EXPECTED_VMAP_LOG_LIKELIHOOD_IMAGE_PLANE_REPEAT_SOLVED,
    rtol=1e-4,
    err_msg="point_source/image_plane: JAX vmap likelihood mismatch (repeat solved)",
)


"""
__Path A: jit-wrap ``analysis.fit_from`` (FitPositionsImagePairRepeatSolved)__

Expected to JIT end-to-end exactly like the modelled-centre ``FitPositionsImagePairAll``
block above — no try/except; a failure here is a real regression.
"""
analysis_repeat_solved_np = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeatSolved,
    use_jax=False,
)
fit_repeat_solved_np = analysis_repeat_solved_np.fit_from(instance=instance_solved)
print(
    "NumPy fit.log_likelihood (repeat solved):",
    float(fit_repeat_solved_np.log_likelihood),
)

analysis_repeat_solved_jit = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeatSolved,
    use_jax=True,
)
fit_repeat_solved_jit_fn = jax.jit(analysis_repeat_solved_jit.fit_from)
fit_repeat_solved = fit_repeat_solved_jit_fn(instance_solved)

print("JIT fit.log_likelihood (repeat solved):", fit_repeat_solved.log_likelihood)
assert isinstance(
    fit_repeat_solved.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit_repeat_solved.log_likelihood)}"
np.testing.assert_allclose(
    float(fit_repeat_solved.log_likelihood),
    float(fit_repeat_solved_np.log_likelihood),
    rtol=1e-4,
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar (repeat solved).")
