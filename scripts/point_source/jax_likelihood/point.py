"""
Func Grad: Point Source Likelihood
===================================

Test that JAX can compute the log-likelihood of a ``PointDataset`` using the
**image-plane** chi-squared (``al.FitPositionsImagePairAll``) via
``AnalysisPoint``, exercising both the batched ``fitness._vmap`` likelihood
and the full ``jax.jit(analysis.fit_from)`` pipeline.

This script doubles as a documented walkthrough of the point-source modeling
API — see the ``__Point Solver__``, ``__Model__`` and ``__Name Pairing__``
sections below — built around the same ``FitPositionsImagePairAll`` variant
tested more tersely in ``point_source/jax_likelihood/image_plane.py``.

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
from jax import grad
from pathlib import Path

import autofit as af
import autolens as al
from autolens import conf


"""
__Dataset__

Load the strong lens point-source dataset `simple`, which is the dataset we will use to perform point source 
lens modeling.
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

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/point_source/simulators/simple.py"],
        check=True,
    )

"""
We now load the point source dataset we will fit using point source modeling.

We load this data as a `PointDataset`, which contains the positions of every point source.
"""
dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)

"""
__Point Solver__

For point-source modeling we require a `PointSolver`, which determines the multiple-images of the mass model for a 
point source at location (y,x) in the source plane. 

It does this by ray tracing triangles from the image-plane to the source-plane and calculating if the 
source-plane (y,x) centre is inside the triangle. The method gradually ray-traces smaller and smaller triangles so 
that the multiple images can be determine with sub-pixel precision.

The `PointSolver` requires a starting grid of (y,x) coordinates in the image-plane which defines the first set
of triangles that are ray-traced to the source-plane. It also requires that a `pixel_scale_precision` is input, 
which is the resolution up to which the multiple images are computed. The lower the `pixel_scale_precision`, the
longer the calculation, with the value of 0.001 below balancing efficiency with precision.

Strong lens mass models have a multiple image called the "central image". However, the image is nearly always 
significantly demagnified, meaning that it is not observed and cannot constrain the lens model. As this image is a
valid multiple image, the `PointSolver` will locate it irrespective of whether its so demagnified it is not observed.
To ensure this does not occur, we set a `magnification_threshold=0.1`, which discards this image because its
magnification will be well below this threshold.

If your dataset contains a central image that is observed you should reduce to include it in
the analysis.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 - The source galaxy's light is a point `Point` [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

__Name Pairing__

Every point-source dataset in the `PointDataset` has a name, which in this example was `point_0`. This `name` pairs 
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` it will raise an error.

In this example, where there is just one source, name pairing appears unnecessary. However, point-source datasets may
have many source galaxies in them, and name pairing is necessary to ensure every point source in the lens model is 
fitted to its particular lensed images in the `PointDataset`.

__Coordinates__

The model fitting default settings assume that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/modeling/imaging/customize/priors.py`).
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.centre.centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.8)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

point_0 = af.Model(al.ps.PointFlux)

point_0.centre.centre_0 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)
point_0.centre.centre_1 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)
# point_0.flux = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

# Overall Lens Model:

# Cosmology

cosmology = af.Model(al.cosmo.FlatLambdaCDM)

cosmology.H0 = af.UniformPrior(lower_limit=0.0, upper_limit=150.0)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), cosmology=cosmology
)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    # fit_positions_cls=al.FitPositionsSource,
    fit_positions_cls=al.FitPositionsImagePairAll,
)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 1

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector = jnp.array(model.physical_values_from_prior_medians)

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
    err_msg="point: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``analysis.fit_from``__

The Path A round-trip uses a model *without* free ``cosmology`` because the
cosmology distance calc in PyAutoGalaxy caches intermediate values in global
state, which triggers a JAX ``UnexpectedTracerError`` under ``jit`` even though
the vmap path above handles it fine. Once that library-level leak is fixed,
this block can reuse ``model`` directly.
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
centre-free ``FitPositionsImagePairAllSolved`` variant. This script is in
``smoke_tests.txt``, so only this one solved variant is added here — the
Repeat variant and the source-plane / fluxes+time-delays solved coverage live
in ``image_plane.py``, ``source_plane.py`` and ``fluxes_time_delays.py``.
"""

point_0_solved = af.Model(al.ps.PointSolved)

source_solved = af.Model(al.Galaxy, redshift=1.0, point_0=point_0_solved)

model_solved = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_solved), cosmology=cosmology
)

print(model_solved.info)

analysis_all_solved = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAllSolved,
)

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

EXPECTED_VMAP_LOG_LIKELIHOOD_POINT_ALL_SOLVED = -82.33883111

np.testing.assert_allclose(
    np.array(result_all_solved),
    EXPECTED_VMAP_LOG_LIKELIHOOD_POINT_ALL_SOLVED,
    rtol=1e-4,
    err_msg="point: JAX vmap likelihood mismatch (all solved)",
)


"""
__Path A: jit-wrap ``analysis.fit_from`` (FitPositionsImagePairAllSolved)__
"""

model_solved_jit = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_solved)
)

instance_solved = model_solved_jit.instance_from_prior_medians()

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
    float(fit_all_solved.log_likelihood),
    float(fit_all_solved_np.log_likelihood),
    rtol=1e-4,
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar (all solved).")
