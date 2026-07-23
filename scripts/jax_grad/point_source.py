"""
Tests JAX gradients of the point-source **source-plane** chi-squared
(``al.FitPositionsSource``), in two stages:

 1. Finiteness — ``jax.value_and_grad`` returns a finite log-likelihood and a
    finite, non-zero gradient vector.
 2. Correctness — the autodiff gradient agrees with central finite differences
    parameter-by-parameter (see ``util.py``).

The source-plane likelihood ray-traces the observed image-plane positions to
the source plane and penalises their distance to the modelled source centre,
weighted by per-position magnifications computed from the Hessian of the
deflection field — so this validates second-derivative flow through the mass
profiles.

Notable: the forward ``jax.jit`` of this likelihood is blocked
(``Grid2DIrregular.grid_2d_via_deflection_grid_from`` does not propagate
``xp``), but ``jax.value_and_grad`` succeeds — the gradient path is usable by
samplers today even though the jitted forward pass is not. Because of that,
the finite-difference sweep here uses the eager likelihood (no jit), unlike
the imaging scripts.

Parameters with no positional information (``point_0.flux``, ``H0``) have
legitimately zero gradients — both autodiff and finite differences agree on
zero, and the script asserts the positional parameters are live.

Setup mirrors ``scripts/jax_likelihood_functions/point_source/source_plane.py``.
"""
# ENV: jax full_datasets
# Drive jax.value_and_grad + finite-difference gradient checks;
# need JAX enabled and full-resolution float64 data.

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

import autofit as af
import autolens as al

import util

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

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector = jnp.array(model.physical_values_from_prior_medians)

key = jax.random.PRNGKey(42)
perturbation = jax.random.uniform(
    key, shape=param_vector.shape, minval=0.001, maxval=0.005
)
param_vector = param_vector + perturbation

value, grad = jax.value_and_grad(fitness.call)(param_vector)

print(f"Log likelihood = {float(value):.6f}")
print(f"Gradient shape = {grad.shape}")

assert np.isfinite(float(value)), "Log likelihood is not finite"
assert grad.shape == (
    model.total_free_parameters,
), f"Gradient shape mismatch: {grad.shape}"
assert np.all(
    np.isfinite(np.array(grad))
), f"Gradient contains non-finite values: {np.array(grad)}"
assert not np.all(np.array(grad) == 0.0), "Gradient is all zeros"

"""
__Finite-Difference Correctness__

Eager evaluation throughout — the forward jit of this likelihood is blocked
(see module docstring), which the FD sweep tolerates because the positions
dataset is tiny.
"""
param_names = util.parameter_names_from(model)

comparison = util.compare_gradients(
    fitness.call,
    param_vector,
    param_names=param_names,
)

util.assert_gradients_match(comparison)

# The positional parameters (everything except flux and H0) must be live.
positional_indices = [
    i for i, name in enumerate(param_names) if "flux" not in name and "H0" not in name
]
assert np.all(
    np.abs(comparison["ad"][positional_indices]) > 0.0
), "A positional parameter has zero gradient — evaluation point is degenerate."

print("point_source.py JAX gradient checks passed.")
