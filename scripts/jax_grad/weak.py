"""
Tests JAX gradients of the weak-lensing shear log-likelihood (``FitWeak`` via
``al.AnalysisWeak(use_jax=True)``), in two stages:

 1. Finiteness — ``jax.value_and_grad`` returns a finite log-likelihood and a
    finite, non-zero gradient vector.
 2. Correctness — the autodiff gradient agrees with central finite differences
    parameter-by-parameter (see ``util.py``).

The model shear is derived from the tracer's mass profiles via
``LensCalc.shear_yx_2d_via_hessian_from``, so the gradient flows through the
Hessian of the deflection field. Both the plain likelihood and the per-galaxy
redshift-scaled (sigma_crit scaling) variant are checked — the scale factors
are concrete constants computed outside the trace, so they must not disturb
gradient flow.

Setup mirrors ``scripts/jax_likelihood_functions/weak/shear.py`` (the value
parity script for PyAutoLens feature/weak-sigma-crit-jax, issue #590).
"""

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
dataset_path = Path("dataset") / "weak" / "simple"

"""
__Dataset Auto-Simulation__
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/weak/simulator.py"],
        check=True,
    )

dataset = al.from_json(file_path=dataset_path / "dataset.json")

"""
__Model__
"""


def model_from():
    mass = af.Model(al.mp.Isothermal)

    mass.centre.centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
    mass.centre.centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
    mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
    mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
    mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.8)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    source = af.Model(al.Galaxy, redshift=1.0)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Redshift-scaled variant dataset__

Per-galaxy sigma_crit scaling: concrete per-dataset constants computed before any trace.
"""
redshifts = list(np.random.default_rng(2).uniform(0.6, 2.0, dataset.n_galaxies))
dataset_scaled = al.WeakDataset(
    shear_yx=dataset.shear_yx,
    noise_map=dataset.noise_map,
    name="simple_scaled",
    redshifts=redshifts,
)

from autofit.non_linear.fitness import Fitness

for variant, dataset_variant in [
    ("weak shear", dataset),
    ("weak shear, redshift-scaled", dataset_scaled),
]:
    print(f"\n=== {variant} ===")

    model = model_from()

    analysis = al.AnalysisWeak(dataset=dataset_variant, use_jax=True)

    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    param_vector = jnp.array(model.physical_values_from_prior_medians)

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
    """
    f_jit = jax.jit(fitness.call)

    util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

    comparison = util.compare_gradients(
        fitness.call,
        param_vector,
        param_names=util.parameter_names_from(model),
        f_fd=f_jit,
    )

    util.assert_gradients_match(comparison)

    print(f"{variant}: autodiff matches finite differences.")

print("\nweak.py JAX gradient checks passed.")
