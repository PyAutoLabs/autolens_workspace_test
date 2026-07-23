"""
Func Grad: Weak Lensing Shear Chi-Squared
=========================================

Test that JAX can compute the log-likelihood of a ``WeakDataset`` via ``al.AnalysisWeak``
with ``use_jax=True``: the fitness ``_vmap`` path (which registers the model pytrees and
forces tracer propagation) must match the eager NumPy likelihood exactly.

Covers the ``FitWeak`` xp-threaded statistics and the ``AnalysisWeak._register_fit_weak_pytrees``
registration added by PyAutoLens feature/weak-sigma-crit-jax (issue #590). The redshift
scale factors are concrete per-dataset constants, so a dataset carrying per-galaxy redshifts
exercises the eager-scaling + traced-statistics combination too.

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
from pathlib import Path

import autoarray as aa
import autofit as af
import autolens as al

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
mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.centre.centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.8)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

source = af.Model(al.Galaxy, redshift=1.0)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__NumPy reference__
"""
analysis_numpy = al.AnalysisWeak(dataset=dataset, use_jax=False)

instance = model.instance_from_prior_medians()

log_likelihood_numpy = analysis_numpy.log_likelihood_function(instance=instance)

print(f"NumPy log likelihood : {log_likelihood_numpy}")

"""
__JAX vmap parity__
"""
from autofit.non_linear.fitness import Fitness
import time

analysis = al.AnalysisWeak(dataset=dataset, use_jax=True)

batch_size = 2

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
result = fitness._vmap(parameters)
print(f"JAX vmap log likelihood : {np.array(result)}")
print("JAX Time To VMAP + JIT Function", time.time() - start)

np.testing.assert_allclose(
    np.array(result),
    log_likelihood_numpy,
    rtol=1e-6,
    err_msg="weak/shear: JAX vmap likelihood mismatch vs eager NumPy",
)

"""
__Redshift-scaled variant__

Per-galaxy sigma_crit scaling: the scale factors are concrete constants computed before any
trace, so the scaled likelihood must also match between NumPy and vmap paths.
"""
redshifts = list(np.random.default_rng(2).uniform(0.6, 2.0, dataset.n_galaxies))
dataset_scaled = al.WeakDataset(
    shear_yx=dataset.shear_yx,
    noise_map=dataset.noise_map,
    name="simple_scaled",
    redshifts=redshifts,
)

analysis_scaled_numpy = al.AnalysisWeak(dataset=dataset_scaled, use_jax=False)
log_likelihood_scaled_numpy = analysis_scaled_numpy.log_likelihood_function(
    instance=instance
)

analysis_scaled = al.AnalysisWeak(dataset=dataset_scaled, use_jax=True)
fitness_scaled = Fitness(
    model=model,
    analysis=analysis_scaled,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)
result_scaled = fitness_scaled._vmap(parameters)

print(f"NumPy scaled log likelihood : {log_likelihood_scaled_numpy}")
print(f"JAX scaled log likelihood   : {np.array(result_scaled)}")

np.testing.assert_allclose(
    np.array(result_scaled),
    log_likelihood_scaled_numpy,
    rtol=1e-6,
    err_msg="weak/shear: scaled JAX vmap likelihood mismatch vs eager NumPy",
)

assert not np.isclose(
    log_likelihood_scaled_numpy, log_likelihood_numpy
), "scaling changed nothing — redshift scale factors not applied"

print("WEAK JAX PARITY PASSED")
