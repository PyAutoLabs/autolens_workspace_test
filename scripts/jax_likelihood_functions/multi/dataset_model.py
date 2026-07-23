"""
JAX Likelihood: Multi-Wavelength with DatasetModel Offsets
===========================================================
Tests that JAX can compute batched log-likelihoods and jit-wrap
``factor_graph.log_likelihood_function`` for a multi-wavelength imaging model
that includes an ``al.DatasetModel`` on every dataset after the first to
shift its grid relative to the reference dataset.

The first dataset (``i == 0``) keeps the ``DatasetModel`` defaults — fixed
``grid_offset = (0.0, 0.0)``. Every subsequent dataset has its
``grid_offset.grid_offset_0/1`` replaced by ``af.UniformPrior(-1.0, 1.0)``,
making the offset a free parameter that JAX must trace through cleanly.

Why this test exists: ``DatasetModel`` is a plain autoarray class with no
explicit pytree registration. ``register_model`` walks the factor graph and
builds generic flatten/unflatten functions for every ``Model.cls`` it finds,
so it should pick up ``DatasetModel`` automatically. The mixed registration
(i=0 stores ``grid_offset`` as a constant Python tuple while i>0 stores it
as a ``TuplePrior``) is a subtle stress on the classifier's ``setdefault``
behaviour — this script is the regression marker for that path.

Path layout follows ``multi/lp.py``: vmap evaluation through
``fitness._vmap`` and then a separate ``jax.jit`` wrap around
``instance_from_vector → log_likelihood_function``.
"""

"""
__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path
import autofit as af
import autolens as al

waveband_list = ["g", "r"]
pixel_scales = 0.1
mask_radius = 3.0

dataset_path = path.join("dataset", "multi", "lens_sersic")

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/multi/simulator.py"],
        check=True,
    )

dataset_list = [
    al.Imaging.from_fits(
        data_path=path.join(dataset_path, f"{band}_data.fits"),
        psf_path=path.join(dataset_path, f"{band}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{band}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for band in waveband_list
]

mask_list = [
    al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    for dataset in dataset_list
]

dataset_list = [
    dataset.apply_mask(mask=mask) for dataset, mask in zip(dataset_list, mask_list)
]

for dataset in dataset_list:
    dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

"""
__Model__
"""
bulge = af.Model(al.lp_linear.Sersic)
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

source_bulge = af.Model(al.lp_linear.Sersic)
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

dataset_model = af.Model(al.DatasetModel)

model = af.Collection(
    dataset_model=dataset_model,
    galaxies=af.Collection(lens=lens, source=source),
)

print(model.info)

"""
__Per-band models__

The first band keeps the ``DatasetModel`` defaults (fixed ``grid_offset =
(0.0, 0.0)``). Every later band gets a free 2D offset prior. The source's
``ell_comps`` are also per-band as in ``multi/lp.py`` to keep this script
structurally parallel.
"""
model_per_band_list = []
for i in range(len(waveband_list)):
    model_analysis = model.copy()
    model_analysis.galaxies.source.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
        mean=0.0, sigma=0.5
    )
    model_analysis.galaxies.source.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
        mean=0.0, sigma=0.5
    )
    if i > 0:
        model_analysis.dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
            lower_limit=-1.0, upper_limit=1.0
        )
        model_analysis.dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
            lower_limit=-1.0, upper_limit=1.0
        )
    model_per_band_list.append(model_analysis)

"""
__FactorGraphModel (vmap path)__
"""
analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

analysis_factor_list = [
    af.AnalysisFactor(prior_model=m, analysis=analysis)
    for m, analysis in zip(model_per_band_list, analysis_list)
]

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

print(factor_graph.global_prior_model.info)

from autofit.non_linear.fitness import Fitness
import time

batch_size = 3

fitness = Fitness(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

parameters = np.zeros(
    (batch_size, factor_graph.global_prior_model.total_free_parameters)
)
for i in range(batch_size):
    parameters[i, :] = (
        factor_graph.global_prior_model.physical_values_from_prior_medians
    )
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

EXPECTED_VMAP_LOG_LIKELIHOOD = -2699469.80625228

np.testing.assert_allclose(
    np.array(result),
    EXPECTED_VMAP_LOG_LIKELIHOOD,
    rtol=1e-4,
    err_msg="multi/dataset_model: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``factor_graph.log_likelihood_function``__

Same shape as ``multi/lp.py``: build a parallel NumPy ``FactorGraphModel`` to
get a reference scalar from the un-jitted ``log_likelihood_function``, then
build a JAX-enabled twin and ``jax.jit`` a parameter-vector entry point.
``register_model`` is what registers ``DatasetModel`` (along with every other
class in the model tree) as a JAX pytree.
"""


analysis_np_list = [
    al.AnalysisImaging(dataset=dataset, use_jax=False) for dataset in dataset_list
]
analysis_factor_np_list = [
    af.AnalysisFactor(prior_model=m, analysis=a)
    for m, a in zip(model_per_band_list, analysis_np_list)
]
factor_graph_np = af.FactorGraphModel(*analysis_factor_np_list, use_jax=False)

params_np = np.array(
    factor_graph_np.global_prior_model.physical_values_from_prior_medians
)
instance_np = factor_graph_np.global_prior_model.instance_from_vector(
    vector=params_np, xp=np
)
log_l_np = float(factor_graph_np.log_likelihood_function(instance_np))
print("NumPy log_likelihood_function:", log_l_np)

analysis_jit_list = [
    al.AnalysisImaging(dataset=dataset, use_jax=True) for dataset in dataset_list
]
analysis_factor_jit_list = [
    af.AnalysisFactor(prior_model=m, analysis=a)
    for m, a in zip(model_per_band_list, analysis_jit_list)
]
factor_graph_jit = af.FactorGraphModel(*analysis_factor_jit_list, use_jax=True)


@jax.jit
def log_l_jit_fn(parameters):
    instance = factor_graph_jit.global_prior_model.instance_from_vector(
        vector=parameters, xp=jnp
    )
    return factor_graph_jit.log_likelihood_function(instance)


params_jit = jnp.array(
    factor_graph_jit.global_prior_model.physical_values_from_prior_medians
)
log_l_jit = log_l_jit_fn(params_jit)

print("JIT log_likelihood_function:", log_l_jit)
assert isinstance(log_l_jit, jnp.ndarray), f"expected jax.Array, got {type(log_l_jit)}"
np.testing.assert_allclose(float(log_l_jit), log_l_np, rtol=1e-4)
print("PASS: jit(log_likelihood_function) round-trip matches NumPy scalar.")
