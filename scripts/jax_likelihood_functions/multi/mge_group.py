"""
Func Grad: Multi-Wavelength MGE Group
=====================================
Tests that JAX can compute batched log-likelihoods and jit-wrap the
multi-wavelength ``FactorGraphModel`` for an MGE lens bulge + MGE source +
extra galaxies across both g and r bands.

Uses **option B** — per-band source MGE ``ell_comps`` priors via
``model.copy()`` + ``af.GaussianPrior`` on each ``AnalysisFactor``. The lens
MGE bulge, lens mass, shear, and extra-galaxy parameters remain shared across
the g and r bands.

Path A uses ``jax.jit`` on a parameter-vector entry point that mirrors
``fitness._vmap`` (``instance_from_vector`` → ``log_likelihood_function``),
because ``FactorGraphModel`` has no ``fit_from`` method.
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

"""
__Dataset Auto-Simulation__
"""
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

dataset_list = [
    dataset.apply_over_sampling(over_sample_size_lp=1) for dataset in dataset_list
]

"""
__Group Centres__

The multi simulator does not include extra galaxies, so the extra-galaxy
components here have no data support. They still exercise the MGE +
``extra_galaxies`` wiring through the JAX factor graph.
"""
centre_list = [(0.0, 1.0), (1.0, 0.0)]

"""
__Model__
"""
# Lens MGE bulge
total_gaussians = 20
gaussian_per_basis = 2
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []
for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )
    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]
    bulge_gaussian_list += gaussian_list

bulge = af.Model(al.lp_basis.Basis, profile_list=bulge_gaussian_list)

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Source MGE bulge
total_gaussians = 20
gaussian_per_basis = 1
log10_sigma_list = np.linspace(-2, np.log10(1.0), total_gaussians)

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

source_gaussian_list = []
for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )
    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]
    source_gaussian_list += gaussian_list

source_bulge = af.Model(al.lp_basis.Basis, profile_list=source_gaussian_list)
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Extra galaxies
extra_galaxies_list = []
for extra_galaxy_centre in centre_list:
    total_gaussians = 8
    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    eg_gaussian_list = af.Collection(
        af.Model(al.lp_linear.GaussianSph) for _ in range(total_gaussians)
    )
    for i, gaussian in enumerate(eg_gaussian_list):
        gaussian.centre.centre_0 = extra_galaxy_centre[0]
        gaussian.centre.centre_1 = extra_galaxy_centre[1]
        gaussian.sigma = 10 ** log10_sigma_list[i]

    extra_galaxy_bulge = af.Model(al.lp_basis.Basis, profile_list=eg_gaussian_list)

    eg_mass = af.Model(al.mp.IsothermalSph)
    eg_mass.centre = extra_galaxy_centre
    eg_mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    extra_galaxy = af.Model(
        al.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge, mass=eg_mass
    )
    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), extra_galaxies=extra_galaxies
)

"""
__Per-band models (option B)__

Each band gets its own ``model.copy()`` with independent source-MGE ``ell_comps``
priors. All gaussians within the source Basis share one ell_comps prior pair
per basis (the model helper ties them together); we re-tie them to a fresh
pair per factor so each band's source can take a different shape. The lens MGE
and extra galaxies stay shared.
"""
model_per_band_list = []
for _ in waveband_list:
    model_analysis = model.copy()
    ec_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    ec_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    for gaussian in model_analysis.galaxies.source.bulge.profile_list:
        gaussian.ell_comps.ell_comps_0 = ec_0
        gaussian.ell_comps.ell_comps_1 = ec_1
    model_per_band_list.append(model_analysis)

"""
__FactorGraphModel__
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

EXPECTED_VMAP_LOG_LIKELIHOOD = -2088049.36654626

np.testing.assert_allclose(
    np.array(result),
    EXPECTED_VMAP_LOG_LIKELIHOOD,
    rtol=1e-4,
    err_msg="multi/mge_group: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap parameter-vector entry point__
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(factor_graph.global_prior_model)

analysis_np_list = [
    al.AnalysisImaging(dataset=dataset, use_jax=False) for dataset in dataset_list
]
factor_graph_np = af.FactorGraphModel(
    *[
        af.AnalysisFactor(prior_model=m, analysis=a)
        for m, a in zip(model_per_band_list, analysis_np_list)
    ],
    use_jax=False,
)

params_np = np.array(factor_graph_np.global_prior_model.physical_values_from_prior_medians)
instance_np = factor_graph_np.global_prior_model.instance_from_vector(
    vector=params_np, xp=np
)
log_l_np = float(factor_graph_np.log_likelihood_function(instance_np))
print("NumPy log_likelihood_function:", log_l_np)

analysis_jit_list = [
    al.AnalysisImaging(dataset=dataset, use_jax=True) for dataset in dataset_list
]
factor_graph_jit = af.FactorGraphModel(
    *[
        af.AnalysisFactor(prior_model=m, analysis=a)
        for m, a in zip(model_per_band_list, analysis_jit_list)
    ],
    use_jax=True,
)


@jax.jit
def log_l_jit_fn(parameters):
    instance = factor_graph_jit.global_prior_model.instance_from_vector(
        vector=parameters, xp=jnp
    )
    return factor_graph_jit.log_likelihood_function(instance)


params_jit = jnp.array(factor_graph_jit.global_prior_model.physical_values_from_prior_medians)
log_l_jit = log_l_jit_fn(params_jit)

print("JIT log_likelihood_function:", log_l_jit)
assert isinstance(log_l_jit, jnp.ndarray), (
    f"expected jax.Array, got {type(log_l_jit)}"
)
np.testing.assert_allclose(float(log_l_jit), log_l_np, rtol=1e-4)
print("PASS: jit(log_likelihood_function) round-trip matches NumPy scalar.")
