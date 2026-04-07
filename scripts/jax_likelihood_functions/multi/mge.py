"""
Func Grad: Multi-Wavelength MGE
================================
Tests that JAX can compute batched log-likelihood evaluations for a multi-wavelength
imaging model using the `FactorGraphModel` API. Two imaging datasets (g and r bands)
are fitted simultaneously with a shared Isothermal+ExternalShear lens mass and shared
MGE source.
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

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
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

for dataset in dataset_list:
    dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

# Model: shared across both bands
bulge_lens = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=10, centre_prior_is_uniform=True
)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge_lens,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)

bulge_source = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=10, centre_prior_is_uniform=False
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge_source)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

# FactorGraphModel: same model instance for both analyses (fully shared parameters)
analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

analysis_factor_list = [
    af.AnalysisFactor(prior_model=model, analysis=analysis)
    for analysis in analysis_list
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

param_vector = jnp.array(
    factor_graph.global_prior_model.physical_values_from_prior_medians
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

np.testing.assert_allclose(
    np.array(result),
    -2174335.96508048,
    rtol=1e-4,
    err_msg="multi/mge: JAX vmap likelihood mismatch",
)

print("multi/mge.py checks passed.")
