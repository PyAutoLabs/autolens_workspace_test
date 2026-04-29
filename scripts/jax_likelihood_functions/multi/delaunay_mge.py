"""
Func Grad: Multi-Wavelength Delaunay + MGE Lens
===============================================
Tests that JAX can compute batched log-likelihoods and jit-wrap the
multi-wavelength ``FactorGraphModel`` for an MGE lens bulge + Delaunay
pixelization source (Hilbert image-mesh) across both g and r bands.

Uses **option B** — per-band source ``regularization.inner_coefficient``
priors via ``model.copy()`` + ``af.GaussianPrior`` on each ``AnalysisFactor``.
The MGE lens bulge, lens mass, shear, and mesh parameters remain shared.

Path A asserts ``vmap == JIT round-trip``; see ``rectangular.py`` for
the rationale (pixelized ``log_likelihood_function`` differs between
``use_jax=True`` and ``use_jax=False``).
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
    dataset.apply_over_sampling(over_sample_size_lp=1, over_sample_size_pixelization=1)
    for dataset in dataset_list
]

"""
__Image-Plane Mesh & Adapt Images (per band)__
"""
pixels = 500
edge_pixels_total = 30

adapt_images_list = []
for dataset, mask in zip(dataset_list, mask_list):
    galaxy_image_name_dict = {
        "('galaxies', 'lens')": dataset.data,
        "('galaxies', 'source')": dataset.data,
    }
    image_mesh = al.image_mesh.Hilbert(
        pixels=pixels, weight_power=3.5, weight_floor=0.01
    )
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
    )
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )
    adapt_images_list.append(
        al.AdaptImages(
            galaxy_name_image_dict=galaxy_image_name_dict,
            galaxy_name_image_plane_mesh_grid_dict={
                "('galaxies', 'source')": image_plane_mesh_grid
            },
        )
    )

"""
__Model__
"""
bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=10, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
mass.einstein_radius = af.UniformPrior(lower_limit=1.55, upper_limit=1.65)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.045, upper_limit=0.060)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=0.04, upper_limit=0.06)
shear.gamma_2 = af.UniformPrior(lower_limit=0.04, upper_limit=0.06)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(pixels=pixels, zeroed_pixels=edge_pixels_total),
    regularization=al.reg.AdaptSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__Per-band models (option B)__

Each band gets its own ``model.copy()`` with an independent prior on the
source regularization ``inner_coefficient``.
"""
model_per_band_list = []
for _ in waveband_list:
    model_analysis = model.copy()
    model_analysis.galaxies.source.pixelization.regularization.inner_coefficient = (
        af.GaussianPrior(mean=1.0, sigma=0.5)
    )
    model_per_band_list.append(model_analysis)

"""
__FactorGraphModel__
"""
analysis_list = [
    al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        raise_inversion_positions_likelihood_exception=False,
    )
    for dataset, adapt_images in zip(dataset_list, adapt_images_list)
]

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

EXPECTED_VMAP_LOG_LIKELIHOOD = -545.25327979

np.testing.assert_allclose(
    np.array(result),
    EXPECTED_VMAP_LOG_LIKELIHOOD,
    rtol=1e-4,
    err_msg="multi/delaunay_mge: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap parameter-vector entry point__
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(factor_graph.global_prior_model)


@jax.jit
def log_l_jit_fn(parameters):
    instance = factor_graph.global_prior_model.instance_from_vector(
        vector=parameters, xp=jnp
    )
    return factor_graph.log_likelihood_function(instance)


params_jit = jnp.array(factor_graph.global_prior_model.physical_values_from_prior_medians)
log_l_jit = log_l_jit_fn(params_jit)

print("JIT log_likelihood_function:", log_l_jit)
assert isinstance(log_l_jit, jnp.ndarray), (
    f"expected jax.Array, got {type(log_l_jit)}"
)
np.testing.assert_allclose(float(log_l_jit), EXPECTED_VMAP_LOG_LIKELIHOOD, rtol=1e-4)
print("PASS: jit(log_likelihood_function) round-trip matches vmap scalar.")
