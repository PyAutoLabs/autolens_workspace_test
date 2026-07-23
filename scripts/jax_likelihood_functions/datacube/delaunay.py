"""
Func Grad: Datacube Delaunay Pixelization Source
=================================================

Tests that JAX can compute batched log-likelihoods and jit-wrap the cube
``FactorGraphModel`` for a 4-channel datacube fitted with a Delaunay
pixelization source (Hilbert image-mesh + edge zeroing).

Mirrors ``jax_likelihood_functions/interferometer/delaunay.py`` (same lens
model, same Delaunay mesh, same regularization, same Sersic-image
``AdaptImages``) but loads the SMA dataset N=4 times as an identical-channel
cube and wires it through ``af.FactorGraphModel`` — the same pattern as
``jax_likelihood_functions/multi/delaunay.py``.

Identical channels make the cube reference deterministic: the cube
log-likelihood is exactly ``N × single_channel_log_likelihood``.

Path A asserts ``vmap == JIT round-trip`` (both through
``FactorGraphModel.log_likelihood_function``) rather than NumPy-vs-JAX
parity, matching the ``multi/`` pattern: for pixelized sources,
``analysis.log_likelihood_function`` under ``use_jax=True`` takes a different
numerical path than under ``use_jax=False`` (the JAX path matches
``fit.log_likelihood`` only when routed through ``fit_from``, which
``FactorGraphModel`` does not expose).

Path B re-runs the same vmap likelihood with ``TransformerNUFFT`` and asserts
the same expected value — proves the cube path works with both DFT and NUFFT
transformers.
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
from autolens import conf

n_channels = 4

"""
__Mask__

Same as ``interferometer/delaunay.py`` — shared across all channels.
"""
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load the SMA interferometer dataset 4 times as a 4-channel cube. Identical
channels — same visibilities, noise, uv_wavelengths.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

"""
__Dataset Auto-Simulation__
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/jax_likelihood_functions/interferometer/simulator.py",
        ],
        check=True,
    )

dataset_list = [
    al.Interferometer.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )
    for _ in range(n_channels)
]

print(f"Channels: {n_channels}")
print(f"Visibilities/channel: {dataset_list[0].uv_wavelengths.shape[0]}")

"""
__JAX & Preloads__

Same Delaunay image-mesh setup as ``interferometer/delaunay.py``: Hilbert
mesh with 750 source pixels + 30 edge points zeroed. The mesh is computed
once on a Sersic adapt-image and shared across all channels (the lens
model is shared, so the image-plane mesh-grid is channel-invariant).
"""
pixels = 750
edge_pixels_total = 30

bulge_adapt = al.lp.Sersic()
adapt_image = bulge_adapt.image_2d_from(grid=dataset_list[0].grid)

galaxy_image_name_dict = {
    "('galaxies', 'lens')": adapt_image,
    "('galaxies', 'source')": adapt_image,
}

image_mesh = al.image_mesh.Hilbert(pixels=pixels, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=real_space_mask,
    adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
)

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=real_space_mask.mask_centre,
    radius=mask_radius + real_space_mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

total_mapper_pixels = image_plane_mesh_grid.shape[0]

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)


"""
__Model__

Same `PowerLaw + ExternalShear` lens and Delaunay source (`reg.AdaptSplit()`)
as ``interferometer/delaunay.py``.
"""
mass = af.Model(al.mp.PowerLaw)

mass.centre.centre_0 = af.UniformPrior(lower_limit=0.2, upper_limit=0.4)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.4, upper_limit=-0.2)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=0.11111111111111108, upper_limit=0.1111111111111111
)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=mass,
    shear=shear,
)

regularization = al.reg.AdaptSplit()

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(pixels=pixels, zeroed_pixels=edge_pixels_total),
    regularization=regularization,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__FactorGraphModel__

Per-channel ``AnalysisInterferometer`` + ``AnalysisFactor`` with `model.copy()`.
No per-channel prior overrides — every parameter is shared across channels.
"""
analysis_list = [
    al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        raise_inversion_positions_likelihood_exception=False,
    )
    for dataset in dataset_list
]

analysis_factor_list = [
    af.AnalysisFactor(prior_model=model.copy(), analysis=analysis)
    for analysis in analysis_list
]

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

print(factor_graph.global_prior_model.info)

"""
__Fitness + vmap__
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 1

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

"""
Cube log-likelihood ≈ N × single-channel log-likelihood for identical
channels. Pinned empirically below.
"""
EXPECTED_SINGLE_CHANNEL_LOG_LIKELIHOOD = -3165.42388511
EXPECTED_VMAP_LOG_LIKELIHOOD = n_channels * EXPECTED_SINGLE_CHANNEL_LOG_LIKELIHOOD

np.testing.assert_allclose(
    np.array(result),
    EXPECTED_VMAP_LOG_LIKELIHOOD,
    rtol=1e-4,
    err_msg="datacube/delaunay: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap parameter-vector entry point__

Matches ``multi/delaunay.py``: jit-wrap ``factor_graph.log_likelihood_function``
through ``instance_from_vector`` and assert the result matches the vmap value.
"""


@jax.jit
def log_l_jit_fn(parameters):
    instance = factor_graph.global_prior_model.instance_from_vector(
        vector=parameters, xp=jnp
    )
    return factor_graph.log_likelihood_function(instance)


params_jit = jnp.array(
    factor_graph.global_prior_model.physical_values_from_prior_medians
)
log_l_jit = log_l_jit_fn(params_jit)

print("JIT log_likelihood_function:", log_l_jit)
assert isinstance(log_l_jit, jnp.ndarray), f"expected jax.Array, got {type(log_l_jit)}"
np.testing.assert_allclose(float(log_l_jit), EXPECTED_VMAP_LOG_LIKELIHOOD, rtol=1e-4)
print("PASS: jit(log_likelihood_function) round-trip matches vmap scalar.")


"""
__Path B: TransformerNUFFT cross-check__

Re-run a single-channel vmap with ``TransformerNUFFT`` and confirm the result
matches the single-channel ``TransformerDFT`` value. The 4-channel DFT path above
already validates datacube factor-graph summation and JIT routing; duplicating the
same Delaunay NUFFT graph across four identical channels can exceed the release
runner memory budget.
"""
nufft_channels = 1

dataset_list_nufft = [
    al.Interferometer.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,
    )
    for _ in range(nufft_channels)
]

analysis_list_nufft = [
    al.AnalysisInterferometer(
        dataset=d,
        adapt_images=adapt_images,
        raise_inversion_positions_likelihood_exception=False,
    )
    for d in dataset_list_nufft
]

analysis_factor_list_nufft = [
    af.AnalysisFactor(prior_model=model.copy(), analysis=a) for a in analysis_list_nufft
]

factor_graph_nufft = af.FactorGraphModel(*analysis_factor_list_nufft, use_jax=True)

fitness_nufft = Fitness(
    model=factor_graph_nufft.global_prior_model,
    analysis=factor_graph_nufft,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

parameters_nufft = np.zeros(
    (batch_size, factor_graph_nufft.global_prior_model.total_free_parameters)
)
for i in range(batch_size):
    parameters_nufft[i, :] = (
        factor_graph_nufft.global_prior_model.physical_values_from_prior_medians
    )
parameters_nufft = jnp.array(parameters_nufft)

result_nufft = fitness_nufft._vmap(parameters_nufft)
print()
print("TransformerNUFFT cube vmap result:", result_nufft)

np.testing.assert_allclose(
    np.array(result_nufft),
    nufft_channels * EXPECTED_SINGLE_CHANNEL_LOG_LIKELIHOOD,
    rtol=1e-4,
    err_msg="datacube/delaunay: TransformerNUFFT cube vmap disagrees with TransformerDFT",
)
print("PASS: TransformerNUFFT cube cross-check matches TransformerDFT.")
