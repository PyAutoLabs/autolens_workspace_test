"""
Func Grad: Datacube Rectangular Pixelization Source
====================================================

Tests that JAX can compute batched log-likelihoods and jit-wrap the cube
``FactorGraphModel`` for a 4-channel datacube fitted with a
``RectangularAdaptDensity`` pixelization source.

Mirrors ``jax_likelihood_functions/interferometer/rectangular.py`` (same lens
model, same mesh, same regularization, same adapt-images setup) but loads
the SMA dataset N=4 times as an identical-channel cube and wires it through
``af.FactorGraphModel`` — the same pattern as
``jax_likelihood_functions/multi/delaunay.py``.

Identical channels make the cube reference deterministic: the cube
log-likelihood is exactly ``N × single_channel_log_likelihood``. The
expected literal is pinned empirically below.

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

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al
from autoconf import conf

n_channels = 4

"""
__Mask__

Same as ``interferometer/rectangular.py`` — shared across all channels.
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
channels — same visibilities, noise, uv_wavelengths. Regression assertion is
exactly N × single-channel below.
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
__Over Sampling__

Interferometer does not oversample — same as ``interferometer/rectangular.py``.

__Mesh Shape__

Same 8×8 rectangular mesh as ``interferometer/rectangular.py``.
"""
mesh_pixels_yx = 8
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Model__

Same lens (`Isothermal + ExternalShear`) and source (`RectangularAdaptDensity`
+ `reg.Adapt()`) as ``interferometer/rectangular.py``.
"""
mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
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

mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = al.reg.Adapt()
pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Adapt Images__

Same Sersic-image reference as ``interferometer/rectangular.py``. Shared across
all channels in the cube (same `dataset.grid` since channels share the mask).
"""
bulge = al.lp.Sersic()
image = bulge.image_2d_from(grid=dataset_list[0].grid)

galaxy_name_image_dict = {
    "('galaxies', 'lens')": image,
    "('galaxies', 'source')": image,
}

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

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
Cube log-likelihood ≈ N × single-channel log-likelihood (-3164.286252) for
identical channels. Pinned empirically below.
"""
EXPECTED_VMAP_LOG_LIKELIHOOD = n_channels * -3164.286252

np.testing.assert_allclose(
    np.array(result),
    EXPECTED_VMAP_LOG_LIKELIHOOD,
    rtol=1e-4,
    err_msg="datacube/rectangular: JAX vmap likelihood mismatch",
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

Re-run the same cube vmap with ``TransformerNUFFT`` and confirm the result
matches the ``TransformerDFT`` value. Proves the cube path works with both
transformers; mirrors the single-channel ``interferometer/rectangular.py``
Path B assertion summed over channels.
"""
dataset_list_nufft = [
    al.Interferometer.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,
    )
    for _ in range(n_channels)
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
    EXPECTED_VMAP_LOG_LIKELIHOOD,
    rtol=1e-4,
    err_msg="datacube/rectangular: TransformerNUFFT cube vmap disagrees with TransformerDFT",
)
print("PASS: TransformerNUFFT cube cross-check matches TransformerDFT.")
