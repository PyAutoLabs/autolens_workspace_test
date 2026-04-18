"""
Func Grad: Interferometer MGE + Extra Galaxies
===============================================
Tests that JAX can compute batched log-likelihood evaluations for an interferometer
model with extra galaxies, using the same dataset as interferometer/mge.py.
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path
import autofit as af
import autolens as al

mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
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

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

print(f"Total Visibilities: {dataset.uv_wavelengths.shape[0]}")

# Lens mass + ExternalShear
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source: MGE with 10 Gaussians
bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=10, centre_prior_is_uniform=False
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Extra galaxies: 3 galaxies with IsothermalSph mass at fixed centres
extra_galaxy_centres = [(0.5, 1.0), (-0.5, 1.5), (1.0, -0.5)]
extra_galaxies_list = []

for centre in extra_galaxy_centres:
    mass_extra = af.Model(al.mp.IsothermalSph)
    mass_extra.centre = centre
    mass_extra.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.3)
    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass_extra)
    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    extra_galaxies=extra_galaxies,
)

print(model.info)

analysis = al.AnalysisInterferometer(
    dataset=dataset,
    raise_inversion_positions_likelihood_exception=False,
)

from autofit.non_linear.fitness import Fitness
import time

batch_size = 3

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
    -3154.194645,
    rtol=1e-4,
    err_msg="interferometer/mge_group: JAX vmap likelihood mismatch",
)

print("interferometer/mge_group.py checks passed.")
