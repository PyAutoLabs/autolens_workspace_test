"""
Func Grad: Interferometer Rectangular Pixelization Double Source Plane
=======================================================================

This script tests if JAX can successfully compute the gradient of the log likelihood
of an `Interferometer` double source plane (DSPL) dataset with a model which uses
rectangular pixelization sources at two redshifts.

Mirrors `imaging/rectangular_dspl.py` but loads from `dataset/interferometer/dspl/`
via `Interferometer.from_fits` and uses `AnalysisInterferometer`. No
apply_over_sampling — interferometer does not oversample.

The `should_simulate` bootstrap invokes `simulator_dspl.py` to generate the dataset.
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al
from autoconf import conf

"""
__Mask__

We define the 'real_space_mask' which defines the grid the image the strong lens is
evaluated using.
"""
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load the interferometer DSPL dataset from .fits files.
"""
dataset_path = path.join("dataset", "interferometer", "dspl")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running
the corresponding simulator script.
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/jax_likelihood_functions/interferometer/simulator_dspl.py",
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

print(f"Total Visiblities: {dataset.uv_wavelengths.shape[0]}")

"""
__Over Sampling__

Interferometer does not observe galaxies in a way where over sampling is necessary,
therefore all interferometer calculations are performed without over sampling.

__Mesh Shape__
"""
image_mesh = None
mesh_shape = (30, 30)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

"""
__Model__

Three-galaxy double source plane system:
 - lens_0 at z=0.5: MGE bulge, `Isothermal` mass, `ExternalShear`.
 - lens_1 at z=1.0: `Isothermal` mass, rectangular pixelization (intermediate source).
 - source at z=2.0: rectangular pixelization.
"""
# Lens 0 (z=0.5):

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=30,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=0.11111111111111108, upper_limit=0.1111111111111111
)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)

lens_0 = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=mass,
    shear=shear,
)

# Lens 1 (z=1.0) with intermediate rectangular pixelization:

mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
mass.einstein_radius = af.UniformPrior(lower_limit=0.4, upper_limit=0.6)
mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=0.11111111111111108, upper_limit=0.1111111111111111
)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

mesh = al.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0)

regularization = al.reg.Adapt()

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

lens_1 = af.Model(al.Galaxy, redshift=1.0, mass=mass, pixelization=pixelization)

# Source (z=2.0):

mesh = al.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0)

regularization = al.reg.Adapt()

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=2.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens_0, lens_1=lens_1, source=source))

# Use a Sersic image as adapt data to avoid negative dirty-image values.
bulge_adapt = al.lp.Sersic()
adapt_image = bulge_adapt.image_2d_from(grid=dataset.grid)

galaxy_name_image_dict = {
    "('galaxies', 'lens_0')": adapt_image,
    "('galaxies', 'lens_1')": adapt_image,
    "('galaxies', 'source')": adapt_image,
}

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisInterferometer` object defines the `log_likelihood_function` which will
be used to determine if JAX can compute its gradient.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness`
class in **PyAutoFit**, which pairs the model with likelihood.
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 1

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
    -3170.6680826,
    rtol=1e-4,
    err_msg="interferometer/rectangular_dspl: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``analysis.fit_from``__
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(model)

instance = model.instance_from_prior_medians()

analysis_np = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=True,
)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(
    fit.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit.log_likelihood)}"
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
