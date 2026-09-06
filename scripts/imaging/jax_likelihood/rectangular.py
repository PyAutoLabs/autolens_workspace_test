"""
Func Grad: Light Parametric Operated
====================================

This script test if JAX can successfully compute the gradient of the log likelihood of an `Imaging` dataset with a
model which uses operated light profiles.

 __Operated Fitting__

It is common for galaxies to have point-source emission, for example bright emission right at their centre due to
an active galactic nuclei or very compact knot of star formation.

This point-source emission is subject to blurring during data accquisiton due to the telescope optics, and therefore
is not seen as a single pixel of light but spread over multiple pixels as a convolution with the telescope
Point Spread Function (PSF).

It is difficult to model this compact point source emission using a point-source light profile (or an extremely
compact Gaussian / Sersic profile). This is because when the model-image of a compact point source of light is
convolved with the PSF, the solution to this convolution is extremely sensitive to which pixel (and sub-pixel) the
compact model emission lands in.

Operated light profiles offer an alternative approach, whereby the light profile is assumed to have already been
convolved with the PSF. This operated light profile is then fitted directly to the point-source emission, which as
discussed above shows the PSF features.

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
import jax
from os import path

import autofit as af
import autolens as al

sub_size = 2
psf_shape_2d = (21, 21)

"""
__Dataset__

Load and plot the galaxy dataset via .fits files.
"""
dataset_path = path.join("dataset", "imaging", "jax_test")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator/simple.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.3,
    over_sample_size_lp=sub_size,
    over_sample_size_pixelization=sub_size,
)


"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[2, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

snr_no_lens = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "snr_no_lens.fits"), pixel_scales=0.3
)

signal_to_noise_threshold = 3.0
over_sample_size_pixelization = np.where(
    snr_no_lens.native > signal_to_noise_threshold,
    2,
    1,
)
over_sample_size_pixelization = al.Array2D(
    values=over_sample_size_pixelization, mask=mask
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=over_sample_size_pixelization,
)

# dataset = dataset.apply_sparse_operator(batch_size=64)


"""
__Mesh Shape__

The `mesh_shape` parameter defines number of pixels used by the rectangular mesh to reconstruct the source,
set below to 20 x 20.

The mesh must not have more source pixels than the mask has image pixels, or the inversion is
under-determined. The 3.5" mask holds ~428 pixels at the dataset's 0.3" pixel scale, so 20 x 20
(400) is the largest square mesh that stays determined.

The `mesh_shape` must be fixed before modeling and cannot be a free parameter of the model, because JAX uses the
mesh shape to define static shaped arrays which use the mesh to reconstruct the source. For a rectangular
mesh, the same number of pixels must be used in the y and x directions.

__Edge Zeroing__

By default, all pixels at the edge of the mesh in the source-plane are forced to solutions of zero brightness by 
the linear algebra solver. This prevents unphysical solutions where pixels at the edge of the mesh reconstruct 
bright surface brightnesses, often because they fit residuals from the lens light subtraction.

For a rectangular mesh, the source code computes edge pixels internally using the known
pixels at the edge of the mesh. 
"""
mesh_pixels_yx = 20
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's point source emission is a parametric operated `Gaussian` centred on the bulge [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
# # Lens:

"""
The simulated lens galaxy (`scripts/imaging/simulator/simple.py`) has a `Sersic` bulge, an
`Exponential` disk, an `Isothermal` mass and a small external shear. The model below carries
every one of those components, and each prior is a uniform prior centred on the simulated
value — the likelihood here is evaluated at the prior medians, so a matched model means the
medians sit at the truth and the evaluation is a high-likelihood one rather than an arbitrary
point in parameter space.
"""
bulge_ell_comps = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
disk_ell_comps = al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0)
mass_ell_comps = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)

bulge = af.Model(al.lp_linear.Sersic)
bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=bulge_ell_comps[0] - 0.05, upper_limit=bulge_ell_comps[0] + 0.05
)
bulge.ell_comps.ell_comps_1 = af.UniformPrior(
    lower_limit=bulge_ell_comps[1] - 0.05, upper_limit=bulge_ell_comps[1] + 0.05
)
bulge.effective_radius = af.UniformPrior(lower_limit=0.5, upper_limit=0.7)
bulge.sersic_index = af.UniformPrior(lower_limit=2.5, upper_limit=3.5)

disk = af.Model(al.lp_linear.Exponential)
disk.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
disk.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
disk.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=disk_ell_comps[0] - 0.05, upper_limit=disk_ell_comps[0] + 0.05
)
disk.ell_comps.ell_comps_1 = af.UniformPrior(
    lower_limit=disk_ell_comps[1] - 0.05, upper_limit=disk_ell_comps[1] + 0.05
)
disk.effective_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)

mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=mass_ell_comps[0] - 0.05, upper_limit=mass_ell_comps[0] + 0.05
)
mass.ell_comps.ell_comps_1 = af.UniformPrior(
    lower_limit=mass_ell_comps[1] - 0.05, upper_limit=mass_ell_comps[1] + 0.05
)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    disk=disk,
    mass=mass,
    shear=shear,
)

# Source:

mesh = al.mesh.RectangularBilinearAdaptImage(shape=mesh_shape, weight_power=1.0)

regularization = al.reg.Adapt()

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

galaxy_name_image_dict = {
    "('galaxies', 'lens')": dataset.data,
    "('galaxies', 'source')": dataset.data,
}

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""
import jax.numpy as jnp

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    settings=al.Settings(
        use_border_relocator=True,
        use_positive_only_solver=True,
        use_mixed_precision=True,
    ),
)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
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
    717.264911,
    rtol=1e-4,
    err_msg="rectangular: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``analysis.fit_from``__
"""


instance = model.instance_from_prior_medians()

analysis_np = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    settings=al.Settings(
        use_border_relocator=True,
        use_positive_only_solver=True,
        use_mixed_precision=True,
    ),
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    settings=al.Settings(
        use_border_relocator=True,
        use_positive_only_solver=True,
        use_mixed_precision=True,
    ),
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
