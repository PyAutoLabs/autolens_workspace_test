"""
JAX Likelihood: HilbertSNR Image Mesh With Soft S/N Adapt Images
===============================================================

The default adaptive set-up of the workspaces (adapt-mesh-weighting, PyAutoArray#714): a `HilbertSNR` image mesh
drawn from the source's signal-to-noise map, with edge points from `edge_pixels`, and the soft S/N maps of
`soft_snr_maps_from` as the adapt images that `AdaptSplit` adapts to. Galaxy keys are written short (`"source"`).

This pins the NumPy figure of merit (log evidence), checks the JAX `vmap` path against it and
`jax.jit(analysis.fit_from)` against the NumPy log likelihood, so the new default stays JAX-compatible. `imaging/jax_likelihood/delaunay.py` keeps the older `Hilbert` set-up.

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
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al

sub_size = 4

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
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/imaging/simulator/simple.py"
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
mask_radius = 2.6

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

# dataset = dataset.apply_over_sampling(over_sample_size_lp=2)

# positions = al.Grid2DIrregular(
#     al.from_json(file_path=path.join(dataset_path, "positions.json"))
# )

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=1,
)


"""
__Adapt Images__

The source's S/N map. The dataset is a simulated lens with no lens light model to subtract, so the S/N map of
the source is the data divided by the noise map (in a pipeline, `al.galaxy_snr_maps_from(result)` or
`al.AdaptImages.from_result` compute it from the previous fit).
"""
snr_map = al.Array2D(
    values=np.asarray(dataset.data.slim) / np.asarray(dataset.noise_map.slim),
    mask=dataset.mask,
)
snr_maps = {"('galaxies', 'source')": snr_map}

image_mesh = al.image_mesh.HilbertSNR(pixels=200, edge_pixels=30)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=snr_map
)

assert image_plane_mesh_grid.shape[0] == image_mesh.pixels + image_mesh.edge_pixels

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=al.soft_snr_maps_from(snr_maps),
    galaxy_name_image_plane_mesh_grid_dict={"source": image_plane_mesh_grid},
)

"""
__Model__
"""
# # Lens:

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
)

# Source:

regularization = al.reg.AdaptSplit()

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(pixels=image_mesh.pixels, zeroed_pixels=image_mesh.edge_pixels),
    regularization=regularization,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

# External Shear: a property of the system, not of the lens galaxy, so it is held in an
# `al.MassField` (a container like a galaxy, carrying no light) in its own `fields=` slot.
field = af.Model(al.MassField, redshift=0.5, shear=shear)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)


"""
__NumPy Figure of Merit (pinned)__
"""
instance = model.instance_from_prior_medians()

analysis_np = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))
print("NumPy fit.figure_of_merit (log evidence):", float(fit_np.figure_of_merit))

# Pinned 2026-11-26 (autolens_workspace#693): HilbertSNR(pixels=200, edge_pixels=30) defaults
# (8-point cap, frame-independent interpolation) on the jax_test dataset, soft S/N adapt images.
np.testing.assert_allclose(
    float(fit_np.figure_of_merit),
    -443.73145939,
    rtol=1e-4,
    err_msg="hilbert_snr: NumPy figure of merit mismatch",
)

"""
__JAX: vmap and jit(fit_from)__
"""
from autofit.non_linear.fitness import Fitness

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
)

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
    batch_size=3,
)
parameters = jnp.array(
    np.tile(model.physical_values_from_prior_medians, (fitness.batch_size, 1))
)
result = np.asarray(fitness._vmap(parameters))
print("JAX vmap log likelihoods:", result)

# The vmapped fitness returns the figure of merit (the log evidence for a pixelized source).
np.testing.assert_allclose(
    result,
    float(fit_np.figure_of_merit),
    rtol=1e-4,
    err_msg="hilbert_snr: JAX vmap figure of merit differs from NumPy",
)

analysis_jit = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=True,
)
fit = jax.jit(analysis_jit.fit_from)(instance)
print("JIT fit.log_likelihood:", fit.log_likelihood)

np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-8
)
print("PASS: HilbertSNR + soft S/N adapt images agree across NumPy, vmap and jit.")
