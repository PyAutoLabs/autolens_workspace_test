"""
Simulator: Simple
=================

This script simulates `Imaging` of a strong lens where:

 - The pixel scale is 0.3" and the PSF is a 0.8"-FWHM Gaussian, i.e. ground-based-seeing-like
   sampling, sized so that the models fitted to this dataset resolve the data they fit.

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

from os import path
import autolens as al

"""
The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/psf.fits`.
"""
dataset_path = path.join("dataset", "imaging", "jax_test")

"""
For simulating an image of a strong lens, we recommend using a Grid2DIterate object. This represents a grid of (y,x) 
coordinates like an ordinary Grid2D, but when the light-profile`s image is evaluated below (using the Tracer) the 
sub-size of the grid is iteratively increased (in steps of 2, 4, 8, 16, 24) until the input fractional accuracy of 
99.99% is met.

This ensures that the divergent and bright central regions of the source galaxy are fully resolved when determining the
total flux emitted within a pixel.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.3)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.35, pixel_scales=grid.pixel_scales, normalize=True
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=2000.0,
    psf=psf,
    background_sky_level=1.0,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

"""
Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.5,
        effective_radius=1.6,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.001, gamma_2=0.001),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=1.0,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Output the simulated dataset to the dataset path as .fits files.
"""
al.output_to_fits(
    values=dataset.data.native,
    file_path=path.join(dataset_path, "data.fits"),
    overwrite=True,
)
al.output_to_fits(
    values=dataset.psf.kernel.native,
    file_path=path.join(dataset_path, "psf.fits"),
    overwrite=True,
)
al.output_to_fits(
    values=dataset.noise_map.native,
    file_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

"""
Save the `Tracer` in the dataset folder as a .json file.
"""
al.output_to_json(
    obj=tracer,
    file_path=path.join(dataset_path, "tracer.json"),
)


"""
Produce the no lens light data.
"""
lens_image = lens_galaxy.padded_image_2d_from(
    grid=grid,
    psf_shape_2d=psf.kernel.shape_native,
)

lens_image = psf.convolved_image_from(image=lens_image, blurring_image=None)

lens_image = lens_image.trimmed_after_convolution_from(
    kernel_shape=psf.kernel.shape_native
)

snr_no_lens = (dataset.data - lens_image) / dataset.noise_map

al.output_to_fits(
    values=snr_no_lens.native,
    file_path=path.join(dataset_path, "snr_no_lens.fits"),
    overwrite=True,
)

"""
Compute and save multiple-image positions for scripts that use a PositionsLH penalty.
"""
solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

al.output_to_json(
    file_path=path.join(dataset_path, "positions.json"),
    obj=positions,
)

print("Dataset written to", dataset_path)
