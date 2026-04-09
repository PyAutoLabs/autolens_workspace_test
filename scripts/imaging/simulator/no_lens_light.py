"""
Simulator: SIE
==============

This script simulates `Imaging` of a strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's `LightProfile` is an `Sersic`.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/psf.fits`.
"""
dataset_label = "build"
dataset_type = "imaging"
dataset_name = "no_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

"""
For simulating an image of a strong lens, we recommend using a Grid2DIterate object. This represents a grid of (y,x) 
coordinates like an ordinary Grid2D, but when the light-profile`s image is evaluated below (using the Tracer) the 
sub-size of the grid is iteratively increased (in steps of 2, 4, 8, 16, 24) until the input fractional accuracy of 
99.99% is met.

This ensures that the divergent and bright central regions of the source galaxy are fully resolved when determining the
total flux emitted within a pixel.
"""
grid = al.Grid2D.uniform(
    shape_native=(120, 120),
    pixel_scales=0.2,
)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.2, pixel_scales=grid.pixel_scales
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens.

For lens modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the `convert` module to determine the elliptical components from the axis-ratio and angle.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.IsothermalSph(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialSph(
        centre=(0.0, 0.1),
        intensity=0.3,
        effective_radius=0.1,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid))

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
aplt.plot_array(array=dataset.data)

"""
Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

positions = al.Grid2DIrregular(values=[(1.6, 0.0), (0.0, 1.6)])
al.output_to_json(
    obj=positions,
    file_path=path.join(dataset_path, "positions.json"),
)

"""
Output a subplot of the simulated dataset, the image and a subplot of the `Tracer`'s quantities to the dataset path 
as .png files.
"""
aplt.plot_array(array=dataset.data, output=aplt.Output(path=dataset_path, format="png"))
aplt.subplot_tracer(
    tracer=tracer, grid=grid, output=aplt.Output(path=dataset_path, format="png")
)

"""
Pickle the `Tracer` in the dataset folder, ensuring the true `Tracer` is safely stored and available if we need to 
check how the dataset was simulated in the future. 

This will also be accessible via the `Aggregator` if a model-fit is performed using the dataset.
"""
al.output_to_json(
    obj=tracer,
    file_path=path.join(dataset_path, "tracer.json"),
)

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/no_lens_light/mass_sie__source_sersic`.
"""
