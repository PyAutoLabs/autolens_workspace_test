"""
Modeling: Mass Total + Source Inversion
=======================================

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is an `Inversion`.

An `Inversion` reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
this reconstruction to be smooth. This uses `Pixelization`  objects and in this example we will
use their simplest forms, a `RectangularAdaptDensity` `Pixelization` and `Constant` `Regularization`.scheme.

Inversions are covered in detail in chapter 4 of the **HowToLens** lectures.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the strong lens dataset `mass_sie__source_sersic` via .fits files, which we will fit with the lens model.
"""
dataset_label = "build"
dataset_type = "imaging"
dataset_name = "with_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator/with_lens_light.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

aplt.plot_array(array=dataset.data)

"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the lens model to the data, which we define
and use to set up the `Imaging` object that the lens model fits.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.plot_array(array=dataset.data)

"""
__Positions__

This fit also uses the arc-second positions of the multiply imaged lensed source galaxy, which were drawn onto the
image via the GUI described in the file `autolens_workspace/*/imaging/preprocess/gui/positions.py`.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)

"""
__Model__

"""
bulge = af.Model(al.lp.DevVaucouleursSph)
bulge.centre.centre_0 = 0.0
bulge.centre.centre_1 = 0.0

mass = af.Model(al.mp.IsothermalSph)
mass.centre.centre_0 = 0.0
mass.centre.centre_1 = 0.0

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

image_mesh = al.image_mesh.Overlay(shape=(26, 26))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)

edge_pixels_total = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)


pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(pixels=100),
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

adapt_images = al.AdaptImages(
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)


"""
__Search__

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm
Nautilus.

A full description of the settings below is given in the beginner modeling scripts, if anything is unclear.
"""
search = af.Nautilus(
    path_prefix=path.join("build", "model_fit", "imaging"),
    n_live=50,
    n_like_max=300,
    number_of_cores=2,
)

"""
__Position Likelihood__

"""
positions_likelihood = al.PositionsLH(positions=positions, threshold=0.1)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset. 
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    adapt_images=adapt_images,
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

"""

"""
The `Result` object also contains:

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Galaxies` and `FitImaging` objects.Information on the posterior as estimated by the `Dynesty` non-linear search. 
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_cornerpy(samples=result.samples)

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""
