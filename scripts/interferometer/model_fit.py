"""
Modeling: Mass Total + Source Inversion
=======================================

This script fits `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `Delaunay` `Pixelization` and `Constant`
   regularization.
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
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(100, 100),
    pixel_scales=0.2,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `mass_sie__source_sersic` from .fits files , which we will fit 
with the lens model.
"""
dataset_label = "build"
dataset_type = "interferometer"
dataset_name = "with_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator/with_lens_light.py"],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
__Inversion Settings (Run Times)__

"""


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
    centre=real_space_mask.mask_centre,
    radius=mask_radius + real_space_mask.pixel_scale / 2.0,
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
    path_prefix=path.join("build", "model_fit", "interferometer"),
    name=dataset_name,
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

The `AnalysisInterferometer` object defines the `log_likelihood_function` used by the non-linear search to fit the 
model to the `Interferometer`dataset.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    adapt_images=adapt_images,
)

"""
__Likelihood Sanity__

Mirror of the imaging guard against regressions like PyAutoLens PR #504, where
the CPU branch of ``AnalysisImaging.log_likelihood_function`` silently returned
``fit.log_likelihood`` instead of ``fit.figure_of_merit``. The interferometer
analysis returns ``figure_of_merit`` correctly today, but this guard catches
the same asymmetry from being introduced here in future.

For a pixelization source ``figure_of_merit`` and ``log_likelihood`` differ by
the regularization log-det terms of the Bayesian log evidence. The sanity
analysis is built without ``positions_likelihood_list`` so there is no
``log_likelihood_penalty`` term to subtract from the comparison.
"""
import pytest
from autofit.non_linear.fitness import Fitness


def _assert_likelihood_sanity(label, analysis, model):
    instance = model.instance_from_prior_medians()
    analysis_value = analysis.log_likelihood_function(instance=instance)
    fit = analysis.fit_from(instance=instance)
    assert float(analysis_value) == pytest.approx(float(fit.figure_of_merit)), (
        f"{label}: log_likelihood_function ({analysis_value}) does not match "
        f"fit.figure_of_merit ({fit.figure_of_merit}) — regression of PR #504"
    )
    assert float(fit.figure_of_merit) != pytest.approx(
        float(fit.log_likelihood), rel=1e-6
    ), (
        f"{label}: figure_of_merit == log_likelihood — pixelization regularization "
        f"log-det terms are zero, this script no longer exercises the bug PR #504 fixed"
    )
    fitness = Fitness(
        model=model,
        analysis=analysis,
        paths=None,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )
    call_wrap_value = fitness.call_wrap(model.physical_values_from_prior_medians)
    assert float(call_wrap_value) == pytest.approx(float(fit.figure_of_merit)), (
        f"{label}: Fitness.call_wrap ({call_wrap_value}) does not match "
        f"fit.figure_of_merit ({fit.figure_of_merit})"
    )
    print(f"  PASS {label}: LLF == figure_of_merit != log_likelihood == call_wrap")


sanity_analysis_cpu = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    use_jax=False,
)
_assert_likelihood_sanity("CPU", sanity_analysis_cpu, model)


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
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(
    tracer=result.max_log_likelihood_tracer, grid=real_space_mask.derive_grid.all_false
)

aplt.subplot_fit_interferometer(fit=result.max_log_likelihood_fit)

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""
