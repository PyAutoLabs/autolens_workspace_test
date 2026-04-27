"""
Database: Model-Fit
===================

Tests that general results can be loaded from hard-disk via a database built via a scrape.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from astropy.io import fits
import numpy as np
from os import path
import os

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking__
"""
dataset_label = "build"
dataset_name = "no_lens_light"
dataset_path = path.join("dataset", dataset_label, "imaging", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator/no_lens_light.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
__Model__
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.Sersic)

extra_galaxies = af.Collection(
    extra_galaxy=af.Model(al.Galaxy, redshift=0.5, bulge=al.lp_linear.Sersic)
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), extra_galaxies=extra_galaxies
)

"""
__Search + Analysis + Model-Fit__
"""
name = "general"

search = af.Nautilus(
    name=name,
    path_prefix=path.join("database", "scrape"),
    unique_tag=dataset_name,
    n_live=50,
    n_like_max=300,
)

analysis = al.AnalysisImaging(dataset=masked_dataset)

result = search.fit(model=model, analysis=analysis, info={"hi": "there"})

"""
__Database__

Add results to database.
"""
from autofit.database.aggregator import Aggregator

database_file = "database_directory_general.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

agg = Aggregator.from_database(database_file)
agg.add_directory(path.join("output", "database", "scrape", dataset_name, name))

assert len(agg) > 0

"""
__Samples + Results__

Make sure database + agg can be used.
"""
print("\n\n***********************")
print("****RESULTS TESTING****")
print("***********************\n")

for samples in agg.values("samples"):
    print(samples.parameter_lists[0])

mp_instances = [samps.median_pdf() for samps in agg.values("samples")]
print(mp_instances)


"""
__Queries__
"""
print("\n\n***********************")
print("****QUERIES TESTING****")
print("***********************\n")

unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "mass_sie__source_sersic__1")
samples_gen = agg_query.values("samples")

unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "incorrect_name")
samples_gen = agg_query.values("samples")

name = agg.search.name
agg_query = agg.query(name == "database_example")
print("Total Queried Results via search name = ", len(agg_query), "\n\n")

lens = agg.model.galaxies.lens
agg_query = agg.query(lens.mass == al.mp.Isothermal)
print("Total Samples Objects via `Isothermal` model query = ", len(agg_query), "\n")

source = agg.model.galaxies.source
agg_query = agg.query(source.disk == None)
print("Total Samples Objects via `Isothermal` model query = ", len(agg_query), "\n")

mass = agg.model.galaxies.lens.mass
agg_query = agg.query((mass == al.mp.Isothermal) & (mass.einstein_radius > 1.0))
print(
    "Total Samples Objects In Query `Isothermal and einstein_radius > 3.0` = ",
    len(agg_query),
    "\n",
)

extra_galaxy_bulge = agg.model.extra_galaxies.extra_galaxy.bulge
agg_query = agg.query(extra_galaxy_bulge == al.lp_linear.Sersic)
print(
    "Total Samples Objects via `Sersic` extra galaxy model query = ",
    len(agg_query),
    "\n",
)


"""
__Files__

Check that all other files stored in database (e.g. model, search) can be loaded and used.
"""
print("\n\n***********************")
print("*****FILES TESTING*****")
print("***********************\n")

for model in agg.values("model"):
    print(f"\n****Model Info (model)****\n\n{model.info}")
    assert model.info[0] == "T"

for search in agg.values("search"):
    print(f"\n****Search (search)****\n\n{search}")
    assert search.paths.name == "general"

for samples_summary in agg.values("samples_summary"):
    instance = samples_summary.max_log_likelihood()
    print(f"\n****Max Log Likelihood (samples_summary)****\n\n{instance}")
    assert instance.galaxies.lens.mass.einstein_radius > 0.0
    assert instance.extra_galaxies.extra_galaxy.bulge.effective_radius > 0.0

for info in agg.values("info"):
    print(f"\n****Info****\n\n{info}")
    assert info["hi"] == "there"

"""
__Aggregator Module__
"""
print("\n\n***********************")
print("***AGG MODULE TESTING***")
print("***********************\n\n")

tracer_agg = al.agg.TracerAgg(aggregator=agg)
tracer_gen = tracer_agg.max_log_likelihood_gen_from()

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for tracer_list in tracer_gen:
    # Only one `Analysis` so take first and only tracer.
    tracer = tracer_list[0]

    try:
        aplt.plot_array(array=tracer.convergence_2d_from(grid=grid))
        aplt.plot_array(array=tracer.potential_2d_from(grid=grid))

    except al.exc.ProfileException:
        print("TracerAgg with linear light profiles raises correct ProfileException")

    assert tracer.galaxies[0].mass.einstein_radius > 0.0
    assert tracer.galaxies[1].bulge.effective_radius > 0.0  # Is an extra galaxy

    print("TracerAgg Checked")

imaging_agg = al.agg.ImagingAgg(aggregator=agg)
imaging_gen = imaging_agg.dataset_gen_from()

for dataset_list in imaging_gen:
    dataset = dataset_list[0]

    aplt.plot_array(array=dataset.data)

    assert dataset.pixel_scales[0] > 0.0

    print("ImagingAgg Checked")

fit_agg = al.agg.FitImagingAgg(
    aggregator=agg,
    settings=al.Settings(use_border_relocator=False),
)
fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_imaging_gen:
    fit = fit_list[0]

    aplt.subplot_fit_imaging(fit=fit)

    assert fit.tracer.galaxies[0].mass.einstein_radius > 0.0
    assert fit.tracer.galaxies[1].bulge.effective_radius > 0.0  # Is an extra galaxy

    print("FitImagingAgg Checked")
