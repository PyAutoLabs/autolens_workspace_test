"""
Database: Scaling Relation
===========================

Tests that model-fit results using a luminosity scaling relation on extra galaxies
can be scraped into a database and queried.

The scaling relation parameterizes the Einstein radius of an extra galaxy's mass
profile as a function of its luminosity:

    einstein_radius = scaling_factor * luminosity ** scaling_exponent

where `scaling_factor` and `scaling_exponent` are free parameters inferred by the
non-linear search.  This is the same relational model API used in production group
and cluster lens modeling (see `autolens_base_project/scripts/group.py`).

__Known issue__

The database serialisation of scaling-relation models (where a free parameter is
linked to an expression involving other free parameters) may fail at the aggregator
reconstruction step.  This test is deliberately written to surface that failure, so
that the exact error can be diagnosed and fixed.

__Model__

- Lens:         `al.mp.Isothermal` + `al.mp.ExternalShear`
- Source:       `al.lp_linear.Sersic`
- Extra galaxy: `al.mp.Isothermal` with Einstein radius set by scaling relation
  (`scaling_factor * luminosity ** scaling_exponent`, luminosity = 0.9)

The extra galaxy is placed at a known centre and included via `extra_galaxies`.

__Structure__

This test mirrors `general.py` exactly, adding only the scaling relation model
for the extra galaxy.
"""

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
__Scaling Relation Model__

The extra galaxy has a known centre and luminosity. Its Einstein radius is
parameterized via the scaling relation:

    einstein_radius = scaling_factor * luminosity ** scaling_exponent

`scaling_factor` and `scaling_exponent` are the two free parameters of the scaling
relation.  They are defined outside the extra-galaxy loop so that a single shared
relation governs all extra galaxies (here there is only one).

The luminosity is a pre-measured float — in a real analysis this would come from a
prior photometric fit or luminosity measurement.
"""
extra_galaxy_centre = (1.5, 0.5)
extra_galaxy_luminosity = 0.9

scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
scaling_exponent = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

extra_galaxy_mass = af.Model(al.mp.Isothermal)
extra_galaxy_mass.centre = extra_galaxy_centre
extra_galaxy_mass.ell_comps = (0.0, 0.0)
extra_galaxy_mass.einstein_radius = (
    scaling_factor * extra_galaxy_luminosity**scaling_exponent
)

extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=extra_galaxy_mass)

"""
__Model__
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.Sersic)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    extra_galaxies=af.Collection(extra_galaxy=extra_galaxy),
)

"""
__Search + Analysis + Model-Fit__
"""
name = "scaling_relation"

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

TODO: The database scrape below is disabled due to a bug in PyAutoFit.
TODO: See https://github.com/rhayes777/PyAutoFit/issues/1171

When a model parameter is set via a relational expression
(`mass.einstein_radius = scaling_factor * luminosity ** scaling_exponent`),
`agg.add_directory` crashes during scraping with:

    KeyError: "Could not find any of the following keys in kwargs
    (('extra_galaxies', 'extra_galaxy', 'mass', 'einstein_radius', 'left_'),)"

The scraper calls `samples_summary.max_log_likelihood()` to build the model instance
during scraping.  The relational expression is represented internally as an arithmetic
expression tree, and the path resolver looks for `left_` / `right_` sub-paths that do
not exist in the stored sample parameter paths.

Re-enable by removing the `if False:` block once the bug is fixed.
"""

# TODO: https://github.com/rhayes777/PyAutoFit/issues/1171
if False:
    from autofit.database.aggregator import Aggregator

    database_file = "database_directory_scaling_relation.sqlite"

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

    Standard queries as in general.py, plus scaling-relation-specific queries.
    """
    print("\n\n***********************")
    print("****QUERIES TESTING****")
    print("***********************\n")

    unique_tag = agg.search.unique_tag
    agg_query = agg.query(unique_tag == "mass_sie__source_sersic__1")
    samples_gen = agg_query.values("samples")

    name_q = agg.search.name
    agg_query = agg.query(name_q == "database_example")
    print("Total Queried Results via search name = ", len(agg_query), "\n\n")

    lens_q = agg.model.galaxies.lens
    agg_query = agg.query(lens_q.mass == al.mp.Isothermal)
    print("Total Samples Objects via `Isothermal` model query = ", len(agg_query), "\n")

    mass_q = agg.model.galaxies.lens.mass
    agg_query = agg.query((mass_q == al.mp.Isothermal) & (mass_q.einstein_radius > 1.0))
    print(
        "Total Samples Objects In Query `Isothermal and einstein_radius > 1.0` = ",
        len(agg_query),
        "\n",
    )

    """
    __Scaling Relation Queries__

    Query on the extra galaxy mass model and the scaling relation parameters.
    """
    print("\n\n*****************************")
    print("****SCALING RELATION QUERIES****")
    print("*****************************\n")

    extra_galaxy_q = agg.model.extra_galaxies.extra_galaxy
    agg_query = agg.query(extra_galaxy_q.mass == al.mp.Isothermal)
    print(
        "Total Samples Objects via extra galaxy `Isothermal` model query = ",
        len(agg_query),
        "\n",
    )

    """
    __Files__

    Check that all other files stored in database (e.g. model, search) can be loaded.
    """
    print("\n\n***********************")
    print("*****FILES TESTING*****")
    print("***********************\n")

    for model in agg.values("model"):
        print(f"\n****Model Info (model)****\n\n{model.info}")
        assert model.info[0] == "T"

    for search in agg.values("search"):
        print(f"\n****Search (search)****\n\n{search}")
        assert search.paths.name == "scaling_relation"

    for samples_summary in agg.values("samples_summary"):
        instance = samples_summary.max_log_likelihood()
        print(f"\n****Max Log Likelihood (samples_summary)****\n\n{instance}")
        assert instance.galaxies.lens.mass.einstein_radius > 0.0

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
        tracer = tracer_list[0]

        try:
            tracer_plotter = aplt.Tracer(tracer=tracer, grid=grid)
            tracer_plotter.figures_2d(convergence=True, potential=True)

        except al.exc.ProfileException:
            print(
                "TracerAgg with linear light profiles raises correct ProfileException"
            )

        assert tracer.galaxies[0].mass.einstein_radius > 0.0

        print("TracerAgg Checked")

    imaging_agg = al.agg.ImagingAgg(aggregator=agg)
    imaging_gen = imaging_agg.dataset_gen_from()

    for dataset_list in imaging_gen:
        dataset = dataset_list[0]

        dataset_plotter = aplt.Imaging(dataset=dataset)
        dataset_plotter.subplot_dataset()

        assert dataset.pixel_scales[0] > 0.0

        print("ImagingAgg Checked")

    fit_agg = al.agg.FitImagingAgg(
        aggregator=agg,
        settings=al.Settings(use_border_relocator=False),
    )
    fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()

    for fit_list in fit_imaging_gen:
        fit = fit_list[0]

        fit_plotter = aplt.FitImaging(fit=fit)
        fit_plotter.subplot_fit()

        assert fit.tracer.galaxies[0].mass.einstein_radius > 0.0

        print("FitImagingAgg Checked")
