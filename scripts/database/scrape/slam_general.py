"""
Database: SLaM General
======================

Runs a SLaM pipeline (linear light profiles) and scrapes the results to a database,
verifying that queries and aggregator modules work correctly.

Based on autolens_workspace/scripts/imaging/features/linear_light_profiles/slam.py
"""

from astropy.io import fits
import numpy as np
import os
from os import path


def fit():
    import pytest

    from pathlib import Path
    import autofit as af
    import autolens as al
    import autolens.plot as aplt

    """
    __Pipeline Functions__
    """

    def source_lp(
        settings_search,
        dataset,
        mask_radius,
        redshift_lens,
        redshift_source,
        n_batch=50,
    ):
        analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

        lens_bulge = af.Model(al.lp_linear.Sersic)

        source_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=redshift_lens,
                    bulge=lens_bulge,
                    disk=None,
                    mass=af.Model(al.mp.Isothermal),
                    shear=af.Model(al.mp.ExternalShear),
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=redshift_source,
                    bulge=source_bulge,
                ),
            ),
        )

        search = af.Nautilus(
            name="source_lp[1]",
            **settings_search.search_dict,
            n_live=200,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    def source_pix_1(
        settings_search,
        dataset,
        source_lp_result,
        mesh_init,
        regularization_init,
        n_batch=20,
    ):
        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=source_lp_result
        )

        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            positions_likelihood_list=[
                source_lp_result.positions_likelihood_from(
                    factor=3.0, minimum_threshold=0.2
                )
            ],
        )

        mass = al.util.chaining.mass_from(
            mass=source_lp_result.model.galaxies.lens.mass,
            mass_result=source_lp_result.model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )
        shear = source_lp_result.model.galaxies.lens.shear

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.lens.redshift,
                    bulge=source_lp_result.instance.galaxies.lens.bulge,
                    disk=source_lp_result.instance.galaxies.lens.disk,
                    mass=mass,
                    shear=shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.source.redshift,
                    pixelization=af.Model(
                        al.Pixelization,
                        mesh=mesh_init,
                        regularization=regularization_init,
                    ),
                ),
            ),
        )

        search = af.Nautilus(
            name="source_pix[1]",
            **settings_search.search_dict,
            n_live=150,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    def source_pix_2(
        settings_search,
        dataset,
        source_lp_result,
        source_pix_result_1,
        mesh,
        regularization,
        n_batch=20,
    ):
        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=source_pix_result_1
        )

        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            use_jax=True,
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.lens.redshift,
                    bulge=source_lp_result.instance.galaxies.lens.bulge,
                    disk=source_lp_result.instance.galaxies.lens.disk,
                    mass=source_pix_result_1.instance.galaxies.lens.mass,
                    shear=source_pix_result_1.instance.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.source.redshift,
                    pixelization=af.Model(
                        al.Pixelization,
                        mesh=mesh,
                        regularization=regularization,
                    ),
                ),
            ),
        )

        search = af.Nautilus(
            name="source_pix[2]",
            **settings_search.search_dict,
            n_live=75,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    def light_lp(
        settings_search,
        dataset,
        source_result_for_lens,
        source_result_for_source,
        n_batch=20,
    ):
        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=source_result_for_lens
        )

        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
        )

        lens_bulge = af.Model(al.lp_linear.Sersic)

        source = al.util.chaining.source_custom_model_from(
            result=source_result_for_source, source_is_model=False
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                    bulge=lens_bulge,
                    disk=None,
                    mass=source_result_for_lens.instance.galaxies.lens.mass,
                    shear=source_result_for_lens.instance.galaxies.lens.shear,
                ),
                source=source,
            ),
        )

        search = af.Nautilus(
            name="light[1]",
            **settings_search.search_dict,
            n_live=150,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    def mass_total(
        settings_search,
        dataset,
        source_result_for_lens,
        source_result_for_source,
        light_result,
        n_batch=20,
    ):
        mass = af.Model(al.mp.PowerLaw)

        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=source_result_for_lens
        )

        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            positions_likelihood_list=[
                source_result_for_source.positions_likelihood_from(
                    factor=3.0, minimum_threshold=0.2
                )
            ],
        )

        mass = al.util.chaining.mass_from(
            mass=mass,
            mass_result=source_result_for_lens.model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )

        bulge = light_result.instance.galaxies.lens.bulge
        disk = light_result.instance.galaxies.lens.disk

        source = al.util.chaining.source_from(result=source_result_for_source)

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                    bulge=bulge,
                    disk=disk,
                    mass=mass,
                    shear=source_result_for_lens.model.galaxies.lens.shear,
                ),
                source=source,
            ),
        )

        search = af.Nautilus(
            name="mass_total[1]",
            **settings_search.search_dict,
            n_live=150,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    """
    __Dataset + Masking__
    """
    dataset_name = "with_lens_light"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    if not path.exists(dataset_path):
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "scripts/imaging/simulator/with_lens_light.py"],
            check=True,
        )

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        pixel_scales=0.2,
    )

    mask_radius = 3.0

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    settings_search = af.SettingsSearch(
        path_prefix=path.join("database", "scrape", "slam_general"),
        number_of_cores=1,
        session=None,
        info={"hi": "there"},
    )

    redshift_lens = 0.5
    redshift_source = 1.0

    mesh_pixels_yx = 28
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

    """
    __SLaM Pipeline__
    """
    source_lp_result = source_lp(
        settings_search=settings_search,
        dataset=dataset,
        mask_radius=mask_radius,
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
    )

    source_pix_result_1 = source_pix_1(
        settings_search=settings_search,
        dataset=dataset,
        source_lp_result=source_lp_result,
        mesh_init=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
        regularization_init=al.reg.Adapt,
    )

    source_pix_result_2 = source_pix_2(
        settings_search=settings_search,
        dataset=dataset,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
        regularization=al.reg.Adapt,
    )

    light_result = light_lp(
        settings_search=settings_search,
        dataset=dataset,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
    )

    mass_result = mass_total(
        settings_search=settings_search,
        dataset=dataset,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
    )

    """
    __Database__

    Add results to database.
    """
    from autofit.database.aggregator import Aggregator

    database_file = "database_directory_slam_general.sqlite"

    try:
        os.remove(path.join("output", database_file))
    except FileNotFoundError:
        pass

    agg = Aggregator.from_database(database_file)
    agg.add_directory(
        directory=path.join("output", "database", "scrape", "slam_general")
    )

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
        print(search.paths.name)
        assert "[" in search.paths.name

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

    agg = agg.query(agg.search.name == "mass_total[1]")

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
            print(
                "TracerAgg with linear light profiles raises correct ProfileException"
            )

        assert tracer.galaxies[0].mass.einstein_radius > 0.0

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
    )
    fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()

    for fit_list in fit_imaging_gen:
        fit = fit_list[0]

        aplt.subplot_fit_imaging(fit=fit)

        assert fit.tracer.galaxies[0].mass.einstein_radius > 0.0

        print("FitImagingAgg Checked")

    fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()

    for fit_list in fit_imaging_gen:
        fit = fit_list[0]

        assert fit.adapt_images is None

        print("FitImagingAgg Adapt Images Checked")


if __name__ == "__main__":

    fit()
