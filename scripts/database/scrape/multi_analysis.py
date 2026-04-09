"""
Database: Multi-Wavelength Simultaneous SLaM
=============================================

Runs a simultaneous multi-wavelength SLaM pipeline using factor graphs and scrapes
the results to a database, verifying that queries and aggregator modules work correctly.

Based on autolens_workspace/scripts/multi/features/slam/simultaneous.py
"""

import os
from os import path


def fit():
    import numpy as np
    from pathlib import Path
    import autofit as af
    import autolens as al
    import autolens.plot as aplt

    """
    __Pipeline Functions__
    """

    def source_lp(
        settings_search,
        analysis_list,
        lens_bulge,
        source_bulge,
        redshift_lens,
        redshift_source,
        dataset_model,
        mass_centre=(0.0, 0.0),
        n_batch=50,
    ):
        mass = af.Model(al.mp.Isothermal)
        mass.centre = mass_centre

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=redshift_lens,
                    bulge=lens_bulge,
                    disk=None,
                    mass=mass,
                    shear=af.Model(al.mp.ExternalShear),
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=redshift_source,
                    bulge=source_bulge,
                ),
            ),
            dataset_model=dataset_model,
        )

        analysis_factor_list = []

        for analysis in analysis_list:
            analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
            analysis_factor_list.append(analysis_factor)

        factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

        search = af.Nautilus(
            name="source_lp[1]",
            **settings_search.search_dict,
            n_live=200,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(
            model=factor_graph.global_prior_model,
            analysis=factor_graph,
            **settings_search.fit_dict,
        )

    def source_pix_1(
        settings_search,
        analysis_list,
        source_lp_result,
        mesh_shape,
        dataset_model,
        n_batch=20,
    ):
        analysis_factor_list = []

        for i, analysis in enumerate(analysis_list):
            mass = al.util.chaining.mass_from(
                mass=source_lp_result[i].model.galaxies.lens.mass,
                mass_result=source_lp_result[i].model.galaxies.lens.mass,
                unfix_mass_centre=True,
            )

            if i > 0:
                mass.centre = model.galaxies.lens.mass.centre

            shear = source_lp_result[i].model.galaxies.lens.shear

            model = af.Collection(
                galaxies=af.Collection(
                    lens=af.Model(
                        al.Galaxy,
                        redshift=source_lp_result[i].instance.galaxies.lens.redshift,
                        bulge=source_lp_result[i].instance.galaxies.lens.bulge,
                        disk=source_lp_result[i].instance.galaxies.lens.disk,
                        mass=mass,
                        shear=shear,
                    ),
                    source=af.Model(
                        al.Galaxy,
                        redshift=source_lp_result[i].instance.galaxies.source.redshift,
                        pixelization=af.Model(
                            al.Pixelization,
                            mesh=af.Model(
                                al.mesh.RectangularAdaptDensity, shape=mesh_shape
                            ),
                            regularization=al.reg.Adapt,
                        ),
                    ),
                ),
                dataset_model=dataset_model,
            )

            analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
            analysis_factor_list.append(analysis_factor)

        factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

        search = af.Nautilus(
            name="source_pix[1]",
            **settings_search.search_dict,
            n_live=150,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(
            model=factor_graph.global_prior_model,
            analysis=factor_graph,
            **settings_search.fit_dict,
        )

    def source_pix_2(
        settings_search,
        analysis_list,
        source_lp_result,
        source_pix_result_1,
        mesh_shape,
        dataset_model,
        n_batch=20,
    ):
        analysis_factor_list = []

        for i, analysis in enumerate(analysis_list):
            model = af.Collection(
                galaxies=af.Collection(
                    lens=af.Model(
                        al.Galaxy,
                        redshift=source_lp_result[i].instance.galaxies.lens.redshift,
                        bulge=source_lp_result[i].instance.galaxies.lens.bulge,
                        disk=source_lp_result[i].instance.galaxies.lens.disk,
                        mass=source_pix_result_1[i].instance.galaxies.lens.mass,
                        shear=source_pix_result_1[i].instance.galaxies.lens.shear,
                    ),
                    source=af.Model(
                        al.Galaxy,
                        redshift=source_lp_result[i].instance.galaxies.source.redshift,
                        pixelization=af.Model(
                            al.Pixelization,
                            mesh=af.Model(
                                al.mesh.RectangularAdaptImage, shape=mesh_shape
                            ),
                            regularization=al.reg.Adapt,
                        ),
                    ),
                ),
                dataset_model=dataset_model,
            )

            analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
            analysis_factor_list.append(analysis_factor)

        factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

        search = af.Nautilus(
            name="source_pix[2]",
            **settings_search.search_dict,
            n_live=75,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(
            model=factor_graph.global_prior_model,
            analysis=factor_graph,
            **settings_search.fit_dict,
        )

    def light_lp(
        settings_search,
        analysis_list,
        source_result_for_lens,
        source_result_for_source,
        lens_bulge,
        dataset_model,
        n_batch=20,
    ):
        analysis_factor_list = []

        for i, analysis in enumerate(analysis_list):
            source = al.util.chaining.source_custom_model_from(
                result=source_result_for_source[i], source_is_model=False
            )

            model = af.Collection(
                galaxies=af.Collection(
                    lens=af.Model(
                        al.Galaxy,
                        redshift=source_result_for_lens[
                            i
                        ].instance.galaxies.lens.redshift,
                        bulge=lens_bulge,
                        disk=None,
                        mass=source_result_for_lens[i].instance.galaxies.lens.mass,
                        shear=source_result_for_lens[i].instance.galaxies.lens.shear,
                    ),
                    source=source,
                ),
                dataset_model=dataset_model,
            )

            analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
            analysis_factor_list.append(analysis_factor)

        factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

        search = af.Nautilus(
            name="light[1]",
            **settings_search.search_dict,
            n_live=250,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(
            model=factor_graph.global_prior_model,
            analysis=factor_graph,
            **settings_search.fit_dict,
        )

    def mass_total(
        settings_search,
        analysis_list,
        source_result_for_lens,
        source_result_for_source,
        light_result,
        dataset_model,
        n_batch=20,
    ):
        mass = af.Model(al.mp.PowerLaw)

        analysis_factor_list = []

        for i, analysis in enumerate(analysis_list):
            mass_i = al.util.chaining.mass_from(
                mass=mass,
                mass_result=source_result_for_lens[i].model.galaxies.lens.mass,
                unfix_mass_centre=True,
            )

            source = al.util.chaining.source_from(
                result=source_result_for_source[i],
            )

            model = af.Collection(
                galaxies=af.Collection(
                    lens=af.Model(
                        al.Galaxy,
                        redshift=source_result_for_lens[
                            i
                        ].instance.galaxies.lens.redshift,
                        bulge=light_result[i].instance.galaxies.lens.bulge,
                        disk=light_result[i].instance.galaxies.lens.disk,
                        mass=mass_i,
                        shear=source_result_for_lens[i].model.galaxies.lens.shear,
                    ),
                    source=source,
                ),
                dataset_model=dataset_model,
            )

            analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
            analysis_factor_list.append(analysis_factor)

        factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

        search = af.Nautilus(
            name="mass_total[1]",
            **settings_search.search_dict,
            n_live=150,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(
            model=factor_graph.global_prior_model,
            analysis=factor_graph,
            **settings_search.fit_dict,
        )

    """
    __Dataset__
    """
    dataset_waveband_list = ["g", "r"]
    pixel_scale_list = [0.12, 0.08]

    dataset_name = "lens_sersic"
    dataset_main_path = path.join("dataset", "multi", dataset_name)

    if not path.exists(dataset_main_path):
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "scripts/multi/simulator.py"],
            check=True,
        )

    dataset_list = []

    for dataset_waveband, pixel_scale in zip(dataset_waveband_list, pixel_scale_list):
        dataset = al.Imaging.from_fits(
            data_path=path.join(dataset_main_path, f"{dataset_waveband}_data.fits"),
            noise_map_path=path.join(
                dataset_main_path, f"{dataset_waveband}_noise_map.fits"
            ),
            psf_path=path.join(dataset_main_path, f"{dataset_waveband}_psf.fits"),
            pixel_scales=pixel_scale,
        )

        mask_radius = 3.0

        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=mask_radius,
        )

        dataset = dataset.apply_mask(mask=mask)

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=dataset.grid,
            sub_size_list=[4, 2, 1],
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0)],
        )

        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        dataset_list.append(dataset)

    settings_search = af.SettingsSearch(
        path_prefix=path.join("database", "scrape", "multi_analysis"),
        number_of_cores=1,
        session=None,
        info={"hi": "there"},
    )

    redshift_lens = 0.5
    redshift_source = 1.0

    mesh_pixels_yx = 28
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

    dataset_model = af.Model(al.DatasetModel)

    """
    __SLaM Pipeline__
    """
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    analysis_list = [
        al.AnalysisImaging(dataset=dataset, use_jax=True) for dataset in dataset_list
    ]

    source_lp_result = source_lp(
        settings_search=settings_search,
        analysis_list=analysis_list,
        lens_bulge=lens_bulge,
        source_bulge=source_bulge,
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
        dataset_model=dataset_model,
    )

    positions_likelihood = source_lp_result.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    )

    adapt_images_list = []

    for result in source_lp_result:
        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=result
        )
        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)
        adapt_images_list.append(adapt_images)

    analysis_list = [
        al.AnalysisImaging(
            dataset=result.max_log_likelihood_fit.dataset,
            adapt_images=adapt_images,
            positions_likelihood_list=[positions_likelihood],
            use_jax=True,
        )
        for result, adapt_images in zip(source_lp_result, adapt_images_list)
    ]

    source_pix_result_1 = source_pix_1(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_lp_result=source_lp_result,
        mesh_shape=mesh_shape,
        dataset_model=dataset_model,
    )

    adapt_images_list = []

    for result in source_pix_result_1:
        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=result
        )
        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)
        adapt_images_list.append(adapt_images)

    analysis_list = [
        al.AnalysisImaging(
            dataset=result.max_log_likelihood_fit.dataset,
            adapt_images=adapt_images,
            use_jax=True,
        )
        for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
    ]

    source_pix_result_2 = source_pix_2(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        mesh_shape=mesh_shape,
        dataset_model=dataset_model,
    )

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    analysis_list = [
        al.AnalysisImaging(
            dataset=result.max_log_likelihood_fit.dataset,
            adapt_images=adapt_images,
            raise_inversion_positions_likelihood_exception=False,
        )
        for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
    ]

    light_result = light_lp(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        lens_bulge=lens_bulge,
        dataset_model=dataset_model,
    )

    positions_likelihood = source_pix_result_1[0].positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    )

    analysis_list = [
        al.AnalysisImaging(
            dataset=result.max_log_likelihood_fit.dataset,
            adapt_images=adapt_images,
            positions_likelihood_list=[positions_likelihood],
        )
        for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
    ]

    mass_result = mass_total(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        dataset_model=dataset_model,
    )

    """
    __Database__

    Add results to database.
    """
    from autofit.database.aggregator import Aggregator

    database_file = "database_directory_multi_analysis.sqlite"

    try:
        os.remove(path.join("output", database_file))
    except FileNotFoundError:
        pass

    agg = Aggregator.from_database(database_file)
    agg.add_directory(
        directory=path.join("output", "database", "scrape", "multi_analysis")
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
        print(samples.log_likelihood_list)
        print(samples.log_likelihood_list[0])

    ml_vector = [
        samps.max_log_likelihood(as_instance=False) for samps in agg.values("samples")
    ]
    print(ml_vector, "\n\n")

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

    for search in agg.values("search"):
        print(f"\n****Search (search)****\n\n{search}")

    for samples_summary in agg.values("samples_summary"):
        instance = samples_summary.max_log_likelihood()
        print(f"\n****Max Log Likelihood (samples_summary)****\n\n{instance}")

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
        tracer = tracer_list[0]

        try:
            aplt.plot_array(array=tracer.convergence_2d_from(grid=grid))
            aplt.plot_array(array=tracer.potential_2d_from(grid=grid))

        except al.exc.ProfileException:
            print(
                "TracerAgg with linear light profiles raises correct ProfileException"
            )

        print("TracerAgg Checked")

    imaging_agg = al.agg.ImagingAgg(aggregator=agg)
    imaging_gen = imaging_agg.dataset_gen_from()

    for dataset_list in imaging_gen:
        dataset = dataset_list[0]

        aplt.plot_array(array=dataset.data)

        print("ImagingAgg Checked")

    fit_agg = al.agg.FitImagingAgg(
        aggregator=agg,
    )
    fit_imaging_gen = fit_agg.max_log_likelihood_gen_from()

    for fit_list in fit_imaging_gen:
        fit = fit_list[0]

        print(fit.dataset_model.grid_offset.grid_offset_1)

        print(fit_list[0].dataset_model.grid_offset.grid_offset_1)
        print(fit_list[1].dataset_model.grid_offset)

        print("FitImagingAgg Checked")


if __name__ == "__main__":
    fit()
