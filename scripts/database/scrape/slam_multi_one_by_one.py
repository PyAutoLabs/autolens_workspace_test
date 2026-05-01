"""
Database: SLaM Multi One-by-One
===============================

Runs a SLaM pipeline on a primary dataset, then fits secondary multi-wavelength datasets
with fixed mass, and scrapes results to a database for verification.

Based on autolens_workspace/scripts/multi/features/slam/independent.py
"""


def fit():
    import pytest

    import numpy as np
    import os
    from os import path
    from pathlib import Path

    import autofit as af
    import autolens as al
    import autolens.plot as aplt

    """
    __Pipeline Functions__
    """

    def source_lp(
        settings_search,
        analysis,
        lens_bulge,
        source_bulge,
        redshift_lens,
        redshift_source,
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
        analysis,
        source_lp_result,
        mesh_shape,
        n_batch=20,
    ):
        mass = al.util.chaining.mass_from(
            mass=source_lp_result.model.galaxies.lens.mass,
            mass_result=source_lp_result.model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.lens.redshift,
                    bulge=source_lp_result.instance.galaxies.lens.bulge,
                    disk=source_lp_result.instance.galaxies.lens.disk,
                    mass=mass,
                    shear=source_lp_result.model.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.source.redshift,
                    pixelization=af.Model(
                        al.Pixelization,
                        mesh=af.Model(
                            al.mesh.RectangularAdaptDensity, shape=mesh_shape
                        ),
                        regularization=al.reg.Adapt,
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
        analysis,
        source_lp_result,
        source_pix_result_1,
        mesh_shape,
        n_batch=20,
    ):
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
                        mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                        regularization=al.reg.Adapt,
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
        analysis,
        source_result_for_lens,
        source_result_for_source,
        lens_bulge,
        n_batch=20,
    ):
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
        analysis,
        source_result_for_lens,
        source_result_for_source,
        light_result,
        n_batch=20,
    ):
        mass = af.Model(al.mp.PowerLaw)

        mass = al.util.chaining.mass_from(
            mass=mass,
            mass_result=source_result_for_lens.model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )

        source = al.util.chaining.source_from(result=source_result_for_source)

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                    bulge=light_result.instance.galaxies.lens.bulge,
                    disk=light_result.instance.galaxies.lens.disk,
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

    def source_lp_secondary(
        settings_search,
        analysis,
        light_result,
        mass_result,
        source_bulge,
        dataset_model,
        redshift_lens=0.5,
        redshift_source=1.0,
        n_batch=50,
    ):
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=redshift_lens,
                    bulge=light_result.instance.galaxies.lens.bulge,
                    disk=None,
                    point=light_result.instance.galaxies.lens.point,
                    mass=mass_result.instance.galaxies.lens.mass,
                    shear=mass_result.instance.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=redshift_source,
                    bulge=source_bulge,
                ),
            ),
            dataset_model=dataset_model,
        )

        search = af.Nautilus(
            name="source_lp[1]",
            **settings_search.search_dict,
            n_live=200,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    def source_pix_1_secondary(
        settings_search,
        analysis,
        source_lp_result,
        mesh_shape,
        dataset_model,
        n_batch=20,
    ):
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.lens.redshift,
                    bulge=source_lp_result.instance.galaxies.lens.bulge,
                    disk=source_lp_result.instance.galaxies.lens.disk,
                    mass=source_lp_result.instance.galaxies.lens.mass,
                    shear=source_lp_result.instance.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.source.redshift,
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

        search = af.Nautilus(
            name="source_pix[1]",
            **settings_search.search_dict,
            n_live=150,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    def source_pix_2_secondary(
        settings_search,
        analysis,
        source_lp_result,
        source_pix_result_1,
        mesh_shape,
        dataset_model,
        n_batch=20,
    ):
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
                        mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                        regularization=al.reg.Adapt,
                    ),
                ),
            ),
            dataset_model=dataset_model,
        )

        search = af.Nautilus(
            name="source_pix[2]",
            **settings_search.search_dict,
            n_live=75,
            n_batch=n_batch,
            n_like_max=300,
        )

        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    """
    __Dataset + Masking (Primary)__
    """
    dataset_name = "lens_sersic"
    dataset_main_path = Path("dataset", "multi", dataset_name)

    if not dataset_main_path.exists():
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "scripts/multi/simulator.py"],
            check=True,
        )

    dataset_waveband = "g"

    dataset = al.Imaging.from_fits(
        data_path=Path(dataset_main_path, f"{dataset_waveband}_data.fits"),
        noise_map_path=Path(dataset_main_path, f"{dataset_waveband}_noise_map.fits"),
        psf_path=Path(dataset_main_path, f"{dataset_waveband}_psf.fits"),
        pixel_scales=0.08,
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

    settings_search = af.SettingsSearch(
        path_prefix=path.join("database", "scrape", "slam_multi_one_by_one"),
        number_of_cores=1,
        session=None,
        info={"hi": "there"},
    )

    redshift_lens = 0.5
    redshift_source = 1.0

    mesh_pixels_yx = 28
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

    """
    __SLaM Pipeline (Primary)__
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

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    source_lp_result = source_lp(
        settings_search=settings_search,
        analysis=analysis,
        lens_bulge=lens_bulge,
        source_bulge=source_bulge,
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
    )

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
        use_jax=True,
    )

    source_pix_result_1 = source_pix_1(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        mesh_shape=mesh_shape,
    )

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    source_pix_result_2 = source_pix_2(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        mesh_shape=mesh_shape,
    )

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    light_result = light_lp(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        lens_bulge=lens_bulge,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_pix_result_2.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    mass_result = mass_total(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
    )

    """
    __Secondary Dataset Fits__
    """
    dataset_waveband_list = ["r"]
    pixel_scale_list = [0.12]

    for dataset_waveband, pixel_scale in zip(dataset_waveband_list, pixel_scale_list):

        dataset = al.Imaging.from_fits(
            data_path=Path(dataset_main_path, f"{dataset_waveband}_data.fits"),
            noise_map_path=Path(
                dataset_main_path, f"{dataset_waveband}_noise_map.fits"
            ),
            psf_path=Path(dataset_main_path, f"{dataset_waveband}_psf.fits"),
            pixel_scales=pixel_scale,
        )

        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=mask_radius,
        )

        dataset = dataset.apply_mask(mask=mask)

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=dataset.grid,
            sub_size_list=[4, 2, 1],
            radial_list=[0.1, 0.3],
            centre_list=[(0.0, 0.0)],
        )

        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        settings_search_secondary = af.SettingsSearch(
            path_prefix=path.join("database", "scrape", "slam_multi_one_by_one"),
            unique_tag=f"{dataset_name}_data_{dataset_waveband}",
            number_of_cores=1,
            session=None,
            info={"hi": "there"},
        )

        dataset_model = af.Model(al.DatasetModel)

        dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )
        dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )

        centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
        centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

        total_gaussians = 20
        gaussian_per_basis = 1

        log10_sigma_list = np.linspace(-3, np.log10(1.0), total_gaussians)

        bulge_gaussian_list = []

        for j in range(gaussian_per_basis):
            gaussian_list = af.Collection(
                af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
            )

            for i, gaussian in enumerate(gaussian_list):
                gaussian.centre.centre_0 = centre_0
                gaussian.centre.centre_1 = centre_1
                gaussian.ell_comps = gaussian_list[0].ell_comps
                gaussian.sigma = 10 ** log10_sigma_list[i]

            bulge_gaussian_list += gaussian_list

        source_bulge = af.Model(
            al.lp_basis.Basis,
            profile_list=bulge_gaussian_list,
        )

        analysis = al.AnalysisImaging(dataset=dataset)

        secondary_source_lp_result = source_lp_secondary(
            settings_search=settings_search_secondary,
            analysis=analysis,
            light_result=light_result,
            mass_result=mass_result,
            source_bulge=source_bulge,
            dataset_model=dataset_model,
            redshift_lens=redshift_lens,
            redshift_source=redshift_source,
        )

        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=secondary_source_lp_result
        )

        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

        dataset_model.grid_offset.grid_offset_0 = (
            secondary_source_lp_result.instance.dataset_model.grid_offset[0]
        )
        dataset_model.grid_offset.grid_offset_1 = (
            secondary_source_lp_result.instance.dataset_model.grid_offset[1]
        )

        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            raise_inversion_positions_likelihood_exception=False,
        )

        secondary_source_pix_result_1 = source_pix_1_secondary(
            settings_search=settings_search_secondary,
            analysis=analysis,
            source_lp_result=secondary_source_lp_result,
            mesh_shape=mesh_shape,
            dataset_model=dataset_model,
        )

        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=secondary_source_pix_result_1
        )

        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
        )

        secondary_multi_result = source_pix_2_secondary(
            settings_search=settings_search_secondary,
            analysis=analysis,
            source_lp_result=secondary_source_lp_result,
            source_pix_result_1=secondary_source_pix_result_1,
            mesh_shape=mesh_shape,
            dataset_model=dataset_model,
        )

    """
    __Database__

    Add results to database.
    """
    from autofit.database.aggregator import Aggregator

    database_file = "database_directory_slam_multi_one_by_one.sqlite"

    try:
        os.remove(path.join("output", database_file))
    except FileNotFoundError:
        pass

    agg = Aggregator.from_database(database_file)
    agg.add_directory(
        directory=path.join("output", "database", "scrape", "slam_multi_one_by_one")
    )

    assert len(agg) > 0

    print("\n\n***********************")
    print("****RESULTS TESTING****")
    print("***********************\n")

    for samples in agg.values("samples"):
        print(samples.parameter_lists[0])

    mp_instances = [samps.median_pdf() for samps in agg.values("samples")]
    print(mp_instances)

    print("\n\n***********************")
    print("***AGG MODULE TESTING***")
    print("***********************\n\n")

    agg_mass = agg.query(agg.search.name == "mass_total[1]")

    tracer_agg = al.agg.TracerAgg(aggregator=agg_mass)
    tracer_gen = tracer_agg.max_log_likelihood_gen_from()

    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

    for tracer_list in tracer_gen:
        tracer = tracer_list[0]

        try:
            aplt.plot_array(array=tracer.convergence_2d_from(grid=grid))
        except al.exc.ProfileException:
            print(
                "TracerAgg with linear light profiles raises correct ProfileException"
            )

        assert tracer.galaxies[0].mass.einstein_radius > 0.0
        print("TracerAgg Checked")


if __name__ == "__main__":
    fit()
