"""
Simulator: Multi-Wavelength
===========================
Simulates two-band (g and r) `Imaging` datasets of a strong lens for use by
the multi-wavelength JAX likelihood function tests in this folder.

Each band shares the same lens mass but has its own source intensity,
written to `dataset/multi/lens_sersic/`.
"""
# ENV: jax full_datasets
# JAX likelihood functions test JIT compilation; need JAX enabled
# and full-size datasets.

from os import path
import autolens as al
import autolens.plot as aplt

dataset_path = path.join("dataset", "multi", "lens_sersic")

grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.1)

psf = al.Convolver.from_gaussian(
    shape_native=(21, 21), sigma=0.1, pixel_scales=grid.pixel_scales, normalize=True
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

# g-band: source intensity 0.3
source_g = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.8,
        sersic_index=1.5,
    ),
)

# r-band: source intensity 0.5
source_r = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.5,
        effective_radius=0.8,
        sersic_index=1.5,
    ),
)

for band, source_galaxy in [("g", source_g), ("r", source_r)]:
    simulator = al.SimulatorImaging(
        exposure_time=2000.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
        noise_seed=1 if band == "g" else 2,
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    al.output_to_fits(
        values=dataset.data.native,
        file_path=path.join(dataset_path, f"{band}_data.fits"),
        overwrite=True,
    )
    al.output_to_fits(
        values=dataset.psf.kernel.native,
        file_path=path.join(dataset_path, f"{band}_psf.fits"),
        overwrite=True,
    )
    al.output_to_fits(
        values=dataset.noise_map.native,
        file_path=path.join(dataset_path, f"{band}_noise_map.fits"),
        overwrite=True,
    )
    al.output_to_json(
        obj=tracer,
        file_path=path.join(dataset_path, f"{band}_tracer.json"),
    )
    print(f"Saved {band}-band dataset")

print("Multi-wavelength datasets written to", dataset_path)
