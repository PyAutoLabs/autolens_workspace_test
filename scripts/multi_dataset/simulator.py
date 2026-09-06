"""
Simulator: Multi-Wavelength
===========================
Simulates two-band (g and r) `Imaging` datasets of a strong lens for use by
the multi-wavelength JAX likelihood function tests in this folder.

Each band shares the same lens mass but has its own source intensity.

Two datasets are written:

 - `dataset/multi_dataset/lens_sersic/` — a mass-only lens (no lens light), read by the
   scripts whose model has no lens light component.
 - `dataset/multi_dataset/lens_sersic_light/` — the same system plus a lens `Sersic` light
   profile, read by the scripts whose model carries a lens light (MGE) component, so that
   every model fits data that contains what it models.

The grid is 80 x 80 at 0.2" per pixel with a 0.6"-FWHM Gaussian PSF, sized so that the
models fitted to these datasets resolve the data they fit.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

from os import path
import autolens as al
import autolens.plot as aplt

dataset_path = path.join("dataset", "multi_dataset", "lens_sersic")
dataset_light_path = path.join("dataset", "multi_dataset", "lens_sersic_light")

grid = al.Grid2D.uniform(shape_native=(80, 80), pixel_scales=0.2)

psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.25, pixel_scales=grid.pixel_scales, normalize=True
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

"""
The lens galaxy of the `lens_sersic_light` dataset: the same mass and shear, plus a `Sersic`
light profile whose intensity varies per band. This is the dataset read by the scripts whose
model includes a lens light (MGE) component.
"""


def lens_galaxy_light_from(intensity):
    return al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            intensity=intensity,
            effective_radius=0.8,
            sersic_index=3.0,
        ),
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

def simulate_and_output(output_path, band, lens, source_galaxy):
    simulator = al.SimulatorImaging(
        exposure_time=2000.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
        noise_seed=1 if band == "g" else 2,
    )
    tracer = al.Tracer(galaxies=[lens, source_galaxy])
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    al.output_to_fits(
        values=dataset.data.native,
        file_path=path.join(output_path, f"{band}_data.fits"),
        overwrite=True,
    )
    al.output_to_fits(
        values=dataset.psf.kernel.native,
        file_path=path.join(output_path, f"{band}_psf.fits"),
        overwrite=True,
    )
    al.output_to_fits(
        values=dataset.noise_map.native,
        file_path=path.join(output_path, f"{band}_noise_map.fits"),
        overwrite=True,
    )
    al.output_to_json(
        obj=tracer,
        file_path=path.join(output_path, f"{band}_tracer.json"),
    )
    print(f"Saved {band}-band dataset to {output_path}")


for band, source_galaxy in [("g", source_g), ("r", source_r)]:
    simulate_and_output(dataset_path, band, lens_galaxy, source_galaxy)

# The lens-light dataset: lens `Sersic` intensity 0.6 (g) / 1.0 (r).
for band, source_galaxy, lens_intensity in [
    ("g", source_g, 0.6),
    ("r", source_r, 1.0),
]:
    simulate_and_output(
        dataset_light_path,
        band,
        lens_galaxy_light_from(intensity=lens_intensity),
        source_galaxy,
    )

print("Multi-wavelength datasets written to", dataset_path, "and", dataset_light_path)
