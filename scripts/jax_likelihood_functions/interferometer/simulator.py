"""
Simulator: Interferometer
=========================

This script simulates an `Interferometer` dataset of a strong gravitational lens
used by the JAX likelihood function tests in this folder.

It saves the dataset to `dataset/interferometer/simple/`, overwriting the existing
files. The mge.py and rectangular.py JAX tests load from this fixed dataset so that
their likelihood values are deterministic and can be numerically asserted.

__Dataset__

The simulation uses the SMA uv_wavelengths stored in `dataset/interferometer/uv_wavelengths/sma.fits`
(190 visibilities), giving fast and deterministic interferometer likelihood evaluations.

__Model__

 - Lens: `Isothermal` mass + `ExternalShear`.
 - Source: `SersicCore` light profile (cored, avoiding over-sampling issues).
"""

from os import path
from pathlib import Path
import autolens as al
import autolens.plot as aplt

dataset_type = "interferometer"
dataset_name = "simple"

dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Grid__

Real-space grid on which the lens image is evaluated before Fourier transforming.
We use 256 x 256 at 0.1" per pixel — consistent with the mge.py JAX test.
"""
grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

"""
__UV Wavelengths__

Load the SMA uv-coverage stored in the repository's shared uv_wavelengths folder (190 baselines).
"""
uv_wavelengths = al.ndarray_via_fits_from(
    file_path=path.join("dataset", "interferometer", "uv_wavelengths", "sma.fits"), hdu=0
)

print(f"Total visibilities: {uv_wavelengths.shape[0]}")

"""
__Simulator__
"""
simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=al.TransformerDFT,
    noise_seed=1,
)

"""
__Ray Tracing__

Lens: Isothermal + ExternalShear.
Source: SersicCore centred slightly off-axis to produce clear multiple images.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Simulate__
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
__Output__

Overwrite the dataset files in `dataset/interferometer/simple/`.
"""
Path(dataset_path).mkdir(parents=True, exist_ok=True)

import numpy as np

al.output_to_fits(
    values=np.stack([dataset.data.real, dataset.data.imag], axis=-1),
    file_path=path.join(dataset_path, "data.fits"),
    overwrite=True,
)
al.output_to_fits(
    values=np.stack([dataset.noise_map.real, dataset.noise_map.imag], axis=-1),
    file_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)
al.output_to_fits(
    values=dataset.uv_wavelengths,
    file_path=path.join(dataset_path, "uv_wavelengths.fits"),
    overwrite=True,
)

"""
__Multiple Image Positions__

Compute and save the multiple-image positions so that mge.py can use a
PositionsLH penalty.
"""
solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

al.output_to_json(
    file_path=path.join(dataset_path, "positions.json"),
    obj=positions,
)

print(f"Saved {len(positions)} multiple-image positions to positions.json")
print("Dataset written to", dataset_path)
