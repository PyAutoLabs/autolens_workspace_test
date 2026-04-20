"""
Simulator: Interferometer Double Source Plane
=============================================

This script simulates an `Interferometer` dataset of a strong gravitational lens
with two source planes (double source plane, DSPL) used by the JAX likelihood
function tests in this folder.

It saves the dataset to `dataset/interferometer/dspl/`, overwriting existing files.
The `rectangular_dspl.py` JAX test loads from this fixed dataset so that its
likelihood values are deterministic and can be numerically asserted.

Mirrors `imaging/simulator_dspl.py` but uses `SimulatorInterferometer` and writes
the interferometer file layout (data.fits with real+imag stacked, noise_map.fits
with real+imag stacked, uv_wavelengths.fits, positions.json).

__Dataset__

The simulation uses the SMA uv_wavelengths stored in
`dataset/interferometer/uv_wavelengths/sma.fits` (190 visibilities).

__Model__

 - Lens 0 at z=0.5: `Sersic` bulge + disk, `Isothermal` mass + `ExternalShear`.
 - Lens 1 at z=1.0: `Isothermal` mass (acts as intermediate lens for source).
 - Source at z=2.0: `SersicCore` light profile.

The redshifts match `imaging/simulator_dspl.py`.
"""

from os import path
from pathlib import Path
import numpy as np
import autolens as al

dataset_type = "interferometer"
dataset_name = "dspl"

dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Grid__

Real-space grid on which the lens image is evaluated before Fourier transforming.
"""
grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

"""
__UV Wavelengths__

Load the SMA uv-coverage stored in the repository's shared uv_wavelengths folder.
"""
uv_wavelengths = al.ndarray_via_fits_from(
    file_path=path.join("dataset", "interferometer", "uv_wavelengths", "sma.fits"),
    hdu=0,
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

Three-galaxy system: two lenses + one source (double source plane).
Redshifts mirror `imaging/simulator_dspl.py`.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=2.0,
        effective_radius=1.6,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.0, gamma_2=0.0),
)

lens_galaxy_1 = al.Galaxy(
    redshift=1.0,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=0.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.SersicCore(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, lens_galaxy_1, source_galaxy])

"""
__Simulate__
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
__Output__

Write the dataset to `dataset/interferometer/dspl/` using the interferometer file
layout: data.fits (real+imag stacked), noise_map.fits (real+imag stacked),
uv_wavelengths.fits, positions.json.
"""
Path(dataset_path).mkdir(parents=True, exist_ok=True)

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

Compute and save the multiple-image positions from the source at z=2.0.
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
